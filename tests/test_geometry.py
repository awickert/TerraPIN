#! /usr/bin/env python
"""
Firm tests for TerraPIN's polygon-algebra geometry (terrapin.geometry).

These are the seed of the test suite for the Shapely-based rewrite. They pin
down the invariants the model must satisfy no matter how it is implemented:

  * the angle-of-repose wall bends correctly at every layer contact;
  * eroded per-material volumes match analytic triangles/trapezoids;
  * materials never overlap, and solid + void fills the domain (conservation);
  * conservation survives a long random sequence of incisions -- the
    "millions of calls" robustness concern, in miniature.

Run in the dedicated environment, e.g.:  conda run -n terrapin python -m pytest
"""
import numpy as np
import pytest

pytest.importorskip("shapely")
pytest.importorskip("scipy")
from shapely.geometry import box, Polygon
from shapely.ops import unary_union

from terrapin.geometry import (repose_wall, eroded_wedge, incise, aggrade,
                               colluvial_pile, position_repose_surface, widen,
                               _deposit_below_repose, _offset_bracket)

# --- A concrete two-material valley: bedrock below -10 m, alluvium above. ---
TAN75 = np.tan(np.deg2rad(75.0))
TAN32 = np.tan(np.deg2rad(32.0))
STACK = [(-10.0, 75.0, "bedrock"), (0.0, 32.0, "alluvium")]
DOMAIN = box(-100.0, -50.0, 0.0, 5.0)


def fresh_bodies():
    return {"bedrock": box(-100.0, -50.0, 0.0, -10.0),
            "alluvium": box(-100.0, -10.0, 0.0, 0.0)}


def test_repose_wall_single_material():
    # A wall that only climbs bedrock (surface taken as the bedrock top).
    wall = repose_wall(-20.0, [(-10.0, 75.0, "bedrock")])
    assert wall[0] == (0.0, -20.0)
    assert np.isclose(wall[-1][0], -10.0 / TAN75)
    assert np.isclose(wall[-1][1], -10.0)


def test_repose_wall_bends_at_contact():
    wall = repose_wall(-20.0, STACK)
    zs = [p[1] for p in wall]
    xs = [p[0] for p in wall]
    assert zs == sorted(zs)                     # climbs monotonically in z
    assert xs == sorted(xs, reverse=True)       # marches up-valley (x decreasing)
    # Vertex at the bedrock/alluvium contact: 10 m of bedrock at 75 deg.
    assert np.isclose(wall[1][0], -10.0 / TAN75)
    assert np.isclose(wall[1][1], -10.0)
    # Vertex at the surface: bedrock run plus 10 m of alluvium at 32 deg.
    assert np.isclose(wall[2][0], -10.0 / TAN75 - 10.0 / TAN32)
    assert np.isclose(wall[2][1], 0.0)


def test_incision_above_surface_removes_nothing():
    _, eroded = incise(fresh_bodies(), 3.0, STACK)   # channel above ground
    assert np.isclose(sum(eroded.values()), 0.0)
    assert eroded_wedge(3.0, STACK).is_empty


@pytest.mark.parametrize("z_ch,expected_bedrock", [
    (-20.0, 0.5 * (10.0 / TAN75) * 10.0),   # cuts 10 m into bedrock
    (-15.0, 0.5 * (5.0 / TAN75) * 5.0),     # cuts 5 m into bedrock
    (-10.0, 0.0),                           # just reaches the bedrock top
    (-5.0, 0.0),                            # stays within the alluvium
])
def test_eroded_bedrock_matches_analytic(z_ch, expected_bedrock):
    _, eroded = incise(fresh_bodies(), z_ch, STACK)
    assert np.isclose(eroded["bedrock"], expected_bedrock)


def test_eroded_alluvium_matches_analytic():
    # Removed alluvium is a trapezoid over the 10 m of alluvium thickness.
    _, eroded = incise(fresh_bodies(), -20.0, STACK)
    width_bottom = 10.0 / TAN75                    # wall x at the contact
    width_top = 10.0 / TAN75 + 10.0 / TAN32        # wall x at the surface
    expected = 0.5 * (width_bottom + width_top) * 10.0
    assert np.isclose(eroded["alluvium"], expected)


def test_no_overlap_and_domain_conservation():
    new, _ = incise(fresh_bodies(), -20.0, STACK)
    overlap = new["bedrock"].intersection(new["alluvium"]).area
    assert np.isclose(overlap, 0.0)
    solids = unary_union(list(new.values()))
    void = DOMAIN.difference(solids).area
    assert np.isclose(solids.area + void, DOMAIN.area)


def test_conservation_under_random_incision_sequence():
    # Robustness in miniature: hammer incise() and require, at every step, that
    # the volume reported eroded equals the drop in total body area.
    rng = np.random.default_rng(0)
    bodies = fresh_bodies()
    for _ in range(200):
        z_ch = float(rng.uniform(-45.0, -1.0))
        before = sum(p.area for p in bodies.values())
        bodies, eroded = incise(bodies, z_ch, STACK)
        after = sum(p.area for p in bodies.values())
        assert np.isclose(before - after, sum(eroded.values()), atol=1e-6)


# ------------------------------- aggradation -------------------------------

@pytest.mark.parametrize("z_fill,expected", [
    (-15.0, 0.5 * (5.0 / TAN75) * 5.0),     # fill stays within the bedrock wedge
    (-10.0, 0.5 * (10.0 / TAN75) * 10.0),   # fill exactly to the bedrock top
])
def test_aggrade_fills_wedge_to_analytic(z_fill, expected):
    incised, _ = incise(fresh_bodies(), -20.0, STACK)
    _, deposited = aggrade(incised, z_fill, DOMAIN)
    assert np.isclose(deposited, expected)


def test_aggrade_surface_is_flat_at_fill_level():
    incised, _ = incise(fresh_bodies(), -20.0, STACK)
    new, _ = aggrade(incised, -15.0, DOMAIN)
    assert np.isclose(new["alluvium_fill"].bounds[3], -15.0)   # top of the fill


def test_aggrade_deposit_nests_without_overlap():
    incised, _ = incise(fresh_bodies(), -20.0, STACK)
    new, deposited = aggrade(incised, -12.0, DOMAIN)
    fill = new["alluvium_fill"]
    for name in ("bedrock", "alluvium"):
        assert np.isclose(fill.intersection(new[name]).area, 0.0)
    # The void shrinks by exactly the deposited volume.
    void_before = DOMAIN.difference(unary_union([incised["bedrock"],
                                                 incised["alluvium"]])).area
    void_after = DOMAIN.difference(unary_union([new["bedrock"], new["alluvium"],
                                               fill])).area
    assert np.isclose(void_before - void_after, deposited)


# -------------------------------- colluvium --------------------------------
# The colluvial pile fits a talus of prescribed (fluffed) volume against the
# valley wall by positioning its repose surface -- the PLIC / area-conservation
# problem. Two wall shapes: a single 75-deg segment, and a piecewise 75/32 wall.

STACK_1 = [(0.0, 75.0, "bedrock")]           # incising leaves one straight wall
VOID_1 = eroded_wedge(-20.0, STACK_1)        # capacity ~53.6
VOID_2 = eroded_wedge(-20.0, STACK)          # piecewise wall, capacity ~120.2
LAMBDA = 0.35
METHODS = ["brent", "bisect", "secant", "analytic"]


def test_area_below_repose_is_monotonic():
    slope = np.tan(np.deg2rad(20.0))
    lo, hi = _offset_bracket(VOID_2, slope)
    areas = [_deposit_below_repose(c, VOID_2, slope).area
             for c in np.linspace(lo, hi, 40)]
    assert np.all(np.diff(areas) >= -1e-9)


@pytest.mark.parametrize("void", [VOID_1, VOID_2])
@pytest.mark.parametrize("method", METHODS)
def test_colluvium_conserves_volume_all_methods(method, void):
    eroded = 20.0
    pile, overflow = colluvial_pile(eroded, void, alpha_c=20.0,
                                    lambda_p=LAMBDA, method=method)
    assert np.isclose(overflow, 0.0)
    assert np.isclose(pile.area, eroded / (1.0 - LAMBDA), atol=1e-8)


def test_colluvium_methods_agree():
    eroded = 25.0
    areas = [colluvial_pile(eroded, VOID_2, 20.0, LAMBDA, method=m)[0].area
             for m in METHODS]
    assert np.allclose(areas, areas[0], atol=1e-8)


def test_colluvium_free_surface_at_repose_angle():
    # The gentlest edge of the deposit is its free (repose) surface.
    pile, _ = colluvial_pile(20.0, VOID_1, alpha_c=20.0, lambda_p=LAMBDA)
    xs, zs = pile.exterior.xy
    slopes = []
    for i in range(len(xs) - 1):
        dx, dz = xs[i + 1] - xs[i], zs[i + 1] - zs[i]
        if abs(dx) > 1e-9 and abs(dz) > 1e-9:
            slopes.append(abs(dz / dx))
    assert np.isclose(min(slopes), np.tan(np.deg2rad(20.0)), atol=1e-6)


def test_colluvium_overflow_becomes_sediment():
    # More colluvium than the void can hold: it fills, and the rest overflows
    # (delivered to the channel as sediment in the coupled setting).
    cap = VOID_1.area
    eroded = (cap + 10.0) * (1.0 - LAMBDA)      # fluffed volume exceeds capacity
    pile, overflow = colluvial_pile(eroded, VOID_1, 20.0, LAMBDA)
    assert np.isclose(pile.area, cap)
    assert np.isclose(overflow, 10.0)
    assert np.isclose(pile.area + overflow, eroded / (1.0 - LAMBDA))


def test_halfplane_clip_is_robust_across_offsets():
    # Regression: the repose half-plane must stay a valid polygon even when the
    # line dips well below the void, or GEOS raises a TopologyException.
    z_ch = -20.0
    void = Polygon([(0, z_ch), (-12.0, z_ch), (-17.359, 0.0), (0.0, 0.0)])
    slope = np.tan(np.deg2rad(20.0))
    lo, hi = _offset_bracket(void, slope)
    for c in np.linspace(lo - 1.0, hi + 1.0, 60):
        area = _deposit_below_repose(c, void, slope).area
        assert -1e-9 <= area <= void.area + 1e-9


def test_colluvium_matches_andy_closed_form():
    # Oracle: Andy's beta-corrected talus height for a straight wall. A talus
    # rests on a flat channel floor, which the pointed eroded_wedge lacks (the
    # channel has no width yet), so we build a floor-bearing void here.
    z_ch, alpha_c, beta = -20.0, 20.0, 75.0
    A = 10.0                                     # deposit area, small -> it fits
    tan_a, tan_b = np.tan(np.deg2rad(alpha_c)), np.tan(np.deg2rad(beta))
    height = np.sqrt(2 * A * tan_a / (1 - tan_a / tan_b))
    base = 2 * A / height
    w = base + 2.0                               # floor wide enough for the toe
    x_top = -w - (0 - z_ch) / tan_b              # wall meets the ground surface
    void = Polygon([(0.0, z_ch), (-w, z_ch), (x_top, 0.0), (0.0, 0.0)])
    pile, _ = colluvial_pile(A * (1 - LAMBDA), void, alpha_c, LAMBDA)
    assert np.isclose(pile.bounds[3] - z_ch, height, atol=1e-6)


def test_colluvium_from_incision_end_to_end():
    _, eroded = incise(fresh_bodies(), -25.0, STACK)
    void = eroded_wedge(-25.0, STACK)
    pile, overflow = colluvial_pile(eroded["bedrock"], void, 20.0, LAMBDA)
    assert pile.area > eroded["bedrock"]                        # fluffed up
    assert np.isclose(pile.area + overflow,
                      eroded["bedrock"] / (1.0 - LAMBDA))


# --------------------------------- widening --------------------------------
# Widening is the same notch cut as incision, but undercut bedrock piles as
# colluvium on the newly-beveled strath floor (the caller supplies the new
# half-width; TerraPIN does the geometry and mass balance).

def test_incise_floor_half_width_backward_compatible():
    # Default floor_half_width=0 reproduces the old pointed-notch incision.
    _, e0 = incise(fresh_bodies(), -20.0, STACK)
    _, e1 = incise(fresh_bodies(), -20.0, STACK, floor_half_width=0.0)
    assert e0 == e1


def test_widen_opens_flat_strath_floor():
    # Widening to half-width w adds a flat strath slab of width w x depth.
    pointed = eroded_wedge(-20.0, STACK, 0.0).area
    floored = eroded_wedge(-20.0, STACK, 8.0).area
    assert np.isclose(floored - pointed, 8.0 * 20.0)


def test_widen_stores_colluvium_on_floor_and_conserves():
    incised, _ = incise(fresh_bodies(), -20.0, STACK, floor_half_width=0.0)
    new, bal = widen(incised, -20.0, 8.0, STACK, alpha_c=20.0, lambda_p=LAMBDA)
    # colluvium is fluffed bedrock: stored + overflow == fluffed eroded volume
    assert np.isclose(bal["colluvium_stored"] + bal["colluvium_overflow"],
                      bal["bedrock_eroded"] / (1.0 - LAMBDA))
    assert np.isclose(new["colluvium"].bounds[1], -20.0)   # rests on the strath
    assert np.isclose(bal["sediment_out"],
                      bal["colluvium_overflow"] + bal["alluvium_eroded"])


def test_widen_reports_correct_per_material_volumes():
    # Independent oracle: incise reports eroded volume by body name, so widen's
    # bedrock/alluvium split must match it (catches piling alluvium as colluvium).
    incised, _ = incise(fresh_bodies(), -20.0, STACK, floor_half_width=0.0)
    _, eroded_ref = incise(incised, -20.0, STACK, floor_half_width=8.0)
    _, bal = widen(incised, -20.0, 8.0, STACK, alpha_c=20.0, lambda_p=LAMBDA)
    assert np.isclose(bal["bedrock_eroded"], eroded_ref["bedrock"])
    assert np.isclose(bal["alluvium_eroded"], eroded_ref["alluvium"])
    assert bal["alluvium_eroded"] > 0.0                    # alluvium really removed


def test_widen_alluvium_carried_off_only_bedrock_piles():
    incised, _ = incise(fresh_bodies(), -20.0, STACK, floor_half_width=0.0)
    _, bal = widen(incised, -20.0, 8.0, STACK, alpha_c=20.0, lambda_p=LAMBDA)
    assert bal["alluvium_eroded"] > 0.0                    # alluvium was removed
    # only bedrock contributes to the pile; alluvium goes straight to sediment
    assert np.isclose(bal["colluvium_stored"] + bal["colluvium_overflow"],
                      bal["bedrock_eroded"] / (1.0 - LAMBDA))


def test_widen_overflow_when_valley_too_tight():
    # In an all-bedrock valley the fluffed colluvium (x1.54) always exceeds the
    # notch it came from, so some overflows to sediment.
    bodies = {"bedrock": box(-100.0, -50.0, 0.0, 0.0)}
    incised, _ = incise(bodies, -20.0, STACK_1, floor_half_width=0.0)
    _, bal = widen(incised, -20.0, 8.0, STACK_1, alpha_c=20.0, lambda_p=LAMBDA)
    assert bal["colluvium_overflow"] > 0.0
    assert np.isclose(bal["colluvium_stored"] + bal["colluvium_overflow"],
                      bal["bedrock_eroded"] / (1.0 - LAMBDA))
    assert np.isclose(bal["sediment_out"], bal["colluvium_overflow"])   # no alluvium
