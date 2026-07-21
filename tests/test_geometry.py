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
from shapely.geometry import box
from shapely.ops import unary_union

from terrapin.geometry import repose_wall, eroded_wedge, incise

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
