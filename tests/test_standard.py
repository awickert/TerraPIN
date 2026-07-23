#! /usr/bin/env python
"""
Tests for the standard TerraPIN model (terrapin.standard).

The keystone is the REPRODUCTION invariant, checked across a variety of
situations: a centered standard valley (channel at x = 0) is exactly the mirrored
symmetric one, because symmetric is one instance of the shared one-wall unit. The
`test_centered_reproduces_symmetric` table runs matched sequences on the symmetric
Terrapin (one wall) and a centered StandardTerrapin (two walls) and asserts the
standard's left half equals the symmetric result. As the standard model gains
aggrade / a widen / etc., those shared ops join the sequences here; migrate and
avulse to off-center positions have no symmetric analog and get their own tests.

Run in the dedicated environment:  conda run -n terrapin python -m pytest
"""
import numpy as np
import pytest

pytest.importorskip("shapely")
pytest.importorskip("scipy")
from shapely.geometry import box
from shapely.affinity import scale
from shapely.ops import unary_union

from terrapin import Terrapin
from terrapin.standard import StandardTerrapin

REPOSE = {"bedrock": 75.0, "alluvium": 32.0, "colluvium": 20.0}
LEFT_HALF = box(-1.0e4, -1.0e4, 0.0, 1.0e4)      # x <= 0


def reflect(geom, x0=0.0):
    return scale(geom, xfact=-1.0, yfact=1.0, origin=(x0, 0.0))


def alluvium_over_bedrock(x_lo, x_hi, contact=-8.0):
    return {"bedrock":  box(x_lo, -60.0, x_hi, contact),
            "alluvium": box(x_lo, contact, x_hi, 0.0)}


# Matched (left-half, full-valley) initial-condition pairs.
def std_pair():
    return (lambda: alluvium_over_bedrock(-80.0, 0.0),
            lambda: alluvium_over_bedrock(-80.0, 80.0))

def thick_pair():          # 20 m of alluvium instead of 8 m
    return (lambda: alluvium_over_bedrock(-80.0, 0.0, contact=-20.0),
            lambda: alluvium_over_bedrock(-80.0, 80.0, contact=-20.0))


# label, (left_fn, full_fn) pair, channel width, [(op, level), ...]
REPRO_CASES = [
    ("point-channel",  std_pair(),    0.0, [("incise", -15.0)]),
    ("narrow",         std_pair(),    8.0, [("incise", -15.0)]),
    ("wide",           std_pair(),   40.0, [("incise", -15.0)]),
    ("shallow",        std_pair(),   22.0, [("incise", -5.0)]),
    ("deep",           std_pair(),   22.0, [("incise", -30.0)]),
    ("two-step",       std_pair(),   16.0, [("incise", -12.0), ("incise", -22.0)]),
    ("thick-alluvium", thick_pair(), 22.0, [("incise", -15.0)]),
    ("incise-aggrade", std_pair(),   16.0, [("incise", -20.0), ("aggrade", -8.0)]),
    ("cut-fill-recut", std_pair(),   16.0, [("incise", -15.0), ("aggrade", -6.0),
                                            ("incise", -22.0)]),
]


@pytest.mark.parametrize("label,pair,width,steps", REPRO_CASES,
                         ids=[c[0] for c in REPRO_CASES])
def test_centered_reproduces_symmetric(label, pair, width, steps):
    # A centered channel has no lateral asymmetry, so any sequence of the ops that
    # the symmetric model also has must reproduce the symmetric (one-wall) result
    # in the standard model's left half.
    left_fn, full_fn = pair
    sym = Terrapin()
    sym.set_bodies(left_fn())
    sym.set_repose_angles(REPOSE)
    sym.set_channel_elevation(0.0)
    sym.set_channel_width(width)
    std = StandardTerrapin()
    std.set_bodies(full_fn())
    std.set_repose_angles(REPOSE)
    std.set_channel_position(0.0)
    std.set_channel_elevation(0.0)
    std.set_channel_width(width)
    for op, level in steps:
        getattr(sym, op)(level)
        getattr(std, op)(level)
    for name in sym.bodies:
        std_left = std.bodies[name].intersection(LEFT_HALF).area
        assert np.isclose(std_left, sym.bodies[name].area), (label, name)


# --- Dedicated standard-model checks (no symmetric analog) ---

def fresh(x_ch=0.0, width=22.0):
    st = StandardTerrapin()
    st.set_bodies(alluvium_over_bedrock(-80.0, 80.0))
    st.set_repose_angles(REPOSE)
    st.set_channel_position(x_ch)
    st.set_channel_elevation(0.0)
    st.set_channel_width(width)
    return st


def test_centered_incision_is_symmetric_about_the_channel():
    st = fresh(0.0, 22.0)
    st.incise(-15.0)
    for g in st.bodies.values():
        assert np.isclose(g.area, reflect(g, 0.0).area)


def test_offcenter_incision_is_symmetric_about_x_ch():
    st = fresh(25.0, 22.0)
    st.incise(-15.0)
    for g in st.bodies.values():
        assert np.isclose(g.area, reflect(g, 25.0).area)


def test_offcenter_eroded_matches_centered():
    a = fresh(0.0, 22.0); a.incise(-15.0)
    b = fresh(30.0, 22.0); b.incise(-15.0)
    assert np.isclose(sum(a.eroded.values()), sum(b.eroded.values()))
    for name in a.eroded:
        assert np.isclose(a.eroded[name], b.eroded[name])


def test_incision_exports_sediment_and_conserves_mass():
    st = fresh(0.0, 16.0)
    before = sum(g.area for g in st.bodies.values())
    st.incise(-20.0)
    after = sum(g.area for g in st.bodies.values())
    assert st.sediment_out > 0.0
    assert np.isclose(st.sediment_out, sum(st.eroded.values()))
    assert np.isclose(before - after, st.sediment_out)


# --- migrate ---

def test_migrate_leaves_the_retreating_wall_and_conserves_mass():
    st = fresh(0.0, 22.0)
    st.incise(-15.0)
    retreating = box(-80.0, -60.0, -11.0 - 1e-9, 5.0)      # left of the channel floor
    left_before = unary_union(list(st.bodies.values())).intersection(retreating).area
    before = sum(g.area for g in st.bodies.values())
    st.migrate(25.0)
    after = sum(g.area for g in st.bodies.values())
    left_after = unary_union(list(st.bodies.values())).intersection(retreating).area
    assert st.x_ch == 25.0
    assert np.isclose(left_before, left_after)            # retreating wall untouched
    assert st.sediment_out > 0.0
    assert np.isclose(before - after, st.sediment_out)    # mass conserved


def test_migrate_to_same_position_is_a_noop():
    st = fresh(0.0, 22.0)
    st.incise(-15.0)
    before = {n: g.area for n, g in st.bodies.items()}
    st.migrate(0.0)
    assert st.sediment_out == 0.0
    for name, area in before.items():
        assert np.isclose(st.bodies[name].area, area)


def test_migrate_left_and_right_are_mirror_images():
    # From a symmetric initial valley, migrating right by d and left by d give
    # mirror-image results (equal areas, equal eroded volumes).
    r = fresh(0.0, 22.0); r.incise(-15.0); r.migrate(20.0)
    l = fresh(0.0, 22.0); l.incise(-15.0); l.migrate(-20.0)
    for name in r.bodies:
        assert np.isclose(r.bodies[name].area, l.bodies[name].area)
        assert np.isclose(r.eroded[name], l.eroded[name])


def test_migrate_planes_the_swept_corridor_to_bed_level():
    from shapely.geometry import Point
    st = fresh(0.0, 22.0)
    st.incise(-15.0)
    st.migrate(25.0)
    solid = unary_union(list(st.bodies.values()))
    # a point just above the bed inside the swept corridor is now open (planed)
    assert not solid.covers(Point(18.0, st.z_ch + 0.5))


# --- avulse ---

def test_avulse_erodes_one_channel_depth_and_width():
    st = fresh(0.0, 22.0)
    st.set_channel_depth(4.0)
    st.incise(-15.0)
    st.avulse(50.0)                        # land on flat floodplain (surface z = 0)
    assert st.x_ch == 50.0
    assert np.isclose(st.z_ch, -4.0)       # one channel depth below the surface
    assert np.isclose(st.sediment_out, 22.0 * 4.0)      # width x depth
    assert np.isclose(st.sediment_out, sum(st.eroded.values()))


def test_avulse_preserves_the_abandoned_belt():
    # Unlike migrate, avulse does not plane the corridor: the ground between the
    # old channel and the landing site is left untouched.
    st = fresh(0.0, 22.0)
    st.set_channel_depth(4.0)
    st.incise(-15.0)
    corridor = box(12.0, -60.0, 38.0, 5.0)    # between old channel (+11) and new (39)
    before = unary_union(list(st.bodies.values())).intersection(corridor).area
    st.avulse(50.0)
    after = unary_union(list(st.bodies.values())).intersection(corridor).area
    assert np.isclose(before, after)


def test_avulse_conserves_mass():
    st = fresh(0.0, 22.0)
    st.set_channel_depth(4.0)
    st.incise(-15.0)
    before = sum(g.area for g in st.bodies.values())
    st.avulse(50.0)
    after = sum(g.area for g in st.bodies.values())
    assert np.isclose(before - after, st.sediment_out)


# --- provenance and terraces (incise / aggrade) ---

def test_initial_bodies_have_unknown_formation_age():
    st = fresh(0.0, 22.0)
    assert st.provenance["bedrock"] == {"kind": "initial",
                                        "lithology": "bedrock", "age": None}
    assert st.provenance["alluvium"]["kind"] == "initial"


def test_aggrade_records_a_floodplain_deposit():
    st = fresh(0.0, 16.0)
    st.incise(-20.0, age=1.0)
    st.aggrade(-8.0, age=5.0)
    prov = st.provenance["alluvium_fill_0"]
    assert prov["kind"] == "floodplain"
    assert prov["age"] == 5.0


def test_fill_terrace_reports_abandonment_and_deposition():
    st = StandardTerrapin()
    st.set_bodies(alluvium_over_bedrock(-80.0, 80.0))
    st.set_repose_angles(REPOSE)
    st.set_channel_position(0.0)
    st.set_channel_elevation(0.0)
    st.set_channel_width(16.0)
    st.incise(-15.0, age=10.0)
    st.aggrade(-6.0, age=20.0)
    st.set_channel_width(0.0)
    st.incise(-20.0, age=30.0)              # strands the fill top at -6
    fills = [t for t in st.terraces()
             if t["kind"] == "floodplain" and np.isclose(t["z"], -6.0)]
    assert fills                             # a floodplain terrace survives
    t = fills[0]
    assert t["age"] == 30.0                  # the terrace age: abandonment
    assert t["deposit_age"] == 20.0          # the fill's own deposition age


def test_terraces_are_symmetric_for_a_centered_channel():
    # A centered channel strands terraces symmetrically: each appears on both
    # sides at the same elevation with the same age.
    st = fresh(0.0, 0.0)
    st.incise(-15.0, age=10.0)               # strands the initial margins at z=0
    terr = st.terraces()
    left = sorted(t["z"] for t in terr if t["x_near"] <= 0.0)
    right = sorted(t["z"] for t in terr if t["x_far"] >= 0.0)
    assert left and np.allclose(left, right)


def test_migrate_strath_becomes_a_terrace_on_later_incision():
    # A migration strath needs no explicit record: the later incision that strands
    # it captures it as a strath terrace, abandoned at that incision.
    st = fresh(0.0, 16.0)
    st.incise(-15.0, age=10.0)
    st.migrate(30.0)                         # sweep right; strath extends that way
    st.set_channel_width(0.0)
    st.incise(-25.0, age=30.0)               # incise below; strand the strath
    straths = [t for t in st.terraces()
               if t["kind"] == "strath" and np.isclose(t["z"], -15.0)]
    assert straths
    assert all(t["age"] == 30.0 for t in straths)      # abandoned by the -25 incision


def test_avulse_records_the_abandoned_channel_at_the_avulsion_age():
    st = fresh(0.0, 22.0)
    st.set_channel_depth(4.0)
    st.incise(-15.0, age=10.0)
    st.avulse(50.0, age=20.0)                # abandon the old channel at -15
    surf = st._surface_at(-15.0)
    assert surf is not None and surf["kind"] == "channel"
    assert surf["abandoned"] == 20.0


# --- channel-belt deposition (migrate at capacity) ---

def test_migrate_at_capacity_leaves_a_channel_belt_deposit():
    st = fresh(0.0, 16.0)
    st.set_channel_depth(4.0)
    st.incise(-15.0)
    st.migrate(30.0, at_capacity=True, age=5.0)
    # the migration belt is the channel deposit dated to the migration (incise also
    # leaves channel-associated bank alluvium)
    belts = [n for n in st.bodies
             if "channel_belt" in n and st.provenance[n]["age"] == 5.0]
    assert belts
    belt = st.bodies[belts[0]]
    assert np.isclose(belt.bounds[1], -15.0)              # bottom at the strath
    assert belt.bounds[3] <= -15.0 + 4.0 + 1e-6           # top no higher than bank-top
    assert st.provenance[belts[0]]["kind"] == "channel"


def test_at_capacity_exports_less_than_erosional():
    a = fresh(0.0, 16.0); a.set_channel_depth(4.0); a.incise(-15.0)
    a.migrate(30.0, at_capacity=False)
    b = fresh(0.0, 16.0); b.set_channel_depth(4.0); b.incise(-15.0)
    b.migrate(30.0, at_capacity=True, age=5.0)
    assert b.deposited > 0.0
    assert b.sediment_out < a.sediment_out                # belt retained, less exported
    assert np.isclose(a.sediment_out - b.deposited, b.sediment_out)   # net export


def test_at_capacity_keeps_the_active_channel_open_with_belt_behind():
    from shapely.geometry import Point
    st = fresh(0.0, 16.0)
    st.set_channel_depth(4.0)
    st.incise(-15.0)
    st.migrate(30.0, at_capacity=True, age=5.0)
    solid = unary_union(list(st.bodies.values()))
    assert not solid.covers(Point(30.0, -14.5))           # active channel stays open
    assert solid.covers(Point(10.0, -14.5))               # belt fills the swept corridor


def test_abandoned_channel_survives_as_a_paleochannel_terrace():
    # Avulse preserves the old channel in place; when the new channel later cuts
    # below it, it stands as a channel terrace dated to the AVULSION, not the
    # later incision.
    st = fresh(0.0, 22.0)
    st.set_channel_depth(4.0)
    st.incise(-8.0, age=10.0)                # shallow channel at x=0, z=-8
    st.avulse(50.0, age=20.0)                # hop to x=50; old channel abandoned
    st.set_channel_width(0.0)
    st.incise(-15.0, age=30.0)               # new channel cuts below -8
    chans = [t for t in st.terraces()
             if t["kind"] == "channel" and np.isclose(t["z"], -8.0)]
    assert chans
    assert chans[0]["age"] == 20.0           # the avulsion, not the -15 incision
