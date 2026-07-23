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


# label, (left_fn, full_fn) pair, channel width, incision steps
REPRO_CASES = [
    ("point-channel",  std_pair(),    0.0, [-15.0]),
    ("narrow",         std_pair(),    8.0, [-15.0]),
    ("wide",           std_pair(),   40.0, [-15.0]),
    ("shallow",        std_pair(),   22.0, [-5.0]),
    ("deep",           std_pair(),   22.0, [-30.0]),
    ("two-step",       std_pair(),   16.0, [-12.0, -22.0]),
    ("thick-alluvium", thick_pair(), 22.0, [-15.0]),
]


@pytest.mark.parametrize("label,pair,width,steps", REPRO_CASES,
                         ids=[c[0] for c in REPRO_CASES])
def test_centered_reproduces_symmetric(label, pair, width, steps):
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
    for z in steps:
        sym.incise(z)
        std.incise(z)
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
