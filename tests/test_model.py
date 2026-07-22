#! /usr/bin/env python
"""
Tests for the Terrapin state object (terrapin.model): configuration, the
imperative operations, the emergent width, and agreement with the geometry
functions it wraps.

Run in the dedicated environment:  conda run -n terrapin python -m pytest
"""
import numpy as np
import pytest

pytest.importorskip("shapely")
pytest.importorskip("scipy")
from shapely.geometry import box

from terrapin import geometry
from terrapin import Terrapin

REPOSE = {"bedrock": 75.0, "alluvium": 32.0, "colluvium": 20.0}


def fresh():
    tp = Terrapin()
    tp.set_bodies({"bedrock": box(-100.0, -50.0, 0.0, -10.0),
                   "alluvium": box(-100.0, -10.0, 0.0, 0.0)})
    tp.set_repose_angles(REPOSE)
    tp.set_channel_elevation(0.0)
    return tp


def test_incise_carves_channel_width_and_reports_mass():
    tp = fresh()
    tp.set_channel_width(16.0)
    tp.incise(-20.0)
    assert tp.z_ch == -20.0
    assert np.isclose(tp.compute_valley_width(), 16.0)        # carved exactly b
    assert tp.sediment_out > 0.0
    assert np.isclose(tp.sediment_out, sum(tp.eroded.values()))


def test_default_channel_width_is_a_point():
    tp = fresh()
    tp.incise(-20.0)                                          # default width 0
    assert np.isclose(tp.compute_valley_width(), 0.0, atol=1e-6)


def test_plane_laterally_widens_and_exports_sediment():
    tp = fresh()
    tp.incise(-20.0)
    tp.plane_laterally(30.0)
    assert tp.channel_width == 30.0
    assert np.isclose(tp.compute_valley_width(), 30.0)
    assert tp.sediment_out > 0.0


def test_aggrade_fills_and_raises_bed():
    tp = fresh()
    tp.set_channel_width(16.0)
    tp.incise(-20.0)
    tp.aggrade(-8.0)
    assert tp.z_ch == -8.0
    assert any("alluvium_fill" in name for name in tp.bodies)
    assert tp.deposited > 0.0


def test_repeated_aggradation_accumulates():
    # Each aggradation is its own body, so successive fills stack up.
    tp = fresh()
    tp.set_channel_width(16.0)
    tp.incise(-20.0)
    tp.aggrade(-14.0)
    tp.aggrade(-10.0)
    fills = [name for name in tp.bodies if "alluvium_fill" in name]
    assert len(fills) == 2                       # both fills persist, not overwritten
    assert all(tp.bodies[name].area > 0.0 for name in fills)


def test_channel_width_can_be_reset_over_time():
    tp = fresh()
    tp.set_channel_width(10.0)
    tp.incise(-15.0)
    assert np.isclose(tp.compute_valley_width(), 10.0)
    tp.set_channel_width(24.0)          # user widens the channel in time
    tp.incise(-20.0)
    assert np.isclose(tp.compute_valley_width(), 24.0)


def test_class_matches_geometry_functions():
    # The stateful wrapper must agree with calling geometry directly.
    bodies = {"bedrock": box(-100.0, -50.0, 0.0, -10.0),
              "alluvium": box(-100.0, -10.0, 0.0, 0.0)}
    ref_bodies, _ = geometry.incise(dict(bodies), -20.0, REPOSE,
                                    floor_half_width=8.0)
    tp = fresh()
    tp.set_channel_width(16.0)
    tp.incise(-20.0)
    for name in ref_bodies:
        assert np.isclose(tp.bodies[name].area, ref_bodies[name].area)
