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


# ------------------------- provenance and terraces --------------------------
#
# A deposit holds the age of its deposition; a surface holds the age of its
# abandonment. A terrace's age is the abandonment age, and nothing else -- the
# deposit it is cut on has its own, separate deposition age. Terraces are the
# flat benches read from the live geometry.


def test_initial_bodies_have_unknown_formation_age():
    # Pre-existing material was not deposited on our clock, so its formation
    # age is unknown; its top is a surface not yet abandoned.
    tp = fresh()
    assert tp.provenance["bedrock"] == {"kind": "initial",
                                        "lithology": "bedrock", "age": None}
    top = tp._surface_at(0.0)
    assert top is not None and top["kind"] == "initial"
    assert top["abandoned"] is None


def test_aggrade_records_deposit_formation_age():
    # The fill is a deposit: it carries its formation age; its fresh top is a
    # surface not yet abandoned (it is the live valley floor).
    tp = fresh()
    tp.incise(-20.0, age=1.0)
    tp.aggrade(-12.0, age=5.0)
    assert tp.provenance["alluvium_fill_0"]["kind"] == "fill"
    assert tp.provenance["alluvium_fill_0"]["age"] == 5.0
    assert tp._surface_at(-12.0)["abandoned"] is None


def test_plane_laterally_logs_a_live_strath():
    # Planation cuts the strath but does not abandon it: the planed floor is the
    # live channel bed, logged as a surface not yet abandoned. The cutting takes
    # time but that duration is not tracked -- the strath's age is the instant of
    # a later abandonment.
    tp = fresh()
    tp.incise(-20.0, age=1.0)
    tp.plane_laterally(30.0)
    strath = tp._surface_at(-20.0)
    assert strath["kind"] == "strath"
    assert strath["abandoned"] is None


def test_incise_abandons_the_surface_it_strands():
    # The first incision strands the original valley-margin surface, stamping it
    # with the incision's age.
    tp = fresh()
    tp.incise(-20.0, age=7.0)
    terr = tp.terraces()
    assert len(terr) == 1
    t = terr[0]
    assert t["z"] == 0.0 and t["kind"] == "initial"
    assert t["deposit_age"] is None and t["age"] == 7.0


def test_fill_terrace_age_is_abandonment_not_deposition():
    # A cut-fill-recut sequence. The terrace's age is the re-incision that
    # stranded it (abandonment) -- NOT the fill's deposition. The deposition age
    # is the deposit's own, reported separately as deposit_age.
    tp = fresh()
    tp.incise(-15.0, age=10.0)
    tp.plane_laterally(44.0)
    tp.aggrade(-6.0, age=20.0)
    tp.set_channel_width(0.0)                      # narrow inner channel
    tp.incise(-20.0, age=30.0)
    fills = [t for t in tp.terraces() if t["kind"] == "fill"]
    assert len(fills) == 1
    t = fills[0]
    assert np.isclose(t["z"], -6.0)
    assert t["age"] == 30.0                        # the terrace age: abandonment
    assert t["deposit_age"] == 20.0               # the deposit's own, separate age
    assert t["body"] == "alluvium_fill_0"


def test_buried_strath_is_not_reported_as_a_terrace():
    # The strath is buried by the fill before any incision strands it, so it is
    # never abandoned and never a terrace: it stays a live surface with no
    # abandonment age.
    tp = fresh()
    tp.incise(-15.0, age=10.0)
    tp.plane_laterally(44.0)
    tp.aggrade(-6.0, age=20.0)
    tp.set_channel_width(0.0)
    tp.incise(-20.0, age=30.0)
    assert not any(np.isclose(t["z"], -15.0) for t in tp.terraces())
    assert tp._surface_at(-15.0)["abandoned"] is None


def test_strath_terrace_age_is_its_abandonment():
    # The strath is planed at -15 and stays the live bed until the river incises
    # below it at t=30. Its terrace age is that abandonment instant; the planation
    # duration is not tracked.
    tp = fresh()
    tp.incise(-15.0, age=10.0)
    tp.plane_laterally(44.0)
    tp.set_channel_width(0.0)
    tp.incise(-25.0, age=30.0)
    straths = [t for t in tp.terraces() if t["kind"] == "strath"]
    assert len(straths) == 1
    t = straths[0]
    assert np.isclose(t["z"], -15.0)
    assert t["age"] == 30.0                        # abandonment (the incision)
    assert t["deposit_age"] is None                # erosional: no deposit beneath
    assert t["lithology"] == "bedrock"             # a strath is cut into rock


def test_terraces_are_read_from_live_geometry():
    # The re-incision wall eats into the fill bench, so the reported tread is
    # shorter than the fill and offset from the channel -- proof the extent comes
    # from the current geometry, not a remembered span.
    tp = fresh()
    tp.incise(-15.0, age=10.0)
    tp.plane_laterally(44.0)
    tp.aggrade(-6.0, age=20.0)
    tp.set_channel_width(0.0)
    tp.incise(-20.0, age=30.0)
    t = [t for t in tp.terraces() if t["kind"] == "fill"][0]
    assert t["x_far"] < t["x_near"] < 0.0          # a wall stands between it and x=0
    assert 0.0 < t["width"] < 22.0                 # shorter than the 22 m half-width
