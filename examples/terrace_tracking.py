#! /usr/bin/env python
"""
Worked example: terrace and provenance tracking with the Terrapin state object.

Drives the Terrapin class -- not the bare geometry functions -- through a
cut / plane / fill / re-cut history, tagging each operation with the time it
happened, and then reads the terraces back out with terraces(). It renders the
final cross-section twice from the same state:

    terrace_tracking.png        the section with the terrace benches marked
    terrace_tracking_aged.png   the same, with each bench labelled by its age

The point the labelled figure makes: a terrace's age is the age at which its
surface was ABANDONED -- when the river left it behind -- and nothing else. The
deposit a terrace is cut on carries its own, separate deposition age (see the
`deposit_age` field printed below); that belongs to the deposit, not the
terrace. Planation both cuts a strath and abandons it as the river sweeps on, so
a strath's age is naturally the sweep's span; a fill terrace is abandoned in the
instant the river re-incises below it.

The unit models one valley wall, mirrored about the channel (x = 0) to read as a
full symmetric valley.

Run in the dedicated environment:
    conda run -n terrapin python examples/terrace_tracking.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import box

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from terrapin import Terrapin

REPOSE = {"bedrock": 75.0, "alluvium": 32.0, "colluvium": 20.0}

# --- Initial condition: alluvium over bedrock, flat land surface at z = 0 ---
tp = Terrapin()
tp.set_bodies({"bedrock": box(-140.0, -80.0, 0.0, -6.0),
               "alluvium": box(-140.0, -6.0, 0.0, 0.0)})
tp.set_repose_angles(REPOSE)
tp.set_channel_elevation(0.0)

# --- A valley history. Incision and aggradation are pinned in time by the caller;
#     planation carries no time, since a strath's age is the instant it is later
#     abandoned, not the (untracked) duration of its cutting. ---
tp.incise(-8.0, age=5.0)                      # cut down; strand the land surface
tp.plane_laterally(95.0)                       # plane a high, wide strath at -8 m
tp.set_channel_width(0.0)
tp.incise(-18.0, age=12.0)                     # re-incise; strand the -8 strath
tp.plane_laterally(40.0)                        # plane a lower strath at -18 m
tp.aggrade(-13.0, age=22.0)                    # bury the -18 strath under a fill
tp.set_channel_width(0.0)
tp.incise(-26.0, age=33.0)                     # re-incise; strand the fill top

# --- What terraces survive, and with which ages? The terrace age is always the
#     abandonment; a fill's deposition age belongs to the deposit, shown apart. ---
print("terraces (channel now at %.0f m):" % tp.z_ch)
for t in tp.terraces():
    print("  %-7s at z=%6.1f m  (w=%4.1f m)  terrace age (abandoned)=%-7s"
          "  deposit age=%s"
          % (t["kind"], t["z"], t["width"], tp._fmt_age(t["age"]),
             tp._fmt_age(t["deposit_age"])))

# --- The same final state, drawn without and with age labels ---
here = os.path.dirname(os.path.abspath(__file__))
for label_ages, suffix in ((False, ""), (True, "_aged")):
    ax = tp.plot(label_ages=label_ages)
    ax.set_ylim(-30.0, 4.0)                    # focus on the terrace zone
    ax.figure.set_size_inches(11.0, 3.4)
    ax.set_title("Terrace tracking (symmetric TerraPIN)"
                 + (" -- ages = abandonment" if label_ages else ""),
                 fontsize=11)
    out = os.path.join(here, "terrace_tracking%s.png" % suffix)
    ax.figure.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(ax.figure)
    print("wrote", out)
