#! /usr/bin/env python
"""
Worked example: the standard TerraPIN model -- a mobile channel that migrates and
avulses, building an asymmetric valley with channel-belt deposits, a preserved
(buried) paleochannel, and dated terraces.

Drives StandardTerrapin through a sequence and draws each stage with the class's
own plot(), colouring bodies by deposit kind and overlaying the terraces:

  1. initial        alluvium over bedrock
  2. incise         a channel cuts down at centre
  3. migrate R      the channel sweeps right AT CAPACITY, planing a strath and
                    leaving a channel-belt of alluvium behind it (asymmetry)
  4. avulse L       the channel hops left; the old channel is abandoned in place
  5. aggrade        overbank floodplain buries the abandoned channel
  6. re-incise      the new channel cuts down, stranding strath, fill, and the
                    paleochannel as terraces

Run in the dedicated environment:
    conda run -n terrapin python examples/standard/standard_terraces.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import box

from terrapin.standard import StandardTerrapin

REPOSE = {"bedrock": 75.0, "alluvium": 32.0, "colluvium": 20.0}
XMIN, XMAX, ZMIN, ZMAX = -120.0, 120.0, -30.0, 8.0

st = StandardTerrapin()
st.set_bodies({"bedrock":  box(-120.0, -70.0, 120.0, -8.0),
               "alluvium": box(-120.0,  -8.0, 120.0,  0.0)})
st.set_repose_angles(REPOSE)
st.set_channel_position(0.0)
st.set_channel_elevation(0.0)
st.set_channel_width(20.0)
st.set_channel_depth(4.0)
st.establish_channel()                         # cut the initial channel into the ground

fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.4), sharex=True, sharey=True)
ax = axes.ravel()

st.plot(ax=ax[0]); ax[0].set_title("1. initial", fontsize=10, fontweight="bold")
st.incise(-12.0, age=5.0)
st.plot(ax=ax[1]); ax[1].set_title("2. incise (centre)", fontsize=10, fontweight="bold")
st.migrate(45.0, at_capacity=True, age=10.0)
st.plot(ax=ax[2]); ax[2].set_title("3. migrate right, at capacity", fontsize=10, fontweight="bold")
st.avulse(-10.0, age=15.0)
st.plot(ax=ax[3]); ax[3].set_title("4. avulse (onto the belt)", fontsize=10, fontweight="bold")
st.aggrade(-2.0, age=20.0)
st.plot(ax=ax[4]); ax[4].set_title("5. aggrade (bury)", fontsize=10, fontweight="bold")
st.set_channel_width(8.0)
st.incise(-18.0, age=30.0)
st.plot(ax=ax[5]); ax[5].set_title("6. re-incise (terraces)", fontsize=10, fontweight="bold")

VE = 4.0                                   # vertical exaggeration, for legibility
for a in ax:
    a.set_xlim(XMIN, XMAX)
    a.set_ylim(ZMIN, ZMAX)
    a.set_aspect(VE)

handles = [Patch(facecolor=c, edgecolor="k", hatch=h, label=k)
           for k, (c, h) in StandardTerrapin._STYLE.items()]
handles.append(Patch(facecolor="#2b7bba", edgecolor="k", label="river"))
handles.append(plt.Line2D([], [], color="#c1272d", lw=2.4, label="terrace"))
fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Standard TerraPIN: migration, avulsion, channel belts, and terraces "
             "(vertical exaggeration %g×)" % VE, fontsize=13, fontweight="bold")
fig.subplots_adjust(left=0.05, right=0.98, top=0.89, bottom=0.13,
                    hspace=0.55, wspace=0.22)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "standard_terraces.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)

print("\nterraces at the end (channel at x=%.0f, z=%.0f):" % (st.x_ch, st.z_ch))
for t in st.terraces():
    print("  %-10s z=%6.1f  x[%6.1f,%6.1f]  age=%-6s deposit_age=%s"
          % (t["kind"], t["z"], t["x_far"], t["x_near"],
             st._fmt_age(t["age"]), st._fmt_age(t["deposit_age"])))
