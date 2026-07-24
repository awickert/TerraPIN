#! /usr/bin/env python
"""
Worked example: terrace formation with symmetric TerraPIN.

Drives the geometry engine through a physically-ordered sequence and draws each
stage as a labelled cross-section:

  1. initial        alluvium over bedrock
  2. incise         river cuts down; wall fails at the angle of repose
  3. widen          the sweeping river planes a flat strath and carries ALL the
                    eroded rock off as sediment -- it cannot leave talus in its
                    own path, so the strath is clean
  4. shed talus     the river has narrowed away from the wall; the now-abandoned
                    wall sheds a modest talus apron onto the quiet strath (the
                    shed volume is a physics input -- illustrative here)
  5. aggrade        a valley fill buries strath and talus
  6. re-incise      the river cuts a new inner channel into the fill, leaving a
                    fill (cut-in-fill) terrace; the strath is buried beneath it

The unit models one valley wall; we mirror it about the channel (x = 0) so the
figure reads as a full symmetric valley -- the symmetric model, in which the
channel holds its lateral position and the valley incises, aggrades, and planes
symmetrically around it (the channel's own position/motion is not resolved).

Run in the dedicated environment:
    conda run -n terrapin python examples/low_unit_terraces.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import box

from terrapin.geometry import incise, widen, aggrade, eroded_wedge, colluvial_pile
from terrapin import plotting

# lithology -> legend label; the colours/hatches come from terrapin.plotting.STYLE.
# This example distinguishes an alluvial *fill* from the pre-existing alluvium.
LABELS = {"bedrock": "bedrock", "alluvium": "alluvium",
          "fill": "alluvial fill / terrace", "colluvium": "colluvium (talus)"}


def lithology(name):
    if "bedrock" in name:
        return "bedrock"
    if "colluvium" in name:
        return "colluvium"
    if "fill" in name:
        return "fill"
    return "alluvium"


def draw(ax, bodies, z_ch, title, subtitle):
    plotting.draw_bodies(ax, bodies, lithology, mirror=True)   # full valley
    ax.plot(0, z_ch, marker="v", color="#1f6fb2", ms=9, zorder=3,
            markeredgecolor="k")                      # channel
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=16)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=7.8, color="0.35")  # caption above the panel
    ax.set_xlim(-XMAX, XMAX)
    ax.set_ylim(ZMIN, ZMAX)
    ax.set_aspect("equal")
    ax.axhline(0, color="0.6", lw=0.5, ls=":", zorder=0)
    ax.tick_params(labelsize=8)


# --- Domain and initial condition: alluvium over bedrock, flat surface at 0 ---
XMAX, ZMIN, ZMAX = 42.0, -28.0, 5.0
REPOSE = {"bedrock": 75.0, "alluvium": 32.0, "colluvium": 20.0}
LAMBDA, ALPHA_C = 0.35, 20.0
DOMAIN = box(-80.0, -60.0, 0.0, ZMAX)

bodies = {"bedrock": box(-80.0, -60.0, 0.0, -8.0),
          "alluvium": box(-80.0, -8.0, 0.0, 0.0)}
snaps = [(dict(bodies), 0.0, "1. Initial", "alluvium over bedrock")]

# 2. Incise into bedrock -> a repose wall forms
bodies, _ = incise(bodies, -15.0, REPOSE, floor_half_width=0.0)
snaps.append((dict(bodies), -15.0, "2. Incise",
              "river cuts to -15 m; wall fails at repose"))

# 3. Widen: the river planes a flat strath and exports all the rock
pre_widen = dict(bodies)
bodies, bal = widen(bodies, -15.0, 22.0, REPOSE)
snaps.append((dict(bodies), -15.0, "3. Widen (plane strath)",
              "flat strath; %.0f m$^2$ bedrock swept to sediment"
              % bal["bedrock_eroded"]))

# 4. River has narrowed away from the wall; the abandoned wall sheds a talus
#    apron onto the quiet strath. The shed volume is a physics input (external);
#    here we place an illustrative amount and let the engine fit the apron.
void = eroded_wedge(-15.0, pre_widen, REPOSE, 22.0)
talus, _ = colluvial_pile(15.0, void, ALPHA_C, LAMBDA)
bodies = dict(bodies)
bodies["colluvium"] = talus
snaps.append((dict(bodies), -15.0, "4. Wall sheds talus (river absent)",
              "abandoned wall sheds a %.0f m$^2$ apron; strath still open"
              % talus.area))

# 5. Aggrade a valley fill over strath and talus
bodies, dep = aggrade(bodies, -6.0, DOMAIN, name="alluvium_fill")
snaps.append((dict(bodies), -6.0, "5. Aggrade fill",
              "valley fills to -6 m (+%.0f m$^2$ alluvium)" % dep))

# 6. Re-incise a narrow inner channel -> fill and strath left as terraces
bodies, _ = incise(bodies, -20.0, REPOSE, floor_half_width=0.0)
snaps.append((dict(bodies), -20.0, "6. Re-incise (fill terrace)",
              "river cuts to -20 m; fill left as a terrace (strath buried)"))

# --- Figure: 2 rows x 3 cols ---
fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.2), sharex=True, sharey=True)
for ax, (bd, zc, title, sub) in zip(axes.ravel(), snaps):
    draw(ax, bd, zc, title, sub)
for ax in axes[:, 0]:
    ax.set_ylabel("elevation [m]", fontsize=10)
fig.supxlabel("cross-valley distance [m]  (one wall, mirrored about the channel)",
              fontsize=10, y=0.06)

handles = [Patch(facecolor=plotting.STYLE[k][0], edgecolor="k",
                 hatch=plotting.STYLE[k][1], label=LABELS[k])
           for k in ("bedrock", "alluvium", "fill", "colluvium")]
handles.append(plt.Line2D([], [], marker="v", color="#1f6fb2", ls="",
                          markeredgecolor="k", label="channel"))
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Strath planation, talus, and fill-terrace formation "
             "(symmetric TerraPIN)",
             fontsize=13, fontweight="bold", y=0.99)
fig.tight_layout(rect=(0, 0.08, 1, 0.95))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "symmetric_valley_terraces.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
