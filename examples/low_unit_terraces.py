#! /usr/bin/env python
"""
Worked example: terrace formation with the low (one-wall) TerraPIN unit.

Drives the geometry engine through the classic sequence -- incise, laterally
plane (widen) leaving a strath and a colluvial pile, aggrade a valley fill, then
re-incise to strand that fill as a terrace -- and draws each stage as a labelled
cross-section. The unit models one valley wall; we mirror it about the channel
(x = 0) so the figure reads as a full symmetric valley, which is exactly what
"low" means (one wall, mirrored).

Run in the dedicated environment:
    conda run -n terrapin python examples/low_unit_terraces.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import box

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from terrapin.geometry import incise, widen, aggrade

# --- Material palette (geoscience-flavoured), keyed by lithology ---
STYLE = {
    "bedrock":  dict(fc="#b8926a", hatch="//",  label="bedrock"),
    "alluvium": dict(fc="#e6cf7a", hatch="..",  label="alluvium"),
    "fill":     dict(fc="#d7a43c", hatch="..",  label="alluvial fill / terrace"),
    "colluvium": dict(fc="#9a8f7d", hatch="xx", label="colluvium (talus)"),
}


def lithology(name):
    if "bedrock" in name:
        return "bedrock"
    if "colluvium" in name:
        return "colluvium"
    if "fill" in name:
        return "fill"
    return "alluvium"


def draw(ax, bodies, z_ch, title, subtitle):
    for name, geom in bodies.items():
        if geom.is_empty:
            continue
        st = STYLE[lithology(name)]
        parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for p in parts:
            xs, zs = p.exterior.xy
            xs = np.asarray(xs)
            for sign in (1, -1):                      # mirror about the channel
                ax.fill(sign * xs, zs, facecolor=st["fc"], edgecolor="k",
                        linewidth=0.6, hatch=st["hatch"], zorder=1)
    ax.plot(0, z_ch, marker="v", color="#1f6fb2", ms=10, zorder=3,
            markeredgecolor="k")                      # channel
    ax.set_title(title, fontsize=11, fontweight="bold", pad=18)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=8, color="0.35")    # caption above the panel
    ax.set_xlim(-XMAX, XMAX)
    ax.set_ylim(ZMIN, ZMAX)
    ax.set_aspect("equal")
    ax.axhline(0, color="0.6", lw=0.5, ls=":", zorder=0)
    ax.tick_params(labelsize=8)


# --- Domain and initial condition: alluvium over bedrock, flat surface at 0 ---
XMAX, ZMIN, ZMAX = 42.0, -28.0, 5.0
STACK = [(-8.0, 75.0, "bedrock"), (0.0, 32.0, "alluvium")]
LAMBDA, ALPHA_C = 0.35, 20.0

bodies = {"bedrock": box(-80.0, -60.0, 0.0, -8.0),
          "alluvium": box(-80.0, -8.0, 0.0, 0.0)}

snaps = [(dict(bodies), 0.0, "1. Initial",
          "alluvium over bedrock; river at the surface")]

# 2. Incise into bedrock -> a repose wall forms
bodies, er = incise(bodies, -15.0, STACK, floor_half_width=0.0)
snaps.append((dict(bodies), -15.0,
              "2. Incise", "river cuts to -15 m; wall fails at repose"))

# 3. Widen (lateral planation) -> strath floor + colluvial pile
bodies, bal = widen(bodies, -15.0, 22.0, STACK, ALPHA_C, LAMBDA)
snaps.append((dict(bodies), -15.0, "3. Widen (plane strath)",
              "bedrock %.0f -> talus %.0f m$^2$; %.0f to sediment"
              % (bal["bedrock_eroded"], bal["colluvium_stored"],
                 bal["sediment_out"])))

# 4. Aggrade a valley fill
bodies, dep = aggrade(bodies, -5.0, box(-80.0, -60.0, 0.0, ZMAX),
                      name="alluvium_fill")
snaps.append((dict(bodies), -5.0, "4. Aggrade fill",
              "valley fills to -5 m (+%.0f m$^2$ alluvium)" % dep))

# 5. Re-incise -> the fill is stranded as a terrace
bodies, er = incise(bodies, -20.0, STACK, floor_half_width=0.0)
snaps.append((dict(bodies), -20.0, "5. Re-incise (strand terrace)",
              "river cuts to -20 m; fill left as a terrace"))

# --- Figure ---
fig, axes = plt.subplots(1, len(snaps), figsize=(3.05 * len(snaps), 3.0),
                         sharey=True)
for ax, (bd, zc, title, sub) in zip(axes, snaps):
    draw(ax, bd, zc, title, sub)
axes[0].set_ylabel("elevation [m]", fontsize=10)
fig.supxlabel("cross-valley distance [m]  (one wall, mirrored about the channel)",
              fontsize=10, y=0.10)

handles = [Patch(facecolor=s["fc"], edgecolor="k", hatch=s["hatch"],
                 label=s["label"]) for s in STYLE.values()]
handles.append(plt.Line2D([], [], marker="v", color="#1f6fb2", ls="",
                          markeredgecolor="k", label="channel"))
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Strath and fill terraces from the low (one-wall) TerraPIN unit",
             fontsize=13, fontweight="bold", y=1.0)
fig.tight_layout(rect=(0, 0.12, 1, 0.94))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "low_unit_terraces.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
