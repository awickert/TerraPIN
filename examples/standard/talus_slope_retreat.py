#! /usr/bin/env python
"""
Talus by parallel slope retreat -- the mechanism and its geometry.

An exposed slope standing at its angle of repose sheds material by PARALLEL
RETREAT: the face steps back into the hillslope keeping its angle, and the shed
rock piles as talus at the base. The pile buries the lower face, so only the
still-exposed face above the talus keeps retreating -- a moving boundary that
makes the retreat decelerate as the apron grows, and it leaves the cliff with a
GRADED (differential-retreat) profile: the higher it is, the longer it stayed
exposed, the farther it retreated. (This is the river-ABSENT talus that a
migrating channel unlocks; undercutting by a present channel instead sends
material straight to the river. See issue awickert/TerraPIN#9.)

The apron rests on the strath and leans on the cliff, its free surface at the
colluvium repose angle a_c, its area equal to the (porosity-fluffed) shed volume.

  * The FIRST fall, against a straight cliff, is a triangle -- closed form,
    no root-finding (a_c, a_w = colluvium, wall angles; t = their tangents):
        A = 1/2 b h,  h = b (t_c t_w)/(t_w - t_c)
      => b = sqrt(2A (t_w - t_c)/(t_c t_w)),   h = sqrt(2A (t_c t_w)/(t_w - t_c))
    (area is 1/2 b h for ANY wall angle; the wall angle only relates b and h).

  * After that it is NO LONGER a triangle: differential retreat kinks the cliff,
    so the apron bulges out along the retreated steps. The free surface is still
    one repose line, so the general area-matching solver -- colluvial_pile /
    position_repose_surface, given the BOUNDED wall-strath corner as its void --
    places it against whatever cliff shape is actually there. That is what this
    example uses, so the apron is correct on every fall.

Driven here by a fixed retreat step dx per event; the shed volume V = dx * exposed
height falls out, and mass is conserved (talus_area * (1 - lambda_p) == rock removed).

Run in the dedicated environment:
    conda run -n terrapin python examples/standard/talus_slope_retreat.py
"""
import os
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import Polygon

from terrapin.geometry import colluvial_pile

LAMBDA = 0.35                          # sediment porosity (fluffs rock into talus)
ALPHA_C, ALPHA_W = 20.0, 75.0          # colluvium repose; bedrock wall face [deg]
T_W = math.tan(math.radians(ALPHA_W))
Z_TOP, DX, X_RIGHT = 40.0, 2.5, 100.0  # cliff height [m]; retreat per shed [m]; valley extent

ZG = np.linspace(0.0, Z_TOP, 801)      # height grid tracking the cliff face position
xface = -ZG / T_W                      # face x at each height (cliff base at x = 0)


def corner_void(xf):
    """The bounded wall-strath corner the talus can occupy: cliff face on the left,
    strath below, the valley to the right, capped at the cliff top."""
    return Polygon([(xf[i], ZG[i]) for i in range(len(ZG))] + [(X_RIGHT, Z_TOP), (X_RIGHT, 0.0)])


def cliff_solid(xf):
    """Bedrock: floor plus the wall left of the (retreating) face profile."""
    from shapely.geometry import box
    wall = [(xf[i], ZG[i]) for i in range(len(ZG))] + [(-100.0, Z_TOP), (-100.0, 0.0)]
    return box(-100.0, -10.0, X_RIGHT, 0.0).union(Polygon(wall))


total_V, talus_top = 0.0, 0.0
history = [(cliff_solid(xface.copy()), None)]
print("shed |     V | run-out | talus top | mass: talus*(1-lam) == rock removed")
for k in range(5):
    exposed = ZG > (talus_top - 1e-9 if talus_top < 1e-6 else talus_top)  # base bares only when uncovered
    V = DX * (Z_TOP - talus_top)                     # exposed face swept back by dx
    xface[exposed] -= DX
    total_V += V
    apron, overflow = colluvial_pile(total_V, corner_void(xface), ALPHA_C, LAMBDA)
    talus_top = apron.bounds[3] if not apron.is_empty else 0.0
    toe = apron.bounds[2] if not apron.is_empty else 0.0
    history.append((cliff_solid(xface.copy()), apron))
    print("  %d  | %5.1f |  %6.2f |   %6.2f  |   %7.2f == %7.2f"
          % (k + 1, V, toe, talus_top, apron.area * (1.0 - LAMBDA), total_V))

panels = [0, 1, 3, 5]
titles = ["initial cliff", "after shed 1 (triangle)", "after shed 3", "after shed 5"]
fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.3), sharex=True, sharey=True)
for ax, i, title in zip(axes, panels, titles):
    solid, apron = history[i]
    for geom, fc, hatch in [(solid, "#b8926a", "//"), (apron, "#9a8f7d", "xx")]:
        if geom is None or geom.is_empty:
            continue
        for p in (geom.geoms if geom.geom_type != "Polygon" else [geom]):
            xs, zs = p.exterior.xy
            ax.fill(xs, zs, facecolor=fc, edgecolor="k", linewidth=0.7, hatch=hatch)
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_xlim(-50, 50); ax.set_ylim(-6, 44); ax.set_aspect(1.0)
    ax.set_xlabel("cross-valley distance [m]")
axes[0].set_ylabel("elevation [m]")

handles = [Patch(facecolor="#b8926a", edgecolor="k", hatch="//", label="bedrock (retreating cliff)"),
           Patch(facecolor="#9a8f7d", edgecolor="k", hatch="xx", label="talus (colluvium)")]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Talus by parallel slope retreat: the apron grows against the retreating cliff, buries "
             "its base, and (after fall 1) is no longer a triangle", fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0.06, 1, 0.96))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "talus_slope_retreat.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
