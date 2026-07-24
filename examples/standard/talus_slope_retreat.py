#! /usr/bin/env python
"""
Talus by parallel slope retreat -- the mechanism, with a closed-form repose wedge.

An exposed slope standing at its angle of repose sheds material by PARALLEL
RETREAT: the face steps back into the hillslope keeping its angle, and the shed
rock piles as talus at the base. The pile buries the lower face, so only the
still-exposed face above the talus keeps retreating -- a moving boundary that
makes the retreat decelerate as the apron grows. (This is the river-ABSENT talus
that a migrating channel unlocks; undercutting by a present channel instead sends
material straight to the river. See issue awickert/TerraPIN#9.)

The apron is a triangle at the colluvium repose angle a_c, resting on the strath
and leaning on the wall (angle a_w). Its area equals the (porosity-fluffed) shed
volume, so with the corner at the origin it is CLOSED FORM -- no root-finding:

    A = 1/2 * b * h,   h = b * (t_c t_w) / (t_w - t_c)      (t = tangents)

  =>  b = sqrt( 2A (t_w - t_c) / (t_c t_w) )    run-out of the toe on the strath
      h = sqrt( 2A (t_c t_w) / (t_w - t_c) )    height it buries the wall

The area is 1/2 * b * h for ANY wall angle (a horizontal base b and apex height h);
the wall angle only sets how h and b relate. When the apron grows past a material
contact in the wall it stops being one triangle -- then the general area-matching
solver (geometry.position_repose_surface) takes over.

Driven here by a fixed retreat step dx per event; the shed volume V = dx * exposed
height falls out, and mass is conserved (talus_area * (1 - lambda_p) == rock removed).

Run in the dedicated environment:
    conda run -n terrapin python examples/standard/talus_slope_retreat.py
"""
import os
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import box, Polygon
from shapely.affinity import translate

LAMBDA = 0.35                          # sediment porosity (fluffs rock into talus)
ALPHA_C, ALPHA_W = 20.0, 75.0          # colluvium repose; bedrock wall face [deg]
T_C, T_W = math.tan(math.radians(ALPHA_C)), math.tan(math.radians(ALPHA_W))
Z_TOP, DX = 40.0, 2.0                  # wall height [m]; retreat per shed [m]
DOMAIN = box(-100.0, -10.0, 100.0, 45.0)
CX, CZ = 0.0, 0.0                      # wall-strath corner (the apron's anchor)


def talus_wedge(area):
    """The closed-form talus triangle of (fluffed) `area`, anchored at the corner:
    toe running out `b` on the strath, apex at height `h` on the wall face."""
    if area <= 1.0e-12:
        return Polygon(), 0.0, 0.0
    b = math.sqrt(2.0 * area * (T_W - T_C) / (T_C * T_W))
    h = math.sqrt(2.0 * area * (T_C * T_W) / (T_W - T_C))
    apex = (CX - h / T_W, CZ + h)
    return Polygon([(CX, CZ), (CX + b, CZ), apex]), b, h


def retreat(solid, h):
    """Parallel-retreat the EXPOSED face (above the current talus top `h`) by DX,
    returning the new solid and the shed rock volume V."""
    void = DOMAIN.difference(solid)
    exposed = box(-100.0, CZ + h + 1.0e-9, 100.0, 45.0)
    slab = solid.intersection(translate(void, xoff=-DX)).intersection(exposed)
    return solid.difference(slab), slab.area


x_top = -Z_TOP / T_W
solid = Polygon([(-100, -10), (100, -10), (100, 0), (0, 0), (x_top, Z_TOP), (-100, Z_TOP)])
total_V = 0.0
history = [(solid, Polygon(), 0.0, 0.0)]
print("shed |  exposed h |     V | run-out b | burial h | mass: talus*(1-lam) == rock removed")
for k in range(5):
    _, _, h = talus_wedge(total_V / (1.0 - LAMBDA))
    solid, V = retreat(solid, h)
    total_V += V
    apron, b, h_new = talus_wedge(total_V / (1.0 - LAMBDA))
    history.append((solid, apron, b, h_new))
    print("  %d  |   %6.2f  | %5.1f |   %6.2f  |  %6.2f  |   %7.2f == %7.2f"
          % (k + 1, h, V, b, h_new, apron.area * (1.0 - LAMBDA), total_V))

panels = [0, 1, 3, 5]
titles = ["initial cliff", "after shed 1", "after shed 3", "after shed 5"]
fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.3), sharex=True, sharey=True)
for ax, i, title in zip(axes, panels, titles):
    sd, ap, b, h = history[i]
    for geom, fc, hatch in [(sd, "#b8926a", "//"), (ap, "#9a8f7d", "xx")]:
        if geom.is_empty:
            continue
        xs, zs = geom.exterior.xy
        ax.fill(xs, zs, facecolor=fc, edgecolor="k", linewidth=0.7, hatch=hatch)
    if not ap.is_empty:
        ax.plot([CX + b], [CZ], "v", color="#2b7bba", ms=7, markeredgecolor="k")
        ax.annotate("h=%.1f" % h, xy=(CX - h / T_W, CZ + h), xytext=(6, 0),
                    textcoords="offset points", fontsize=8, va="center")
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_xlim(-50, 50); ax.set_ylim(-6, 44); ax.set_aspect(1.0)
    ax.set_xlabel("cross-valley distance [m]")
axes[0].set_ylabel("elevation [m]")

handles = [Patch(facecolor="#b8926a", edgecolor="k", hatch="//", label="bedrock (retreating cliff)"),
           Patch(facecolor="#9a8f7d", edgecolor="k", hatch="xx", label="talus (colluvium)"),
           plt.Line2D([], [], marker="v", color="#2b7bba", ls="", markeredgecolor="k", label="apron toe")]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Talus by parallel slope retreat: closed-form repose wedge grows, buries the "
             "wall base, and the retreat decelerates", fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0.06, 1, 0.96))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "talus_slope_retreat.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
