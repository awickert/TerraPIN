#! /usr/bin/env python
"""
Example: terraces from a random-walk channel-elevation history.

Drives Terrapin with a stochastic bed history -- an incision-dominated random
walk followed by a re-aggradation phase -- as an external driver (a stochastic
forcing, or a long-profile model) would, and draws the resulting cross-section.
Ported from the old randWalk.py demo onto the polygon library.

Run in the dedicated environment:
    conda run -n terrapin python examples/random_walk_terraces.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import box

from terrapin import Terrapin

rng = np.random.default_rng(3)

tp = Terrapin()
tp.set_bodies({"bedrock": box(-160.0, -80.0, 0.0, -8.0),
               "alluvium": box(-160.0, -8.0, 0.0, 0.0)})
tp.set_repose_angles({"bedrock": 75.0, "alluvium": 32.0, "colluvium": 20.0})
tp.set_channel_elevation(0.0)

# A stochastic history: the river mostly cuts down a narrow channel, but now and
# then it lingers and planes a strath. Each strath is inset (narrower) into the
# one above -- as a real valley narrows while it incises -- so the wider, older
# straths higher up survive as terraces: a random history builds a staircase.
NARROW = 6.0
z, strath_width = 0.0, 120.0
for step in range(22):
    z -= rng.uniform(1.0, 3.0)             # cut down a random amount ...
    tp.set_channel_width(NARROW)           # ... in a narrow channel
    tp.incise(z)
    if rng.random() < 0.65:                # often linger and plane a strath
        tp.plane_laterally(strath_width)
        strath_width = max(strath_width * rng.uniform(0.7, 0.86), NARROW + 4.0)

ax = tp.plot()
ax.set_title("Terraces from a random-walk elevation history")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "random_walk_terraces.png")
ax.figure.savefig(out, dpi=140, bbox_inches="tight")
print("final bed %.2f m, valley width %.2f m"
      % (tp.z_ch, tp.compute_valley_width()))
print("wrote", out)
