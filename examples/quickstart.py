#! /usr/bin/env python
"""
Quickstart: the smallest useful TerraPIN run.

Configure a cross-section, drive it through incision, lateral planation, and
aggradation, read the emergent valley width, and plot it. (The richer,
annotated terrace story is in vertical_only_terraces.py.)

Run in the dedicated environment:
    conda run -n terrapin python examples/quickstart.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import box

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from terrapin import Terrapin

# Configure: alluvium over bedrock, river at the surface.
tp = Terrapin()
tp.set_bodies({"bedrock": box(-200.0, -100.0, 0.0, -8.0),
               "alluvium": box(-200.0, -8.0, 0.0, 0.0)})
tp.set_repose_angles({"bedrock": 75.0, "alluvium": 32.0, "colluvium": 20.0})
tp.set_channel_elevation(0.0)
tp.set_channel_width(20.0)          # the river carves a 20 m-wide channel

# Drive it (the motions would come from an external model or forcing).
tp.incise(-15.0)                    # cut down to -15 m
print("after incision : valley width = %.1f m" % tp.compute_valley_width())
tp.plane_laterally(50.0)            # widen the valley floor to 50 m
print("after planation : valley width = %.1f m, %.0f m2 to sediment"
      % (tp.compute_valley_width(), tp.sediment_out))
tp.aggrade(-6.0)                    # fill to -6 m

ax = tp.plot()
ax.set_title("TerraPIN quickstart")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quickstart.png")
ax.figure.savefig(out, dpi=120, bbox_inches="tight")
print("wrote", out)
