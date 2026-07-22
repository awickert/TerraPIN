#! /usr/bin/env python
"""
Benchmark the colluvial-pile repose-surface solvers.

Positioning the repose surface to trap a target area is the PLIC / volume-
conservation-enforcement problem from volume-of-fluid methods. We offer four
interchangeable backends; this script measures which is cheapest on our own
geometry, counting Shapely clips (the real cost) and wall-clock per solve.

Verdict (typical run): the analytic piecewise-quadratic solve uses the fewest
clips and ties Brent on wall-clock; Brent is the robust boring default; plain
bisection is ~5x slower. All agree to ~1e-14, so the default (Brent) is chosen
for robustness, not speed.

Run in the dedicated environment:
    conda run -n terrapin python benchmarks/pile_solvers.py
"""
import os
import sys
import time

import numpy as np

# Make the package importable however this script is launched.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shapely.geometry import box

import terrapin.geometry as g
from terrapin.geometry import (eroded_wedge, position_repose_surface,
                               _deposit_below_repose)

REPOSE = {"bedrock": 75.0, "alluvium": 32.0}
_BEDROCK = {"bedrock": box(-100.0, -60.0, 0.0, 0.0)}
_LAYERED = {"bedrock": box(-100.0, -60.0, 0.0, -10.0),
            "alluvium": box(-100.0, -10.0, 0.0, 0.0)}
WALLS = {
    "single 75deg":    eroded_wedge(-20.0, _BEDROCK, REPOSE),
    "piecewise 75/32": eroded_wedge(-20.0, _LAYERED, REPOSE),
}
METHODS = ["brent", "bisect", "secant", "analytic"]
ALPHA_C = 20.0
FRACTIONS = (0.2, 0.5, 0.85)      # fill fractions spanning the regimes


def main():
    slope = np.tan(np.deg2rad(ALPHA_C))
    # Count clips by wrapping the one function that touches Shapely.
    original = g._deposit_below_repose
    counter = {"n": 0}

    def counting(*args, **kwargs):
        counter["n"] += 1
        return original(*args, **kwargs)

    print("%-16s %-9s %8s %10s %11s"
          % ("wall", "method", "clips", "us/solve", "max|err|"))
    for wname, void in WALLS.items():
        cap = void.area
        for method in METHODS:
            g._deposit_below_repose = counting
            clips, reps = [], 200
            t0 = time.perf_counter()
            for _ in range(reps):
                for frac in FRACTIONS:
                    counter["n"] = 0
                    position_repose_surface(frac * cap, void, ALPHA_C, method=method)
                    clips.append(counter["n"])
            per_solve = (time.perf_counter() - t0) / (reps * len(FRACTIONS))
            g._deposit_below_repose = original

            errs = []
            for frac in FRACTIONS:
                c = position_repose_surface(frac * cap, void, ALPHA_C, method=method)
                errs.append(abs(_deposit_below_repose(c, void, slope).area
                                - frac * cap))
            print("%-16s %-9s %8.1f %10.1f %11.2e"
                  % (wname, method, np.mean(clips), per_solve * 1e6, max(errs)))


if __name__ == "__main__":
    sys.exit(main())
