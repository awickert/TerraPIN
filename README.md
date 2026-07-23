[![DOI](https://zenodo.org/badge/29247260.svg)](https://zenodo.org/badge/latestdoi/29247260)
[![tests](https://github.com/MNiMORPH/TerraPIN/actions/workflows/tests.yml/badge.svg)](https://github.com/MNiMORPH/TerraPIN/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/terrapin-valley.svg)](https://pypi.org/project/terrapin-valley/)

# TerraPIN

**Terrapin** (or **TerraPIN**) stands for "Terraces Put Into Numerics". It
generates the terraces — both strath and fill — that a river leaves as it
incises, aggrades, and planes laterally across its valley.

STARTED 09 SEPTEMBER 2013 (written by ADW)

## What it is

TerraPIN models a river-valley cross-section as material **polygons**
(Shapely/GEOS) and expresses the physics as polygon algebra: incision and
lateral planation remove material, aggradation and talus add it, and eroded and
deposited volumes are polygon areas — so mass is conserved by construction
rather than by bookkeeping.

It is a **driver-agnostic geometry and mass-balance engine**. You tell it what
happened — "incise to `z`", "plane to channel width `w`", "aggrade to level" —
and it reports back the updated geometry, the mass balance (eroded and deposited
volumes, sediment out), and the emergent valley width. It does *not* decide
rates, erosion laws, or channel migration; those live in whatever drives it (for
example [GRLP](https://github.com/awickert/GRLP) for vertical change, a separate
lateral-migration model, or a test).

## Terraces and provenance

Each material body is a **deposit** that carries the age of its **formation**;
each stranded **surface** carries the age of its **abandonment**. A terrace is an
abandoned surface, so its age is the age of abandonment — and the deposit it is
cut on keeps its own, separate deposition age. `Terrapin.terraces()` reads the
benches straight from the live geometry and returns both, so a cut–fill–recut
history is legible as strath and fill terraces annotated with when they formed
and when the river left them behind.

## A minimal run

```python
from shapely.geometry import box
from terrapin import Terrapin

tp = Terrapin()
tp.set_bodies({"bedrock":  box(-140.0, -80.0, 0.0, -6.0),
               "alluvium": box(-140.0,  -6.0, 0.0,  0.0)})
tp.set_repose_angles({"bedrock": 75.0, "alluvium": 32.0, "colluvium": 20.0})
tp.set_channel_elevation(0.0)

tp.incise(-8.0, age=5.0)                     # cut down; strand the land surface
tp.plane_laterally(95.0)                      # plane a strath (its age is set when abandoned)
tp.set_channel_width(0.0)
tp.aggrade(-3.0, age=22.0)                    # a valley fill
tp.incise(-20.0, age=33.0)                    # re-incise; strand the fill top

for t in tp.terraces():
    print(t["kind"], "at z =", t["z"],
          "| terrace age (abandoned) =", t["age"],
          "| deposit age =", t["deposit_age"])
```

## Install and dependencies

TerraPIN is a small pure-Python package; install it from source (`pip install
-e .` or by putting the repository on your `PYTHONPATH`). The geometry engine
requires **Python 3**, **NumPy 2**, **Shapely** (GEOS), and **SciPy**;
**Matplotlib** is used for plotting. A dedicated environment keeps these
consistent — for example:

```bash
conda create -n terrapin numpy shapely scipy matplotlib pytest
conda run -n terrapin python -m pytest
```

## Learn more

- **[`docs/architecture.md`](docs/architecture.md)** — the model: polygon
  algebra, the driver-agnostic engine, the one-wall unit, symmetric vs. the
  standard model, and terrace/provenance tracking.
- **[`examples/`](examples)** — worked cross-sections, in
  [`symmetric/`](examples/symmetric) (e.g. `terrace_tracking.py` — terraces read
  back and labelled by age) and [`standard/`](examples/standard) (e.g.
  `standard_terraces.py` — migration, avulsion, channel belts, and terraces).
- **[AlluvStrat](https://github.com/awickert/alluvstrat)** — ADW's earlier raster
  model of alluvial stratigraphy in strike-section.

## Citing

If you use TerraPIN, please cite it: GitHub builds a ready-made citation from
[`CITATION.cff`](CITATION.cff) (the "Cite this repository" button), and every
release is archived on Zenodo with a DOI.
