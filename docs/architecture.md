# TerraPIN architecture

TerraPIN models a river-valley cross-section as **polygons** (Shapely/GEOS) and
expresses the physics as polygon algebra: incision and lateral planation remove
material, aggradation and talus add it, and eroded/deposited volumes are polygon
areas — so mass is conserved by construction. See `terrapin/geometry.py`.

## Driver-agnostic engine

TerraPIN is a **geometry + mass-balance engine**. It is *told* what happened
("incise to z", "the channel is now at x", "widen to half-width w", "aggrade to
level") and reports back updated geometry plus a mass balance (eroded and
deposited volumes, sediment out). It does **not** decide rates, migration rules,
erosion laws, or discharge — those live in whatever drives it (e.g. GRLP for
vertical change, a separate model for lateral channel motion, a stochastic
forcing, or a test). The only properties TerraPIN holds are geometric/material
(repose angles, porosity), and even those are supplied by the caller.

A key disposition rule: **the active river's zone exports everything.** Wherever
the river is currently working — incising *or* laterally planing — all eroded
material is carried off as sediment and the swept floor is left clean; the river
cannot leave a talus pile in its own path. Talus is a separate, *river-absent*
process (see below).

## The one-wall unit; vertical-only vs. the standard model

The atomic unit is a **one-wall half-section**: a single valley wall, the flat
strath floor from the channel (`x = 0`) out to the wall base at `x = -w`, and the
channel bank on that side. Every operation in `geometry.py` acts on one such unit.

TerraPIN's two configurations differ only in **lateral channel dynamics**:

- **vertical-only** (the simplification) — the channel moves only vertically
  (incise / aggrade) and holds its lateral position; the valley widens
  symmetrically around it. Built as *one* unit, **mirrored** about the channel to
  give a full symmetric valley (width `= 2w`). Because the river is always present
  at the (single, mirrored) wall, **persistent talus is essentially absent** —
  which is honest; talus needs a wall the river has left.

- **standard** (the default, unqualified TerraPIN) — adds **lateral migration and
  avulsion**: *two* units (left + right) share a **mobile channel** on the floor
  between them. An external driver moves the channel; whichever wall it is against
  gets undercut (planed → sediment), while the wall it has **left** sheds a talus
  apron onto the quiet strath, which is **re-eroded** when the channel migrates
  back. This yields valley asymmetry, one-sided terrace preservation, and an
  **emergent** valley width — none of which the symmetric vertical-only model can
  produce. The unit code is shared, so vertical-only is literally one instance of
  the same building block, not a special case of a heavier one.

Persistent, dynamic talus is therefore intrinsically a **standard-model**
phenomenon; the colluvial-pile machinery is built and validated but does little
physical work until the mobile channel exists.
