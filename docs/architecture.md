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

## The one-wall unit; symmetric vs. the standard model

The atomic unit is a **one-wall half-section**: a single valley wall, the flat
strath floor from the channel (`x = 0`) out to the wall base at `x = -w`, and the
channel bank on that side. Every operation in `geometry.py` acts on one such unit.

TerraPIN's two configurations differ only in **lateral channel dynamics**:

- **symmetric** (the simplification) — the channel holds its lateral position
  (no migration or avulsion); the valley incises, aggrades, and planes laterally
  in symmetric bulk, without resolving where the channel is. Built as *one* unit,
  **mirrored** about the channel to
  give a full symmetric valley (width `= 2w`). Because the river is always present
  at the (single, mirrored) wall, **persistent talus is essentially absent** —
  which is honest; talus needs a wall the river has left.

- **standard** (the default, unqualified TerraPIN) — adds **lateral migration and
  avulsion**: *two* units (left + right) share a **mobile channel** on the floor
  between them. An external driver moves the channel; whichever wall it is against
  gets undercut (planed → sediment), while the wall it has **left** sheds a talus
  apron onto the quiet strath, which is **re-eroded** when the channel migrates
  back. This yields valley asymmetry, one-sided terrace preservation, and an
  **emergent** valley width — none of which the symmetric model can
  produce. The unit code is shared, so symmetric is literally one instance of
  the same building block, not a special case of a heavier one.

Persistent, dynamic talus is therefore intrinsically a **standard-model**
phenomenon; the colluvial-pile machinery is built and validated but does little
physical work until the mobile channel exists.

## Terrace and provenance tracking

TerraPIN records enough history to read its terraces back out, and it keeps two
times cleanly apart:

- A **deposit** (a material body) holds the age of its **formation**. A fill
  carries the age at which it aggraded; pre-existing bodies have no known
  formation age. This lives in `Terrapin.provenance`,
  `{name: {kind, lithology, age}}`.
- A **surface** holds the age of its **abandonment** — when the river left it
  behind. This lives in `Terrapin.surfaces`, `{kind, z, abandoned}`.

A **terrace is an abandoned surface**, so **its age is the age of abandonment,
and nothing else**. A fill's deposition age belongs to the deposit the terrace is
cut on, not to the terrace, so a fill terrace reports two distinct times — the
fill's deposition and the terrace's abandonment — and never conflates them.

Which operation writes which age follows the physics:

- `aggrade(z, age)` lays down a deposit and stamps it with its **formation** age.
- `plane_laterally(w)` **cuts** a strath but does *not* abandon it: the planed
  floor is the live channel bed and stays the bed until a later incision drops
  below it. The cutting takes time, but that duration is not tracked — a strath's
  age is the *instant* it is abandoned — so this operation carries no time. The
  strath is logged as a live surface, to be stamped when it is stranded.
- `incise(z, age)` is the **abandoning** event for everything it strands above the
  new bed — the old channel floor (a fill top or a planed strath), a
  buried-then-exposed bench, the valley margin — stamping each newly stranded
  surface with the incision's age. This is what turns a strath or a fill top into
  a terrace, and its age is the terrace age.

The timing is an **input the caller pins**, driver-agnostic like every other
quantity: incision and aggradation take an optional `age`, either a point or a
`(start, end)` span for a process that acts over a duration.

`Terrapin.terraces()` then reports the terraces present now. It reads the flat
benches straight from the **live geometry** (`geometry.treads_above`), so a
re-incision that eats into a bench shortens what is reported to what actually
survives — the extent is measured, not remembered. Each bench is returned with
its **abandonment** age (its terrace age) and, kept separate, the **deposition**
age of the fill it caps (if any), plus its elevation, extent, kind
(`strath` / `fill` / `initial`), and lithology.
