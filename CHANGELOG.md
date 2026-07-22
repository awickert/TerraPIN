# Changelog

All notable changes to TerraPIN are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-23

First release of the rewritten library. The geometry core is now polygon algebra
on Shapely/GEOS, and TerraPIN is organized as a driver-agnostic geometry and
mass-balance engine.

### Changed
- Rewrote the geometry core as **polygon algebra** on Shapely/GEOS, replacing the
  legacy hand-rolled line-intersection and point-classification model. Mass
  conservation is now structural (eroded area = deposited area = sediment out).
- Ported the code to **Python 3 / NumPy 2**.

### Added
- **`Terrapin` state object** — a driver-agnostic engine told what happened
  (`incise`, `aggrade`, `plane_laterally`) that returns updated geometry and a
  mass balance, plus `compute_valley_width()` and `plot()`.
- **Material-following repose walls** — failure walls bend at each material
  contact, with a per-lithology angle of repose.
- **Colluvial-pile (talus) placement** via a PLIC volume-conservation solver.
- **Terrace and provenance tracking** — deposits carry their deposition age,
  surfaces their abandonment age; `terraces()` reads the stranded benches from
  the live geometry and reports each terrace's age (its abandonment).
- Runtime dependency declarations, a modern `pyproject.toml` build, and PyPI
  packaging; a pytest suite; worked examples and an architecture document.

### Removed
- The legacy line-intersection model and the Python-2 `ez_setup` build bootstrap.

[Unreleased]: https://github.com/awickert/TerraPIN/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/awickert/TerraPIN/compare/v0.0.0...v0.1.0
