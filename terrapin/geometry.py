#! /usr/bin/env python
"""
Polygon-algebra geometry for TerraPIN cross-sections, built on Shapely.

The cross-section is a stack of geologic layers, each represented as a Shapely
polygon. The physics is expressed as polygon algebra: incision removes material
along an angle-of-repose failure wall (a difference), aggradation adds a wedge
of alluvium (a union), and eroded and deposited volumes are simply polygon
areas -- so mass is conserved by construction rather than by bookkeeping.

We work on the left half of a symmetric valley: x runs from a far-field wall
(a large but finite stand-in for -infinity) up to the channel at x = 0.

This module is the robust replacement for the old hand-rolled line-intersection
and point-classification machinery, which re-derived topology from noisy
coordinates and so needed scattered rounding to survive. Here the geometric
decisions are delegated to GEOS, which evaluates them consistently.
"""
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from scipy.optimize import brentq, bisect, newton


def repose_wall(z_ch, stack):
    """Vertices of the failure wall rising from the channel bottom.

    Starting at the channel bottom (0, z_ch), the wall climbs up-valley and
    bends at every layer contact: each layer holds its own angle of repose, so
    the wall stands steep in strong rock and lies back in loose sediment.

    stack: layers from the bottom up, each (top_elevation, repose_angle_deg,
           lithology). The top of the last (uppermost) layer is the ground
           surface.

    Returns a list of (x, z) vertices from the channel bottom up to where the
    wall meets the ground surface. If the channel sits at or above the surface
    there is nothing to climb, and the single starting vertex is returned.
    """
    x, z = 0.0, float(z_ch)
    wall = [(x, z)]
    for z_top, angle, lithology in stack:
        if z >= z_top:
            # The channel is already at or above this layer; nothing to climb.
            continue
        slope = np.tan(np.deg2rad(angle))   # rise in z per unit horizontal run
        x -= (z_top - z) / slope            # step up-valley to the contact
        z = z_top
        wall.append((float(x), float(z)))
    return wall


def eroded_wedge(z_ch, stack):
    """Polygon of material removed by incising the channel down to z_ch.

    Bounded below-right by the repose wall, across the top by the old ground
    surface, and closed down the channel axis at x = 0. Returns an empty
    polygon when the channel is at or above the surface (nothing is removed).
    """
    wall = repose_wall(z_ch, stack)
    if len(wall) < 2:
        return Polygon()
    z_surface = stack[-1][0]
    return Polygon(wall + [(0.0, z_surface)])


def incise(bodies, z_ch, stack):
    """Incise the channel to z_ch; return updated bodies and eroded volumes.

    bodies: dict {name: Polygon} of the material bodies to cut.

    Returns (new_bodies, eroded) where new_bodies holds the trimmed polygons
    and eroded[name] is the area (volume per unit valley length) removed from
    each body. Because eroded area is measured as the intersection with the
    same wedge that is differenced away, removed volume and the drop in body
    area agree to floating-point tolerance -- mass is conserved by construction.
    """
    wedge = eroded_wedge(z_ch, stack)
    new_bodies, eroded = {}, {}
    for name, poly in bodies.items():
        eroded[name] = poly.intersection(wedge).area
        new_bodies[name] = poly.difference(wedge)
    return new_bodies, eroded


def aggrade(bodies, z_fill, domain, name="alluvium_fill"):
    """Fill the valley with alluvium up to the level z_fill (a union).

    Sediment drops into whatever open void lies below the fill level, giving a
    flat aggradation surface at z_fill. bodies is a dict {name: Polygon}; the
    deposit is added under `name`.

    Returns (new_bodies, deposited) where deposited is the area (volume per
    unit valley length) laid down. Because the deposit is exactly the slice of
    void below z_fill, it nests against the existing bodies with no overlap.
    """
    minx, miny, maxx, _ = domain.bounds
    void = domain.difference(unary_union(list(bodies.values())))
    below = box(minx, miny, maxx, z_fill)
    fill = void.intersection(below)
    new_bodies = dict(bodies)
    new_bodies[name] = fill
    return new_bodies, fill.area


# --- Colluvial pile: fit a talus of prescribed volume against the valley wall ---
#
# Failed bedrock does not vanish; it piles as talus against the freshly-cut
# valley wall. Loose colluvium takes up more room than the solid rock it came
# from, by the porosity "fluffing" factor 1 / (1 - lambda_p). We must place a
# deposit of that area whose free surface stands at the colluvium angle of
# repose, resting on the floor and leaning on the wall -- whatever the wall's
# shape (it is generally non-vertical and, where incision crossed materials,
# piecewise).
#
# This is a known, well-worn problem: positioning a line of fixed orientation
# (the repose surface) to cut a polygon (the eroded void) to a target area. In
# volume-of-fluid CFD it is PLIC interface positioning / volume conservation
# enforcement. The area swept below the repose line is monotone in the line's
# offset, so we simply solve area(offset) = target. Nothing new; boring on
# purpose.


def _repose_halfplane(offset, void, slope):
    """The half-plane below a repose line z = offset - slope * x, as a polygon.

    Sized to cover the void, so intersecting it with the void gives everything
    lying below the repose surface.
    """
    minx, miny, maxx, _ = void.bounds
    x0, x1 = minx - 1.0, maxx + 1.0
    z_top_x0 = offset - slope * x0
    z_top_x1 = offset - slope * x1
    # Keep the box bottom below both the void and the (sloped) top edge, so the
    # trapezoid never self-intersects when the repose line dips low.
    z_floor = min(miny, z_top_x0, z_top_x1) - 1.0
    return Polygon([(x0, z_floor), (x1, z_floor),
                    (x1, z_top_x1), (x0, z_top_x0)])


def _deposit_below_repose(offset, void, slope):
    """The part of the void lying below the repose surface at this offset."""
    return void.intersection(_repose_halfplane(offset, void, slope))


def _offset_bracket(void, slope):
    """Offsets that empty / fill the void, bracketing the solution.

    A boundary vertex (x, z) lies below the line z = offset - slope*x exactly
    when offset >= z + slope*x, so the extreme values of (z + slope*x) over the
    void bracket every partial fill.
    """
    e = np.array([z + slope * x for x, z in void.exterior.coords])
    return float(e.min()), float(e.max())


def position_repose_surface(target_area, void, alpha_c, method="brent",
                            first_guess=None):
    """Offset of the repose surface that traps `target_area` inside the void.

    area(offset) is continuous and monotonically increasing, so any bracketed
    or seeded root-finder converges. Methods (all agree to ~1e-12):
      'brent'    -- Brent-Dekker on the bracket (the boring default)
      'bisect'   -- bisection on the bracket (simplest; ~4x more clips)
      'secant'   -- secant seeded by `first_guess` (few clips when the guess is
                    good; this is where a closed-form first guess pays off)
      'analytic' -- locate the event interval containing the solution, where
                    area(offset) is exactly quadratic, and solve that quadratic
    """
    slope = np.tan(np.deg2rad(alpha_c))
    lo, hi = _offset_bracket(void, slope)

    def excess(offset):
        return _deposit_below_repose(offset, void, slope).area - target_area

    if method == "brent":
        return brentq(excess, lo, hi, xtol=1e-12)
    if method == "bisect":
        return bisect(excess, lo, hi, xtol=1e-12)
    if method == "secant":
        if first_guess is None:
            first_guess = lo + (hi - lo) * (target_area / void.area)
        return newton(excess, x0=first_guess, tol=1e-12, maxiter=100)
    if method == "analytic":
        # Events are the offsets at which a void vertex crosses the line; the
        # solution lies between two consecutive events, where the swept area is
        # a quadratic in the offset. Sample three points there and solve it.
        events = np.unique([z + slope * x for x, z in void.exterior.coords])
        a = lo
        for b in events[1:]:
            if excess(b) >= 0.0:
                break
            a = b
        xs = np.array([a, 0.5 * (a + b), b])
        ys = np.array([_deposit_below_repose(x, void, slope).area for x in xs])
        q2, q1, q0 = np.polyfit(xs, ys, 2)
        roots = np.roots([q2, q1, q0 - target_area])
        roots = roots[np.isreal(roots)].real
        roots = roots[(roots >= a - 1e-9) & (roots <= b + 1e-9)]
        return float(roots[0])
    raise ValueError("Unknown method: %r" % method)


def colluvial_pile(eroded_bedrock_area, void, alpha_c, lambda_p,
                   method="brent", first_guess=None):
    """Fit a talus of the (fluffed) eroded volume against the valley wall.

    The colluvium fills the eroded `void` from the floor upward, its free
    surface a straight line at the repose angle alpha_c, positioned so the
    deposit area equals the porosity-fluffed eroded volume

        A = eroded_bedrock_area / (1 - lambda_p).

    If the void cannot hold it all, the remainder overflows -- in the coupled
    setting, that overflow is colluvium delivered to the channel as sediment.

    Returns (pile, overflow): the talus Polygon and the area that would not fit.
    """
    A = eroded_bedrock_area / (1.0 - lambda_p)
    if void.area <= A:
        return void, A - void.area
    offset = position_repose_surface(A, void, alpha_c, method, first_guess)
    slope = np.tan(np.deg2rad(alpha_c))
    return _deposit_below_repose(offset, void, slope), 0.0
