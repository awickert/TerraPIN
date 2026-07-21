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
