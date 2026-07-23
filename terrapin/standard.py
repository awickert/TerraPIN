#! /usr/bin/env python
"""
The standard TerraPIN cross-section: two one-wall units sharing a mobile channel.

Where the symmetric model (terrapin.Terrapin) holds the channel at a fixed lateral
position and evolves the valley in symmetric bulk, the standard model resolves the
channel's position on the valley floor. It is built from two one-wall
half-sections -- a left unit occupying x <= x_ch and a right unit occupying
x >= x_ch -- that share a channel at the mobile position x_ch. An external driver
supplies x_ch (and the vertical motions); TerraPIN does the geometry.

The one-wall engine in terrapin.geometry is reused unchanged: it builds a wall
rising up-valley to the left of a channel at x = 0. This module is the thin
composition layer that applies it to both walls at an arbitrary x_ch, by working
in a frame translated so the channel sits at x = 0 and reflecting that frame about
the channel for the right unit. So "the unit code is shared" is literal --
geometry.py stays a one-wall engine, and asymmetry lives only in the composition.

This is a work in progress: incision (both walls) is in place; migration,
avulsion, aggradation, and talus dynamics are to follow.
"""
import numpy as np
from shapely.geometry import box, LineString
from shapely.affinity import scale, translate
from shapely.ops import unary_union

from . import geometry

__all__ = ["StandardTerrapin"]


def _reflect(geom, x0=0.0):
    """Reflect a geometry about the vertical line x = x0."""
    return scale(geom, xfact=-1.0, yfact=1.0, origin=(x0, 0.0))


class StandardTerrapin(object):
    """
    Terraces Put Into Numerics -- the standard model: two one-wall units (left and
    right) sharing a mobile channel on the valley floor between them.
    """
    def __init__(self):
        self.bodies = None          # {name: shapely Polygon}: the material bodies
        self.x_ch = 0.              # channel lateral position [m] (mobile)
        self.z_ch = None            # channel-bed elevation [m]
        self.channel_width = 0.     # flat width the incising river carves [m]
        self.channel_depth = 0.     # channel depth an avulsion cuts on landing [m]
        self.repose_angles = None   # {lithology: angle of repose [degrees]}
        self.lambda_p = 0.35        # sediment porosity (fluffs eroded rock into colluvium)
        self.eroded = None          # {name: area}: material removed by the last cut
        self.deposited = 0.         # material laid down by the last aggradation [area]
        self.sediment_out = 0.      # material exported by the last operation [area]
        self._n_fill = 0            # counter so each aggradation gets its own body

    # ----------------------------- configuration -----------------------------

    def set_bodies(self, bodies):
        """Set the material bodies: a dict {name: shapely Polygon} spanning the
        full valley (both sides of the channel)."""
        self.bodies = dict(bodies)

    def set_channel_position(self, x_ch):
        """Set the channel's lateral position on the valley floor."""
        self.x_ch = x_ch

    def set_channel_elevation(self, z_ch):
        """Set the channel-bed elevation."""
        self.z_ch = z_ch

    def set_channel_width(self, channel_width):
        """Set the flat channel width the incising river carves (default 0)."""
        self.channel_width = channel_width

    def set_channel_depth(self, channel_depth):
        """Set the channel depth an avulsion cuts into the surface where it lands."""
        self.channel_depth = channel_depth

    def set_repose_angles(self, repose_angles):
        """Set the angle of repose of each lithology: {lithology: degrees}."""
        self.repose_angles = repose_angles

    def set_porosity(self, lambda_p):
        """Set the sediment porosity used to fluff eroded rock into colluvium."""
        self.lambda_p = lambda_p

    # -------------------------- operations (told to it) ----------------------

    def incise(self, z_ch):
        """
        Incise the channel bed to z_ch at the current position x_ch, cutting BOTH
        walls: a flat channel of the current width with a material-following repose
        wall rising up-valley on each side. Eroded material is swept away as
        sediment. Returns nothing; updates bodies and reports the mass balance in
        self.eroded / self.sediment_out.
        """
        notch = self._two_wall_wedge(z_ch, self.channel_width / 2.)
        self._remove(notch)
        self.z_ch = z_ch

    def migrate(self, x_new):
        """
        Migrate the channel laterally to x_new at the current bed elevation,
        planing a strath across the swept corridor and undercutting the wall it
        advances INTO. The retreating side's wall is left untouched (its strath is
        abandoned, and can shed talus later). All eroded material is swept away as
        sediment. Geometrically this rebuilds only the advancing wall: a one-wall
        wedge whose floor reaches from the old channel to x_new plus half a channel
        width, on whichever side the channel moved.
        """
        if x_new == self.x_ch:
            self.eroded = {n: 0.0 for n in self.bodies}
            self.sediment_out = 0.0
            return
        side = "right" if x_new > self.x_ch else "left"
        floor_half_width = abs(x_new - self.x_ch) + self.channel_width / 2.
        wedge = self._wall_wedge(self.z_ch, floor_half_width, self.x_ch, side)
        self._remove(wedge)
        self.x_ch = x_new

    def avulse(self, x_new):
        """
        Avulse the channel to x_new: a discontinuous hop. Unlike migration, the old
        channel is abandoned IN PLACE and the ground between old and new positions
        is NOT planed -- so the vacated belt is preserved, to be buried by later
        aggradation. Where it lands, the new channel cuts down one channel depth
        over one channel width, eroding a block of material from the local surface
        downward; that block is exported as sediment. The new bed sits one channel
        depth below the surface at x_new.
        """
        surface = self._surface_elevation(x_new)
        z_bed = surface - self.channel_depth
        half_width = self.channel_width / 2.
        block = box(x_new - half_width, z_bed, x_new + half_width, surface)
        self._remove(block)
        self.x_ch = x_new
        self.z_ch = z_bed

    def aggrade(self, z_fill):
        """
        Fill the valley with alluvium up to the level z_fill. Each aggradation is
        stored as its own body ('alluvium_fill_N'), so repeated fills accumulate.
        Aggradation does not depend on the channel position: it fills whatever open
        void lies below z_fill across the valley.
        """
        name = "alluvium_fill_%d" % self._n_fill
        self.bodies, self.deposited = geometry.aggrade(
            self.bodies, z_fill, self._domain(z_fill), name=name)
        self._n_fill += 1
        self.z_ch = z_fill
        self.sediment_out = 0.

    # -------------------------------- helpers --------------------------------

    def _remove(self, wedge):
        """Difference an eroded wedge from the bodies, recording the mass balance."""
        self.eroded = {n: g.intersection(wedge).area for n, g in self.bodies.items()}
        self.bodies = {n: g.difference(wedge) for n, g in self.bodies.items()}
        self.sediment_out = sum(self.eroded.values())

    def _wall_wedge(self, z_ch, floor_half_width, x_axis, side):
        """One wall's eroded wedge, with the channel axis at x_axis.

        The one-wall engine builds a wall rising up-valley to the LEFT of a channel
        at x = 0. For the left wall it is used directly; for the right wall it is
        run in a frame reflected about the channel, then reflected back. In both
        cases the frame is translated so the channel axis sits at x = 0.
        """
        shifted = {n: translate(g, xoff=-x_axis) for n, g in self.bodies.items()}
        if side == "right":
            shifted = {n: _reflect(g) for n, g in shifted.items()}
        wedge = geometry.eroded_wedge(z_ch, shifted, self.repose_angles, floor_half_width)
        if side == "right":
            wedge = _reflect(wedge)
        return translate(wedge, xoff=x_axis)

    def _two_wall_wedge(self, z_ch, floor_half_width):
        """The combined eroded wedge of both walls' notch at the current x_ch."""
        return unary_union([
            self._wall_wedge(z_ch, floor_half_width, self.x_ch, "left"),
            self._wall_wedge(z_ch, floor_half_width, self.x_ch, "right")])

    def _domain(self, z_top):
        """A bounding box that spans the bodies and reaches above z_top."""
        minx, miny, maxx, maxy = unary_union(list(self.bodies.values())).bounds
        return box(minx, miny, maxx, max(maxy, z_top) + 1.)

    def _surface_elevation(self, x, _reach=1.0e6):
        """Elevation of the top of the solid column at cross-valley position x."""
        solid = unary_union([g for g in self.bodies.values() if not g.is_empty])
        hit = LineString([(x, -_reach), (x, _reach)]).intersection(solid)
        return None if hit.is_empty else hit.bounds[3]      # bounds[3] = maxy = top
