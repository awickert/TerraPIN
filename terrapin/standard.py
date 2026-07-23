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
        self.repose_angles = None   # {lithology: angle of repose [degrees]}
        self.lambda_p = 0.35        # sediment porosity (fluffs eroded rock into colluvium)
        self.eroded = None          # {name: area}: material removed by the last cut
        self.sediment_out = 0.      # material exported by the last operation [area]

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
        self.eroded = {n: g.intersection(notch).area for n, g in self.bodies.items()}
        self.bodies = {n: g.difference(notch) for n, g in self.bodies.items()}
        self.z_ch = z_ch
        self.sediment_out = sum(self.eroded.values())

    # -------------------------------- helpers --------------------------------

    def _two_wall_wedge(self, z_ch, floor_half_width):
        """The combined eroded wedge of both walls' notch at the current x_ch.

        Work in a frame translated so the channel sits at x = 0. The left wall is
        the one-wall engine directly; the right wall is the same engine run in a
        frame reflected about the channel, then reflected back. Translate the union
        back to the channel's true position.
        """
        shifted = {n: translate(g, xoff=-self.x_ch) for n, g in self.bodies.items()}
        left = geometry.eroded_wedge(z_ch, shifted, self.repose_angles, floor_half_width)
        mirrored = {n: _reflect(g) for n, g in shifted.items()}
        right = _reflect(geometry.eroded_wedge(z_ch, mirrored, self.repose_angles,
                                               floor_half_width))
        return translate(unary_union([left, right]), xoff=self.x_ch)
