#! /usr/bin/env python
"""
The Terrapin cross-section: a river valley built from material polygons and
evolved by incision, aggradation, and lateral planation.

Terrapin holds the model state -- the material bodies, the channel elevation and
width, and the material properties -- and exposes the geometric operations as
methods. It is driver-agnostic: the caller supplies the vertical and lateral
motions (from a long-profile model, a lateral-migration model, or a test) and
Terrapin does the geometry and reports the mass balance. The heavy geometry
lives in terrapin.geometry; this is the stateful wrapper around it.

Configure with the set_* methods, then drive with incise / aggrade /
plane_laterally; read the emergent width with compute_valley_width().
"""
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import box
from shapely.ops import unary_union

from . import geometry


class Terrapin(object):
    """
    Terraces Put Into Numerics: one evolving river-valley cross-section.
    """
    def __init__(self):
        self.bodies = None          # {name: shapely Polygon}: the material bodies
        self.z_ch = None            # channel-bed elevation [m]
        self.channel_width = 0.     # flat width the incising river carves [m]; 0 is a point
        self.repose_angles = None   # {lithology: angle of repose [degrees]}
        self.lambda_p = 0.35        # sediment porosity (fluffs eroded rock into colluvium)
        self.t = 0.                 # model time
        self.eroded = None          # {name: area}: material removed by the last cut
        self.sediment_out = 0.      # material exported by the last operation [area]
        self._n_fill = 0            # counter so each aggradation gets its own body

    # ----------------------------- configuration -----------------------------

    def set_bodies(self, bodies):
        """
        Set the material bodies: a dict {name: shapely Polygon}. Each name holds
        its lithology ('bedrock', 'alluvium', 'colluvium'), which selects the
        angle of repose.
        """
        self.bodies = dict(bodies)

    def set_channel_elevation(self, z_ch):
        """Set the channel-bed elevation."""
        self.z_ch = z_ch

    def set_channel_width(self, channel_width):
        """
        Set the flat channel width the incising river carves. May be reset at any
        time; default 0 (the channel is a point).
        """
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
        Incise the channel bed to z_ch, carving a flat channel of the current
        width. The eroded material is swept away by the river as sediment.
        """
        self.bodies, self.eroded = geometry.incise(
            self.bodies, z_ch, self.repose_angles,
            floor_half_width=self.channel_width / 2.)
        self.z_ch = z_ch
        self.sediment_out = sum(self.eroded.values())

    def aggrade(self, z_fill):
        """
        Fill the valley with alluvium up to the level z_fill. Each aggradation is
        stored as its own body ('alluvium_fill_N'), so repeated fills accumulate
        rather than overwrite one another.
        """
        name = "alluvium_fill_%d" % self._n_fill
        self.bodies, self.deposited = geometry.aggrade(
            self.bodies, z_fill, self._domain(z_fill), name=name)
        self._n_fill += 1
        self.z_ch = z_fill
        self.sediment_out = 0.

    def plane_laterally(self, channel_width):
        """
        Widen the valley by lateral planation to a new channel width, at the
        current bed elevation. The swept rock is exported as sediment; the river
        cannot leave talus in its own path (see terrapin.geometry.widen).
        """
        self.bodies, self.balance = geometry.widen(
            self.bodies, self.z_ch, channel_width / 2., self.repose_angles)
        self.channel_width = channel_width
        self.sediment_out = self.balance["sediment_out"]

    # -------------------------------- outputs --------------------------------

    def compute_valley_width(self):
        """
        Emergent width of the valley floor at the current bed elevation: the
        cross-valley distance out to the first thing higher than the floor.
        """
        self.valley_width = geometry.valley_width(self.bodies, self.z_ch)
        return self.valley_width

    # -------------------------------- helpers --------------------------------

    def _domain(self, z_top):
        """A bounding box that spans the bodies and reaches above z_top."""
        minx, miny, maxx, maxy = unary_union(list(self.bodies.values())).bounds
        return box(minx, miny, maxx, max(maxy, z_top) + 1.)

    def plot(self, mirror=True):
        """
        Draw the cross-section, filling each material and (by default) mirroring
        one wall about the channel to show a full symmetric valley.
        """
        fill = {'bedrock': ('#b8926a', '//'), 'alluvium': ('#e6cf7a', '..'),
                'colluvium': ('#9a8f7d', 'xx')}
        def lithology(name):
            for key in fill:
                if key in name:
                    return key
            return 'alluvium'
        fig, ax = plt.subplots()
        for name, geom in self.bodies.items():
            if geom.is_empty:
                continue
            facecolor, hatch = fill[lithology(name)]
            parts = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
            for p in parts:
                xs, zs = p.exterior.xy
                xs = np.asarray(xs)
                for sign in ((1, -1) if mirror else (1,)):
                    ax.fill(sign * xs, zs, facecolor=facecolor, edgecolor='k',
                            linewidth=0.6, hatch=hatch)
        ax.plot(0, self.z_ch, 'v', color='#1f6fb2', markeredgecolor='k')
        ax.set_xlabel('cross-valley distance [m]')
        ax.set_ylabel('elevation [m]')
        ax.set_aspect('equal')
        return ax
