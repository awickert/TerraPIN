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

__all__ = ["Terrapin"]


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
        self.provenance = {}        # {name: {kind, lithology, age}}: deposit + its formation age
        self.surfaces = []          # [{kind, z, abandoned, on}]: surfaces + their abandonment age

    # ----------------------------- configuration -----------------------------

    def set_bodies(self, bodies):
        """
        Set the material bodies: a dict {name: shapely Polygon}. Each name holds
        its lithology ('bedrock', 'alluvium', 'colluvium'), which selects the
        angle of repose. The initial bodies are pre-existing, so their formation
        age is unknown (None); later deposits stamp their own.
        """
        self.bodies = dict(bodies)
        for name in self.bodies:
            self._record_deposit(name, kind="initial", age=None)
        miny = unary_union(list(self.bodies.values())).bounds[1]
        for z, _x_far, _x_near in geometry.treads_above(self.bodies, miny - 1.):
            self._record_surface("initial", z, abandoned=None)

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

    def incise(self, z_ch, age=None):
        """
        Incise the channel bed to z_ch, carving a flat channel of the current
        width. The eroded material is swept away by the river as sediment.

        Incision is the abandoning event: every surface it strands above the new
        bed -- the old channel floor, buried-then-re-exposed benches, the valley
        margin -- is a terrace from now on. The optional `age` (a point, or a
        (start, end) span if the incision took time) is stamped on each surface
        newly abandoned by this cut; surfaces already abandoned keep their age.
        """
        self._abandon_stranded(z_ch, age)
        self.bodies, self.eroded = geometry.incise(
            self.bodies, z_ch, self.repose_angles,
            floor_half_width=self.channel_width / 2.)
        self.z_ch = z_ch
        self.sediment_out = sum(self.eroded.values())

    def aggrade(self, z_fill, age=None):
        """
        Fill the valley with alluvium up to the level z_fill. Each aggradation is
        stored as its own body ('alluvium_fill_N'), so repeated fills accumulate
        rather than overwrite one another.

        The fill is a deposit, so it carries its formation age: the optional
        `age` (a point, or a (start, end) span) is recorded as the deposit's
        formation time. Its top is the live valley floor until a later incision
        strands it as a fill terrace.
        """
        name = "alluvium_fill_%d" % self._n_fill
        self.bodies, self.deposited = geometry.aggrade(
            self.bodies, z_fill, self._domain(z_fill), name=name)
        self._record_deposit(name, kind="fill", age=age)
        self._record_surface("fill", z_fill, abandoned=None)
        self._n_fill += 1
        self.z_ch = z_fill
        self.sediment_out = 0.

    def plane_laterally(self, channel_width):
        """
        Widen the valley by lateral planation to a new channel width, at the
        current bed elevation. The swept rock is exported as sediment; the river
        cannot leave talus in its own path (see terrapin.geometry.widen).

        Planation cuts the strath but does not abandon it: the planed floor is the
        live channel bed and stays the bed until a later incision drops below it
        and strands it. The strath is therefore logged here as a live surface,
        with no age -- the planation takes time but that duration is not tracked;
        the strath's age is the instant of its abandonment, stamped by the
        incision that strands it (as for a fill top). So this operation takes no
        timing argument.
        """
        self.bodies, self.balance = geometry.widen(
            self.bodies, self.z_ch, channel_width / 2., self.repose_angles)
        self._record_surface("strath", self.z_ch, abandoned=None)
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

    def terraces(self):
        """
        The terraces present now: flat benches of ground stranded above the
        channel, read from the live geometry (so re-incision that eats into a
        bench shortens it to what actually survives).

        A terrace's age is the age at which its surface was ABANDONED -- when the
        river left it behind -- and nothing else. A fill's deposition age
        (`deposit_age`) belongs to the deposit the terrace is cut on, not to the
        terrace, and is carried here only as provenance.

        Returns a list of dicts, valley-floor upward, each with:
          z            elevation of the tread
          x_near, x_far  cross-valley edges (near / away from the channel at x=0)
          width        tread width (per unit valley length)
          kind         surface origin ('strath', 'fill', 'initial')
          age          the terrace age: when it was abandoned (point or span)
          body         name of the deposit the terrace is cut on / capped by
          deposit_age  that deposit's own deposition age (None if not deposited)
          lithology    its lithology
        """
        out = []
        for z, x_far, x_near in geometry.treads_above(self.bodies, self.z_ch):
            xm = 0.5 * (x_far + x_near)
            body = geometry._material_at(self.bodies, xm, z - self._PROBE)
            prov = self.provenance.get(body, {})
            surf = self._surface_at(z) or {}
            out.append({
                "z": z,
                "x_near": x_near,
                "x_far": x_far,
                "width": x_near - x_far,
                "kind": surf.get("kind", "initial"),
                "age": surf.get("abandoned"),
                "body": body,
                "deposit_age": prov.get("age"),
                "lithology": prov.get("lithology"),
            })
        return out

    # -------------------------------- helpers --------------------------------

    _PROBE = 1e-4       # vertical nudge to sample the material just under a surface

    def _record_deposit(self, name, kind, age):
        """Log a body's provenance: its origin and formation age."""
        self.provenance[name] = {
            "kind": kind,
            "lithology": geometry._lithology(name),
            "age": age,
        }

    def _record_surface(self, kind, z, abandoned=None):
        """Log a surface and the age at which it was abandoned (its terrace age).
        A surface is created live (abandoned=None) and stamped when an incision
        strands it."""
        self.surfaces.append({"kind": kind, "z": z, "abandoned": abandoned})

    def _surface_at(self, z, _tol=1e-5):
        """The most recently logged surface at elevation z, or None."""
        for surf in reversed(self.surfaces):
            if abs(surf["z"] - z) <= _tol:
                return surf
        return None

    def _abandon_stranded(self, z_new, age):
        """Stamp `age` on every exposed surface this incision strands above z_new.

        Called before the cut, on the pre-incision geometry, so the surfaces are
        read while still intact. The old channel floor and any bench above the
        new bed are abandoned now; surfaces already abandoned (or buried, hence
        not exposed) are left untouched. A stranded surface not yet logged -- the
        original valley margin -- is recorded here as it becomes a terrace.
        """
        for z, x_far, x_near in geometry.treads_above(self.bodies, z_new):
            surf = self._surface_at(z)
            if surf is None:                        # a freshly cut floor, now stranded
                xm = 0.5 * (x_far + x_near)
                body = geometry._material_at(self.bodies, xm, z - self._PROBE)
                kind = "strath" if geometry._lithology(body) == "bedrock" else "fill"
                self._record_surface(kind, z, abandoned=age)
            elif surf["abandoned"] is None:
                surf["abandoned"] = age

    def _domain(self, z_top):
        """A bounding box that spans the bodies and reaches above z_top."""
        minx, miny, maxx, maxy = unary_union(list(self.bodies.values())).bounds
        return box(minx, miny, maxx, max(maxy, z_top) + 1.)

    @staticmethod
    def _fmt_age(age):
        """A compact string for a terrace age -- a point or a (start, end) span."""
        if age is None:
            return "?"
        if isinstance(age, tuple):
            return "%g–%g" % age                 # start-end (en dash)
        return "%g" % age

    def plot(self, mirror=True, show_terraces=True, label_ages=True):
        """
        Draw the cross-section, filling each material and (by default) mirroring
        one wall about the channel to show a full symmetric valley.

        With show_terraces, the stranded benches from terraces() are drawn as
        bold risers on the section; with label_ages, each is annotated with its
        age -- the age of abandonment, which is the terrace's age. Pass
        label_ages=False for the same scene without the annotations.
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
            if geom.geom_type == 'Polygon':
                parts = [geom]
            else:                       # MultiPolygon or GeometryCollection
                parts = [g for g in geom.geoms if g.geom_type == 'Polygon']
            for p in parts:
                xs, zs = p.exterior.xy
                xs = np.asarray(xs)
                for sign in ((1, -1) if mirror else (1,)):
                    ax.fill(sign * xs, zs, facecolor=facecolor, edgecolor='k',
                            linewidth=0.6, hatch=hatch)
        if show_terraces:
            for t in self.terraces():
                for sign in ((1, -1) if mirror else (1,)):
                    ax.plot([sign * t['x_far'], sign * t['x_near']],
                            [t['z'], t['z']], color='#c1272d', lw=2.4,
                            solid_capstyle='butt', zorder=4)
                if label_ages:
                    xm = 0.5 * (t['x_far'] + t['x_near'])   # label the left unit
                    ax.annotate("%s  t=%s" % (t['kind'], self._fmt_age(t['age'])),
                                xy=(xm, t['z']), xytext=(0, 4),
                                textcoords='offset points', ha='center',
                                va='bottom', fontsize=7.5, color='#7a1116',
                                zorder=5)
        ax.plot(0, self.z_ch, 'v', color='#1f6fb2', markeredgecolor='k')
        ax.set_xlabel('cross-valley distance [m]')
        ax.set_ylabel('elevation [m]')
        ax.set_aspect('equal')
        return ax
