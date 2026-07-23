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
from matplotlib import pyplot as plt
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
        self._n_belt = 0            # counter so each channel-belt deposit gets its own body
        self.provenance = {}        # {name: {kind, lithology, age}}: deposit + its formation age
        self.surfaces = []          # [{kind, z, abandoned}]: surfaces + their abandonment age

    # ----------------------------- configuration -----------------------------

    def set_bodies(self, bodies):
        """Set the material bodies: a dict {name: shapely Polygon} spanning the
        full valley (both sides of the channel). The initial bodies are
        pre-existing, so their formation age is unknown (None)."""
        self.bodies = dict(bodies)
        for name in self.bodies:
            self._record_deposit(name, kind="initial", age=None)
        miny = unary_union(list(self.bodies.values())).bounds[1]
        for z, _x_far, _x_near in geometry.treads_above(self.bodies, miny - 1.):
            self._record_surface("initial", z, abandoned=None)

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

    def incise(self, z_ch, age=None):
        """
        Incise the channel bed to z_ch at the current position x_ch, cutting BOTH
        walls: a flat channel of the current width with a material-following repose
        wall rising up-valley on each side. Eroded material is swept away as
        sediment.

        Incision is an abandoning event: every surface it strands above the new bed
        -- straths, buried-then-exposed benches, the valley margins on both sides --
        becomes a terrace. The optional `age` (a point, or a (start, end) span) is
        stamped on each surface newly abandoned by this cut.
        """
        self._abandon_stranded(z_ch, age)
        notch = self._two_wall_wedge(z_ch, self.channel_width / 2.)
        self._remove(notch)
        self.z_ch = z_ch

    def migrate(self, x_new, at_capacity=False, age=None):
        """
        Migrate the channel laterally to x_new at the current bed elevation,
        planing a strath across the swept corridor and undercutting the wall it
        advances INTO. The retreating side's wall is left untouched (its strath is
        abandoned, and can shed talus later). Geometrically this rebuilds only the
        advancing wall: a one-wall wedge whose floor reaches from the old channel to
        x_new plus half a channel width, on whichever side the channel moved.

        Whether the channel leaves sediment behind depends on the caller (it can
        change through time): with at_capacity=True the channel is transporting at
        capacity and leaves ~one channel depth of channel-belt alluvium above the
        planed bedrock strath (a deposit, with formation age `age`), everywhere it
        swept except the active channel itself; with at_capacity=False (default) it
        is net erosional and planes a clean strath, exporting everything. The
        reported sediment_out is the net export (eroded minus any belt deposited).
        """
        if x_new == self.x_ch:
            self.eroded = {n: 0.0 for n in self.bodies}
            self.deposited = 0.0
            self.sediment_out = 0.0
            return
        side = "right" if x_new > self.x_ch else "left"
        floor_half_width = abs(x_new - self.x_ch) + self.channel_width / 2.
        wedge = self._wall_wedge(self.z_ch, floor_half_width, self.x_ch, side)
        self._remove(wedge)
        self.deposited = 0.0
        if at_capacity:
            self._deposit_channel_belt(x_new, age)
        self.x_ch = x_new

    def avulse(self, x_new, age=None):
        """
        Avulse the channel to x_new: a discontinuous hop. Unlike migration, the old
        channel is abandoned IN PLACE and the ground between old and new positions
        is NOT planed -- so the vacated belt is preserved, to be buried by later
        aggradation. Where it lands, the new channel cuts down one channel depth
        over one channel width, eroding a block of material from the local surface
        downward; that block is exported as sediment. The new bed sits one channel
        depth below the surface at x_new.

        The avulsion abandons the old channel now, so the optional `age` is stamped
        on the abandoned channel floor -- its terrace age is this avulsion, not a
        later incision that might strand it.
        """
        self._record_surface("channel", self.z_ch, abandoned=age)
        surface = self._surface_elevation(x_new)
        z_bed = surface - self.channel_depth
        half_width = self.channel_width / 2.
        block = box(x_new - half_width, z_bed, x_new + half_width, surface)
        self._remove(block)
        self.x_ch = x_new
        self.z_ch = z_bed

    def aggrade(self, z_fill, age=None):
        """
        Fill the valley with alluvium up to the level z_fill. Each aggradation is
        stored as its own body ('alluvium_fill_N'), so repeated fills accumulate.
        Aggradation does not depend on the channel position: it fills whatever open
        void lies below z_fill across the valley -- overbank floodplain, and burying
        any abandoned channel left by an avulsion.

        The fill is a floodplain deposit and carries its formation age: the optional
        `age` (a point or a (start, end) span). Its top is the live floodplain
        surface until a later incision strands it as a fill terrace.
        """
        name = "alluvium_fill_%d" % self._n_fill
        self.bodies, deposited = geometry.aggrade(
            self.bodies, z_fill, self._domain(z_fill), name=name)
        # The channel is incised into its fresh floodplain: its bed sits one channel
        # depth below the new surface, and the fill leaves that channel open (so the
        # river is bank-bounded, not perched atop the fill).
        self.z_ch = z_fill - self.channel_depth
        half_width = self.channel_width / 2.
        channel = box(self.x_ch - half_width, self.z_ch, self.x_ch + half_width, z_fill)
        self.bodies[name] = self.bodies[name].difference(channel)
        self.deposited = self.bodies[name].area
        self._record_deposit(name, kind="floodplain", age=age)
        self._record_surface("floodplain", z_fill, abandoned=None)
        self._n_fill += 1
        self.sediment_out = 0.

    # -------------------------------- outputs --------------------------------

    def terraces(self):
        """
        The terraces present now: flat benches of ground stranded above the
        channel, read from the live geometry across the whole valley (both sides),
        so re-incision or migration that eats into a bench shortens it to what
        survives. A terrace's age is its ABANDONMENT age -- when the river left it
        -- and nothing else; a fill's deposition age is the deposit's own, carried
        as `deposit_age`.

        Returns a list of dicts, valley-floor upward, each with z, x_near/x_far
        edges and width, kind ('strath', 'floodplain', 'initial'), age (the terrace
        age), body (the deposit it caps), deposit_age, and lithology.
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

    @staticmethod
    def _fmt_age(age):
        """A compact string for an age -- a point or a (start, end) span."""
        if age is None:
            return "?"
        if isinstance(age, tuple):
            return "%g–%g" % age
        return "%g" % age

    # colour + hatch by deposit kind (bedrock and colluvium by lithology)
    _STYLE = {"bedrock":    ("#b8926a", "//"),
              "initial":    ("#e6cf7a", ".."),
              "floodplain": ("#d7a43c", ".."),
              "channel":    ("#c07b34", "xx"),   # channel-belt / paleochannel fill
              "colluvium":  ("#9a8f7d", "xx")}

    def _style(self, name):
        litho = geometry._lithology(name)
        if litho in ("bedrock", "colluvium"):
            return self._STYLE[litho]
        kind = self.provenance.get(name, {}).get("kind", "initial")
        return self._STYLE.get(kind, self._STYLE["initial"])

    def plot(self, ax=None, show_terraces=True, label_ages=True):
        """
        Draw the full-valley cross-section, each body coloured by deposit kind
        (bedrock, initial alluvium, floodplain, channel-belt), the channel marked
        at (x_ch, z_ch), and (by default) the terraces overlaid as bold benches,
        optionally labelled by their abandonment age. Draws into `ax` if given,
        else makes a new figure; returns the axes.
        """
        if ax is None:
            _, ax = plt.subplots()
        for name, geom in self.bodies.items():
            if geom.is_empty:
                continue
            facecolor, hatch = self._style(name)
            parts = geom.geoms if geom.geom_type != "Polygon" else [geom]
            for p in parts:
                if p.geom_type != "Polygon":
                    continue
                xs, zs = p.exterior.xy
                ax.fill(xs, zs, facecolor=facecolor, edgecolor="k",
                        linewidth=0.6, hatch=hatch)
        if show_terraces:
            for t in self.terraces():
                ax.plot([t["x_far"], t["x_near"]], [t["z"], t["z"]],
                        color="#c1272d", lw=2.4, solid_capstyle="butt", zorder=4)
                if label_ages:
                    ax.annotate("%s t=%s" % (t["kind"], self._fmt_age(t["age"])),
                                xy=(0.5 * (t["x_far"] + t["x_near"]), t["z"]),
                                xytext=(0, 4), textcoords="offset points",
                                ha="center", va="bottom", fontsize=7,
                                color="#7a1116", zorder=5)
        half_width = self.channel_width / 2.
        if half_width > 0:                          # finite channel: a river-blue box
            x0, x1 = self.x_ch - half_width, self.x_ch + half_width
            z1 = self.z_ch + self.channel_depth
            ax.fill([x0, x1, x1, x0], [self.z_ch, self.z_ch, z1, z1],
                    facecolor="#2b7bba", edgecolor="k", linewidth=0.6, zorder=6)
        else:                                       # zero width: a point marker
            ax.plot(self.x_ch, self.z_ch, "v", color="#1f6fb2",
                    markeredgecolor="k", zorder=6)
        ax.set_xlabel("cross-valley distance [m]")
        ax.set_ylabel("elevation [m]")
        ax.set_aspect("equal")
        return ax

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
        """Log a surface and the age at which it was abandoned (its terrace age)."""
        self.surfaces.append({"kind": kind, "z": z, "abandoned": abandoned})

    def _surface_at(self, z, _tol=1e-5):
        """The most recently logged surface at elevation z, or None."""
        for surf in reversed(self.surfaces):
            if abs(surf["z"] - z) <= _tol:
                return surf
        return None

    def _abandon_stranded(self, z_new, age):
        """Stamp `age` on every exposed surface this incision strands above z_new.

        Read on the pre-incision geometry so the surfaces are intact. Surfaces
        already abandoned (or buried, hence not exposed) are left untouched; a
        stranded surface not yet logged -- a freshly cut floor -- is recorded here.
        """
        for z, x_far, x_near in geometry.treads_above(self.bodies, z_new):
            surf = self._surface_at(z)
            if surf is None:
                xm = 0.5 * (x_far + x_near)
                body = geometry._material_at(self.bodies, xm, z - self._PROBE)
                kind = "strath" if geometry._lithology(body) == "bedrock" else "floodplain"
                self._record_surface(kind, z, abandoned=age)
            elif surf["abandoned"] is None:
                surf["abandoned"] = age

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

    def _deposit_channel_belt(self, x_new, age):
        """Lay a channel-belt of alluvium, one channel depth thick, bank to bank.

        Because this is a MIGRATION (not aggradation), the belt is defined by the
        sweep: it fills the corridor from the retreating bank to the advancing bank
        -- min(old, new) - half_width to max(old, new) + half_width -- in the band
        from the strath (z_ch) up to bank-top (z_ch + channel_depth), minus the
        active channel. The alluvium comes from the channel's load, so it is a
        deposit (a sink) and reduces the net sediment exported: sediment_out
        becomes eroded minus deposited. (self.x_ch is still the old position here.)
        """
        z0, z1 = self.z_ch, self.z_ch + self.channel_depth
        half_width = self.channel_width / 2.
        lo = min(self.x_ch, x_new) - half_width     # retreating bank
        hi = max(self.x_ch, x_new) + half_width     # advancing bank
        solid = unary_union([g for g in self.bodies.values() if not g.is_empty])
        active = box(x_new - half_width, z0, x_new + half_width, z1)
        belt = box(lo, z0, hi, z1).difference(solid).difference(active)
        if belt.is_empty:
            return
        name = "channel_belt_%d" % self._n_belt
        self.bodies[name] = belt
        self._record_deposit(name, kind="channel", age=age)
        self._n_belt += 1
        self.deposited = belt.area
        self.sediment_out -= belt.area          # net export = eroded - deposited

    def _domain(self, z_top):
        """A bounding box that spans the bodies and reaches above z_top."""
        minx, miny, maxx, maxy = unary_union(list(self.bodies.values())).bounds
        return box(minx, miny, maxx, max(maxy, z_top) + 1.)

    def _surface_elevation(self, x, _reach=1.0e6):
        """Elevation of the top of the solid column at cross-valley position x."""
        solid = unary_union([g for g in self.bodies.values() if not g.is_empty])
        hit = LineString([(x, -_reach), (x, _reach)]).intersection(solid)
        return None if hit.is_empty else hit.bounds[3]      # bounds[3] = maxy = top
