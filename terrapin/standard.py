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
from . import plotting

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

    def establish_channel(self):
        """Cut the initial channel into the current ground at x_ch: its top sits at
        the local surface and its bed one channel depth below, carved out of the
        existing material (which forms its banks). Use once at setup so the model
        starts with a real, bank-bounded channel rather than a bed drawn at the
        surface. No-op with zero channel depth or width."""
        if self.channel_depth <= 0 or self.channel_width <= 0:
            return
        surface = self._surface_elevation(self.x_ch)
        if surface is None:
            return
        self.z_ch = surface - self.channel_depth
        half_width = self.channel_width / 2.
        channel = box(self.x_ch - half_width, self.z_ch,
                      self.x_ch + half_width, surface)
        self.bodies = {n: g.difference(channel) for n, g in self.bodies.items()}
        self._coalesce_bodies()

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
        self._fill_banks(age)
        self._coalesce_bodies()

    def migrate(self, x_new, at_capacity=False, age=None):
        """
        Migrate the channel laterally to x_new at the current bed elevation,
        planing a strath across the swept corridor and undercutting the wall it
        advances INTO. The channel bed sweeps the whole corridor, so it erodes
        EVERYTHING above the bed there: the channel cannot slide under an overhang
        (an isolated body or valley-wall alluvium hanging over the swept ground is
        cut through, not left overhanging). Geometrically the removed region is the
        full column across the corridor -- from the retreating bank to the advancing
        bank -- unioned with a repose wedge that grades the advancing wall beyond it.
        The retreating side's wall is left untouched (its strath is abandoned, and
        can shed talus later).

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
        # The channel bed sweeps the whole corridor, so it erodes EVERYTHING above
        # the bed there: it cannot slide under an overhang. Remove the full column
        # across the swept corridor (bank to bank), then let the advancing wall
        # grade back to repose beyond it.
        half_width = self.channel_width / 2.
        lo = min(self.x_ch, x_new) - half_width         # retreating bank
        hi = max(self.x_ch, x_new) + half_width         # advancing bank
        ceiling = unary_union(list(self.bodies.values())).bounds[3] + 1.0
        corridor = box(lo, self.z_ch, hi, ceiling)
        self._remove(unary_union([corridor, wedge]))
        self.deposited = 0.0
        if at_capacity:
            self._deposit_channel_belt(x_new, age)
        self.x_ch = x_new
        self._coalesce_bodies()

    def avulse(self, x_new, age=None):
        """
        Avulse the channel to x_new: a discontinuous hop. Unlike migration, the old
        channel is abandoned IN PLACE and the ground between old and new positions
        is NOT planed -- so the vacated belt is preserved, to be buried by later
        aggradation. Where it lands, the new channel cuts down one channel depth
        below the local surface, a vertical-banked slot one channel width wide.

        Above the channel top, where the cut passes through HIGHER valley-wall
        material, that material is undercut and fails back to its angle of repose --
        so the exposed wall snaps to repose, it is not left standing vertical, and it
        is never left overhanging (the removal reaches the full column above the bed).
        On flat ground nothing stands above the channel top, so the banks stay
        vertical and the eroded volume is just width x depth. The eroded material is
        exported as sediment.

        The avulsion abandons the old channel now, so the optional `age` is stamped
        on the abandoned channel floor -- its terrace age is this avulsion, not a
        later incision that might strand it.
        """
        self._record_surface("channel", self.z_ch, abandoned=age)
        surface = self._surface_elevation(x_new)
        z_bed = surface - self.channel_depth
        half_width = self.channel_width / 2.
        # The channel slot: a vertical-banked box (width x depth), plus the full
        # column above it so nothing is left overhanging.
        ceiling = unary_union(list(self.bodies.values())).bounds[3] + 1.0
        block = box(x_new - half_width, z_bed, x_new + half_width, ceiling)
        # Grade the valley-wall material ABOVE the channel top back to repose, so the
        # exposed wall lies at its angle of repose rather than standing vertical.
        walls = unary_union([
            self._wall_wedge(surface, half_width, x_new, "left"),
            self._wall_wedge(surface, half_width, x_new, "right")])
        self._remove(unary_union([block, walls]))
        self.x_ch = x_new
        self.z_ch = z_bed
        self._coalesce_bodies()

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
        self.bodies, _ = geometry.aggrade(
            self.bodies, z_fill, self._domain(z_fill), name=name)
        fill = self.bodies[name]
        self.z_ch = z_fill - self.channel_depth
        half_width = self.channel_width / 2.
        self.deposited = fill.area
        if self.channel_depth > 0 and half_width > 0:
            # The channel aggrades with its floodplain: its bed rises to one channel
            # depth below the new surface, leaving CHANNEL-BELT deposit in its column
            # below the new bed and the channel itself open above it. Outside the
            # column, the fill is overbank floodplain.
            column = box(self.x_ch - half_width, z_fill - 1.0e6,
                         self.x_ch + half_width, z_fill)
            channel = box(self.x_ch - half_width, self.z_ch,
                          self.x_ch + half_width, z_fill)          # open channel
            belt = fill.intersection(column).difference(channel)   # channel-belt below the bed
            self.bodies[name] = fill.difference(column)            # overbank floodplain
            if not belt.is_empty:
                belt_name = "channel_belt_%d" % self._n_belt
                self.bodies[belt_name] = belt
                self._record_deposit(belt_name, kind="channel", age=age)
                self._n_belt += 1
            self.deposited = self.bodies[name].area + belt.area
        self._record_deposit(name, kind="floodplain", age=age)
        self._record_surface("floodplain", z_fill, abandoned=None)
        self._n_fill += 1
        self.sediment_out = 0.
        self._coalesce_bodies()

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
            surf = self._surface_at(z)
            if surf is None or surf.get("abandoned") is None:
                continue                       # a terrace is an ABANDONED surface
            xm = 0.5 * (x_far + x_near)
            body = geometry._material_at(self.bodies, xm, z - self._PROBE)
            prov = self.provenance.get(body, {})
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

    # the standard model's fill categories, a view of the shared style table
    # (bedrock/colluvium by lithology; alluvium bodies by deposit kind)
    _STYLE = {k: plotting.STYLE[k]
              for k in ("bedrock", "initial", "floodplain", "channel", "colluvium")}

    def _category(self, name):
        """The body's fill category: lithology for bedrock/colluvium, else the
        deposit kind (initial/floodplain/channel) from its provenance."""
        litho = geometry._lithology(name)
        if litho in ("bedrock", "colluvium"):
            return litho
        return self.provenance.get(name, {}).get("kind", "initial")

    def plot(self, ax=None, show_terraces=True, label_ages=True):
        """
        Draw the full-valley cross-section, each body coloured by deposit kind
        (bedrock, initial alluvium, floodplain, channel-belt), the channel at
        (x_ch, z_ch), and (by default) the terraces overlaid as bold benches,
        optionally labelled by their abandonment age. Draws into `ax` if given,
        else makes a new figure; returns the axes.
        """
        if ax is None:
            _, ax = plt.subplots()
        plotting.draw_bodies(ax, self.bodies, self._category)
        if show_terraces:
            plotting.draw_terraces(ax, self.terraces(), self._fmt_age,
                                   label_ages=label_ages)
        if self.channel_width > 0 and self.channel_depth > 0:
            plotting.draw_channel_box(ax, self.x_ch, self.z_ch,
                                      self.channel_width, self.channel_depth)
        elif self._surface_elevation(self.x_ch) is not None:
            plotting.draw_channel_marker(ax, self.x_ch, self.z_ch, zorder=6)
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

    def _fill_banks(self, age=None):
        """Line the valley floor around the channel with a channel-depth alluvial
        layer, up to bank-top, leaving the channel open -- so the channel is
        bank-bounded and its top sits at the solid material beside it (review point
        2, "good enough for now"). The banks are channel-associated alluvium; being
        a deposit, they reduce the net exported sediment. No-op with zero channel
        depth or width, so the symmetric-reproduction invariant is unaffected.
        """
        if self.channel_depth <= 0 or self.channel_width <= 0:
            return
        z0, z1 = self.z_ch, self.z_ch + self.channel_depth
        half_width = self.channel_width / 2.
        minx, _, maxx, _ = unary_union(list(self.bodies.values())).bounds
        solid = unary_union([g for g in self.bodies.values() if not g.is_empty])
        channel = box(self.x_ch - half_width, z0, self.x_ch + half_width, z1)
        banks = box(minx, z0, maxx, z1).difference(solid).difference(channel)
        if banks.is_empty:
            return
        name = "channel_belt_%d" % self._n_belt
        self.bodies[name] = banks
        self._record_deposit(name, kind="channel", age=age)
        self._n_belt += 1
        self.deposited = banks.area
        self.sediment_out -= banks.area

    def _domain(self, z_top):
        """A bounding box that spans the bodies and reaches above z_top."""
        minx, miny, maxx, maxy = unary_union(list(self.bodies.values())).bounds
        return box(minx, miny, maxx, max(maxy, z_top) + 1.)

    def _surface_elevation(self, x, _reach=1.0e6):
        """Elevation of the top of the solid column at cross-valley position x."""
        solid = unary_union([g for g in self.bodies.values() if not g.is_empty])
        hit = LineString([(x, -_reach), (x, _reach)]).intersection(solid)
        return None if hit.is_empty else hit.bounds[3]      # bounds[3] = maxy = top

    def _provenance_key(self, name):
        """The full attribute signature of a body: kind, lithology, and age.
        Two bodies are the SAME material only if all three match."""
        p = self.provenance.get(name, {})
        return (p.get("kind"), p.get("lithology"), p.get("age"))

    def _coalesce_bodies(self, _tol=1.0e-9):
        """Merge bodies that are spatially contiguous AND share every attribute
        (kind, lithology, age) into a single polygon. Distinct deposits stay
        distinct: two channel belts of different ages, or alluvium against bedrock,
        never amalgamate -- only genuinely identical material that shares a boundary
        does, so the model keeps one polygon per (attributes, connected region)
        rather than accumulating redundant congruent pieces. Contiguity requires a
        shared edge (positive-length contact), not a bare corner touch."""
        merged = True
        while merged:
            merged = False
            names = [n for n, g in self.bodies.items() if not g.is_empty]
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    if self._provenance_key(a) != self._provenance_key(b):
                        continue
                    shared = self.bodies[a].intersection(self.bodies[b])
                    if shared.is_empty or shared.length <= _tol:
                        continue                       # disjoint or corner-touch only
                    self.bodies[a] = unary_union([self.bodies[a], self.bodies[b]])
                    del self.bodies[b]
                    self.provenance.pop(b, None)
                    merged = True
                    break
                if merged:
                    break
