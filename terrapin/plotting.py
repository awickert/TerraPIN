"""
Shared cross-section drawing for TerraPIN.

One place for the fill styles and the draw loops that both the symmetric
(`Terrapin`) and standard (`StandardTerrapin`) models -- and the examples --
use, so the colour table and the body/terrace/channel rendering are defined
once rather than copied per plotter.

Each model supplies its own `category_of(name)` (the symmetric model styles by
lithology; the standard model styles alluvium bodies by deposit kind), because
what a body IS differs by context; everything downstream of that -- the colours,
the fill loop, the mirror handling, the terraces, the channel -- is shared.
"""
import numpy as np

# fill colour + hatch by category. `initial`/`alluvium` and `floodplain`/`fill`
# are aliases (same look, different names the two contexts use as legend labels).
STYLE = {
    "bedrock":    ("#b8926a", "//"),
    "initial":    ("#e6cf7a", ".."),
    "alluvium":   ("#e6cf7a", ".."),
    "floodplain": ("#d7a43c", ".."),
    "fill":       ("#d7a43c", ".."),
    "channel":    ("#c07b34", "xx"),   # channel-belt / paleochannel fill
    "colluvium":  ("#9a8f7d", "xx"),
}

RIVER = "#2b7bba"       # active-channel blue
MARKER = "#1f6fb2"      # zero-width channel marker
TERRACE = "#c1272d"     # terrace riser
LABEL = "#7a1116"       # terrace-age label


def draw_bodies(ax, bodies, category_of, mirror=False):
    """Fill each material body, coloured by its category. With mirror, also draw
    the reflection about x = 0 (the symmetric model's full valley)."""
    signs = (1.0, -1.0) if mirror else (1.0,)
    for name, geom in bodies.items():
        if geom.is_empty:
            continue
        facecolor, hatch = STYLE[category_of(name)]
        parts = [geom] if geom.geom_type == "Polygon" else \
                [g for g in geom.geoms if g.geom_type == "Polygon"]
        for p in parts:
            xs, zs = p.exterior.xy
            xs = np.asarray(xs)
            for sign in signs:
                ax.fill(sign * xs, zs, facecolor=facecolor, edgecolor="k",
                        linewidth=0.6, hatch=hatch)
    return ax


def draw_terraces(ax, terraces, fmt_age=str, mirror=False, label_ages=True,
                  fontsize=7, label_fmt="%s t=%s"):
    """Draw the stranded benches as bold risers, optionally annotated by age."""
    signs = (1.0, -1.0) if mirror else (1.0,)
    for t in terraces:
        for sign in signs:
            ax.plot([sign * t["x_far"], sign * t["x_near"]], [t["z"], t["z"]],
                    color=TERRACE, lw=2.4, solid_capstyle="butt", zorder=4)
        if label_ages:
            xm = 0.5 * (t["x_far"] + t["x_near"])
            ax.annotate(label_fmt % (t["kind"], fmt_age(t["age"])),
                        xy=(xm, t["z"]), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=fontsize, color=LABEL,
                        zorder=5)


def draw_channel_box(ax, x_ch, z_ch, width, depth):
    """The river as its prescribed width x depth, pinned bottom-centre at (x_ch, z_ch)."""
    hw = width / 2.0
    top = z_ch + depth
    ax.fill([x_ch - hw, x_ch + hw, x_ch + hw, x_ch - hw],
            [z_ch, z_ch, top, top],
            facecolor=RIVER, edgecolor="k", linewidth=0.6, zorder=6)


def draw_channel_marker(ax, x_ch, z_ch, zorder=None):
    """A zero-width channel drawn as a downward triangle at (x_ch, z_ch)."""
    kw = {} if zorder is None else {"zorder": zorder}
    ax.plot(x_ch, z_ch, "v", color=MARKER, markeredgecolor="k", **kw)
