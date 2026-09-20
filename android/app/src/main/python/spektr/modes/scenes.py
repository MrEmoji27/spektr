"""Scene modes — things with a horizon, a depth axis, or their own particles."""

from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..palette import RAMP_STEPS
from ..render import (
    SPACE,
    cell_max,
    frac,
    pack_braille,
)
from . import (
    Ctx,
    band_columns,
    bg_contrast,
    contrast_ramp,
    empty,
    mode,
    spread,
)
from . import (
    angular_bands as _angular_bands,
)
from . import (
    angular_lut as _angular_lut,
)
from . import (
    polar_grid as _polar,
)

_FULL = ord("█")


@mode("Retro", group="scenes", blurb="sunset grid, with the spectrum as the horizon")
def retro(ctx: Ctx):
    """The other mode that was still a nested Python loop — 18 ms per frame at
    200x50, and the reason fullscreen Retro was dropping to 33 fps.

    Every layer here is a broadcast mask over the whole dot grid instead. The
    picture is identical; it just costs about a twentieth as much.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 12 or dc < 12:
        return empty(ctx.w, ctx.h)

    horizon = max(2, dr * 2 // 5)
    floor_r = dr - horizon
    cx = (dc - 1) / 2.0

    rows = np.arange(dr)[:, None]
    cols = np.arange(dc)[None, :]

    grid = np.zeros((dr, dc), dtype=np.int8)   # 0 empty, 1 grid, 2 wave, 3 sun

    # ── sun: a half disc above the horizon, cut by scanline slits ──
    sun_r = horizon * 0.85
    rd = (horizon - rows).astype(np.float64)              # distance up from horizon
    above = (rows < horizon) & (rd <= sun_r)
    hw = np.sqrt(np.maximum(sun_r * sun_r - rd * rd, 0.0))

    # Slits, not a 50/50 chop. This used to band on ``(rd // slice_w) % 2``,
    # a duty cycle whose gap is as tall as the lit stripe — and with
    # ``slice_w`` landing around 4 dots, that gap is exactly one full braille
    # row, so it rendered as a blank *text line* straight through the sun,
    # splitting it into disconnected slabs rather than scoring it with
    # scanlines. Keeping the slit to a small fraction of the period is what
    # stops a gap from ever swallowing a whole cell row; widening it toward
    # the horizon is the motif this is quoting, where the stripes open up as
    # the disc sinks.
    band_zone = sun_r * 0.62
    period = max(3.0, sun_r * 0.17)
    sink = np.clip(1.0 - rd / max(band_zone, 1e-6), 0.0, 1.0)
    # A *fraction* of the period, and never wider than 3 dots. Both bounds
    # matter and each caught a real blank row: sized absolutely, the slit
    # swallowed 87% of the period once ``period`` hit its 3.0 floor on a
    # short terminal, and left uncapped it grows with ``sun_r`` until it
    # spans a whole cell again on a tall one. At most 3 dot rows means any
    # four consecutive — that is, any braille row — keeps at least one lit,
    # so no size can produce an empty line through the disc.
    slit = np.clip(period * (0.22 + sink * 0.28), 0.8, 3.0)
    into_period = rd - np.floor(rd / period) * period
    banded = (rd < band_zone) & (into_period < slit)

    sun = above & (np.abs(cols - cx) <= hw) & ~banded
    grid[sun] = 3

    # ── horizon line ──
    grid[horizon, :] = 1

    # ── perspective verticals ──
    below = np.arange(horizon + 1, dr)
    if below.size:
        t_ = ((below - horizon) / max(1, floor_r - 1))[:, None]
        bx = (np.arange(19) * (dc - 1) / 18.0)[None, :]
        ix = np.rint(cx + (bx - cx) * t_).astype(np.int32)
        ok = (ix >= 0) & (ix < dc)
        yy = np.repeat(below[:, None], 19, axis=1)
        grid[yy[ok], ix[ok]] = 1

    # ── scrolling horizontals, spaced by z² so they bunch toward the horizon ──
    # Speed is integrated into a phase rather than taken as ``t * speed``.
    # That form is fine only while the speed is constant, which is what this
    # used to be — a fixed rate that ignored the audio entirely, so the one
    # thing carrying a sense of travel never responded to the music. Making
    # the multiplier audio-driven *and* leaving it against ``ctx.t`` would
    # have been worse than either: the phase is time times speed, so a change
    # in speed retroactively rewrites the whole history and the grid teleports
    # rather than accelerating — measured at a 130-turn jump for an ordinary
    # loudness change a couple of minutes into a session, against the 0.02
    # turns a frame of honest motion is worth. Accumulating sidesteps that
    # completely: past phase is banked and only the increment changes.
    sc = ctx.scratch("retro_scroll", lambda: {"v": 0.0})
    sc["v"] = (sc["v"] + (0.30 + ctx.energy * 1.5) * max(ctx.dt, 0.0)) % 1.0
    scroll = sc["v"]
    z = frac((np.arange(10) + scroll) / 10.0)
    ys = horizon + 1 + (z * z * max(1, floor_r - 2)).astype(np.int32)
    ys = ys[(ys > horizon) & (ys < dr)]
    grid[ys, :] = 1

    # ── the spectrum, as a wave sitting on the horizon ──
    levels = spread(ctx.display_bands(), dc)
    max_wv = horizon * 0.85
    wy = np.clip(horizon - (np.maximum(levels, 0.03) * max_wv).astype(np.int32), 0, dr - 1)
    prev = np.empty_like(wy)
    prev[0] = wy[0]
    prev[1:] = wy[:-1]
    lo = np.minimum(wy, prev)[None, :]
    hi = np.maximum(wy, prev)[None, :]
    grid[(rows >= lo) & (rows <= hi)] = 2

    dots = grid != 0
    codes = pack_braille(dots)

    # colour: grid cool, sun graded, wave hot
    heat = np.zeros((dr, dc), dtype=np.float64)
    heat[grid == 1] = 0.12
    # The sun was a single flat value, which is the one thing a sunset can't
    # be — the disc is the only large area on screen, so a constant index
    # across all of it reads as a cut-out shape rather than as light. Graded
    # along the radius instead, deepening toward the horizon. It stays under
    # the wave's 1.0 at every point so the spectrum still reads on top of it
    # rather than dissolving into the disc.
    sun_heat = 0.34 + 0.38 * np.clip(1.0 - rd / max(sun_r, 1e-6), 0.0, 1.0)
    heat = np.where(grid == 3, np.broadcast_to(sun_heat, heat.shape), heat)
    heat[grid == 2] = 1.0
    cidx = ctx.ramp(cell_max(heat))
    return codes, cidx


@mode("Auroras", group="scenes", blurb="a light ribbon whose lower rim rides the spectrum")
def auroras(ctx: Ctx):
    """A ribbon of light across the sky, its lower rim riding the spectrum.

    The obvious implementation — one full-grid mask per curtain — costs a pass
    over 300k dots per curtain at fullscreen. This builds the whole aurora as
    three 1-D profiles across the width (how bright, where the lower rim sits,
    how tall the ribbon is above it), then *shears* them per row with a single
    gather. The billow is real horizontal displacement rather than a per-row
    recomputation, so the cost is the same for six curtains or fourteen.

    The geometry is a ribbon, not a set of hanging panels. Panels that each ran
    from the top of the screen down to their own depth gave a wedge silhouette
    and left most of the width empty — measured 0.082 reactivity over 7% of the
    screen, the least responsive mode in the app by a wide margin. A ribbon is
    continuous across the full width, so every band contributes and the shape
    the spectrum draws is the undulating *lower edge*, which is exactly the
    feature that reads as an aurora.

    Brightness is concentrated at that lower rim rather than at the top: a real
    aurora is a faint diffuse column with a hot lower edge, where the electrons
    finally reach dense air.

    ``Plasma`` is a smooth field with no structure; this has a hard lower
    boundary, vertical striations and a dithered upper fade.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 12 or dc < 16:
        return empty(ctx.w, ctx.h)

    # One curtain per band, not per 110 columns. The old count was
    # ``clip(dc // 110, 2, 5)``, which is *two* curtains on an 80-column
    # terminal — and two curtains means ``display_bands(2)``, so the entire
    # mode was driven by the average of the bottom half of the spectrum and
    # the average of the top half.
    n = int(np.clip(dc // 26, 6, 14))
    lows = ctx.range(0.00, 0.20)
    treble = ctx.range(0.60, 1.00)

    cols = np.arange(dc, dtype=np.float32)
    lv = ctx.display_bands(n).astype(np.float32)
    spacing = dc / n

    # Weighted blend rather than a per-curtain maximum. Taking the max leaves
    # the gutters between curtains at zero, and a zero-width curtain has no
    # rim position at all — there is nothing to divide by. Overlapping weights
    # that sum to a continuous profile give one sheet whose lower edge dips and
    # rises band by band.
    wsum = np.full(dc, 1e-3, dtype=np.float32)
    bsum = np.zeros(dc, dtype=np.float32)      # rim position, weighted
    hsum = np.zeros(dc, dtype=np.float32)      # ribbon height, weighted
    gsum = np.zeros(dc, dtype=np.float32)      # brightness, weighted
    for i in range(n):
        level = float(lv[i])
        centre = (i + 0.5) * spacing + math.sin(ctx.t * (0.21 + 0.06 * i) + i * 2.1) * spacing * 0.35
        d = np.abs(cols - centre)
        edge = np.clip(1.0 - d / (spacing * 0.9), 0.0, 1.0)
        w = edge * edge + np.float32(0.02)
        wsum += w
        # The whole ribbon brightens on the beat and settles through the bar.
        # ``ctx.pulse`` rather than an onset because an aurora should breathe,
        # not flash, and because a swell keyed to discrete hits does nothing at
        # all between them — which on the slow material this mode suits best is
        # most of the time.
        gsum += w * (0.18 + 0.82 * level) * np.float32(1.0 + 0.15 * ctx.pulse)
        bsum += w * (0.34 + 0.52 * level)      # louder pushes the rim lower
        hsum += w * (0.26 + 0.40 * level)      # and makes the ribbon deeper
    inv_w = 1.0 / wsum
    bright = gsum * inv_w
    bottom = bsum * inv_w
    inv_h = 1.0 / np.maximum(hsum * inv_w, 1e-3)

    # fine vertical ribbing — the striations are what make an aurora read as an
    # aurora rather than as a smear, and in 1-D they cost nothing. The period
    # is ~20 cycles across any width rather than a fixed 7.85 dot-columns:
    # at 400 wide a fixed period-8 sine is 100 cycles of shimmer across the
    # screen — sub-resolution noise that also drove the strip builder to emit
    # a segment for every other cell (see the ``_RLE_TOL`` note in render.py).
    # The 160 keeps the original 0.8 rad/dot-column at an 80-column terminal,
    # so the reference look is untouched.
    bright *= 0.66 + 0.34 * np.sin(cols * (0.8 * (160.0 / dc)) + ctx.t * 0.5).astype(np.float32)

    # Tiled three times so the shear can wrap by simple offset. A modulo over
    # the whole dot grid is one of the most expensive things you can do per
    # frame; adding dc to the index and reading from the middle copy is free.
    bright3 = np.tile(bright, 3)
    # Pre-divided in 1-D so the per-dot ``u`` below is one multiply-subtract
    # against two gathers, with no division over the whole grid.
    boh3 = np.tile(bottom * inv_h, 3)
    inv_h3 = np.tile(inv_h, 3)

    # per-row horizontal shear — this is the billow
    y = np.arange(dr, dtype=np.float32) / max(1, dr - 1)
    sway = min((0.03 + 0.10 * treble) * dc, dc * 0.9)
    shift = (
        np.sin(y * 3.1 + ctx.t * 0.9) * sway
        + np.sin(y * 7.7 - ctx.t * 0.55) * sway * 0.4
    )
    # The shear is a whole-column displacement, and every column index is an
    # integer already, so the offset can be truncated once per *row* instead of
    # truncating a full dot grid of floats: ``int(col + s) == col + int(s)``
    # exactly, for integral col and non-negative s. Halves the cost of building
    # the index and drops a 1.3 MB float temporary at 400x100.
    #
    # And once the offset is per-row, the index array is not needed at all. Row
    # ``r`` of the sheared picture is ``bright3[off[r] : off[r] + dc]`` — a
    # contiguous *slice*, not a scatter of arbitrary positions — so a sliding
    # window over the tiled profile turns each of the three shears from an
    # element-wise gather driven by a 1.3 MB index into one row-sized memcpy
    # apiece. The window itself is a view and costs nothing to build.
    #
    # The clip is not decoration: ``sway`` is capped at ``0.9 * dc`` while the
    # two sine terms can reach 1.4x it, so a large enough sway would index past
    # the end of a three-tile profile. Nothing reaches that today — ``treble``
    # is at most 1, which puts sway at 0.13 dc — but the cap says otherwise and
    # a latent out-of-bounds is not worth leaving in for a comparison per row.
    off = np.clip((shift + dc).astype(np.int32), 0, 2 * dc)
    bright_sheared = sliding_window_view(bright3, dc)[off]
    boh_sheared = sliding_window_view(boh3, dc)[off]
    inv_h_sheared = sliding_window_view(inv_h3, dc)[off]

    # Height *within* the ribbon: 0 at the lower rim, 1 at its top edge.
    # Negative below the rim, above 1 over the top, so one pair of comparisons
    # masks the whole shape.
    u = boh_sheared - y[:, None] * inv_h_sheared
    rim = np.clip(1.0 - u * np.float32(5.0), 0.0, 1.0)

    gain = np.float32(0.55 + 0.75 * lows)
    # The weight is built in place in ``rim`` — the only array here nothing
    # else still needs — rather than as four dot-grid temporaries chained by
    # operators. Same arithmetic in the same order, 1.5 ms against 2.4 at
    # 400x100.
    rim *= np.float32(0.90)
    rim += np.float32(0.16) + np.float32(0.34) * (1.0 - u)
    rim *= gain
    heat = bright_sheared
    heat *= rim
    heat *= (u >= 0.0) & (u <= 1.0)

    # Ragged edges, dithered against a *fixed* grain rather than a fresh random
    # field every frame. Per-frame noise over the whole dot grid was both the
    # most expensive operation in the mode and a boil: the ribbon already sways,
    # so a stationary grain for it to move through gives the texture without the
    # whole sheet fizzing in place. Thresholding on ``heat`` rather than on
    # vertical extent is what breaks the faint upper body into scattered dots
    # while the rim stays solid.
    grain = ctx.scratch(
        "aurora_grain",
        lambda: np.random.default_rng(31).random((dr, dc)).astype(np.float32),
    )
    lit = grain < heat * np.float32(1.9)

    codes = pack_braille(lit)
    # Quantised before ramping, the same trick and for the same reason as
    # Chladni: this is a smooth full-width field, so without it almost every
    # adjacent pair of cells lands in a different ramp bucket and the strip
    # builder emits a Segment for each — measured at ~107 runs per row at
    # 400x100, which cost more than building the picture did. Sixteen buckets
    # over a field whose visible range is a fraction of the ramp is finer than
    # the eye resolves here, and it does not depend on the theme being gentle
    # enough for the strip builder's own tolerance to help.
    shade = np.clip(heat, 0.0, 1.0)
    cidx = ctx.ramp(np.round(cell_max(shade) * 16.0) * (1.0 / 16.0))
    return codes, cidx


@mode("Keys", group="scenes", blurb="a lit keyboard; struck bands scroll away as falling notes")
def keys(ctx: Ctx):
    """A piano roll, not another bar chart.

    Every band gets a key at the bottom instead of a bar height: pressing one
    lights it and starts a note sustained for as long as the band stays
    loud, and the note scrolls up and away exactly once, the way a struck
    note leaves the playhead in a DAW roll. ``Ladder``/``Bars`` redraw a
    height every frame from the current level; this only draws something new
    when a band actually crosses into "struck," and what it drew keeps
    existing after the level drops.
    """
    w, h = ctx.w, ctx.h
    if w < 16 or h < 8:
        return empty(w, h)

    n = min(ctx.n_display, max(4, w // 3))
    col_band, active = band_columns(w, n)
    lv = ctx.display_bands(n)
    roll_h = h - 2   # bottom two rows are the keyboard itself

    def spawn():
        return {"roll": np.zeros((roll_h, n), dtype=np.float32), "held": np.zeros(n, dtype=bool)}

    st = ctx.scratch("keys", spawn)
    if st["roll"].shape[1] != n:
        st["roll"] = np.zeros((roll_h, n), dtype=np.float32)
        st["held"] = np.zeros(n, dtype=bool)
    roll = st["roll"]

    struck = (lv > 0.24) & ~st["held"]
    st["held"] = lv > 0.16

    # scroll everything already on the roll up by however many rows this
    # frame's dt is worth, at a fixed pace, so playback speed doesn't drift
    # with frame rate
    acc = ctx.scratch("keys_acc", lambda: {"v": 0.0})
    acc["v"] += (roll_h / 2.6) * ctx.dt
    shift = min(int(acc["v"]), roll_h)
    if shift:
        acc["v"] -= shift
        roll[: roll_h - shift] = roll[shift:]
        roll[roll_h - shift :] = 0.0

    roll *= 0.995   # notes dim slightly as they age, on top of moving away
    strike = np.where(lv > 0.16, np.maximum(lv, np.where(struck, 1.0, 0.0)), 0.0)
    roll[-1] = np.maximum(roll[-1], strike)

    roll_wide = np.where(active[None, :], roll[:, col_band], 0.0)
    lit = roll_wide > 0.05

    codes = np.full((h, w), SPACE, dtype=np.int32)
    cidx = np.zeros((h, w), dtype=np.int32)
    codes[:roll_h][lit] = _FULL
    cidx[:roll_h] = ctx.ramp(roll_wide)

    key0, key1 = h - 2, h - 1
    pressed = active & (lv[col_band] > 0.16)
    idle = active & ~pressed
    codes[key0, active] = _FULL
    codes[key1, active] = _FULL
    cidx[key0, idle] = ctx.palette.index(0.28)
    cidx[key1, idle] = ctx.palette.index(0.20)
    press_heat = ctx.ramp(np.clip(lv[col_band], 0.0, 1.0))
    cidx[key0, pressed] = press_heat[pressed]
    cidx[key1, pressed] = press_heat[pressed]
    return codes, cidx


def _tunnel_geometry(dr: int, dc: int, dist, turn, max_r) -> dict:
    """Everything about a wireframe corridor of this size that does not move.

    Shared by every mode in the tunnel family, so they draw the same strokes,
    the same spokes and the same centre. See :func:`_tunnel` for why each piece
    is the way it is.
    """
    cy, cx = dr / 2.0, dc / 2.0
    s = np.float32(cy / max(cx, 1.0))              # _polar's x stretch
    yy = (np.arange(dr, dtype=np.float32) - np.float32(cy))[:, None]
    xx = (np.arange(dc, dtype=np.float32) - np.float32(cx))[None, :]
    safe = np.maximum(dist, np.float32(0.5))
    near = np.clip(dist / np.float32(max_r), 0.0, 1.0)

    # Rings sit where depth * 0.55 crosses a whole number, i.e. at
    # dist = 0.55 max_r / m. The distance to the nearest one, in dots, is
    # the depth error divided by how fast depth changes per dot here.
    grad = np.sqrt((s * s) * (xx * s) ** 2 + yy * yy) / safe
    per_dot = np.float32(0.55 * max_r) * grad / (safe * safe)
    # Too close to the next ring to resolve: spacing under about three and
    # a half dots in the tightest direction. Tested per frame, below, on
    # each ring's *centre line*, so a ring is drawn all the way round or
    # not at all; a per-dot test clipped a loud ring that straddled the
    # radius and left its outer edge behind as a scatter of single dots.
    cut = np.float32(math.sqrt(3.5 * 0.55 * max_r))
    ring_fade = np.clip((dist - cut) / cut, 0.0, 1.0)

    # Spokes: sixteen straight lines through the centre. The nearest one's
    # direction, taken from the stretched angle and mapped back to dots,
    # gives an exact perpendicular distance in dots.
    spoke = np.rint(turn * np.float32(16.0)).astype(np.int32) % 16
    k = spoke.astype(np.float32) * np.float32(2.0 * math.pi / 16.0)
    ux, uy = np.cos(k) / s, np.sin(k)
    norm = np.sqrt(ux * ux + uy * uy)
    perp = np.abs(xx * uy - yy * ux) / norm
    # Outside ``hole`` the spokes are the tapering strokes; inside it they
    # carry on to the vanishing point as one-dot hairlines. Sixteen lines
    # cannot all get there — neighbours are ``dist * 2pi/16`` apart and fuse
    # into a blob under about two and a half dots — so they stop in a
    # hierarchy, each where it would touch its neighbour: the eight
    # in-between spokes at that radius, the four diagonals where eight
    # lines would fuse, and the four axes where four would, which is a
    # dot or two from the centre. Every line stays one continuous stroke,
    # and all of them dim toward the centre rather than brighten into a
    # starburst.
    step = np.float32(2.0 * math.pi / 16.0)
    fuse = np.float32(2.5)
    hole = np.float32(min(0.25 * max_r, max(0.07 * max_r, 3.0 / float(step))))
    # Stop radii in real dots, not in the stretched distance: stretched,
    # the horizontal axis stopped more than twice as far out as the
    # vertical one, and the centre read as a wide horizontal void.
    dots = np.sqrt(xx * xx + yy * yy)
    reach_in = np.where(spoke % 4 == 0, fuse / (4 * step),
                        np.where(spoke % 2 == 0, fuse / (2 * step), fuse / step)).astype(np.float32)
    outer = (perp <= np.float32(0.55) + np.float32(0.45) * near) & (dist > hole)
    inner = (perp <= np.float32(0.5)) & (dist <= hole) & (dots >= reach_in)
    walls = outer | inner
    # Each spoke fades from its outer weight at ``hole`` to almost the
    # background by the radius where it stops, so no line ends in a visible
    # cut. Stopped at full visibility, the staggered ends drew a ring of
    # stubs round an empty centre — a second cut-off pattern rather than
    # lines meeting. Faded out, every spoke dissolves on its way in and the
    # eye carries it the rest of the way to the vanishing point.
    edge = dots * hole / np.maximum(dist, np.float32(1e-3))   # ``hole`` along this ray, in dots
    fade = np.clip((dots - reach_in) / np.maximum(edge - reach_in, np.float32(1e-3)), 0.0, 1.0)
    spoke_value = np.where(
        outer, np.float32(0.25) + np.float32(0.35) * np.clip((dist - hole) / np.float32(0.8 * max_r), 0.0, 1.0),
        np.float32(0.0)).astype(np.float32)
    spoke_fade = np.where(inner, fade ** np.float32(1.5), np.float32(0.0)).astype(np.float32)

    return {
        "depth055": (np.float32(max_r) / np.maximum(dist, np.float32(0.9)) * np.float32(0.55)).astype(np.float32),
        # half-width in dots: a one-to-two-dot stroke, a touch heavier
        # toward the viewer, and heavier again when its band is loud
        "ring_base": (per_dot * (np.float32(0.55) + np.float32(0.30) * near)).astype(np.float32),
        "ring_loud": (per_dot * np.float32(1.10)).astype(np.float32),
        "ring_depth_cut": float(0.55 * max_r / cut),
        "ring_value": (ring_fade * (np.float32(0.25) + np.float32(0.75) * near)).astype(np.float32),
        "walls": walls,
        "spoke_value": spoke_value,
        "spoke_fade": spoke_fade,
        # how fast depth changes per dot, for modes that draw other ring shapes
        "per_dot": per_dot.astype(np.float32),
    }


def _tunnel(ctx, inward: bool):
    """Ribs travel down the pipe at an audio-reactive speed.

    Shared by both Tunnel and Tunnel In, which are the same corridor differing
    only in which way the ribs travel: ``inward`` flips the sign the
    accumulated phase is applied with, and nothing else. They were separate
    implementations until Tunnel In had been patched twice and still did not
    read as well as the mode it was imitating; folding it onto this body was
    the fix, so keep them sharing it. A change made here that should not apply
    to both is a sign the two have diverged in intent, not an invitation to
    branch on ``inward``.

    Each direction keeps its own phase accumulator (see the scratch key), so
    switching modes does not jump the ribs.

    **Why the phase is accumulated rather than ``ctx.t * speed``.** This is the
    important part of this function and it is not obvious from the arithmetic.

    Since phase is time times speed, changing the speed retroactively rewrites
    the whole history the multiplication represents, not just the rate going
    forward — an ordinary loudness change teleports the ribs by however many
    turns of ``depth * 0.55`` separate the old and new phase at the *current*
    ``t``, and that gap grows without bound the longer the session has been
    running. Traced and measured when the same bug was found in Retro's sun
    scroll: at t=120s an energy change could jump the ring phase by ~130 turns
    in a single frame, and under a gentle energy wobble 53% of frames moved
    more than a quarter of a rib-spacing. A ``ctx.dt``-accumulated phase held
    in scratch is the fix, and it is the same one Retro's scroll and ECG's and
    Spectro's column steps already use — the same bug wearing scenes.py's
    clothes instead of fields.py's or scope.py's.

    ``turn * spin`` below is NOT this bug: that spin rate is a constant
    (0.024), so ``ctx.t * constant`` is an ordinary, correct phase. Only
    multiplying time by a *varying* rate is unsafe.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 8 or dc < 8:
        return empty(ctx.w, ctx.h)

    dist, turn, max_r = _polar(ctx)

    # Wireframe, in dot units. Every line in the corridor is a *stroke* — a set
    # of dots within a half-width of an ideal curve, measured in real braille
    # dots, not in the stretched coordinates ``_polar`` hands back. The picture
    # used to be built from bands of those coordinates instead, and on screen
    # that showed as uneven weight and broken contours, measured at 188x50:
    #
    # * spokes whose width was quantised from a one-dot hairline to a three-
    #   and four-dot band at a single radius (46-56 abrupt steps over four
    #   spokes) and, because the stretch compresses x by the window's aspect,
    #   drew horizontal strokes several times heavier than vertical ones;
    # * rings that were a fixed *fraction* of one depth period — near the rim a
    #   period is tens of dots, so a loud ring was a filled annulus (79% of lit
    #   dots in solid 5x5 blocks) with a per-frame dither punching holes in it;
    # * near the centre the same fraction was under a dot, and ring and dither
    #   together left 60-odd specks under six dots each;
    # * the dither re-rolled every frame, reversing 2-8% of lit dots each frame.
    #
    # So: the distance from a dot to its nearest ring and spoke is taken in
    # dots, using the gradient of the stretched distance so an elliptical
    # contour is the same weight all the way round; each stroke has a
    # half-width in dots with a gentle taper toward the viewer; a ring too
    # close to its neighbours to resolve is not drawn at all, fading in colour
    # before it goes; and there is no dither — depth is carried by colour.
    #
    # The corridor itself does not move, so everything that depends only on
    # where a dot is lives in one cached entry per size. What moves per frame is
    # the rib phase sliding along the depth axis and the per-band level.
    geo = ctx.scratch("tunnel_geo", lambda: _tunnel_geometry(dr, dc, dist, turn, max_r))

    n = min(16, ctx.n_display)
    nrg = _angular_bands(ctx, turn, n, ctx.t * 0.024).astype(np.float32, copy=False)

    speed = 0.6 + ctx.energy * 2.4
    phase = ctx.scratch("tunnel{}_phase".format("_in" if inward else ""), lambda: {"v": 0.0})
    phase["v"] += speed * max(ctx.dt, 0.0)
    direction = -1 if inward else 1

    # distance, in depth units, to the nearest ring
    shift = np.float32(direction * (phase["v"] % 1.0))
    f = geo["depth055"] - shift
    centre = np.rint(f)
    np.subtract(f, centre, out=f)
    np.abs(f, out=f)
    reach = geo["ring_loud"] * nrg
    reach += geo["ring_base"]
    # never more than a seventh of the way to the next ring either side, so a
    # loud ring near the vanishing point of a small terminal stays a line and
    # does not swell into the gap
    np.minimum(reach, np.float32(0.14), out=reach)
    ribs = f <= reach
    # the ring this dot belongs to sits at depth ``centre + shift``; draw it
    # only if that ring is wide enough to resolve
    centre += shift
    np.bitwise_and(ribs, centre <= np.float32(geo["ring_depth_cut"]), out=ribs)

    lit = ribs | geo["walls"]
    codes = pack_braille(lit)

    # Brightness: rings by depth and by their band's level, spokes by depth.
    # Where a ring crosses a spoke the brighter one wins rather than the two
    # adding, so an intersection is not a heavier blob than either line.
    shade = nrg * np.float32(0.55)
    shade += np.float32(0.45)
    shade *= geo["ring_value"]
    shade *= ribs
    np.maximum(shade, geo["spoke_value"], out=shade)
    # Everything except the fading spoke ends keeps a visibility floor of
    # contrast 3.0: these are lines, and a far line at the quietest colour a
    # theme can show read as missing rather than as distant. The ends are the
    # exception — they are meant to disappear — so the colour walk starts much
    # nearer the background, the rest of the picture is lifted to where 3.0
    # sits on it, and the ends run from there down toward the background.
    lo = _faint_walk_point(ctx.palette, _TUNNEL_FADE_FLOOR, 3.0)
    shade *= np.float32(1.0 - lo)
    shade += np.float32(lo)
    np.multiply(shade, lit, out=shade)
    fade = geo["spoke_fade"] * np.float32(lo + (1.0 - lo) * 0.25)
    np.maximum(shade, fade, out=shade)
    return codes, contrast_ramp(ctx.palette, cell_max(shade), faint=_TUNNEL_FADE_FLOOR)


#: How close to the background the fading spoke ends get, as WCAG contrast.
_TUNNEL_FADE_FLOOR = 1.3


def _faint_walk_point(palette, faint: float, target: float) -> float:
    """Where on ``contrast_ramp``'s walk from ``faint`` a contrast of ``target`` sits, 0..1."""
    cr = bg_contrast(palette)
    lo = int(np.argmin(np.abs(cr - faint)))
    hi = int(np.argmax(cr))
    if hi == lo:
        return 0.0
    for v in np.linspace(0.0, 1.0, 65):
        if cr[int(round(lo + (hi - lo) * v))] >= target:
            return float(v)
    return 1.0


@mode("Tunnel", group="scenes", blurb="flying down a pipe, ribbed by the beat")
def tunnel(ctx: Ctx):
    return _tunnel(ctx, inward=False)


@mode("Warp", group="scenes", blurb="starfield, accelerating with the music")
def warp(ctx: Ctx):
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 8 or dc < 8:
        return empty(ctx.w, ctx.h)

    count = int(np.clip(dc * 1.4, 60, 900))

    def spawn():
        rng = np.random.default_rng(7)
        return {
            "ang": rng.uniform(0, 2 * math.pi, count),
            "rad": rng.uniform(0.5, 1.0, count) ** 2,
            "spd": rng.uniform(0.55, 1.6, count),
            "rng": rng,
        }

    st = ctx.scratch("warp", spawn)
    rng = st["rng"]

    max_r = min(dr, dc) / 2.0
    # Percussiveness on top of level: how fast you travel should answer to
    # attack, not only to how loud the mix is. A wall of sustained guitar and
    # a drum break sit at the same energy and should not fly past at the same
    # speed.
    st["rad"] += st["spd"] * ctx.dt * (0.22 + ctx.energy * 1.9 + ctx.drive * 1.2)

    gone = st["rad"] >= 1.0
    if gone.any():
        k = int(gone.sum())
        st["rad"][gone] = rng.uniform(0.02, 0.10, k)
        st["ang"][gone] = rng.uniform(0, 2 * math.pi, k)
        st["spd"][gone] = rng.uniform(0.55, 1.6, k)

    r = st["rad"] * max_r
    cx, cy = dc / 2.0, dr / 2.0
    x = np.clip(np.rint(cx + np.cos(st["ang"]) * r * 2.0).astype(np.int32), 0, dc - 1)
    y = np.clip(np.rint(cy + np.sin(st["ang"]) * r).astype(np.int32), 0, dr - 1)

    field = np.zeros((dr, dc), dtype=np.float64)
    np.add.at(field, (y, x), st["rad"])
    # streak the fastest stars outward by one dot, which reads as motion blur
    fast = st["rad"] > 0.55
    if fast.any():
        x2 = np.clip(x[fast] + np.sign(np.cos(st["ang"][fast])).astype(np.int32), 0, dc - 1)
        np.add.at(field, (y[fast], x2), st["rad"][fast] * 0.7)

    np.clip(field, 0.0, 1.0, out=field)
    dots = field > 0.05
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(field))
    return codes, cidx


_MATRIX_GLYPHS = np.array(
    [ord(c) for c in "0123456789ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ<>|=+*"],
    dtype=np.int32,
)


@mode("Matrix", group="scenes", blurb="digital rain, falling faster when it's loud")
def matrix(ctx: Ctx):
    w, h = ctx.w, ctx.h
    if h < 3 or w < 4:
        return empty(w, h)

    def spawn():
        rng = np.random.default_rng(11)
        return {
            "head": rng.uniform(-h, 0.0, w),
            "spd": rng.uniform(6.0, 22.0, w),
            "len": rng.integers(max(3, h // 5), max(5, h), w).astype(np.float64),
            "glyph": _MATRIX_GLYPHS[rng.integers(0, len(_MATRIX_GLYPHS), (h, w))],
            "rng": rng,
        }

    st = ctx.scratch("matrix", spawn)
    rng = st["rng"]

    # each column is driven by the band that sits at its horizontal position
    drive = spread(ctx.display_bands(), w)
    # ``ctx.drive`` on top of the per-column band: the columns say *where* the
    # rain is heavy, the flux says how percussive the moment is, and the two
    # are different questions. A sustained pad fills the bands and the rain
    # falls steadily; a drum break barely moves them and the rain should still
    # race. Continuous, so it fills the gaps between detected onsets.
    st["head"] += st["spd"] * ctx.dt * (0.35 + drive * 2.2 + ctx.drive * 1.1)

    done = st["head"] - st["len"] > h
    if done.any():
        k = int(done.sum())
        st["head"][done] = rng.uniform(-h * 0.4, 0.0, k)
        st["spd"][done] = rng.uniform(6.0, 22.0, k)
        st["len"][done] = rng.integers(max(3, h // 5), max(5, h), k)

    # churn a few glyphs per frame so the rain doesn't look like sliding text
    churn = max(1, (h * w) // 90)
    ys = rng.integers(0, h, churn)
    xs = rng.integers(0, w, churn)
    st["glyph"][ys, xs] = _MATRIX_GLYPHS[rng.integers(0, len(_MATRIX_GLYPHS), churn)]

    y = np.arange(h)[:, None]
    behind = st["head"][None, :] - y
    lit = (behind >= 0) & (behind < st["len"][None, :])
    bright = (1.0 - behind / np.maximum(st["len"][None, :], 1.0)) * lit

    codes = np.where(lit, st["glyph"], SPACE).astype(np.int32)
    cidx = ctx.ramp(bright ** 1.6)
    # the leading glyph of each drop burns brightest
    headrow = np.rint(st["head"]).astype(np.int32)
    ok = (headrow >= 0) & (headrow < h)
    cols = np.flatnonzero(ok)
    if cols.size:
        cidx[headrow[cols], cols] = RAMP_STEPS - 1
    return codes, cidx


_BOOT_INTRO = [
    "SPEKTR-BIOS (C) 1985 SPEKTR SYSTEMS",
    "CPU: Z80-COMPATIBLE   CLOCK: 3.58MHZ",
    "MEMORY TEST ................. 065536K OK",
    "DETECTING DRIVES ............ A: B: OK",
    "LOADING SPEKTR.SYS",
    "LOADING AUDIO.DRV ........... OK",
    "INIT DISPLAY ADAPTER ........ OK",
    "",
    "SPEKTR OS v0.2  READY",
    "",
]

#: Endless idle chatter once the intro's played out — ``{n}``/``{n2}`` are
#: filled with random digits per line, so the same template doesn't repeat
#: verbatim.
_BOOT_LOOP = [
    "PROC {n:04d} .................. OK",
    "IRQ {n:02d} ACK",
    "READ SECTOR {n:05d} ......... OK",
    "CACHE FLUSH BANK {n:02d}",
    "LOAD AVG 0.{n:02d}",
    "CHECKSUM {n:04X}H OK",
    "CHANNEL {n} SYNC",
    "BAND {n} PEAK {n2:03d}",
    "> RUN VISUALIZER.EXE",
]

_GLITCH_GLYPHS = np.array([ord(c) for c in "#%&@$?!/\\░▒▓█"], dtype=np.int32)


def _boot_advance_line(st: dict, w: int) -> None:
    """Pick the next line to type: drain the intro queue, then loop forever."""
    if st["queue"]:
        text = st["queue"].pop(0)
    else:
        tpl = _BOOT_LOOP[int(st["rng"].integers(0, len(_BOOT_LOOP)))]
        text = tpl.format(
            n=int(st["rng"].integers(0, 10000)), n2=int(st["rng"].integers(0, 999))
        )
    text = text[:w]
    target = np.array([ord(c) for c in text], dtype=np.int32) if text else np.zeros(0, dtype=np.int32)
    st["target"] = target
    st["pos"] = 0
    if target.size:
        st["reveal_t"][: target.size] = -99.0


def _boot_spawn(w: int) -> dict:
    st = {
        "rng": np.random.default_rng(53),
        "queue": list(_BOOT_INTRO),
        "committed": [],       # list of int32 arrays, oldest first
        "committed_t": [],     # matching commit timestamps
        "target": np.zeros(0, dtype=np.int32),
        "reveal_t": np.full(w, -99.0, dtype=np.float64),
        "pos": 0,
        "acc": 0.0,
        "last_reboot": -99.0,
        "onset_fast": 0.0,
        "onset_slow": 0.0,
        "glitch_t0": -99.0,
    }
    _boot_advance_line(st, w)
    return st


@mode("Boot", group="scenes", blurb="an old PC waking up — BIOS POST, a boot log, a blinking cursor")
def boot(ctx: Ctx):
    """A monochrome terminal replaying the moment a machine powers on.

    Text-cell resolution, not the dot grid — Readout is the other mode that
    draws real glyphs rather than braille, for the same reason: sub-cell
    packing would mangle character shapes. Everything here is either an
    accumulator or a function of elapsed wall time (``ctx.t - event_time``),
    never ``ctx.t * speed`` — that pattern is what made Spectro and ECG lag
    then jump earlier this session, and a boot log stuttering mid-scroll
    would be the same bug wearing a different mode.

    The typewriter reveals characters at an audio-reactive rate (louder =
    faster typing, a bass hit bursts a few extra characters through); once
    a line is fully typed it's committed and scrolls up, and the next line
    comes off a short fixed BIOS intro, then an endless pool of idle status
    lines. A hard onset occasionally "power-cycles" the machine — clears the
    log and replays the intro — throttled to at most once every 12 real
    seconds so it reads as a rare dramatic beat, not a flicker.
    """
    w, h = ctx.w, ctx.h
    if w < 24 or h < 8:
        return empty(w, h)

    st = ctx.scratch("boot_term", lambda: _boot_spawn(w))
    rng = st["rng"]
    t = ctx.t
    dt = max(ctx.dt, 0.0)

    bass = ctx.range(0.0, 0.15)
    st["onset_fast"] += (bass - st["onset_fast"]) * min(1.0, dt / 0.03)
    st["onset_slow"] += (bass - st["onset_slow"]) * min(1.0, dt / 0.4)
    onset = st["onset_fast"] - st["onset_slow"]
    hit = onset > 0.12
    flash = 0.0
    if onset > 0.30 and (t - st["last_reboot"]) > 12.0:
        st["queue"] = list(_BOOT_INTRO)
        st["committed"] = []
        st["committed_t"] = []
        _boot_advance_line(st, w)
        st["last_reboot"] = t
        flash = 1.0

    cps = 14.0 + ctx.energy * 46.0
    st["acc"] += cps * dt + (6.0 if hit else 0.0)

    for _ in range(64):
        target = st["target"]
        if st["pos"] < target.size:
            if st["acc"] < 1.0:
                break
            st["acc"] -= 1.0
            st["reveal_t"][st["pos"]] = t
            st["pos"] += 1
            if st["pos"] < target.size:
                continue
        st["committed"].append(target)
        st["committed_t"].append(t)
        cap = h + 2
        if len(st["committed"]) > cap:
            drop = len(st["committed"]) - cap
            del st["committed"][:drop]
            del st["committed_t"][:drop]
        _boot_advance_line(st, w)
        if target.size > 0 and st["acc"] < 1.0:
            break

    codes = np.full((h, w), SPACE, dtype=np.int32)
    bright = np.zeros((h, w), dtype=np.float64)

    visible = st["committed"][-(h - 1):]
    visible_t = st["committed_t"][-(h - 1):]
    for i, (row_codes, ct) in enumerate(zip(visible, visible_t)):
        L = row_codes.size
        if L:
            codes[i, :L] = row_codes
            bright[i, :L] = 0.5 + 0.42 * math.exp(-(t - ct) / 0.8)

    typing_row = len(visible)
    target, pos = st["target"], st["pos"]
    if typing_row < h and pos > 0:
        codes[typing_row, :pos] = target[:pos]
        glow = np.exp(-(t - st["reveal_t"][:pos]) / 0.5)
        bright[typing_row, :pos] = 0.5 + 0.45 * glow
    if typing_row < h and pos < w and (t * 2.2) % 1.0 < 0.5:
        codes[typing_row, pos] = _FULL
        bright[typing_row, pos] = 1.0

    bright[0::2] *= 0.90

    rows_idx = np.arange(h, dtype=np.float64)
    band_pos = (t * (h / 1.3)) % h
    band_dist = np.abs(rows_idx - band_pos)
    band_dist = np.minimum(band_dist, h - band_dist)
    band_boost = np.clip(1.0 - band_dist, 0.0, 1.0) * 0.12
    bright += band_boost[:, None] * (codes != SPACE)

    treble = ctx.range(0.6, 1.0)
    if treble > 0.55 and (t - st["glitch_t0"]) > 0.15:
        st["glitch_t0"] = t
    glitch_env = math.exp(-(t - st["glitch_t0"]) / 0.12)
    if glitch_env > 0.05:
        mask = rng.random((h, w)) < (glitch_env * 0.10)
        n_hit = int(mask.sum())
        if n_hit:
            codes[mask] = _GLITCH_GLYPHS[rng.integers(0, len(_GLITCH_GLYPHS), n_hit)]
            bright[mask] = np.maximum(bright[mask], glitch_env)

    if flash > 0.0:
        bright[:] = np.maximum(bright, 1.0)

    cidx = ctx.ramp(np.clip(bright, 0.0, 1.0))
    return codes, cidx

@mode("Tunnel In", group="scenes", after="Tunnel",
      blurb="rings thrown out of the centre on the beat, rushing past you")
def tunnel_in(ctx: Ctx):
    return _tunnel(ctx, inward=True)


# ── Crosscurrent ─────────────────────────────────────────────────────────────

#: Sides of the inbound stream's rings. Eight, with a corner on every other
#: spoke: a sixteen-sided ring differs from a round one by under a dot at any
#: size a terminal has, so the two streams would read as one. At eight the
#: flat edges bow in by about 8% of the radius between corners, which is three
#: dots on a 40-dot ring and still a dot on the smallest ring drawn.
_XC_SIDES = 8

#: Seconds a spoke's pulse takes to fade. Short on purpose: a transient is a
#: flick of light along the line, not a lamp left on.
_XC_FLASH_TAU = 0.085

#: How far a band has to jump above its own recent level, 0..1, before the
#: spokes in its sector flick. Steady music sits under it; a hit crosses it.
_XC_RISE = 0.045

#: Onset strength that counts as a hit worth a coordinated event, and the
#: strength below which an onset is left to the band-rise pulses alone.
_XC_HIT = 0.68
_XC_TOUCH = 0.3

#: Delay per spoke of the hit ripple, seconds. The pulse runs round the
#: wireframe from the loudest sector in about a seventh of a second — every
#: spoke lights, but one after another, so the event has a direction and the
#: screen never strobes as a whole.
_XC_RIPPLE_S = 0.018

#: How many spokes either side of its origin a hit's pulse reaches before it
#: has faded out, and the least time between two such pulses. A pulse is a
#: region of the wireframe answering a hit, not the whole of it; measured on a
#: percussive folk track, a pulse per hit lighting every spoke had twelve or
#: more of the sixteen lit in 12% of frames.
_XC_RIPPLE_REACH = 6.0
_XC_RIPPLE_GAP = 0.3


#: Angular resolution of the spinning wireframe: a turn in this many steps.
#: 4096 is under a tenth of a degree, finer than the dot grid can show.
_XC_TURN_STEPS = 4096

#: How fast a flickering spoke strobes when no tempo is known, cycles per
#: second. With a tempo it strobes in sixteenth notes instead.
_XC_STROBE_HZ = 9.0

#: A flickering spoke's dash pattern, in real dots along the spoke: this long
#: a period, of which this much stays lit on the dark half of the strobe. The
#: lit part is kept at nine dots or more, so a dash is a piece of line and not
#: a speck.
_XC_DASH = 14.0
_XC_DASH_ON = 0.64


def _crosscurrent_geometry(dr: int, dc: int, dist, turn, max_r, base: dict) -> dict:
    """Everything about the spinning corridor that does not depend on the spin.

    The wireframe turns, so nothing here is a mask: the spokes and the
    octagons are resolved each frame through tables indexed by a dot's angle
    *relative to* the current spin (see :func:`crosscurrent`), and what is
    cached is the per-dot half of that — each dot's angle step, its radius in
    stretched and in real dots, and the tables that do not change as the whole
    thing rotates.

    A ring of the inbound stream is the set of points whose *octagonal* radius
    is the ring's radius: for a point at angle ``phi`` inside its sector, the
    flat edge between two corners sits at ``dist * cos(phi - half) / cos(half)``
    of the corner radius. Two streams put a ring at every half period of depth
    instead of every whole one, so the radius inside which rings are too close
    to resolve moves out by root two.
    """
    n = _XC_TURN_STEPS
    cy, cx = dr / 2.0, dc / 2.0
    s = cy / max(cx, 1.0)
    yy = (np.arange(dr, dtype=np.float32) - np.float32(cy))[:, None]
    xx = (np.arange(dc, dtype=np.float32) - np.float32(cx))[None, :]
    dots = np.sqrt(xx * xx + yy * yy)
    near = np.clip(dist / np.float32(max_r), 0.0, 1.0)
    cut = float(math.sqrt(2.0 * 3.5 * 0.55 * max_r))

    step = 2.0 * math.pi / 16.0
    fuse = 2.5
    hole = float(min(0.25 * max_r, max(0.07 * max_r, 3.0 / step)))

    rel = np.arange(n, dtype=np.float64) / n
    kf = np.rint(rel * 16.0)
    k = kf.astype(np.int32) % 16
    sector = 2.0 * math.pi / _XC_SIDES
    oct_table = np.cos(((rel * _XC_SIDES) % 1.0) * sector - sector / 2.0) / math.cos(sector / 2.0)
    reach_in = np.where(k % 4 == 0, fuse / (4 * step), np.where(k % 2 == 0, fuse / (2 * step), fuse / step))

    inner_idx = np.flatnonzero((dist <= np.float32(hole)).ravel())
    depth055 = base["depth055"].ravel()
    by_depth = np.argsort(depth055, kind="stable")
    return {
        "turn_idx": ((turn * np.float32(n)).astype(np.int32) & (n - 1)).ravel(),
        "turn": turn.ravel(),
        "spoke_index": _xc_spoke_index(dist, turn, hole),
        "by_depth": by_depth,
        "depth_sorted": depth055[by_depth],
        "dist": dist.ravel(),
        "dots": dots.ravel().astype(np.float32),
        "near": near.ravel().astype(np.float32),
        "stretch": s,
        "hole": hole,
        "outer_width": (np.float32(0.55) + np.float32(0.45) * near).ravel().astype(np.float32),
        "outer_value": (np.float32(0.25) + np.float32(0.35) * np.clip((dist - np.float32(hole)) / np.float32(0.8 * max_r), 0.0, 1.0)).ravel().astype(np.float32),
        "inner_idx": inner_idx,
        "edge": (dots * np.float32(hole) / np.maximum(dist, np.float32(1e-3))).ravel()[inner_idx].astype(np.float32),
        "k_table": k,
        "off_table": (kf / 16.0 - rel),
        "face_table": (np.floor(rel * _XC_SIDES) + 0.5) / _XC_SIDES,
        "oct_table": oct_table.astype(np.float32),
        "reach_table": reach_in.astype(np.float32),
        "depth_cut": float(0.55 * max_r / cut),
        "corner": float(0.55 * max_r / (float(dist.max()) - 6.0)),
        "ring_value": (np.clip((dist - np.float32(cut)) / np.float32(cut), 0.0, 1.0)
                       * (np.float32(0.3) + np.float32(0.7) * near)).ravel().astype(np.float32),
        "per_dot": base["per_dot"].ravel(),
        "depth055": base["depth055"].ravel(),
    }


def _xc_spoke_index(dist, turn, hole: float) -> tuple:
    """Where the outer spokes can be, for any spin: dots sorted by angle within
    rings of radius.

    A dot at stretched radius ``dist`` is within a dot of the spoke line at
    angle ``theta`` only if ``dist * |sin(phi - theta)| <= 1`` (the stretch can
    only lengthen the perpendicular), so inside a ring of radius starting at
    ``r`` the candidates for a spoke are the dots within ``asin(1 / r)`` of its
    angle. Dots are grouped into rings of radius a quarter wider each, sorted
    by angle, and each group is keyed ``angle + 4 * group`` into one array, so
    a frame finds every candidate with two ``searchsorted`` calls. Each group
    carries a copy of its dots near 0 and 1 a turn away, so a window across
    the seam is still one contiguous range.
    """
    n = _XC_TURN_STEPS
    dist_r = dist.ravel()
    turn_r = turn.ravel().astype(np.float64)
    far = np.flatnonzero(dist_r > np.float32(hole))
    edges = [hole]
    top = float(dist_r.max()) + 1.0
    while edges[-1] < top:
        edges.append(edges[-1] * 1.25)
    group = np.searchsorted(np.array(edges), dist_r[far].astype(np.float64), side="right") - 1
    wrap = 1.0 / 32.0 + 2.0 / n
    keys, members, groups, halves = [], [], [], []
    for g in range(len(edges) - 1):
        m = far[group == g]
        if not m.size:
            continue
        t = turn_r[m]
        order = np.argsort(t, kind="stable")
        t, m = t[order], m[order]
        head, tail = t < wrap, t >= 1.0 - wrap
        keys.append(np.concatenate([t[tail] - 1.0, t, t[head] + 1.0]) + 4.0 * g)
        members.append(np.concatenate([m[tail], m, m[head]]))
        groups.append(g)
        # never past half a spoke gap: past it a dot belongs to the next spoke
        halves.append(min(math.asin(min(1.0, 1.0 / edges[g])) / (2.0 * math.pi), 1.0 / 32.0) + 2.0 / n)
    return (np.concatenate(keys), np.concatenate(members),
            np.array(groups, dtype=np.float64), np.array(halves, dtype=np.float64))


def _xc_spoke_candidates(xc: dict, spin: float) -> np.ndarray:
    """Indices of every dot that could be on an outer spoke, each once."""
    keys, members, groups, halves = xc["spoke_index"]
    centres = (spin + np.arange(16) / 16.0) % 1.0
    base = 4.0 * groups[:, None] + centres[None, :]
    lo = np.searchsorted(keys, (base - halves[:, None]).ravel(), side="left")
    hi = np.searchsorted(keys, (base + halves[:, None]).ravel(), side="right")
    # Neighbouring windows only overlap when a small terminal's hole forces the
    # half-gap cap, and only then can a dot be listed twice.
    return _xc_ranges(members, lo, hi, unique=bool(halves.max() >= 1.0 / 32.0))


def _xc_depth_bands(xc: dict, bands) -> np.ndarray:
    """Indices of the dots whose round depth lies in any band, each once."""
    ds = xc["depth_sorted"]
    # float32 bounds: a float64 needle makes searchsorted cast the whole table
    lo = np.searchsorted(ds, np.array([a for a, _ in bands], dtype=np.float32), side="left")
    hi = np.searchsorted(ds, np.array([b for _, b in bands], dtype=np.float32), side="right")
    # The bands come in increasing depth but may overlap; clipping each start
    # to the furthest end before it lists every dot once without a sort.
    hi = np.maximum.accumulate(hi)
    lo[1:] = np.maximum(lo[1:], hi[:-1])
    return _xc_ranges(xc["by_depth"], lo, hi, unique=False)


def _xc_ranges(members: np.ndarray, lo: np.ndarray, hi: np.ndarray, unique: bool) -> np.ndarray:
    """``members[lo[i]:hi[i]]`` for every i, gathered at once; sorted and
    de-duplicated only when asked, because the order of the dots never changes
    what is drawn."""
    lengths = np.maximum(hi - lo, 0)
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    starts = np.repeat(lo - (np.cumsum(lengths) - lengths), lengths)
    got = members[starts + np.arange(total)]
    return np.unique(got) if unique else got


def _crosscurrent_spin_tables(xc: dict, spin_idx: int):
    """Per-frame tables for the current spin: how far a dot at each relative
    angle is from its nearest spoke, as a multiple of its stretched radius.

    A spoke at absolute angle ``theta`` is a straight line in real dots. For a
    dot at stretched radius ``dist`` and angle ``phi`` its perpendicular
    distance in dots is ``dist * |sin(theta - phi)| / (s * norm(theta))``, where
    ``norm`` is the length of the spoke's direction mapped back to dots. That
    splits into the per-dot ``dist`` and a table over the relative angle whose
    only dependence on the spin is through ``norm`` — 4096 entries a frame.

    The second table corrects the octagons' stroke width. Depth changes per
    real dot as fast as the octagonal radius does, and on a flat face that
    grows along the face's normal, not along the dot's own radius the round
    rings' rate is measured on. The stretch makes the two rates differ by up to
    an eighth, which is enough to thin a one-dot stroke into broken pieces.
    """
    n = _XC_TURN_STEPS
    s = xc["stretch"]
    spin = spin_idx / n
    theta = (spin + xc["k_table"] / 16.0) * 2.0 * math.pi
    norm = np.sqrt((np.cos(theta) / s) ** 2 + np.sin(theta) ** 2)
    to_spoke = (np.abs(np.sin(2.0 * math.pi * xc["off_table"])) / (s * norm)).astype(np.float32)
    face = (spin + xc["face_table"]) * 2.0 * math.pi
    phi = (spin + np.arange(n) / n) * 2.0 * math.pi
    along_face = np.sqrt((s * np.cos(face)) ** 2 + np.sin(face) ** 2) / math.cos(math.pi / _XC_SIDES)
    along_radius = np.sqrt((s * np.cos(phi)) ** 2 + np.sin(phi) ** 2)
    return to_spoke, (along_face / along_radius).astype(np.float32)


@mode("Crosscurrent", group="scenes", after="Tunnel In",
      blurb="a spinning tunnel of two ring streams, rushing out and drawn in, flickering on the beat")
def crosscurrent(ctx: Ctx):
    """A spinning tunnel carrying two opposing streams, and the music where they meet.

    Round rings travel out toward the viewer and eight-sided rings travel in
    toward the vanishing point, in the same wireframe at the same time, and
    the whole wireframe turns. Wherever a round ring and an octagon coincide
    the round stroke flares, and because a circle and an octagon only meet at
    some angles, a crossing is a spark sliding round the ring.

    **The spin follows the music.** With a tempo, the spokes advance about a
    spoke gap a beat, faster the more active the track is; without one, the
    rate is the activity alone. Hits give it a push, the rate eases toward its
    target over most of a second — so it visibly accelerates into a busy
    passage and coasts down out of one — and silence lets it settle to a slow
    turn. The octagons' corners stay on their spokes as it turns, and the
    brightness pattern of the bands turns with it.

    **The lines flicker.** A band that jumps pulses the spokes in its sector,
    and a pulsed spoke strobes: on the beat's sixteenth notes (about 9 Hz with
    no tempo) it alternates between a solid, bright line and a dashed one whose
    gaps shift on every strobe, so the light visibly stutters along the spoke
    rather than only changing colour. A strong hit sends the pulse sweeping
    out from the sector that jumped most, fading within six spokes; spokes that
    are not pulsing are never touched, and a dashed spoke keeps two thirds of
    its dots, so the tunnel never blinks out.

    The rest follows the tunnels: the lower half of the spectrum pushes the
    outbound stream and the upper half pulls the inbound one, each against its
    own recent level; strokes are bounded widths in real dots; spokes fade into
    a shared vanishing point; rings too close to resolve are not drawn; no
    dither.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 8 or dc < 8:
        return empty(ctx.w, ctx.h)

    dist, turn, max_r = _polar(ctx)
    geo = ctx.scratch("tunnel_geo", lambda: _tunnel_geometry(dr, dc, dist, turn, max_r))
    xc = ctx.scratch("crosscurrent_geo",
                     lambda: _crosscurrent_geometry(dr, dc, dist, turn, max_r, geo))
    dt = max(ctx.dt, 0.0)

    st = ctx.scratch("crosscurrent", lambda: {
        "out": 0.0, "in": 0.5, "spin": 0.0, "spin_v": 0.01, "strobe": 0.0,
        "slow": np.zeros(9, dtype=np.float64),
        "flash": np.zeros(16, dtype=np.float64),
        "due": np.full(16, -1.0), "amp": np.zeros(16),
        "surge": 0.0, "kick": 0.0,
    })

    # ── the music ──
    levels = ctx.display_bands(9).astype(np.float64)
    rise = np.maximum(levels - st["slow"], 0.0)
    st["slow"] += (levels - st["slow"]) * min(1.0, dt / 0.3)
    # Each stream answers its own half of the spectrum, measured against that
    # half's recent level. Absolute levels do not work on real music: across a
    # minute of each of two ordinary tracks the bottom quarter of the bands had
    # a median level of 0.00 and 0.02. The floor keeps near-silence from being
    # amplified into motion.
    low = ctx.range(0.0, 0.45)
    high = ctx.range(0.45, 1.0)
    a = min(1.0, dt / 4.0)
    st["avg_lo"] = st.get("avg_lo", low) + (low - st.get("avg_lo", low)) * a
    st["avg_hi"] = st.get("avg_hi", high) + (high - st.get("avg_hi", high)) * a
    drive_lo = min(1.5, low / max(2.0 * st["avg_lo"], 0.04))
    drive_hi = min(1.5, high / max(2.0 * st["avg_hi"], 0.04))

    flash = st["flash"]
    flash *= math.exp(-dt / _XC_FLASH_TAU)
    k16 = np.arange(16)
    # Spoke k sits k/16 of a turn round from spoke 0. Its band is how far it is
    # from spoke 0's axis, on either side, so the spectrum is mirrored both
    # ways across the turning structure: bass on spokes 0 and 8, treble on 4
    # and 12.
    from_axis = np.minimum(k16 % 8, 8 - k16 % 8)
    sector = from_axis * 2
    sector_rise = rise[sector]
    # Only the two sectors that rose most may flick. A broadband attack lifts
    # every band at once, and with four spokes mirrored to most bands, lighting
    # every sector that cleared the threshold had twelve or more spokes lit at
    # once in 4% of the frames of a percussive track.
    ranked = np.sort(rise[[0, 2, 4, 6, 8]])
    strongest = sector_rise >= ranked[-2]
    np.maximum(flash, np.where(strongest, np.clip((sector_rise - _XC_RISE) * 7.0, 0.0, 1.0), 0.0), out=flash)

    st["surge"] *= math.exp(-dt / 0.45)
    st["kick"] *= math.exp(-dt / 0.2)
    hit = float(min(1.0, ctx.onset_strength)) if ctx.onsets else 0.0
    if hit >= _XC_HIT:
        st["surge"] = max(st["surge"], hit)
        st["kick"] = max(st["kick"], hit)
    if hit >= _XC_HIT and ctx.t - st.get("rippled", -9.0) >= _XC_RIPPLE_GAP:
        # The pulse starts in the sector that jumped most and spreads to its
        # neighbours, fading with distance: a region of the wireframe answers
        # the hit in a quick sweep, not the whole of it.
        origin = int(np.argmax(sector_rise)) if sector_rise.max() > 0 else 0
        gap = np.abs(k16 - origin)
        gap = np.minimum(gap, 16 - gap)
        st["due"] = ctx.t + gap * _XC_RIPPLE_S
        st["amp"] = (0.55 + 0.45 * hit) * np.clip(1.0 - gap / _XC_RIPPLE_REACH, 0.0, 1.0) ** 1.5
        st["rippled"] = ctx.t
    elif hit >= _XC_TOUCH:
        # A smaller onset is a detail: the sector that jumped most and its two
        # neighbours pulse, in proportion to the hit.
        origin = int(np.argmax(sector_rise)) if sector_rise.max() > 0 else int(np.argmax(levels[sector]))
        gap = np.abs(k16 - origin)
        gap = np.minimum(gap, 16 - gap)
        np.maximum(flash, np.where(gap <= 1, (0.45 + 0.4 * hit) * (1.0 - 0.35 * gap), 0.0), out=flash)
    ready = (st["due"] >= 0.0) & (ctx.t >= st["due"])
    if ready.any():
        np.maximum(flash, np.where(ready, st["amp"], 0.0), out=flash)
        st["due"][ready] = -1.0

    st["out"] += (0.12 + 0.9 * drive_lo + 1.1 * st["surge"]) * dt
    st["in"] += (0.12 + 0.9 * drive_hi + 0.8 * st["surge"]) * dt

    # ── the spin ──
    # How active the track is: its attack, how far its two halves sit above
    # their own recent levels (a steady track sits at one half each, so that
    # counts for nothing), and the surge of recent hits. A tempo sets the pace,
    # a spoke gap a beat; without one the pace is that of 96 bpm. Activity
    # scales it on a curve, so the difference between a steady passage and a
    # busy one is a visible change of gear: a sixth of the pace for a pad,
    # about the pace at moderate activity, and up to three and a half times
    # it when everything is going.
    activity = min(1.5, ctx.drive + 0.6 * max(0.0, drive_lo + drive_hi - 1.0) + 0.5 * st["surge"])
    pace = ctx.tempo_bpm / 60.0 / 16.0 if ctx.tempo_bpm > 0 else 0.1
    target = 0.01 if ctx.silent else 0.01 + pace * (0.15 + 1.5 * activity * activity)
    st["spin_v"] += (target - st["spin_v"]) * min(1.0, dt / 0.5)
    if hit >= _XC_HIT:
        st["spin_v"] += 0.2 * pace * hit
    st["spin"] = (st["spin"] + st["spin_v"] * dt) % 1.0

    # the strobe: sixteenth notes with a tempo, a fixed rate without
    strobe_hz = min(16.0, 4.0 * ctx.tempo_bpm / 60.0) if ctx.tempo_bpm > 0 else _XC_STROBE_HZ
    st["strobe"] += strobe_hz * dt
    strobe_dark = (st["strobe"] % 1.0) >= 0.5
    dash_shift = (math.floor(st["strobe"]) * 0.37) % 1.0

    n = min(16, ctx.n_display)
    # The band pattern turns with the wireframe. Only the 512-entry table is
    # built here; each set of dots that needs a level gathers its own, which
    # is exactly lut[idx] without an index over the whole grid.
    lut = _angular_lut(ctx, turn[:1, :1], n, -st["spin"])[0]
    lut_turn = xc["turn"]
    lut_offset = np.float32(float(-st["spin"]) % 1.0)

    def nrg_at(idx):
        return lut[((lut_turn[idx] + lut_offset) * np.float32(512)).astype(np.int32) & 511]

    kick = np.float32(st["kick"])

    # ── the spinning spokes ──
    # Everything below works on the dots that could be on a stroke, found from
    # the geometry cache, rather than on the grid: turning the wireframe means
    # nothing about a stroke can be cached, and at 400x100 the whole-grid form
    # of this frame cost two thirds as much again as the static one.
    steps = _XC_TURN_STEPS
    spin_idx = int(round(st["spin"] * steps)) & (steps - 1)
    gtab, oct_rate = _crosscurrent_spin_tables(xc, spin_idx)
    turn_idx = xc["turn_idx"]
    dist_f = xc["dist"]

    def rel_at(idx):
        return (turn_idx[idx] - spin_idx) & (steps - 1)

    cand = _xc_spoke_candidates(xc, spin_idx / steps)
    outer = cand[dist_f[cand] * gtab[rel_at(cand)] <= xc["outer_width"][cand]]
    inn = xc["inner_idx"]
    inn_rel = rel_at(inn)
    reach_i_in = xc["reach_table"][inn_rel]
    inner_ok = (dist_f[inn] * gtab[inn_rel] <= np.float32(0.5)) & (xc["dots"][inn] >= reach_i_in)
    inner = inn[inner_ok]
    edge = xc["edge"][inner_ok]
    reach_ok = reach_i_in[inner_ok]
    fade = np.clip((xc["dots"][inner] - reach_ok) / np.maximum(edge - reach_ok, np.float32(1e-3)), 0.0, 1.0) ** np.float32(1.5)

    spoke_of_outer = xc["k_table"][rel_at(outer)]
    st["spokes"] = (outer, inner, spoke_of_outer)
    pulse_level = flash[spoke_of_outer]
    if strobe_dark and (pulse_level > 0.2).any():
        # A pulsing spoke's dark strobe: gaps open along it, in real dots, and
        # shift every strobe so the light visibly stutters along the line. The
        # first period out of the hole stays solid, and a piece the screen edge
        # would cut shorter than six dots goes dark with its gap, so a dashed
        # spoke is pieces of line and never specks.
        pulsing = pulse_level > 0.2
        along = xc["dots"][outer]
        lead = along - along * np.float32(xc["hole"]) / np.maximum(dist_f[outer], np.float32(1e-3))
        end = np.zeros(16, dtype=np.float32)
        np.maximum.at(end, spoke_of_outer, along)
        dash = (along / np.float32(_XC_DASH) + np.float32(dash_shift)) % np.float32(1.0)
        piece = np.minimum(np.float32(_XC_DASH_ON * _XC_DASH), end[spoke_of_outer] - (along - dash * np.float32(_XC_DASH)))
        keep = ~pulsing | (lead < np.float32(_XC_DASH)) | ((dash < np.float32(_XC_DASH_ON)) & (piece >= np.float32(6.0)))
        outer, spoke_of_outer, pulse_level = outer[keep], spoke_of_outer[keep], pulse_level[keep]

    # ── both streams, stroke half-widths in real dots ──
    cap = np.float32(0.11)
    near = xc["near"]
    per_o = xc["per_dot"]
    depth055 = xc["depth055"]
    depth_cut = np.float32(xc["depth_cut"])

    def half_at(idx):
        h_ = np.float32(0.45) + np.float32(0.2) * near[idx]
        h_ += np.float32(0.4) * nrg_at(idx)
        h_ += np.float32(0.25) * kick
        return h_

    # outbound: round rings, toward the viewer. A ring only just larger than
    # the screen would show as slivers of a few dots in its corners, so rings
    # reaching within six units of a corner are not drawn at all.
    shift_o = np.float32(st["out"] % 1.0)
    corner_o = np.float32(xc["corner"])

    def out_at(idx):
        f_ = depth055[idx] + shift_o
        c_ = np.rint(f_)
        f_ -= c_
        np.abs(f_, out=f_)
        c_ -= shift_o
        return f_, (c_ <= depth_cut) & (c_ >= corner_o)

    # inbound: octagonal rings, toward the vanishing point, turning with the
    # spokes. The same corner rule, but an octagon's reach toward the corners
    # depends on how far the turn has carried a vertex from them; the corners
    # are a quarter turn apart in stretched coordinates, as the vertices are,
    # so one of them stands for all four.
    shift_i = np.float32(st["in"] % 1.0)
    oct_table = xc["oct_table"]
    corner_i = xc["corner"] / oct_table[rel_at(np.zeros(1, dtype=np.int64))][0]
    depth_scale = np.float32(0.55 * max_r)

    def in_at(idx):
        o_ = oct_table[rel_at(idx)]
        f_ = depth_scale / np.maximum(dist_f[idx] * o_, np.float32(0.9))
        f_ -= shift_i
        c_ = np.rint(f_)
        f_ -= c_
        np.abs(f_, out=f_)
        c_ += shift_i
        return f_, (c_ <= depth_cut) & (c_ >= np.float32(corner_i)), o_

    # Rings sit at whole depths less the stream's shift, so the dots that can
    # be on one lie in a handful of narrow depth bands, looked up in the cache.
    lo_m = math.floor(xc["corner"] + float(shift_o)) - 1
    hi_m = math.ceil(xc["depth_cut"] + float(shift_o)) + 1
    super_o = _xc_depth_bands(xc, [(m - float(shift_o) - 0.111, m - float(shift_o) + 0.111)
                                   for m in range(lo_m, hi_m + 1)])
    fo, ok_o = out_at(super_o)
    cand_o = super_o[(fo <= cap) & ok_o]
    fo_c = fo[(fo <= cap) & ok_o]
    half_o = half_at(cand_o)
    on_o = fo_c <= np.minimum(per_o[cand_o] * half_o, cap)
    idx_o = cand_o[on_o]
    half_o = half_o[on_o]

    # an octagon's depth lies between the round depth and cos(pi/8) of it
    lo_m = math.floor(corner_i - float(shift_i)) - 1
    hi_m = math.ceil(xc["depth_cut"] - float(shift_i)) + 1
    widen = 1.0 / math.cos(math.pi / _XC_SIDES)
    super_i = _xc_depth_bands(xc, [(m + float(shift_i) - 0.111, (m + float(shift_i) + 0.111) * widen + 0.001)
                                   for m in range(lo_m, hi_m + 1)])
    fi, ok_i, oct_scale = in_at(super_i)
    keep_i = (fi <= cap) & ok_i
    cand_i = super_i[keep_i]
    fi_c = fi[keep_i]
    oct_c = oct_scale[keep_i]

    # ── the crossing ──
    # Near a round ring the octagon narrows smoothly to a one-dot hairline, so
    # the pair reads as one line with a thread beside it rather than a heavy
    # double band, and the round stroke flares where the octagon is near.
    fi_o, ok_i_o, oct_o = in_at(idx_o)
    per_i_o = per_o[idx_o] * oct_rate[rel_at(idx_o)] / (oct_o * oct_o)
    flare = idx_o[(fi_o <= per_i_o * (half_o + np.float32(1.5))) & ok_i_o]
    half_c = half_at(cand_i)
    fo_i = out_at(cand_i)[0]
    beside = np.clip(fo_i / np.maximum(per_o[cand_i], np.float32(1e-6)) / np.float32(3.0), 0.0, 1.0)
    half_i = np.float32(0.5) + (half_c - np.float32(0.5)) * beside
    per_i_c = per_o[cand_i] * oct_rate[rel_at(cand_i)] / (oct_c * oct_c)
    idx_i = cand_i[fi_c <= np.minimum(per_i_c * half_i, cap)]

    lit_f = np.zeros(dr * dc, dtype=bool)
    lit_f[outer] = True
    lit_f[inner] = True
    lit_f[idx_o] = True
    lit_f[idx_i] = True
    lit = lit_f.reshape(dr, dc)
    codes = pack_braille(lit)
    st["drawn"] = outer

    # ── brightness ──
    ring_value = xc["ring_value"]
    shade_f = np.zeros(dr * dc, dtype=np.float32)
    # a dot on both streams is listed twice; both writes are the same value
    rings_idx = np.concatenate([idx_o, idx_i])
    lift = np.float32(0.35) + np.float32(0.35) * nrg_at(rings_idx)
    lift += np.float32(0.2) * kick
    shade_f[rings_idx] = lift * ring_value[rings_idx]
    shade_f[flare] = np.maximum(shade_f[flare], np.float32(0.35) + np.float32(0.65) * ring_value[flare])

    rest = xc["outer_value"][outer] * np.float32(0.55)
    pulse = rest + (np.float32(1.0) - rest) * pulse_level.astype(np.float32)
    shade_f[outer] = np.maximum(shade_f[outer], pulse)

    lo = _faint_walk_point(ctx.palette, _TUNNEL_FADE_FLOOR, 3.0)
    # Gathered before it is written back, so a dot listed twice is lifted once.
    drawn_all = np.concatenate([rings_idx, outer, inner])
    part = shade_f[drawn_all]
    part *= np.float32(1.0 - lo)
    part += np.float32(lo)
    shade_f[drawn_all] = part
    shade_f[inner] = np.maximum(shade_f[inner], fade * np.float32(lo + (1.0 - lo) * 0.25))
    return codes, contrast_ramp(ctx.palette, cell_max(shade_f.reshape(dr, dc)), faint=_TUNNEL_FADE_FLOOR)
