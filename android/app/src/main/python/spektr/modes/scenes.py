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
    noise,
    noise_below,
    noise_level,
    pack_braille,
)
from . import (
    Ctx,
    angular_bands as _angular_bands,
    band_columns,
    empty,
    mode,
    polar_grid as _polar,
    spread,
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

    # The corridor itself does not move. Depth is a function of distance from
    # the centre, the spokes are a function of angle and depth, the fade is a
    # function of distance, and none of those three has a time term in it —
    # what moves is the rib phase sliding along a fixed depth axis and the
    # per-band brightness. Recomputing the fixed part every frame cost about a
    # fifth of the mode: at 400x100 the depth divide, the spoke ``frac`` and
    # the fade clip together measured 2.1 ms of a 15.8 ms frame, on 320,000
    # dots that produce the same numbers every time.
    #
    # One scratch entry, not five: the audit caps a mode at four keys, and
    # these five arrays have exactly the same lifetime — they are all
    # "everything about a corridor of this size".
    def build():
        depth = max_r / np.maximum(dist, 0.9)
        walls = frac(turn * 16.0 + depth * 0.03) < 0.09
        near = np.clip(dist / max_r, 0.0, 1.0)
        return {
            # Pre-scaled: the rib phase is subtracted from this every frame,
            # and the multiply was a full pass over the dot grid for a
            # constant.
            "depth055": depth * np.float32(0.55),
            # The spokes and the dead zone around the centre are both fixed
            # masks, so combine them once. ``lit`` below then needs one OR and
            # one AND rather than two ANDs and an OR.
            "walls": walls & (dist > 1.5),
            "far": dist > 1.5,
            "near": near,
            # The dither threshold rises with distance so the far end thins
            # out. Held in the integer space the hash already lives in — see
            # render.noise_level — so the per-frame dither is a comparison and
            # nothing else.
            "dither": noise_level(0.25 + near * np.float32(0.85)),
        }

    geo = ctx.scratch("tunnel_geo", build)
    near = geo["near"]

    n = min(16, ctx.n_display)
    nrg = _angular_bands(ctx, turn, n, ctx.t * 0.024)

    speed = 0.6 + ctx.energy * 2.4
    phase = ctx.scratch("tunnel{}_phase".format("_in" if inward else ""), lambda: {"v": 0.0})
    phase["v"] += speed * max(ctx.dt, 0.0)
    direction = -1 if inward else 1
    rings = frac(geo["depth055"] - direction * phase["v"])
    ribs = rings < (0.10 + 0.22 * nrg)

    # In place from here down. Every one of these is a 320,000-element array,
    # so a temporary is 1.3 MB of allocation and a pass over it; ``ribs`` is
    # already a fresh array nothing else holds, which makes it the buffer.
    np.bitwise_and(ribs, geo["far"], out=ribs)
    np.bitwise_or(ribs, geo["walls"], out=ribs)
    lit = ribs
    np.bitwise_and(lit, noise_below((dr, dc), ctx.frame, geo["dither"]), out=lit)

    codes = pack_braille(lit)
    # Same product, one buffer: multiplication commutes exactly in IEEE, so
    # starting from the only fresh array here and folding the rest in place is
    # bit-identical to ``near * (0.4 + 0.6 * nrg) * lit`` and allocates two
    # fewer dot-grid temporaries.
    shade = nrg * np.float32(0.6)
    shade += np.float32(0.4)
    shade *= near
    shade *= lit
    return codes, ctx.ramp(cell_max(shade))


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
