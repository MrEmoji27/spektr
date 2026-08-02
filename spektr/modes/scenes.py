"""Scene modes — things with a horizon, a depth axis, or their own particles."""

from __future__ import annotations

import math

import numpy as np

from ..palette import RAMP_STEPS
from ..render import SPACE, cell_max, noise, pack_braille
from . import Ctx, band_columns, empty, mode, spread
from .particles import _angular_bands, _polar

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

    # ── sun: a half disc above the horizon, sliced by scanline gaps ──
    sun_r = horizon * 0.85
    rd = (horizon - rows).astype(np.float64)              # distance up from horizon
    above = (rows < horizon) & (rd <= sun_r)
    hw = np.sqrt(np.maximum(sun_r * sun_r - rd * rd, 0.0))
    slice_w = max(1.0, sun_r * 0.15)
    banded = (rd < sun_r * 0.5) & (((rd // slice_w).astype(np.int64)) % 2 == 1)
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
    scroll = math.fmod(ctx.t * 0.48, 1.0)
    z = np.mod((np.arange(10) + scroll) / 10.0, 1.0)
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

    # colour: grid cool, sun mid, wave hot
    heat = np.zeros((dr, dc), dtype=np.float64)
    heat[grid == 1] = 0.12
    heat[grid == 3] = 0.55
    heat[grid == 2] = 1.0
    cidx = ctx.ramp(cell_max(heat))
    return codes, cidx


@mode("Auroras", group="scenes", blurb="light curtains, billowing on the treble")
def auroras(ctx: Ctx):
    """Vertical curtains that hang from the top and billow with the highs.

    The obvious implementation — one full-grid mask per curtain — costs a pass
    over 300k dots per curtain at fullscreen. This builds the curtains once as
    two 1-D profiles across the width (brightness and how far down they hang),
    then *shears* them per row with a single gather. The undulation is real
    horizontal displacement rather than a per-row recomputation, so the cost is
    the same for two curtains or ten.

    ``Plasma`` is a smooth field with no structure; this has discrete curtains
    with a soft vertical falloff and a ragged lower edge, which reads as
    something else entirely.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 12 or dc < 16:
        return empty(ctx.w, ctx.h)

    n = int(np.clip(dc // 110, 2, 5))
    lows = ctx.range(0.00, 0.20)
    treble = ctx.range(0.60, 1.00)

    cols = np.arange(dc, dtype=np.float32)
    bright = np.zeros(dc, dtype=np.float32)
    reach = np.zeros(dc, dtype=np.float32)

    lv = ctx.display_bands(n).astype(np.float32)
    spacing = dc / n
    for i in range(n):
        level = float(lv[i])
        centre = (i + 0.5) * spacing + math.sin(ctx.t * (0.21 + 0.06 * i) + i * 2.1) * spacing * 0.35
        half = spacing * (0.10 + 0.30 * level)
        d = np.abs(cols - centre)
        edge = np.clip(1.0 - d / max(half, 1e-3), 0.0, 1.0)
        # squared falloff: a soft core with visible edges, not a flat band
        np.maximum(bright, edge * edge * (0.25 + 0.75 * level), out=bright)
        np.maximum(reach, edge * (0.30 + 0.65 * level), out=reach)

    # fine vertical ribbing — the striations are what make an aurora read as an
    # aurora rather than as a smear, and in 1-D they cost nothing
    bright *= 0.68 + 0.32 * np.sin(cols * 0.8 + ctx.t * 0.5).astype(np.float32)

    # Tiled three times so the shear can wrap by simple offset. A modulo over
    # the whole dot grid is one of the most expensive things you can do per
    # frame; adding dc to the index and reading from the middle copy is free.
    bright3 = np.tile(bright, 3)
    reach3 = np.tile(reach, 3)
    inv_reach3 = 1.0 / np.maximum(reach3, 1e-3)     # inverted once, in 1-D

    # per-row horizontal shear — this is the billow
    y = np.arange(dr, dtype=np.float32) / max(1, dr - 1)
    sway = min((0.03 + 0.10 * treble) * dc, dc * 0.9)
    shift = (
        np.sin(y * 3.1 + ctx.t * 0.9) * sway
        + np.sin(y * 7.7 - ctx.t * 0.55) * sway * 0.4
    )
    idx = (cols[None, :] + (shift + dc)[:, None]).astype(np.int32)

    # hangs from the top: full strength at the top, gone by the curtain's reach
    falloff = 1.0 - y[:, None] * inv_reach3[idx]
    np.clip(falloff, 0.0, 1.0, out=falloff)

    gain = np.float32(0.55 + 0.75 * lows)
    heat = bright3[idx]
    heat *= (np.float32(0.25) + np.float32(0.75) * falloff) * gain

    # ragged lower edge — dithered where the curtain is fading out
    keep = noise((dr, dc), ctx.frame) < (falloff * np.float32(1.6) + np.float32(0.15))
    lit = (heat > 0.07) & (falloff > 0.0) & keep

    codes = pack_braille(lit)
    cidx = ctx.ramp(cell_max(np.where(lit, np.clip(heat, 0.0, 1.0), 0.0)))
    return codes, cidx


_WINDOW = ord("▀")
_ANTENNA = ord("╹")


@mode("Skyline", group="scenes", blurb="a city at night, windows lit by their band")
def skyline(ctx: Ctx):
    """Bands as buildings, with windows that flicker at the band's own rate.

    Drawn at text-cell resolution rather than on the dot grid: a skyline wants
    hard rectangular edges, and braille would only soften them. ``Retro`` has a
    horizon and a perspective grid; this is flat and front-lit, and the motion
    comes entirely from the windows.
    """
    w, h = ctx.w, ctx.h
    if w < 12 or h < 5:
        return empty(w, h)

    n = ctx.n_display
    col_band, active = band_columns(w, n)      # the gutters become alleys
    lv = ctx.display_bands(n)

    level = np.where(active, lv[col_band], 0.0)
    tops = h - 1 - np.rint(np.clip(level, 0.05, 1.0) * (h - 2)).astype(np.int32)

    rows = np.arange(h, dtype=np.int32)[:, None]
    body = active[None, :] & (rows >= tops[None, :])

    # a fixed random phase per cell, so windows don't all blink together
    phase = ctx.scratch(
        "skyline", lambda: np.random.default_rng(5).uniform(0.0, 2 * math.pi, (h, w))
    )
    # window grid: every other row, every third column of each building
    grid = ((rows % 2) == 1) & ((np.arange(w)[None, :] % 3) == 1)
    rate = 1.2 + level * 7.0
    lit = np.sin(phase + ctx.t * rate[None, :]) > 0.45
    windows = body & grid & lit

    codes = np.where(body, _FULL, SPACE).astype(np.int32)
    codes[windows] = _WINDOW

    # buildings are near-black; the windows carry all the colour
    cidx = np.zeros((h, w), dtype=np.int32)
    cidx[body] = ctx.palette.index(0.06)
    win_heat = np.repeat(ctx.ramp(0.55 + 0.45 * level)[None, :], h, axis=0)
    cidx[windows] = win_heat[windows]

    # a blinking aircraft light on whichever buildings are tallest right now
    tall = np.flatnonzero(active & (level > 0.72))
    if tall.size and math.sin(ctx.t * 3.4) > 0.0:
        r = np.clip(tops[tall] - 1, 0, h - 1)
        codes[r, tall] = _ANTENNA
        cidx[r, tall] = RAMP_STEPS - 1
    return codes, cidx


@mode("Tunnel", group="scenes", blurb="flying down a pipe, ribbed by the beat")
def tunnel(ctx: Ctx):
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 8 or dc < 8:
        return empty(ctx.w, ctx.h)

    dist, turn, max_r = _polar(ctx)
    n = min(16, ctx.n_display)
    nrg = _angular_bands(ctx, turn, n, ctx.t * 0.024)

    # inverse distance is the depth axis: near the centre is far away
    depth = max_r / np.maximum(dist, 0.9)

    speed = 0.6 + ctx.energy * 2.4
    rings = np.mod(depth * 0.55 - ctx.t * speed, 1.0)
    ribs = rings < (0.10 + 0.22 * nrg)

    spokes = np.mod(turn * 16.0 + depth * 0.03, 1.0)
    walls = spokes < 0.09

    lit = (ribs | walls) & (dist > 1.5)
    # fade the far end out so the centre reads as distance, not a hole
    near = np.clip(dist / max_r, 0.0, 1.0)
    lit &= noise((dr, dc), ctx.frame) < (0.25 + near * 0.85)

    codes = pack_braille(lit)
    cidx = ctx.ramp(cell_max(np.where(lit, near * (0.4 + 0.6 * nrg), 0.0)))
    return codes, cidx


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
    st["rad"] += st["spd"] * ctx.dt * (0.22 + ctx.energy * 1.9)

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
    st["head"] += st["spd"] * ctx.dt * (0.35 + drive * 2.2)

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
    bright = np.where(lit, 1.0 - behind / np.maximum(st["len"][None, :], 1.0), 0.0)

    codes = np.where(lit, st["glyph"], SPACE).astype(np.int32)
    cidx = ctx.ramp(bright ** 1.6)
    # the leading glyph of each drop burns brightest
    headrow = np.rint(st["head"]).astype(np.int32)
    ok = (headrow >= 0) & (headrow < h)
    cols = np.flatnonzero(ok)
    if cols.size:
        cidx[headrow[cols], cols] = RAMP_STEPS - 1
    return codes, cidx
