"""Full-field modes: waterfall, plasma, and level meters."""

from __future__ import annotations

import math

import numpy as np

from ..palette import RAMP_STEPS
from ..render import (
    SPACE,
    cell_max,
    pack_braille,
)
from . import Ctx, empty, mode, spread

#: Spectro's scroll rate, in columns per second. Paced in seconds rather than
#: per frame so the time axis means the same thing at any frame rate — see the
#: mode's docstring.
SPECTRO_COLS_PER_SEC = 60.0

_UPPER_HALF = ord("▀")
_FULL = ord("█")
_TICK = ord("┃")
_DOT = ord("·")
_BOLD_DOT = ord("•")
_FLAG = ord("▲")

@mode("VFD", group="fields", blurb="vacuum-fluorescent bargraph with phosphor afterglow")
def vfd(ctx: Ctx):
    """The other Japanese hi-fi display technology, next to the LED meters.

    A vacuum-fluorescent display doesn't turn off instantly — the phosphor
    keeps glowing for a beat after the drive current cuts, so a fast
    transient leaves a fading trail above where the bar now sits instead of a
    hard edge. Reproduced literally: each frame draws a crisp new bar into a
    dot-resolution persistence buffer that decays exponentially, and dots
    stay lit — at fading brightness — until the decay carries them below the
    draw threshold.
    """
    w, h = ctx.w, ctx.h
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 4 or dc < 4:
        return empty(w, h)

    buf = ctx.scratch("vfd", lambda: np.zeros((dr, dc), dtype=np.float32))

    level = spread(ctx.display_bands(), dc)
    rows = np.arange(dr - 1, -1, -1, dtype=np.float32) / dr    # row 0 (top) -> ~1
    bar = (level[None, :] > rows[:, None]).astype(np.float32)

    decay = float(np.exp(-max(ctx.dt, 0.0) / 0.12))
    buf *= decay
    np.maximum(buf, bar, out=buf)

    dots = buf > 0.04
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(buf * dots))
    return codes, cidx


def _vu_level(ctx: Ctx) -> np.ndarray:
    """Per-channel loudness, weighted like a real VU rather than flat.

    Treble weighs more than bass, so the reading tracks perceived loudness
    instead of whatever the kick drum is doing. Shared by both meter modes so
    they can never disagree about how loud the signal is.
    """
    weight = np.linspace(0.55, 1.25, len(ctx.bands_l))
    return np.array(
        [
            float(np.clip((ctx.bands_l * weight).mean() * 1.5, 0.0, 1.0)),
            float(np.clip((ctx.bands_r * weight).mean() * 1.5, 0.0, 1.0)),
        ]
    )


def _peak_hold(st: dict, lvl: np.ndarray, dt: float) -> np.ndarray:
    """Instant attack, held, then a slow fall — the mechanical meter's ballistic."""
    rise = lvl >= st["peak"]
    st["peak"][rise] = lvl[rise]
    st["hold"][rise] = 0.9
    st["hold"][~rise] -= dt
    falling = st["hold"] <= 0
    st["peak"][falling] = np.maximum(lvl[falling], st["peak"][falling] - 0.7 * dt)
    return st["peak"]


@mode("Needle", group="stereo", blurb="analogue VU — one sweeping needle, one red zone")
def needle(ctx: Ctx):
    """A hardware VU meter: pivot, arc, red zone, and a peak flag.

    The other meters are bar graphs. This is the dial — the level is an angle,
    not a length, and the red zone sits where it does on the glass rather than
    being implied by colour. Geometry is cached per size, so a frame is three
    comparisons over the (small) text grid.
    """
    w, h = ctx.w, ctx.h
    if w < 20 or h < 6:
        return empty(w, h)

    def geo():
        cx = (w - 1) / 2.0
        cy = h - 1.0
        # halve x: a text cell is about twice as tall as it is wide, and
        # without this the arc comes out as an ellipse
        dx = (np.arange(w, dtype=np.float64) - cx) * 0.5
        dy = cy - np.arange(h, dtype=np.float64)
        dist = np.sqrt(dx[None, :] ** 2 + dy[:, None] ** 2)
        ang = np.arctan2(dy[:, None] + np.zeros_like(dx)[None, :], dx[None, :] + np.zeros_like(dy)[:, None])
        radius = max(2.0, min(h - 1.5, (w * 0.5) * 0.5 - 1.0))
        return dist, ang, radius

    dist, ang, radius = ctx.scratch("needle", geo)
    st = ctx.scratch("needle_peak", lambda: {"peak": np.zeros(2), "hold": np.zeros(2)})

    lvl = _vu_level(ctx)
    peak = _peak_hold(st, lvl, ctx.dt)
    level = float(lvl.mean())
    peak_level = float(peak.max())

    lo, hi = math.radians(160.0), math.radians(20.0)   # left = empty, right = full
    sweep = lo + (hi - lo) * level
    peak_sweep = lo + (hi - lo) * min(1.0, peak_level)
    red_from = lo + (hi - lo) * 0.78

    on_arc = (np.abs(dist - radius) < 0.75) & (ang <= lo) & (ang >= hi)
    red = on_arc & (ang <= red_from)

    codes, cidx = empty(w, h)
    codes[on_arc] = _DOT
    cidx[on_arc] = ctx.palette.index(0.2)
    codes[red] = _BOLD_DOT
    cidx[red] = RAMP_STEPS - 1

    # scale ticks every 20% of the sweep, just inside the arc
    tol_tick = 1.0 / np.maximum(dist, 1.0)
    for frac in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        a = lo + (hi - lo) * frac
        tick = (np.abs(ang - a) < tol_tick) & (np.abs(dist - radius * 0.86) < 0.8)
        codes[tick] = _TICK
        cidx[tick] = ctx.palette.index(0.3 + 0.5 * frac)

    # the needle: thin at the tip because the angular tolerance shrinks with radius
    tol = 1.1 / np.maximum(dist, 1.0)
    shaft = (np.abs(ang - sweep) < tol) & (dist <= radius * 0.92)
    codes[shaft] = _FULL
    cidx[shaft] = ctx.ramp(np.clip(0.25 + level, 0.0, 1.0))

    flag = (np.abs(ang - peak_sweep) < tol_tick * 0.8) & (np.abs(dist - radius * 1.04) < 0.8)
    codes[flag] = _FLAG
    cidx[flag] = RAMP_STEPS - 1

    pivot = dist < 1.6
    codes[pivot] = _FULL
    cidx[pivot] = ctx.palette.index(0.45)
    return codes, cidx


@mode("VU", group="stereo", blurb="big L/R LED meters with peak hold")
def vu(ctx: Ctx):
    w, h = ctx.w, ctx.h
    if w < 12 or h < 4:
        return empty(w, h)

    def init():
        return {"peak": np.zeros(2), "hold": np.zeros(2)}

    st = ctx.scratch("vu", init)

    lvl = _vu_level(ctx)
    _peak_hold(st, lvl, ctx.dt)

    codes, cidx = empty(w, h)

    gap = 1 if h >= 6 else 0
    band_h = max(1, (h - gap) // 2)
    top = 0
    bottom = band_h + gap

    xs = np.arange(w)
    heat = ctx.ramp(xs / max(1, w - 1))
    gutter = (xs % 4) == 3          # LED segmentation

    for ch, y0 in ((0, top), (1, bottom)):
        n_lit = int(round(lvl[ch] * w))
        lit = (xs < n_lit) & ~gutter
        y1 = min(h, y0 + band_h)
        if y1 <= y0:
            continue
        rows = slice(y0, y1)
        codes[rows, lit] = _FULL
        cidx[rows] = heat

        px = int(round(st["peak"][ch] * (w - 1)))
        if px > 0:
            codes[rows, px] = _TICK
            cidx[rows, px] = RAMP_STEPS - 1

    if gap:
        codes[band_h] = SPACE
    return codes, cidx
