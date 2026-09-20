"""Full-field modes: waterfall, plasma, and level meters."""

from __future__ import annotations

import numpy as np

from ..analysis import resample_bands
from ..render import (
    SHADES,
    shade_cells,
    subcell_rows,
)
from . import Ctx, empty, mode

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


@mode("Spectro", group="fields", blurb="scrolling waterfall — frequency up, time across")
def spectrogram(ctx: Ctx):
    """History is the point of this one, so it keeps a rolling buffer.

    Frequency runs bottom-to-top and time scrolls right-to-left, which is the
    convention every other spectrogram uses; getting it backwards makes the
    display unreadable to anyone who has seen one before.

    The scroll is paced in **columns per second**, not columns per frame.
    Shifting a fixed one column per frame — which this did originally — makes
    the time axis mean whatever the current frame rate happens to be: measured
    at a 4x spread, with one second of audio occupying 30 columns at 30 fps
    against 120 at 120 fps. That also breaks the promise the settings panel
    makes about frame rate ("the motion is timed in seconds, so this changes
    smoothness only"), and it interacts badly with the adaptive pacer in
    widget.py, which retimes fps by +/-6 at runtime: the waterfall visibly
    slowed down and sped back up as the pacer moved, with nothing in the audio
    changing. The fractional remainder is carried in scratch rather than
    rounded away, so a rate that isn't a whole number of columns per frame
    still averages out exactly instead of drifting.
    """
    w, h = ctx.w, ctx.h
    if w < 4 or h < 3:
        return empty(w, h)

    hist = ctx.scratch("spectro", lambda: np.zeros((h, w), dtype=np.float32))
    acc = ctx.scratch("spectro_acc", lambda: {"v": 0.0})

    column = resample_bands(ctx.bands, h)[::-1]     # low frequencies at the bottom

    acc["v"] += SPECTRO_COLS_PER_SEC * max(ctx.dt, 0.0)
    step = int(min(w, acc["v"]))
    if step:
        acc["v"] -= step
        hist[:, :-step] = hist[:, step:]
        # every column this frame covers gets the same reading — the analyser
        # only published one, so inventing detail between them would be a lie
        hist[:, -step:] = column[:, None]
    else:
        # Above the scroll rate — several frames can share one column. Peak-hold
        # into the column still being written rather than skipping the reading:
        # dropping it would lose any transient that happened to land on one of
        # those frames, which on a spectrogram is the one thing you were looking
        # for. Same peak-preserving reasoning as ECG's decimation.
        np.maximum(hist[:, -1], column, out=hist[:, -1])

    lut = np.array([ord(c) for c in SHADES], dtype=np.int32)
    step = np.clip((hist * (len(SHADES) - 1) * 1.35).astype(np.int32), 0, len(SHADES) - 1)
    codes = lut[step]
    cidx = ctx.ramp(hist)
    return codes, cidx


def _plasma(ctx: Ctx, cells: str):
    """Drawn with ``▀`` so each cell carries two colours — foreground for the
    top half, background for the bottom. That doubles the vertical resolution
    for free, which matters a lot for a smooth gradient field.

    ``cells="octant"`` samples the same field on the 4x2 subcell grid instead
    of the 1x2 half-row one and draws it with octant glyphs: four times the
    vertical detail and twice the horizontal, at the same two colours a cell.
    That means evaluating the field over four times as many points, which is
    only affordable because it now runs in float32 — 9.58 ms against 4.09 at
    400x100 for the same arithmetic, and the half-block path went 2.03 to 0.65
    on the same change.

    Because the field is smooth everywhere, almost every cell takes
    :func:`render.shade_cells`' flat path and comes out as a plain half-block
    with two exact colours. What the octant grid buys here is not the glyph but
    the *sampling*: eight points a cell averaged down to two colours instead of
    two points taken raw, so the colours are right even where the glyph has
    nothing to say. The cells that do hold an edge — the ridges, at a small
    terminal — get the mask.

    That trick is also the expensive part. Two colours per cell means
    make_strips run-length-encodes on the (fg, bg) *pair*, not a single index,
    and this field crossed enough ramp buckets per column that most rows were
    forty-odd tiny segments instead of a handful of long ones — 10-11 ms of
    make_strips alone at 400x100, the actual bottleneck, confirmed by profiling
    that isolated it from this function's own ~2 ms. The fix isn't giving up
    the two-colour trick; it's that the swirl frequencies (6/5/4/14 cycles
    across the field) were higher than the terminal's own resolution could
    usefully show — a colour changing several times a *column* reads as noise,
    not gradient, on top of costing real time to encode. Halved, it reads as
    the calmer, larger-scale drift the blurb already promised ("solid colour
    field") and make_strips drops to ~5 ms, which is the whole win.
    """
    w, h = ctx.w, ctx.h
    if w < 2 or h < 2:
        return empty(w, h)

    octant = cells == "octant"
    rows2 = h * subcell_rows() if octant else h * 2
    cols = w * 2 if octant else w

    # float32 throughout, the same deal Flame's and Vinyl's docstrings make: a
    # sine over a grid this size is memory-bound, and float64 moves twice the
    # bytes for a value that ends up quantised to 64 ramp steps.
    def geo():
        y = np.arange(rows2, dtype=np.float32)[:, None] / np.float32(max(1, rows2 - 1))
        x = np.arange(cols, dtype=np.float32)[None, :] / np.float32(max(1, cols - 1))
        # The ripple's distance-from-centre is pure geometry: no time in it and
        # no audio, so it is the same array every frame for a given size. It
        # was being rebuilt each one — two squares, a broadcast add, a square
        # root and a scale over the whole grid — which at 400x100 is five
        # traversals of 80,000 cells to arrive at the number it arrived at last
        # frame. The other three terms are genuinely per-frame; this one is
        # furniture. Scaled by 7 here too, so the per-frame form is one
        # subtract and one sine.
        r = np.sqrt((x - np.float32(0.5)) ** 2 * np.float32(2.2)
                    + (y - np.float32(0.5)) ** 2) * np.float32(7.0)
        return y, x, r.astype(np.float32)

    y, x, ripple_r = ctx.scratch("plasma", geo)

    t = np.float32(ctx.t)
    # fractions rather than fixed indices — this mode has no business knowing
    # how many bands the analyser happens to produce, and docs/plugins.md tells
    # plugin authors exactly this
    lows = np.float32(ctx.range(0.00, 0.20))
    mids = np.float32(ctx.range(0.25, 0.62))
    highs = np.float32(ctx.range(0.70, 1.00))

    v = (
        np.sin((x * np.float32(3.0) + t * np.float32(0.7)) * (np.float32(1.0) + lows * np.float32(1.4)))
        + np.sin((y * np.float32(2.5) - t * np.float32(0.5)) * (np.float32(1.0) + mids * np.float32(1.2)))
        + np.sin(((x + y) * np.float32(2.0) + t * np.float32(0.9)))
        + np.sin(ripple_r - t * np.float32(2.2) * (np.float32(0.4) + highs * np.float32(2.0)))
    )
    field = (v + np.float32(4.0)) * np.float32(0.125)
    field = np.clip(field * np.float32(0.35 + ctx.energy * 1.5), 0.0, 1.0)

    if octant:
        # Shaded, not masked. This field is smooth everywhere, so there is no
        # shape in a cell to cut — thresholding it against the cell's own
        # extremes turns a gradient into texture, which is how the first
        # version of this variant came out looking *more* pixelated than the
        # mode it varies. shade_cells leaves a cell with no edge in it alone.
        return shade_cells(field)

    idx = ctx.ramp(field)
    codes = np.full((h, w), _UPPER_HALF, dtype=np.int32)
    return codes, idx[0::2], idx[1::2]


@mode("Plasma", group="fields",
      blurb="solid colour field, warped by the spectrum")
def plasma(ctx: Ctx):
    return _plasma(ctx, "half")


@mode("Plasma (o)", hidden=True, after="Plasma", group="fields",
      blurb="the same field at eight samples a cell instead of two — needs Unicode 16 octants")
def plasma_fine(ctx: Ctx):
    return _plasma(ctx, "octant")
