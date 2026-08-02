"""Waveform modes, drawn with braille sub-cells.

``Wave`` was one of the two modes still written as nested Python loops over
every braille dot — 15 ms per frame at 200x50, against ~1 ms for the numpy
modes. The line-fill is expressed here as a single broadcast comparison
instead, which is the same maths with none of the interpreter overhead.
"""

from __future__ import annotations

import numpy as np

from ..render import cell_max, pack_braille
from . import Ctx, empty, mode


def _trace_dots(values: np.ndarray, dot_rows: int, dot_cols: int, gain: float = 1.0):
    """Sample a signal across the width and fill between consecutive points.

    Filling between ``y[x-1]`` and ``y[x]`` is what stops a fast waveform from
    breaking into disconnected specks — without it, a steep edge only lights
    one dot per column.
    """
    n = len(values)
    if n == 0:
        return np.zeros((dot_rows, dot_cols), dtype=bool), None

    pick = np.clip((np.arange(dot_cols) * n) // max(1, dot_cols), 0, n - 1)
    s = np.clip(values[pick] * gain, -1.0, 1.0)

    centre = (dot_rows - 1) / 2.0
    y = np.clip(np.rint(centre - s * centre).astype(np.int32), 0, dot_rows - 1)
    prev = np.empty_like(y)
    prev[0] = y[0]
    prev[1:] = y[:-1]

    lo = np.minimum(y, prev)[None, :]
    hi = np.maximum(y, prev)[None, :]
    rows = np.arange(dot_rows, dtype=np.int32)[:, None]
    dots = (rows >= lo) & (rows <= hi)

    heat = np.abs(rows - centre) / max(centre, 1.0)
    return dots, np.where(dots, heat, 0.0)


@mode("Wave", group="scope", blurb="smoothed waveform")
def wave(ctx: Ctx):
    dots, heat = _trace_dots(ctx.wave, ctx.dot_rows, ctx.dot_cols)
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(heat)) if heat is not None else np.zeros_like(codes)
    return codes, cidx


@mode("Scope", group="scope", blurb="trigger-synced oscilloscope — the trace holds still")
def scope(ctx: Ctx):
    """A real oscilloscope, with edge triggering.

    Plain waveform display slides sideways because each frame starts at an
    arbitrary phase. Locking onto a rising zero crossing near the middle of the
    buffer makes a steady tone sit still, which is the whole point of a scope.
    """
    raw = ctx.stereo.mean(axis=1) if ctx.stereo.size else ctx.wave
    n = len(raw)
    if n < 8:
        return empty(ctx.w, ctx.h)

    span = n // 2
    search = raw[:span]
    rising = np.flatnonzero((search[:-1] <= 0.0) & (search[1:] > 0.0))
    if rising.size:
        # the crossing nearest the middle of the search window is the most
        # stable choice frame to frame
        start = int(rising[np.argmin(np.abs(rising - span // 2))])
    else:
        start = 0

    window = raw[start : start + span]
    dots, heat = _trace_dots(window, ctx.dot_rows, ctx.dot_cols)
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(heat))
    return codes, cidx


@mode("ECG", group="scope", blurb="scrolling trace, like a heart monitor")
def ecg(ctx: Ctx):
    """The waveform as history rather than as a snapshot.

    ``Spectro`` scrolls the *spectrum*; nothing scrolled the raw signal. The
    buffer holds one value per dot column and shifts left by however many
    columns the elapsed time is worth, so the scroll speed is in columns per
    second and does not change with the frame rate.

    New columns are decimated peak-preserving — the largest-magnitude sample in
    each block rather than one sampled sample. A transient that lands between
    two sample points still shows up, which is the whole difference between a
    trace that looks like a heartbeat and one that looks like noise.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 8 or dc < 8:
        return empty(ctx.w, ctx.h)

    hist = ctx.scratch("ecg", lambda: np.zeros((2, dc), dtype=np.float32))

    src = ctx.stereo.mean(axis=1) if ctx.stereo.size else ctx.wave
    src = np.asarray(src, dtype=np.float32)

    # ~55% of the width per second, so a full sweep takes just under two
    # seconds at any terminal size
    step = int(min(dc, max(1, round(dc * 0.55 * ctx.dt))))

    # Only the newest slice of the buffer is new: the analyser publishes a
    # ~43 ms window and the render loop reads it every ~17 ms, so consuming the
    # whole thing every frame would smear each column across the same three
    # frames of audio and flatten the trace. Take the tail worth ``dt``.
    take = int(np.clip(src.size * (ctx.dt / 0.043), 8, src.size)) if src.size else 0
    seg = src[-take:] if take else src

    if seg.size >= step * 2:
        starts = (np.arange(step, dtype=np.int64) * seg.size) // step
        hi = np.maximum.reduceat(seg, starts)
        lo = np.minimum.reduceat(seg, starts)
    else:
        v = float(seg[-1]) if seg.size else 0.0
        hi = np.full(step, v, dtype=np.float32)
        lo = hi

    hist[:, :-step] = hist[:, step:]
    hist[0, -step:] = hi
    hist[1, -step:] = lo

    # Filled between the per-column minimum and maximum rather than drawn as a
    # single line. That is how every audio editor draws a waveform, and it is
    # the only honest way to show a signal whose period is shorter than one
    # column — a line would alias into a staircase.
    centre = (dr - 1) / 2.0
    top = np.clip(np.rint(centre - np.clip(hist[0] * 0.95, -1, 1) * centre), 0, dr - 1)
    bot = np.clip(np.rint(centre - np.clip(hist[1] * 0.95, -1, 1) * centre), 0, dr - 1)
    rows = np.arange(dr, dtype=np.float64)[:, None]
    dots = (rows >= top[None, :]) & (rows <= bot[None, :])

    # colour by age: the leading edge is hot and the tail cools behind it,
    # which is what makes the direction of travel readable
    age = (np.arange(dc, dtype=np.float64) / max(1, dc - 1)) ** 2
    heat = np.where(dots, (0.18 + 0.82 * age)[None, :], 0.0)
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(heat))
    return codes, cidx


@mode("Strings", group="scope", blurb="plucked strings, bowed by their own band")
def strings(ctx: Ctx):
    """One vertical string per band, standing-wave bowed from its rest line.

    Drawn by scattering one dot per row per string rather than masking the
    whole dot grid: the shape is a function of the row, so the work is
    ``rows x strings`` (a few thousand cells) instead of ``rows x cols``
    (a hundred thousand). Odd-numbered strings vibrate in their second mode,
    which keeps a wall of them from moving as one block.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 8 or dc < 12:
        return empty(ctx.w, ctx.h)

    n = int(min(16, max(3, ctx.w // 10)))
    lv = ctx.display_bands(n).astype(np.float64)

    y = np.arange(dr, dtype=np.float64) / max(1, dr - 1)
    harmonic = np.where(np.arange(n) % 2 == 1, 2.0, 1.0)
    # nodes at both ends, so the string is pinned top and bottom
    shape = np.sin(np.pi * y[:, None] * harmonic[None, :])

    spacing = dc / n
    rest = (np.arange(n) + 0.5) * spacing
    swing = np.sin(ctx.t * (7.0 + np.arange(n) * 1.7) + np.arange(n) * 0.9)
    amp = lv * spacing * 0.42 * swing

    disp = shape * amp[None, :]
    x = np.clip(np.rint(rest[None, :] + disp), 0, dc - 1).astype(np.int32)

    rows = np.repeat(np.arange(dr, dtype=np.int32)[:, None], n, axis=1)
    dots = np.zeros((dr, dc), dtype=bool)
    heat = np.zeros((dr, dc), dtype=np.float64)
    dots[rows, x] = True
    # brightest where the string is furthest from rest — the bow, effectively
    heat[rows, x] = np.clip(np.abs(disp) / max(spacing * 0.42, 1e-6), 0.06, 1.0)

    # a second dot on the leading side while a string is really swinging, so a
    # loud string reads as a thick blurred line rather than a hairline
    fat = np.abs(disp) > spacing * 0.16
    if fat.any():
        x2 = np.clip(x + np.sign(disp).astype(np.int32), 0, dc - 1)
        dots[rows[fat], x2[fat]] = True
        heat[rows[fat], x2[fat]] = np.maximum(
            heat[rows[fat], x2[fat]], np.abs(disp[fat]) / max(spacing * 0.42, 1e-6) * 0.7
        )

    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(np.clip(heat, 0.0, 1.0)))
    return codes, cidx


@mode("Gonio", group="stereo", blurb="stereo phase scope with phosphor trail")
def goniometer(ctx: Ctx):
    """Lissajous of L against R, rotated 45° so mono collapses to a vertical
    line and out-of-phase content spreads horizontally.

    Keeps a decaying intensity field between frames, which is what gives a real
    phosphor scope its look — and is only possible because the whole grid is a
    numpy array we can multiply in one go.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    # scratch is keyed on (w, h) and the dot grid is derived from those, so the
    # buffer can never be the wrong shape — no defensive re-check needed
    field = ctx.scratch("gonio", lambda: np.zeros((dr, dc), dtype=np.float32))

    # decay is dt-correct, so the trail length doesn't change with frame rate
    field *= float(np.exp(-ctx.dt / 0.14))

    pts = ctx.stereo
    if pts.size and not ctx.silent:
        left = np.clip(pts[:, 0], -1.0, 1.0)
        right = np.clip(pts[:, 1], -1.0, 1.0)
        x = (right - left) * 0.7071
        y = (right + left) * 0.7071

        cx, cy = (dc - 1) / 2.0, (dr - 1) / 2.0
        # squash x so the display stays circular despite non-square cells
        px = np.clip(np.rint(cx + x * cx * 0.95).astype(np.int32), 0, dc - 1)
        py = np.clip(np.rint(cy - y * cy * 0.95).astype(np.int32), 0, dr - 1)
        np.add.at(field, (py, px), 0.6)

    np.clip(field, 0.0, 1.0, out=field)
    dots = field > 0.06
    codes = pack_braille(dots)
    cidx = ctx.ramp(np.sqrt(cell_max(field)))
    return codes, cidx
