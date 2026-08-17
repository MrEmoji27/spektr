"""Waveform modes, drawn with braille sub-cells.

``Wave`` was one of the two modes still written as nested Python loops over
every braille dot — 15 ms per frame at 200x50, against ~1 ms for the numpy
modes. The line-fill is expressed here as a single broadcast comparison
instead, which is the same maths with none of the interpreter overhead.
"""

from __future__ import annotations

import math

import numpy as np

from ..render import cell_max, pack_braille, pack_octant_bits
from . import Ctx, empty, mode

#: How many seconds of audio ``ctx.wave`` spans.
#:
#: The analyser downsamples its mid FFT window — 4096 samples, whatever the
#: rate — to WAVE_POINTS, so this is 85 ms at 48 kHz and 93 ms at 44.1 kHz.
#: ECG needs it to work out how much of the buffer is genuinely new since the
#: last column it committed.
#:
#: It read 0.043 for a long time, left over from the 2048-sample FFT the
#: analyser used before the split into bass and mid windows. Exactly half, so
#: every column reduced twice the audio it should have and the trace came out
#: flatter than the signal. Not exact across rates, and it does not need to
#: be: the result is clipped to the buffer either way.
_WAVE_SPAN_S = 0.085


def _stereo_mix(stereo: np.ndarray) -> np.ndarray | None:
    """Mono mix of a stereo buffer; ``None`` when the buffer is empty.

    The two channels of a ``(N, 2)`` buffer average to one trace; a 1-D buffer
    is its own trace — a mono signal handed to a stereo slot is rendered rather
    than dropped, and ``None`` sends callers to their mono fallback.
    """
    if stereo.ndim == 1:
        return stereo if stereo.size else None
    if stereo.size:
        return stereo.mean(axis=1)
    return None


def _stereo_channels(stereo: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The two channels of a stereo buffer; ``None`` when the buffer is empty.

    A 1-D buffer becomes two identical channels, which is the same mono
    treatment :func:`_stereo_mix` gives the mean-based scopes — one trace
    renders as its own L/R pair instead of crashing the axis indexing.
    """
    if stereo.ndim == 1:
        return (stereo, stereo) if stereo.size else None
    if stereo.size:
        return stereo[:, 0], stereo[:, 1]
    return None


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
    return dots, heat * dots


@mode("Wave", group="scope", blurb="smoothed waveform")
def wave(ctx: Ctx):
    dots, heat = _trace_dots(ctx.wave, ctx.dot_rows, ctx.dot_cols)
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(heat)) if heat is not None else np.zeros_like(codes)
    return codes, cidx


def _scope(ctx: Ctx, octant: bool):
    """A real oscilloscope, with edge triggering.

    Two modes share this body. ``octant=False`` is the original: the trace
    is a chain of separated braille dots, which is exactly what a trace
    should not read as — a fast edge looks like stipple between two columns.
    ``octant=True`` packs the identical dot set into Unicode 16 octant
    glyphs, block mosaic at the same 4x2 resolution, so the fill between
    consecutive sample points becomes a continuous stroke. Nothing else
    differs — same trigger, same window, same colour. See
    :func:`render.pack_octant_bits`.

    The trace is a shape against empty space, so the foreground is the only
    colour either version sets: an octant cell is opaque once it is given a
    background index, and the space around the waveform has to stay the
    terminal's own.

    Plain waveform display slides sideways because each frame starts at an
    arbitrary phase. Locking onto a rising zero crossing near the middle of the
    buffer makes a steady tone sit still, which is the whole point of a scope.
    """
    mix = _stereo_mix(ctx.stereo)
    raw = mix if mix is not None else ctx.wave
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
    codes = pack_octant_bits(dots) if octant else pack_braille(dots)
    cidx = ctx.ramp(cell_max(heat))
    return codes, cidx


@mode("Scope", group="scope",
      blurb="trigger-synced oscilloscope — the trace holds still")
def scope(ctx: Ctx):
    return _scope(ctx, octant=False)


@mode("Scope (o)", hidden=True, after="Scope", group="scope",
      blurb="the same trace as a continuous stroke — needs a terminal that draws Unicode 16 octants")
def scope_fine(ctx: Ctx):
    """Scope on octant cells.

    Separate mode rather than a switch on the original, for the same reason
    Kaleidoscope (o) is: octants are Unicode 16 and an older terminal or
    font draws a grid of tofu, which is a thing to opt into rather than to
    discover when a mode you liked stops working.
    """
    return _scope(ctx, octant=True)


def _ecg(ctx: Ctx, octant: bool):
    """The waveform as history rather than as a snapshot.

    Two modes share this body. ``octant=False`` is the original: the trace
    is a chain of separated braille dots. ``octant=True`` packs the
    identical dot set into Unicode 16 octant glyphs, block mosaic at the
    same 4x2 resolution, so the filled band between each column's minimum
    and maximum reads as a continuous stroke. Nothing else differs — same
    scroll clock, same peak-preserving decimation, same age colouring. See
    :func:`render.pack_octant_bits`.

    Like Scope, the trace is a shape against empty space, so the foreground
    is the only colour either version sets — a background index would paint
    the whole cell opaque and the space around the trace has to stay the
    terminal's own.

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

    # The history and the scroll accumulator are the mode's mutable state,
    # and each version keeps its own: sharing one would make the two modes
    # advance each other's clock, so every frame would come out "different"
    # for no reason.
    hist = ctx.scratch(
        "ecg" if not octant else "ecg_fine", lambda: np.zeros((2, dc), dtype=np.float32)
    )

    mix = _stereo_mix(ctx.stereo)
    src = mix if mix is not None else ctx.wave
    src = np.asarray(src, dtype=np.float32)

    # ~55% of the width per second, so a full sweep takes just under two
    # seconds at any terminal size.
    #
    # The fractional remainder is carried in scratch rather than rounded away
    # each frame. Rounding made the scroll rate wrong in two different
    # directions depending on how ``dc * 0.55 * dt`` landed: at a narrow
    # terminal it quantised to alternating 1- and 2-column steps, so the trace
    # advanced at two visibly different speeds from one frame to the next, and
    # the old ``max(1, ...)`` floor forced a whole column even when the frame
    # was worth a fraction of one — measured at +74% too fast at 60 columns and
    # 120 fps, against -13% too slow at 60 fps. Accumulating instead lets a
    # frame legitimately advance zero columns, which is what keeps the average
    # exact at any width and frame rate.
    acc = ctx.scratch(
        "ecg_acc" if not octant else "ecg_acc_fine",
        lambda: {"v": 0.0, "elapsed": 0.0, "filled": 0},
    )
    acc["v"] += dc * 0.55 * max(ctx.dt, 0.0)
    acc["elapsed"] += max(ctx.dt, 0.0)
    step = int(min(dc, acc["v"]))
    acc["v"] -= step

    if step:
        # Only the newest slice of the buffer is new: the analyser publishes an
        # 85 ms window and the render loop reads it every ~17 ms, so consuming
        # the whole thing every frame would smear each column across the same
        # five frames of audio and flatten the trace. Take the tail worth the
        # time actually elapsed since the last committed column — not this one
        # frame's dt, which would silently drop the audio from any frame that
        # advanced zero columns.
        span = acc["elapsed"]
        acc["elapsed"] = 0.0
        take = int(np.clip(src.size * (span / _WAVE_SPAN_S), 8, src.size)) if src.size else 0
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
        acc["filled"] = min(dc, acc["filled"] + step)

    # Filled between the per-column minimum and maximum rather than drawn as a
    # single line. That is how every audio editor draws a waveform, and it is
    # the only honest way to show a signal whose period is shorter than one
    # column — a line would alias into a staircase.
    centre = (dr - 1) / 2.0
    top = np.clip(np.rint(centre - np.clip(hist[0] * 0.95, -1, 1) * centre), 0, dr - 1)
    bot = np.clip(np.rint(centre - np.clip(hist[1] * 0.95, -1, 1) * centre), 0, dr - 1)
    rows = np.arange(dr, dtype=np.float64)[:, None]
    dots = (rows >= top[None, :]) & (rows <= bot[None, :])

    # Columns the trace has not reached yet draw nothing at all.
    #
    # The buffer starts as zeros and zero is the centre line, so an unwritten
    # column drew a dot exactly halfway up — and a screen's worth of them drew
    # a hard horizontal rule across the middle of the mode. It reads as a
    # feature of the display rather than as an absence of data, and it is
    # visible every time the mode is selected, every resize, and for the whole
    # of any silence, which is precisely when there is nothing else to look at.
    filled = int(acc["filled"])
    if filled < dc:
        dots[:, : dc - filled] = False

    # colour by age: the leading edge is hot and the tail cools behind it,
    # which is what makes the direction of travel readable
    age = (np.arange(dc, dtype=np.float64) / max(1, dc - 1)) ** 2
    heat = (0.18 + 0.82 * age)[None, :] * dots
    codes = pack_octant_bits(dots) if octant else pack_braille(dots)
    cidx = ctx.ramp(cell_max(heat))
    return codes, cidx


@mode("ECG", group="scope",
      blurb="scrolling trace, like a heart monitor")
def ecg(ctx: Ctx):
    return _ecg(ctx, octant=False)


@mode("ECG (o)", hidden=True, after="ECG", group="scope",
      blurb="the same scrolling trace, drawn solid — needs a terminal that draws Unicode 16 octants")
def ecg_fine(ctx: Ctx):
    """ECG on octant cells.

    Separate mode rather than a switch on the original, for the same reason
    Kaleidoscope (o) is: octants are Unicode 16 and an older terminal or
    font draws a grid of tofu, which is a thing to opt into rather than to
    discover when a mode you liked stops working. The two versions also keep
    separate scroll buffers in scratch, so switching between them never
    advances the other's history.
    """
    return _ecg(ctx, octant=True)


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


@mode("Helix", group="stereo", blurb="two strands rotating around a shared axis, split by true L/R phase")
def helix(ctx: Ctx):
    """A corkscrew, not a flat trace.

    ``Gonio`` plots L against R as a static Lissajous shape and ``Strings``
    plucks one line per band; this is the only stereo mode with actual
    rotation — two sine strands spinning around a shared axis, one per
    channel, with a pseudo-3D depth cue (the strand curling away dims)
    instead of a flat line. The strands' phase offset is a real measurement,
    not a fixed 90°: it's the Pearson correlation between L and R turned into
    an angle — 0 when the channels are identical, which correctly collapses
    the two strands onto one, through pi when they're fully inverted, which
    is as far apart as the strands ever swing.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 10 or dc < 20:
        return empty(ctx.w, ctx.h)

    ch = _stereo_channels(ctx.stereo)
    if ch is not None:
        left = np.clip(ch[0], -1.0, 1.0).astype(np.float64)
        right = np.clip(ch[1], -1.0, 1.0).astype(np.float64)
        energy = float((left * left + right * right).mean()) * 0.5
        corr = float((left * right).mean())
        c = 0.0 if energy < 1e-8 else float(np.clip(corr / energy, -1.0, 1.0))
        measured = math.acos(c)
    else:
        measured = 0.0

    st = ctx.scratch("helix", lambda: {"phase": 0.0})
    st["phase"] += (measured - st["phase"]) * min(1.0, ctx.dt / 0.25)

    amp_l = 0.15 + 0.75 * float(ctx.bands_l.mean())
    amp_r = 0.15 + 0.75 * float(ctx.bands_r.mean())

    treble = ctx.range(0.6, 1.0)
    pitch = (2.6 + treble * 1.4) * 2 * math.pi / dc
    spin = ctx.t * 2.1

    cols = np.arange(dc, dtype=np.float64)
    theta = cols * pitch + spin
    centre = (dr - 1) / 2.0
    max_amp = centre * 0.92

    field = np.zeros((dr, dc), dtype=np.float64)
    rows = np.arange(dr, dtype=np.int32)[:, None]

    for theta_off, amp, base_bright in ((0.0, amp_l, 1.0), (st["phase"], amp_r, 0.85)):
        t = theta + theta_off
        depth = (np.cos(t) + 1.0) * 0.5      # 0 = curling away, 1 = facing the viewer
        y = centre - np.sin(t) * amp * max_amp
        yi = np.clip(np.rint(y).astype(np.int32), 0, dr - 1)
        prev = np.empty_like(yi)
        prev[0] = yi[0]
        prev[1:] = yi[:-1]
        lo = np.minimum(yi, prev)[None, :]
        hi = np.maximum(yi, prev)[None, :]
        on = (rows >= lo) & (rows <= hi)
        bright = (0.25 + 0.75 * depth) * base_bright
        np.maximum(field, np.where(on, bright[None, :], 0.0), out=field)

    dots = field > 0.05
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(field))
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

    ch = _stereo_channels(ctx.stereo)
    if ch is not None and not ctx.silent:
        left = np.clip(ch[0], -1.0, 1.0)
        right = np.clip(ch[1], -1.0, 1.0)
        x = (right - left) * 0.7071
        y = (right + left) * 0.7071

        cx, cy = (dc - 1) / 2.0, (dr - 1) / 2.0
        # One radius for both axes, so the display is actually circular.
        #
        # The comment here used to claim a squash and there was none: both axes
        # were scaled by their own half-extent, which draws the trace into
        # whatever rectangle the terminal happens to be. Measured, a fully
        # uncorrelated pair came out 684 x 344 dots at 400x100 — a 2:1 ellipse.
        #
        # That is not cosmetic on this mode. A goniometer is read by its
        # geometry: a circle means uncorrelated, a 45-degree line means mono,
        # a vertical one means out of phase. Stretched 2:1, mono drew at 27
        # degrees and every phase judgement off the display was wrong.
        #
        # A braille dot is square — cells are about twice as tall as wide and
        # hold 2x4 dots — so equal radii in dots really is a circle on screen.
        # ``polar_grid`` normalises the same way for the same reason.
        r = min(cx, cy) * 0.95
        px = np.clip(np.rint(cx + x * r).astype(np.int32), 0, dc - 1)
        py = np.clip(np.rint(cy - y * r).astype(np.int32), 0, dr - 1)
        np.add.at(field, (py, px), 0.6)

    np.clip(field, 0.0, 1.0, out=field)
    dots = field > 0.06
    codes = pack_braille(dots)
    cidx = ctx.ramp(np.sqrt(cell_max(field)))
    return codes, cidx


