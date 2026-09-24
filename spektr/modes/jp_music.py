"""JP Sequencer: a drum machine's step grid, written by the drums it hears.

A drum machine's panel is sixteen pads a row and a light running across them
once a bar. This is that panel with the roles reversed: the song plays the
drums, and the grid writes down where each hit landed -- kick, snare and hat
on their own rows, each at the sixteenth of the bar it fell on. What is left
on screen is the bar's actual drum pattern, and a loop that repeats draws the
same pattern again on top of its own echo.

It stands on two pieces of the 0.6.0 analysis: ``ctx.drums`` says which row a
hit belongs on, and the bar says which step it is. It keeps the family's ink
weights from :mod:`spektr.modes.jp` -- a lit pad, a remembered pad a weight
lighter, an unlit one a dot -- and nothing of the ladder.

And it keeps the family's honesty about what it does not know. The first step
is marked as one only when ``bar_confidence`` says the bar is known. With a
beat but no bar the playhead counts four beats from wherever it started, and
with no beat at all it runs on its own slow time with its lamps dimmed.
"""
from __future__ import annotations

import numpy as np

from ..audio.drums import named
from ..render import SPACE
from . import Ctx, empty, mode
from .jp import _LED, _OFF, _PEAK, _TRAIL, ladder_colours, recede_index

#: Steps across a bar: sixteenths of four beats.
_STEPS = 16

#: The rows, top to bottom, and where each sits on the theme's ramp: the hats
#: hot at the top, the kick deep at the bottom, the way a mixer strip reads.
_ROWS = (("hat", 0.95), ("snare", 0.65), ("kick", 0.35))

#: The part of the spectrum each row reads between hits, same order as
#: :data:`_ROWS`, as shares of the band range: the hats' top end, the
#: snare's middle, the kick's bottom.
_ROW_BANDS = ((0.65, 1.0), (0.25, 0.6), (0.0, 0.15))

#: Below this a row's level draws nothing under the playhead.
_LEVEL_FLOOR = 0.08

#: How sure the bar tracker has to be before step one is called one.
_SURE = 0.4

#: One bar of the playhead when there is no tempo at all, in seconds.
_IDLE_BAR_S = 2.0

#: How long a pad flashes after it is written, in seconds. A hard on and off
#: rather than a fade: a fade smeared the hit across the frames after it, and
#: the grid changed as much between the beats as on them.
_FLASH_S = 0.06


def _playhead(ctx: Ctx, st: dict) -> tuple[float, bool, bool]:
    """Where the playhead is across the bar, 0..1; whether that is the bar
    (step one is known); and whether there is a beat at all."""
    if ctx.bar_confidence >= _SURE and ctx.tempo_bpm > 0.0:
        return float(ctx.bar_phase) % 1.0, True, True
    if ctx.tempo_bpm > 0.0:
        # A beat but no bar: count four beats from wherever the count began,
        # and do not call any of them one.
        if ctx.beat_phase < st["last_beat"] - 0.5:
            st["count"] = (st["count"] + 1) % 4
        st["last_beat"] = ctx.beat_phase
        return (st["count"] + float(ctx.beat_phase)) / 4.0, False, True
    st["idle"] = (st["idle"] + max(ctx.dt, 0.0) / _IDLE_BAR_S) % 1.0
    return st["idle"], False, False


@mode("JP Sequencer", group="jp",
      blurb="a drum machine's step grid, written by the song: kick, snare and hat where each hit landed")
def jp_sequencer(ctx: Ctx):
    """Sixteen steps by three drums, and a light running across once a bar.

    Every hit is written into the grid: the row is the drum ``ctx.drums``
    names it as, the column is the sixteenth of the bar the playhead is on,
    and the pad flashes as it is written. When the bar comes round, what was
    written becomes the last bar -- drawn a weight lighter underneath -- and
    the new bar writes over it, so a steady beat lands pad on pad on its own
    echo, and a fill is the pads that land somewhere new.

    The playhead is a lamp over its step, and only that: marking its column
    down through the grid moved more of the screen between the beats than the
    hits did on them. Step one's lamp keeps a mark when the bar is known;
    without a bar there is no step one, and without a beat the lamps are dim
    and the playhead runs on its own time.
    """
    rows, w = ctx.h, ctx.w
    gap = 1
    pad_w = (w - gap * (_STEPS - 1)) // _STEPS
    if rows < 7 or pad_w < 1:
        return empty(w, rows)

    st = ctx.scratch("jp_seq", lambda: {
        "cur": np.zeros((len(_ROWS), _STEPS), np.float32),
        "prev": np.zeros((len(_ROWS), _STEPS), np.float32),
        "flash": np.zeros((len(_ROWS), _STEPS), np.float32),
        "last": 0.0, "count": 0, "last_beat": 0.0, "idle": 0.0,
    })
    phase, known, beating = _playhead(ctx, st)
    if phase < st["last"] - 0.5:
        # A new bar: what was written is now the last bar.
        st["prev"] = st["cur"].copy()
        st["cur"][:] = 0.0
    st["last"] = phase
    step = int(phase * _STEPS) % _STEPS
    st["flash"] -= np.float32(max(ctx.dt, 0.0))

    if ctx.onsets:
        strength = 0.55 + 0.45 * min(1.0, float(ctx.onset_strength))
        # The drums this hit is (a snare with plenty of hat in it is both);
        # see spektr.audio.drums.named for why this is relative.
        hit = named(ctx.drums or {})
        for r, (name, _) in enumerate(_ROWS):
            if name in hit:
                st["cur"][r, step] = max(st["cur"][r, step], strength)
                st["flash"][r, step] = _FLASH_S

    levels = [float(ctx.range(lo, hi)) for lo, hi in _ROW_BANDS]

    # ── layout: a lamp row, a blank row, then three rows of pads ─────────────
    codes = np.full((rows, w), SPACE, dtype=np.int32)
    cidx = np.full((rows, w), recede_index(ctx.palette), dtype=np.int32)
    ramp = ladder_colours(ctx, 64)
    band = (rows - 2 - (len(_ROWS) - 1)) // len(_ROWS)
    x0 = (w - (pad_w * _STEPS + gap * (_STEPS - 1))) // 2

    for s in range(_STEPS):
        left = x0 + s * (pad_w + gap)
        mid = left + pad_w // 2
        # the lamp over each step
        if s == step:
            codes[0, mid] = _LED if beating else _TRAIL
            cidx[0, mid] = ramp[0]
        elif s == 0 and known:
            codes[0, mid] = _PEAK
            cidx[0, mid] = ramp[8]
        else:
            codes[0, mid] = _OFF
        for r, (_, heat) in enumerate(_ROWS):
            top = 2 + r * (band + 1)
            cells = (slice(top, top + band), slice(left, left + pad_w))
            colour = int(np.asarray(ctx.ramp(np.float32(heat))))
            cur, prev = float(st["cur"][r, s]), float(st["prev"][r, s])
            if cur > 0.0:
                codes[cells] = _LED
                lit = heat + (0.4 if st["flash"][r, s] > 0.0 else 0.0)
                cidx[cells] = int(np.asarray(ctx.ramp(np.float32(min(1.0, lit)))))
            elif s == step and levels[r] > _LEVEL_FLOOR:
                # under the playhead, an empty pad shows its row's part of the
                # spectrum: the grid breathes with the music between the hits,
                # and music with no drums in it still moves it.
                lv = float(levels[r])
                codes[top + band // 2, left:left + pad_w] = _PEAK if lv > 0.5 else _TRAIL
                # dim to bright across the pad, up to how loud the row is: a
                # small meter under the playhead rather than a flat block
                cidx[top + band // 2, left:left + pad_w] = np.asarray(ctx.ramp(
                    np.linspace(0.15, min(1.0, 0.2 + 0.8 * lv), pad_w, dtype=np.float32)))
            elif prev > 0.0:
                # last bar's hit, not yet written over: a weight lighter
                codes[top + band - 1, left:left + pad_w] = _TRAIL
                cidx[top + band - 1, left:left + pad_w] = colour
            else:
                codes[top + band // 2, mid] = _OFF
    return codes, cidx
