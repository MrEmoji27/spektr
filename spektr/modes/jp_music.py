"""JP Sequencer: the LED ladder paged by the bar.

The ladder is :mod:`spektr.modes.jp`'s; what this adds is the bar from the
0.6.0 analysis. Four pages, one per beat of the bar: the live one plays, the
others hold what their beat looked like. Only when ``bar_confidence`` says the
bar is known is the first page called one.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from ..render import SPACE
from . import Ctx, band_columns, mode
from .jp import (
    _LED,
    _OFF,
    _PEAK,
    _QUIET,
    _TRAIL,
    bar_panel,
    ladder_colours,
    recede_index,
)

# ── JP Sequencer ─────────────────────────────────────────────────────────────

_PAGES = 4

#: How long a held page takes to fade, as a time constant. About two bars at
#: 120 BPM: a page is still readable when its beat comes round again.
_HOLD_S = 2.0

#: How sure the bar tracker has to be before the first page is called one.
_SURE = 0.4


def _page_step(ctx: Ctx, st: dict) -> tuple[int, bool]:
    """Which page is live, and whether the bar is known.

    Known: the bar tracker's beat. Not known but a tempo: the pages step on
    every beat, counted from wherever they happened to start. Neither: they
    step on the beats the analyser hands the modes.
    """
    if ctx.beat_in_bar is not None and ctx.bar_confidence >= _SURE:
        return int(ctx.beat_in_bar) % _PAGES, True
    if ctx.tempo_bpm > 0.0:
        wrapped = ctx.beat_phase < st["phase"] - 0.5
        st["phase"] = ctx.beat_phase
        if wrapped:
            st["count"] += 1
    elif ctx.onsets:
        st["count"] += 1
    return st["count"] % _PAGES, False


@mode("JP Sequencer", group="jp",
      blurb="the meter paged by the bar — four pages, one live, three holding their beat")
def jp_sequencer(ctx: Ctx):
    """Four small panels, one for each beat of the bar.

    The panel for the beat that is playing is live. The other three hold what
    the music looked like on their beat, drawn a weight lighter and fading,
    so the screen shows a whole bar at once -- a kick-heavy one and a
    snare-heavy three sit side by side, which is what ``Spectro``'s scrolling
    history shows as a smear and this shows as a pattern.

    A row of step lamps above the pages says which beat it is. When the bar
    tracker knows where the bar starts, the first page is the downbeat and its
    lamp keeps a mark when it is not live; when it does not, the pages still
    step on every beat, but no page is called one.
    """
    rows, w = ctx.h, ctx.w
    gap = 1
    page_w = (w - gap * (_PAGES - 1)) // _PAGES
    if rows < 6 or page_w < 3:
        from . import empty

        return empty(w, rows)
    n = max(2, min(ctx.n_display, page_w))
    st = ctx.scratch("jp_seq", lambda: {
        "snap": np.zeros((_PAGES, n), dtype=np.float32),
        "phase": 0.0,
        "count": 0,
    })
    if st["snap"].shape != (_PAGES, n):
        st["snap"] = np.zeros((_PAGES, n), dtype=np.float32)

    live, known = _page_step(ctx, st)
    now = ctx.display_bands(n).astype(np.float32)
    st["snap"] *= np.float32(np.exp(-max(ctx.dt, 0.0) / _HOLD_S))
    st["snap"][live] = now

    codes = np.full((rows, w), SPACE, dtype=np.int32)
    cidx = np.full((rows, w), recede_index(ctx.palette), dtype=np.int32)
    panel_rows = rows - 2
    sub = dataclasses.replace(ctx, w=page_w)
    col_band, active = band_columns(page_w, n)
    for p in range(_PAGES):
        x0 = p * (page_w + gap)
        levels = np.where(active, st["snap"][p][col_band], 0.0)
        pc, px = bar_panel(sub, panel_rows, levels, active)
        if p != live:
            # A held page is a record, not a reading: one weight lighter.
            pc = np.where(pc == _LED, _TRAIL, pc)
        codes[2:, x0:x0 + page_w] = pc
        cidx[2:, x0:x0 + page_w] = px

        # The step lamp, centred over its page.
        lamp = x0 + page_w // 2
        top = ladder_colours(ctx, rows)[0]
        if p == live:
            codes[0, lamp], cidx[0, lamp] = _LED, top
        elif known and p == 0:
            codes[0, lamp], cidx[0, lamp] = _PEAK, top
        else:
            codes[0, lamp] = _OFF
    if ctx.silent or float(now.max()) < _QUIET:
        codes[0] = np.where(codes[0] == _LED, _OFF, codes[0])
    return codes, cidx


