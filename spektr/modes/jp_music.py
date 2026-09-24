"""The JP panels that hear the music: which drum, which beat of the bar, which key.

The first three JP modes (:mod:`spektr.modes.jp`) are the LED ladder blended
with a mechanic from elsewhere in the app, and read the spectrum. These three
keep both rules -- the ladder first, the ink weights, a partner that already
exists -- and add a third, which is what 0.6.0 is for: each one stands on a
piece of the analysis that did not exist before it, and would be a clone
without it.

* **JP Kit** is the ladder × ``Pulse``'s shockwaves, fired by
  ``ctx.drums``. A kick throws a slow, heavy ring out of the bass corner, a
  snare a ring out of the middle, a hat a quick thin one out of the treble
  corner -- so the panel says *which* drum hit, which no other mode can.
* **JP Sequencer** is the ladder × ``Spectro``'s held history, paged by
  ``ctx.beat_in_bar``. Four pages, one per beat of the bar: the live one
  plays, the others hold what their beat looked like. Only when
  ``bar_confidence`` says the bar is known is the first page called one.
* **JP Key** is the ladder × ``Keys``, fed by ``ctx.chroma`` and ``ctx.key``.
  A Keys panel was cut from the family once because 32 FFT bands cannot name
  a note; chroma can. Twelve ladders, one per pitch class, in circle-of-fifths
  order so that a key is one unbroken block of seven -- and only that block is
  powered.

None of them guesses. A drum likelihood under a floor fires nothing, a bar
that is not known has no downbeat, and a key that is not settled lights no
tonic. The not-knowing is drawn, as a panel with less of it switched on.
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
    bulb_count,
    bulb_row_index,
    ladder_colours,
    peak_bulbs,
    recede_index,
)

# ── JP Kit ───────────────────────────────────────────────────────────────────

#: Where each drum's rings start, as ``(across, up)`` shares of the panel, and
#: how they move: speed in panel-widths a second, life in seconds, and ring
#: thickness as a share of the panel. A kick is slow and heavy and comes out
#: of the bass corner; a hat is quick and thin and comes out of the treble one.
_KIT = {
    "kick": ((0.06, 0.0), 0.95, 0.75, 0.13),
    "snare": ((0.50, 0.35), 1.30, 0.55, 0.09),
    "hat": ((0.94, 0.90), 1.90, 0.35, 0.055),
}

#: A drum below this likelihood fires no ring. The likelihoods are shares of
#: one hit, so a snare with some hat in it lights both -- but a trace of one is
#: not that drum.
_KIT_FLOOR = 0.35

#: Rings alive at once, across all three drums.
_KIT_RINGS = 12

#: Below this a ring is too faint to draw.
_KIT_VISIBLE = 0.18


@mode("JP Kit", group="jp",
      blurb="the meter with the drums in it — kick, snare and hat each throw their own ring")
def jp_kit(ctx: Ctx):
    """The ladder, with a ring thrown through it by every drum hit.

    ``Pulse`` throws a shockwave off the beat. This throws one per drum, from
    where that drum lives on the panel, and only through the unlit bulbs: the
    reading is never overwritten, the rings run through the dark part of the
    panel around it. Their shape carries what hit -- a kick's ring is thick
    and slow and rolls out of the bottom left, a hat's is a quick thin flicker
    at the top right -- so two drums landing together draw two rings.

    Which drum is ``ctx.drums``, and they are likelihoods: a hit is named by
    every drum it is at least :data:`_KIT_FLOOR` of. Music without drums fires
    nothing and the panel is a plain meter, which is the correct reading.
    """
    rows, w = ctx.h, ctx.w
    n = ctx.n_display
    col_band, active = band_columns(w, n)
    lv = ctx.display_bands(n)
    levels = np.where(active, lv[col_band], 0.0)
    peaks = np.where(active, ctx.display_peaks(n)[col_band], 0.0)

    st = ctx.scratch("jp_kit", lambda: {
        "born": np.full(_KIT_RINGS, -99.0),
        "amp": np.zeros(_KIT_RINGS),
        "drum": np.zeros(_KIT_RINGS, dtype=np.int32),
    })
    names = tuple(_KIT)
    if ctx.onsets:
        strength = float(np.clip(ctx.onset_strength, 0.0, 1.0))
        for k, name in enumerate(names):
            like = float(ctx.drums.get(name, 0.0))
            if like < _KIT_FLOOR:
                continue
            slot = int(np.argmin(st["born"]))
            st["born"][slot] = ctx.t
            st["amp"][slot] = like * (0.55 + 0.45 * strength)
            st["drum"][slot] = k

    codes, cidx = bar_panel(ctx, rows, levels, active)
    peak_bulbs(ctx, codes, cidx, peaks, levels, rows)

    nb = bulb_count(rows)
    rws = bulb_row_index(rows)
    if nb < 2 or w < 4:
        return codes, cidx
    # Panel coordinates of every bulb: across 0..1, up 0..1, with "up"
    # stretched by the panel's shape so a ring is round on screen. A bulb row
    # is two screen rows and a cell is about twice as tall as it is wide.
    across = (np.arange(w, dtype=np.float32) / max(1, w - 1))[None, :]
    up = (np.arange(nb, dtype=np.float32) / max(1, nb - 1))[:, None]
    tall = (rows * 2.0) / max(w, 1)

    ring = np.zeros((nb, w), dtype=np.float32)
    for i in np.flatnonzero(st["born"] > -98.0):
        (ox, oy), speed, life, thick = _KIT[names[st["drum"][i]]]
        age = ctx.t - st["born"][i]
        if not (0.0 <= age < life):
            continue
        fade = float(st["amp"][i]) * (1.0 - age / life)
        if fade < _KIT_VISIBLE:
            continue
        radius = speed * age
        d = np.hypot(across - ox, (up - oy) * tall)
        band = np.clip(1.0 - np.abs(d - radius) / thick, 0.0, 1.0)
        np.maximum(ring, band * np.float32(fade), out=ring)

    sub_c, sub_x = codes[rws], cidx[rws]
    show = (ring > _KIT_VISIBLE) & (sub_c == _OFF) & active[None, :]
    sub_c[show] = _LED
    sub_x[show] = np.broadcast_to(ladder_colours(ctx, rows)[rws][:, None],
                                  show.shape)[show]
    codes[rws], cidx[rws] = sub_c, sub_x
    return codes, cidx


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


# ── JP Key ───────────────────────────────────────────────────────────────────

#: The twelve pitch classes in circle-of-fifths order, as chroma indices
#: (C = 0). Adjacent columns are a fifth apart, so the seven notes of any
#: major or natural minor key are seven adjacent columns.
_FIFTHS = tuple((7 * k) % 12 for k in range(12))

#: A pitch class lighter than this share of the loudest lights nothing.
_KEY_FLOOR = 0.12


def _key_window(key: str | None) -> tuple[int | None, np.ndarray]:
    """The tonic's pitch class and which columns the key powers, in fifths
    order. None and every column when no key is known."""
    powered = np.ones(12, dtype=bool)
    if not key:
        return None, powered
    from ..audio.chroma import NOTES

    name, _, quality = key.partition(" ")
    if name not in NOTES:
        return None, powered
    tonic = NOTES.index(name)
    # A minor key has the notes of its relative major, three semitones up.
    major = (tonic + 3) % 12 if quality == "minor" else tonic
    at = _FIFTHS.index(major)
    # One fifth below the major tonic to five above: the seven notes.
    powered = np.zeros(12, dtype=bool)
    for k in range(-1, 6):
        powered[(at + k) % 12] = True
    return tonic, powered


@mode("JP Key", group="jp",
      blurb="twelve ladders, one per note in fifths — the key powers its seven, the tonic gets a lamp")
def jp_key(ctx: Ctx):
    """A ladder per pitch class, and the key drawn as which of them are on.

    The columns run round the circle of fifths rather than up the keyboard,
    because that is the order in which a key is contiguous: any major or
    minor key is seven neighbouring columns, and a change of key is that
    block moving along. Each column's level is how much of that note is
    sounding (``ctx.chroma``), scaled by how loud the music is overall so
    the panel goes dark when the music does.

    The key powers its seven columns: they carry the unlit lattice, the other
    five are left dark, so a note outside the key lights up out of nothing.
    The tonic keeps a lamp at the top of its column. While no key is known --
    too little heard, or ``ctx.key_uncertain`` -- the whole panel is powered
    and nothing is marked, which is the honest picture of not knowing.
    """
    rows, w = ctx.h, ctx.w
    chroma = np.asarray(ctx.chroma, dtype=np.float32)
    if chroma.shape != (12,):
        chroma = np.zeros(12, dtype=np.float32)
    loud = float(np.clip(ctx.energy * 2.5, 0.0, 1.0))
    per_col = chroma[list(_FIFTHS)] * loud
    per_col = np.where(per_col > _KEY_FLOOR * loud, per_col, 0.0)

    known = ctx.key if (ctx.key and not ctx.key_uncertain) else None
    tonic, powered = _key_window(known)

    # Twelve columns, each several cells wide with a gutter.
    col_note, active = band_columns(w, 12)
    levels = np.where(active, per_col[col_note], 0.0)
    codes, cidx = bar_panel(ctx, rows, levels, active)

    # Unpowered columns: no lattice, only what is actually lit.
    dark = ~powered[col_note] & active
    codes[:, dark] = np.where(codes[:, dark] == _OFF, SPACE, codes[:, dark])

    if tonic is not None and rows >= 2:
        lamp_cols = np.flatnonzero(active & (np.array(_FIFTHS)[col_note] == tonic))
        top = bulb_row_index(rows)[-1]
        free = codes[top, lamp_cols] != _LED
        codes[top, lamp_cols[free]] = _PEAK
        cidx[top, lamp_cols[free]] = ladder_colours(ctx, rows)[top]
    return codes, cidx
