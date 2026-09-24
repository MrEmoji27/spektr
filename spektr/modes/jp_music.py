"""The JP displays that hear the music: drums, the bar, and the beat.

The first three JP modes (:mod:`spektr.modes.jp`) are one LED ladder blended
with a mechanic from elsewhere. These are not ladders. They take the other
half of a Japanese head unit's faceplate: the graphic displays, the demo
animations and clock faces printed in segments on the glass, where every
segment is always there and the music decides which are lit. What keeps them
in the family is that glass: segments of dots, with the unlit ones drawn
faintly, never a smooth line and never a bare background.

And each stands on a piece of the 0.6.0 analysis:

* **JP Demo** is the demo-mode sunburst. Rays of segments stretch with the
  bands; a kick blasts a ring of light out through the whole burst, a snare
  turns it one notch, the hats make the tips twinkle.
* **JP Clock** is a clock face that keeps the bar. A hand sweeps once a bar,
  writing what it hears into the outer ring; at each new bar the rings step
  inward, so the face holds the last four bars and a repeating pattern shows
  as the same shape four times over.
* **JP Ribbons** is three flowing ribbons of segments: a slow thick one for the
  kick, a middle one for the snare, a thin fast one for the hats, their waves
  locked to the bar so the pattern repeats as the music does.
* **JP Sequencer** pages the ladder by the bar, and is here until its rework.

None of them guesses. No drum, no ring; no bar, no downbeat; no tempo, and
the clock's hand runs on its own slow time, with its beat marks dark.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from ..render import SPACE, cell_max, pack_braille
from . import Ctx, band_columns, empty, mode
from . import polar_grid as _polar
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

#: Drum envelopes, as time constants in seconds: how long each drum's hit
#: goes on moving the picture. A kick is felt longest, a hat hardly at all.
_DRUM_TAU = {"kick": 0.26, "snare": 0.18, "hat": 0.07}

#: A drum below this likelihood did not hit. The likelihoods are shares of one
#: hit, so a snare with some hat in it moves both, but a trace is not a drum.
_DRUM_FLOOR = 0.35


def _drum_env(ctx: Ctx, st: dict) -> dict:
    """Each drum's envelope, 0..1: jumps on a hit of that drum, then decays."""
    env = st.setdefault("env", {"kick": 0.0, "snare": 0.0, "hat": 0.0})
    dt = max(ctx.dt, 0.0)
    for name, tau in _DRUM_TAU.items():
        env[name] *= math.exp(-dt / tau)
    if ctx.onsets:
        strength = 0.55 + 0.45 * min(1.0, float(ctx.onset_strength))
        for name in env:
            like = float(ctx.drums.get(name, 0.0))
            if like >= _DRUM_FLOOR:
                env[name] = max(env[name], like * strength)
    return env


def _glass(ctx: Ctx, lit: np.ndarray, glass: np.ndarray, heat: np.ndarray):
    """Pack a dot picture: lit segments in the ramp at ``heat``, unlit ones
    faint. ``heat`` is per dot, 0..1, and read only where ``lit`` is."""
    codes = pack_braille(lit | glass)
    field = np.where(lit, heat, np.float32(-1.0)).astype(np.float32)
    hot = cell_max(field)
    glass_idx = recede_index(ctx.palette)
    lit_idx = np.asarray(ctx.ramp(np.clip(hot, 0.0, 1.0)), dtype=np.int32)
    # The glass colour is chosen by contrast, and on some themes it is the
    # hot end of the ramp itself -- on gruvbox it is index 63, the colour of
    # anything lit at full heat, so the brightest thing on the face was drawn
    # as glass. A lit cell never takes the glass colour; it takes the one next
    # to it on the ramp.
    step = -1 if glass_idx > 0 else 1
    lit_idx = np.where(lit_idx == glass_idx, glass_idx + step, lit_idx)
    return codes, np.where(hot < 0.0, glass_idx, lit_idx).astype(np.int32)


# ── JP Demo ──────────────────────────────────────────────────────────────────

#: Rays around the burst, and segments along each ray.
_RAYS = 24
_SEGS = 9


@mode("JP Demo", group="jp",
      blurb="a head-unit demo sunburst: kicks blast it outward, snares turn it, hats twinkle the tips")
def jp_demo(ctx: Ctx):
    """The sunburst a head unit plays in its demo mode, played by the music.

    Twenty-four rays of nine segments. Each ray is a band, mirrored so the
    burst is symmetric, and lights out from the hub as far as its band is
    loud. Around that, each drum does one thing only, so which one hit can be
    read off the burst: a kick sends a ring of light out through every ray,
    hub to rim; a snare turns the whole burst one ray on, quickly, and holds;
    a hat lights a scatter of ray tips for a moment.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 16 or dc < 16:
        return empty(ctx.w, ctx.h)
    dist, turn, max_r = _polar(ctx)

    def geometry():
        r = dist / np.float32(max_r)
        seg_f = (r - np.float32(0.12)) / np.float32(0.84) * _SEGS
        seg = np.floor(seg_f)
        inside = (seg >= 0) & (seg < _SEGS) & ((seg_f - seg) < 0.62)
        return {"seg": np.where(inside, seg, -1).astype(np.int16), "r": r}

    geo = ctx.scratch("jp_demo_geo", geometry)
    st = ctx.scratch("jp_demo", lambda: {"rot": 0.0, "aim": 0.0, "blast": -9.0,
                                         "twinkle": np.zeros(_RAYS, np.float32),
                                         "rng": np.random.default_rng(7)})
    env = _drum_env(ctx, st)
    dt = max(ctx.dt, 0.0)

    if ctx.onsets and float(ctx.drums.get("kick", 0.0)) >= _DRUM_FLOOR:
        st["blast"] = ctx.t
    if ctx.onsets and float(ctx.drums.get("snare", 0.0)) >= _DRUM_FLOOR:
        st["aim"] += 1.0
    # The turn eases onto its new notch in about a tenth of a second.
    st["rot"] += (st["aim"] - st["rot"]) * (1.0 - math.exp(-dt / 0.035))
    st["twinkle"] *= np.float32(math.exp(-dt / 0.09))
    if ctx.onsets and float(ctx.drums.get("hat", 0.0)) >= _DRUM_FLOOR:
        pick = st["rng"].random(_RAYS) < 0.35
        st["twinkle"][pick] = 1.0

    # Which ray a dot is on, and whether it is on the ray or in the gap.
    pos = turn * np.float32(_RAYS) + np.float32(st["rot"])
    ray = np.floor(pos).astype(np.int32) % _RAYS
    on_ray = np.abs((pos % np.float32(1.0)) - np.float32(0.5)) < np.float32(0.26)
    seg = geo["seg"]
    body = on_ray & (seg >= 0)

    half = _RAYS // 2
    lv = ctx.display_bands(half + 1)[: half + 1].astype(np.float32)
    mirror = np.concatenate([lv[:half], lv[half:0:-1]])[:_RAYS]
    # By the square root, so an ordinary mix reaches most of the way out: on
    # the level itself a typical band lit two or three segments of nine and
    # the burst was mostly glass.
    reach = np.floor(np.sqrt(mirror) * np.float32(_SEGS * 1.1)).astype(np.int16)
    reach = np.where(mirror > _QUIET, reach, -1)
    lit = body & (seg < reach[ray])

    # The kick's ring, hub to rim in a third of a second.
    age = ctx.t - st["blast"]
    if 0.0 <= age < 0.34:
        front = age / 0.34 * _SEGS
        ring = body & (np.abs(seg.astype(np.float32) - np.float32(front)) < 0.9)
        lit |= ring & (env["kick"] > 0.15)
    # The hats' tips.
    tips = body & (seg >= _SEGS - 2) & (st["twinkle"][ray] > 0.3)
    lit |= tips

    # The glass: one faint line of dots down the middle of every ray. A
    # scatter over the whole ray drew a diagonal hatch across the burst.
    centre = np.abs((pos % np.float32(1.0)) - np.float32(0.5)) < np.float32(0.07)
    glass = body & centre
    heat = geo["r"] * np.float32(0.7) + np.float32(0.3 * env["kick"])
    return _glass(ctx, lit, glass & ~lit, heat)


# ── JP Clock ─────────────────────────────────────────────────────────────────

#: Rings on the face, outermost first: the bar being written, then the three
#: before it. And steps around each ring: sixteenths of a four-beat bar.
_RINGS = 4
_STEPS = 16

#: How sure the bar tracker has to be before the face trusts its downbeat.
_CLOCK_SURE = 0.4

#: One turn of the hand when there is no tempo at all, in seconds.
_CLOCK_IDLE_S = 4.0


def _clock_phase(ctx: Ctx, st: dict) -> tuple[float, bool, bool]:
    """Where the hand is, 0..1 round the face; whether that is the bar (the
    downbeat is known); and whether there is a beat at all."""
    if ctx.bar_confidence >= _CLOCK_SURE and ctx.tempo_bpm > 0.0:
        return float(ctx.bar_phase) % 1.0, True, True
    if ctx.tempo_bpm > 0.0:
        # A beat but no bar: count four beats round the face from wherever
        # the count started, and claim no downbeat.
        if ctx.beat_phase < st["last_beat"] - 0.5:
            st["count"] = (st["count"] + 1) % 4
        st["last_beat"] = ctx.beat_phase
        return (st["count"] + float(ctx.beat_phase)) / 4.0, False, True
    st["idle"] = (st["idle"] + max(ctx.dt, 0.0) / _CLOCK_IDLE_S) % 1.0
    return st["idle"], False, False


@mode("JP Clock", group="jp",
      blurb="a clock face that keeps the bar: the hand writes each bar in, the last four step inward")
def jp_clock(ctx: Ctx):
    """A segmented clock face with a memory of four bars.

    The hand goes round once a bar. The outer ring is sixteen segments, one
    per sixteenth of the bar, and as the hand passes each it lights with how
    loud the music is at that moment. When the bar comes round the rings step
    inward: the bar just written becomes the second ring, and so on, the
    fourth falling off the middle. So a loop that repeats is four rings with
    the same shape, a fill is one that is not, and a drop is the face filling
    from the outside in.

    Four beat marks outside the face light with the beat. The mark for one
    lights brighter, and only when the bar tracker knows where one is; with a
    beat but no bar, the face counts four and marks no downbeat, and with no
    beat at all the hand runs on its own slow time and the marks stay dark.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 16 or dc < 16:
        return empty(ctx.w, ctx.h)
    dist, turn, max_r = _polar(ctx)

    def geometry():
        r = dist / np.float32(max_r)
        # Rings from 0.84 inward, each 0.15 deep with a gap, and a tick band
        # outside them.
        band = np.floor((np.float32(0.84) - r) / np.float32(0.17))
        within = (np.float32(0.84) - r) - band * np.float32(0.17)
        ring = np.where((band >= 0) & (band < _RINGS) & (within < 0.12), band, -1)
        step_f = turn * np.float32(_STEPS)
        on_step = (step_f - np.floor(step_f)) < np.float32(0.82)
        step = np.floor(step_f).astype(np.int16) % _STEPS
        beat_f = turn * np.float32(4.0)
        tick = (r > 0.88) & (r < 0.97) & (np.abs(beat_f - np.round(beat_f)) < 0.05)
        return {"ring": ring.astype(np.int16), "step": step, "on_step": on_step,
                "tick": tick, "tick_of": np.round(beat_f).astype(np.int16) % 4,
                "r": r}

    geo = ctx.scratch("jp_clock_geo", geometry)
    st = ctx.scratch("jp_clock", lambda: {
        "rec": np.zeros((_RINGS, _STEPS), np.float32), "last": 0.0,
        "count": 0, "last_beat": 0.0, "idle": 0.0,
    })
    # The hand goes clockwise from twelve. ``turn`` also runs clockwise on
    # screen -- rows count downward -- but from three o'clock, so twelve is
    # three quarters of the way round it.
    phase, known, beating = _clock_phase(ctx, st)
    if phase < st["last"] - 0.5:
        st["rec"][1:] = st["rec"][:-1].copy()
        st["rec"][0] = 0.0
    st["last"] = phase
    hand_turn = (0.75 + phase) % 1.0
    step_now = int(hand_turn * _STEPS) % _STEPS
    level = float(np.clip(ctx.energy * 1.6 + ctx.flux * 0.4, 0.0, 1.0))
    if not ctx.silent:
        st["rec"][0, step_now] = max(st["rec"][0, step_now], level)

    ring, step = geo["ring"], geo["step"]
    body = (ring >= 0) & geo["on_step"]
    val = np.where(body, st["rec"][np.clip(ring, 0, _RINGS - 1), step], 0.0)
    # Older bars a little dimmer, so the face reads outside-in as time.
    age_dim = np.float32(1.0) - np.clip(ring, 0, _RINGS - 1).astype(np.float32) * np.float32(0.15)
    lit = body & (val > 0.12)

    # The hand: a spoke from the hub to the rim at the hand's angle.
    d = np.abs(((turn - np.float32(hand_turn)) + 0.5) % 1.0 - 0.5)
    hand = (d < np.float32(0.006 + 0.5 / max(dc, 1))) & (geo["r"] < 0.86) & (geo["r"] > 0.08)

    beat_now = int(phase * 4) % 4
    ticks = geo["tick"]
    # The mark at twelve is quarter 3 of ``turn``; counted clockwise from it,
    # the beat marks are one, two, three and four.
    mark = (geo["tick_of"] + 1) % 4
    tick_lit = ticks & beating & (mark == beat_now)

    # Unlit marks are glass like everything else: half their dots, so a lit
    # mark stands out by how solid it is as well as by its colour.
    checker = (np.arange(dc)[None, :] + np.arange(dr)[:, None]) % 2 == 0
    glass = (body & ~lit & checker) | (ticks & ~tick_lit & checker)
    heat = np.where(hand, np.float32(1.0), val * age_dim).astype(np.float32)
    # One is the brightest mark, and only when the bar says where one is.
    heat = np.where(tick_lit, np.float32(1.0 if known and beat_now == 0 else 0.75), heat)
    return _glass(ctx, lit | hand | tick_lit, glass, heat)


# ── JP Ribbons ───────────────────────────────────────────────────────────────

#: Each ribbon: the drum it rides, where it sits (share of the height from the
#: top), the band range it reads, how many waves fit across the screen, how
#: many times a bar the pattern travels, and its thickness in dot rows.
_RIBBONS = (
    ("kick", 0.78, (0.0, 0.18), 1.0, 1.0, 2.5),
    ("snare", 0.50, (0.25, 0.60), 2.0, 2.0, 1.6),
    ("hat", 0.22, (0.65, 1.0), 4.0, 4.0, 0.8),
)


@mode("JP Ribbons", group="jp",
      blurb="three flowing ribbons of segments — the kick's, the snare's and the hats' — locked to the bar")
def jp_ribbons(ctx: Ctx):
    """Three dashed ribbons flowing across, one per drum.

    The bottom one is the kick's: long slow waves, thick, swelling when a kick
    lands. The middle one is the snare's, the top one the hats' -- thinner and
    quicker each. Between hits a ribbon still rides the level of its part of
    the spectrum, so the picture is alive in a pad; a hit of its own drum
    throws its wave high and lets it settle back.

    The waves are locked to the bar when the bar is known and to the beat
    when only the beat is, so the shape of a bar comes round with the bar
    instead of drifting against it. Without either they run on their own time.

    Each ribbon is dashed like a segment display, and its resting line is
    drawn faintly underneath, which is the glass.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 12 or dc < 16:
        return empty(ctx.w, ctx.h)
    st = ctx.scratch("jp_ribbons", lambda: {"free": 0.0})
    env = _drum_env(ctx, st)
    st["free"] = (st["free"] + max(ctx.dt, 0.0) * 0.5) % 1.0
    if ctx.bar_confidence >= _CLOCK_SURE and ctx.tempo_bpm > 0.0:
        clock = float(ctx.bar_phase)
    elif ctx.tempo_bpm > 0.0:
        clock = float(ctx.beat_phase) / 4.0
    else:
        clock = st["free"]

    xs = np.arange(dc, dtype=np.float32) / np.float32(max(dc - 1, 1))
    rows = np.arange(dr, dtype=np.float32)[:, None]
    # Short, sparse breaks: long regular gaps across a thick ribbon read as a
    # row of vertical bars, which is exactly what this family is not.
    dash = ((np.arange(dc) // 2) % 6 != 5)[None, :]
    lit = np.zeros((dr, dc), dtype=bool)
    glass = np.zeros((dr, dc), dtype=bool)
    heat = np.zeros((dr, dc), dtype=np.float32)
    for k, (drum, home, (lo, hi), waves, speed, thick) in enumerate(_RIBBONS):
        level = float(ctx.range(lo, hi))
        swing = (0.05 + 0.10 * level + 0.16 * env[drum]) * dr
        centre = home * dr
        wave = np.sin((xs * waves - clock * speed) * np.float32(2 * math.pi)
                      + np.float32(k * 1.3))
        y = centre - swing * wave
        near = np.abs(rows - y[None, :]) <= np.float32(thick * (1.0 + 0.6 * env[drum]))
        on = near & dash & ((level > _QUIET) | (env[drum] > 0.05))
        lit |= on
        # A gradient across the screen, and hotter all over on a hit: one flat
        # colour per ribbon used four steps of the theme's ramp and hid its
        # gradient. Across the screen rather than along the wave, which moves
        # every frame and turned the colour into motion between the beats.
        tone = np.float32(0.1 + 0.15 * k) + np.float32(0.5) * xs + np.float32(0.3 * env[drum])
        heat = np.where(on, np.broadcast_to(tone[None, :], heat.shape), heat)
        glass |= (np.abs(rows - np.float32(centre)) < 0.5) & ((np.arange(dc) % 4 == 0)[None, :])
    return _glass(ctx, lit, glass & ~lit, heat)


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


