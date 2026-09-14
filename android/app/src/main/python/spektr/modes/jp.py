"""The JP family — one bar graph, three machines around it.

Every mode here is the *same* segmented LED meter: a column per band, a fixed
ladder of bulbs per column, the unlit bulbs drawn rather than left blank. That
ladder is :func:`bar_panel` and all three modes draw it. What differs is the
mode each one is *blended with*: ``Keys``' note roll, ``Dune``'s sandpile, and
``Radial``'s circle.

Two rules keep this a family rather than a shelf of near-duplicates, and both
were learned by breaking them.

**The bars come first.** The mechanic is layered onto a ladder that stays
legible — the first version of this file lost the bars entirely in three of
the four and had to be rewritten.

**The partner has to be a real mechanic, not a decoration.** A scan light
sweeping the panel was cut because ``Sonar`` already owns the travelling sweep,
and what was left was a bar chart with a line moving over it. JP Keys was
cut for the same reason from the other side: its partner was a piano keyboard,
which is not a mode, and the keys implied note names that 32 log-spaced FFT
bands cannot give. If a new JP mode cannot be named as "``Bars`` ×
*something that already exists*", it is a clone of ``Bars`` and it should not
be added.

Against the flat spectrum group they sit next to: ``Bars`` draws smooth
fractional blocks and ``Ladder`` a contiguous stack, both of which show only
the lit part of a column. These draw the unlit part too, which is the whole
difference between a bar chart with gaps in it and a panel of hardware.
"""

from __future__ import annotations

import numpy as np

from ..palette import RAMP_STEPS
from ..render import SPACE, cell_max, pack_braille
from . import (
    Ctx,
    band_columns,
    empty,
    mode,
    polar_grid as _polar,
)

_LED = ord("▆")          # a lit bulb
_OFF = ord("▁")          # an unlit one: the dark base of the same bulb
_HALF_DOWN = ord("▄")   # a trail bulb: lighter than lit, heavier than unlit

#: Where an unlit bulb and the *foot* of a lit ladder sit on the ramp.
#:
#: A theme's ramp is a hue gradient at full saturation, not a brightness one:
#: measured across the 55 built-ins, index 0 and index 6 differ by under 10/255
#: per channel on every one of them. So "draw the unlit bulbs dimmer" does not
#: exist — an unlit bulb parked near the bottom of the ramp comes out the same
#: colour as a lit bulb at the bottom of a short column, which is exactly the
#: pair that has to stay apart.
#:
#: Two things separate them instead. The unlit bulb gets a *smaller* glyph, so
#: the picture is legible as characters and not only as colour — in a dump, in
#: a screenshot, and on the flatter themes. And the lit ladder starts at
#: :data:`_LIT_FLOOR` rather than at 0, so even its lowest bulb is a step along
#: the gradient from an unlit one.
_IDLE = 0.0
_LIT_FLOOR = 0.22

#: Concurrent chaser slots on the ring. Two is enough to overlap a fast pair
#: of kicks (see particles.pulse for the same policy).
_KW_WAVES = 2

#: Bulbs along one spoke of the ring.
_RING_SEGS = 12

#: Sand per second into a band at full level, for Drift. ``Dune`` uses 0.55
#: against a 160-row dot grid; a twenty-bulb ladder needs about three times
#: that before anything crosses a bulb boundary often enough to read as
#: moving. See :func:`jp_drift`.
_DRIFT_FEED = 1.6


def bulb_rows(rows: int) -> np.ndarray:
    """Which of ``rows`` rows carry bulbs: the bottom one and every other up.

    The unlit rows between them are what make a column read as a stack of
    discrete elements instead of a solid bar, so they are structural rather
    than decorative — a ladder drawn on every row is ``Ladder``.
    """
    return ((rows - 1) - np.arange(rows)) % 2 == 0


def bulb_count(rows: int) -> int:
    """How many bulbs a column of ``rows`` rows holds."""
    return (rows + 1) // 2


def bulb_row_index(rows: int) -> np.ndarray:
    """Screen row of each bulb, counting bulb 0 as the bottom one.

    Anything that travels *along* a ladder has to move in bulbs, not in rows:
    a trail advanced one screen row at a time spends half its life on the dark
    rows between bulbs and flickers. So the roll buffers here are indexed by
    bulb and converted to rows only when they are drawn.
    """
    return rows - 1 - 2 * np.arange(bulb_count(rows))


def crest_bulb(levels: np.ndarray, rows: int) -> np.ndarray:
    """Index of the topmost lit bulb for each level, or -1 for none.

    :func:`bar_panel` lights bulb *k* when ``level > 2k / rows``. This counts
    that same comparison rather than inverting it into ``ceil(level * rows / 2)
    - 1``, which is the closed form and is wrong about one level in forty.
    The two arithmetic paths disagree at exactly the boundaries: a level that
    is a couple of ulps above ``34/40`` lights the bulb, while the closed form
    rounds ``level * 40 / 2`` back down to ``17.0`` and reports the bulb below
    it. Half a screen away that is a trail detaching one bulb inside its own
    bar, which looks like a rendering glitch and is impossible to find by
    reading either function on its own.

    Cheap to do properly — the comparison is ``n`` by ``bulbs``, both small —
    and it cannot drift from :func:`bar_panel` unless the threshold there is
    changed, in which case the same expression has to change here too.
    """
    thresh = 2 * np.arange(bulb_count(rows)) / rows
    return (np.asarray(levels)[:, None] > thresh[None, :]).sum(axis=1).astype(np.int32) - 1


def bar_panel(ctx: Ctx, rows: int, levels: np.ndarray, active: np.ndarray,
              heat: np.ndarray | None = None):
    """The ladder itself — ``(codes, cidx)`` of shape ``(rows, ctx.w)``.

    ``levels`` and ``active`` are per *column*, already gathered through
    :func:`band_columns`, because every caller here needs the column mapping
    for something else as well and gathering it twice would be waste.

    ``heat`` is an optional per-column multiplier on where the lit bulbs land
    on the ramp — Drift pins a collapsing column to the top of it. Values above
    1 are fine, since :meth:`Ctx.ramp` clips. Leave it out and the ladder is
    coloured by height alone, which is the green-amber-red an equalizer is
    expected to be.
    """
    w = ctx.w
    on_row = bulb_rows(rows)
    panel = on_row[:, None] & active[None, :]

    thresh = (np.arange(rows - 1, -1, -1, dtype=np.float64) / rows)[:, None]
    lit = panel & (levels[None, :] > thresh)

    up = np.linspace(1.0, _LIT_FLOOR, rows)   # hot at the top of the ladder
    if heat is None:
        hot = np.repeat(ctx.ramp(up)[:, None], w, axis=1)
    else:
        hot = ctx.ramp(up[:, None] * heat[None, :])

    codes = np.where(lit, _LED, np.where(panel, _OFF, SPACE)).astype(np.int32)
    cidx = np.where(lit, hot, ctx.palette.index(_IDLE))
    return codes, cidx


def peak_bulbs(ctx: Ctx, codes, cidx, peaks: np.ndarray, rows: int,
               row0: int = 0) -> None:
    """Light the bulb the peak is holding at, at the top of the ramp.

    ``spectrum._draw_peaks`` stamps a hairline wherever the cell under it is
    empty. This cannot: on a panel every bulb row is already occupied, by an
    unlit bulb if nothing else. So the peak is a *hotter bulb* rather than a
    different character — which is what the hardware does, and it means a peak
    landing on a bulb the level already lit simply lights it harder instead of
    being dropped.

    The row is snapped **down** onto the nearest bulb, never up. The bottom
    row is always a bulb row and bulb rows alternate, so ``r + 1`` is a bulb
    row whenever ``r`` is not and is always still on the panel — snapping up
    is what walked off the top of a full-scale column. That makes the snap
    total: there is no "no bulb to put it on" case left to filter out, which
    is why the only condition below is the level one.
    """
    on_row = bulb_rows(rows)
    r = rows - 1 - np.floor(np.clip(peaks, 0.0, 0.999) * rows).astype(np.int32)
    r = np.where(on_row[r], r, r + 1)
    cols = np.flatnonzero(peaks > 0.03)
    if cols.size == 0:
        return
    codes[row0 + r[cols], cols] = _LED
    cidx[row0 + r[cols], cols] = RAMP_STEPS - 1


@mode("JP Bars", group="jp",
      blurb="a segmented LED meter whose crest peels off and rises, like notes leaving a key")
def jp_bars(ctx: Ctx):
    """The meter, blended with the note roll out of ``Keys``.

    ``Bars`` shows a level as a height of ink and nothing else, so a quiet
    passage is a nearly empty screen and a loud one is a shape that vanishes
    the instant the level drops. A graphic equalizer's panel is fully there at
    all times — the bulbs above the level are unlit, not absent — and on top
    of that this keeps what the bar *did*: every frame the crest sheds a bulb
    that detaches and climbs the ladder, fading, exactly the way a struck note
    leaves the keyboard and travels up the roll in ``Keys``.

    So the picture holds two things at once. The bright ladder is now; the
    half-height trail above it is the last couple of seconds of this band, and
    a bar that rose fast leaves a staircase behind it because the emission
    point moves with the crest.

    **Not VFD, and the difference is the whole point.** ``VFD`` is the other
    Japanese hi-fi display and it also puts something above the bar, but its
    phosphor decays *in place* — the trail marks where the bar was and stays
    there while it dims. These trails *travel*. One is a bar with a soft edge;
    this is a bar with a history scrolling away from it. Anyone tempted to
    unify them should switch between the two for a few seconds first.
    """
    rows, w = ctx.h, ctx.w
    n = ctx.n_display
    col_band, active = band_columns(w, n)
    lv = ctx.display_bands(n)
    levels = np.where(active, lv[col_band], 0.0)
    peaks = np.where(active, ctx.display_peaks(n)[col_band], 0.0)

    nb = bulb_count(rows)
    st = ctx.scratch("jp_roll", lambda: {
        "roll": np.zeros((nb, n), dtype=np.float32),
        "was": np.full(n, -1, dtype=np.int32),
        "acc": 0.0,
    })
    if st["roll"].shape != (nb, n):
        st["roll"] = np.zeros((nb, n), dtype=np.float32)
        st["was"] = np.full(n, -1, dtype=np.int32)
    roll = st["roll"]

    # Paced in seconds, not frames, like every other scroll in the app: the
    # trail must cross the ladder in the same time at 30 fps and at 144.
    st["acc"] += (nb / 2.2) * ctx.dt
    shift = min(int(st["acc"]), nb)
    if shift:
        st["acc"] -= shift
        roll[shift:] = roll[:nb - shift]
        roll[:shift] = 0.0
    roll *= np.float32(np.exp(-max(ctx.dt, 0.0) / 0.9))

    # Emit only the bulbs a *rise* actually added — the bars that were just
    # raised, which is the thing being left behind. Emitting at the crest
    # every frame instead was the first attempt and it filled the whole panel
    # with trail: the crest wanders, so a continuous emitter smears across
    # every bulb it visits and there is nothing left to read. A held note
    # sheds nothing, and that is correct; it is not doing anything.
    crest = np.clip(crest_bulb(lv, rows), -1, nb - 1)
    kk = np.arange(nb, dtype=np.int32)[:, None]
    raised = (kk > st["was"][None, :]) & (kk <= crest[None, :])
    np.maximum(roll, raised * lv.astype(np.float32)[None, :], out=roll)
    st["was"] = crest

    codes, cidx = bar_panel(ctx, rows, levels, active)
    peak_bulbs(ctx, codes, cidx, peaks, rows)

    # Trails go in only where the ladder is unlit, so the live bar always wins
    # its own cells and the picture never says a bulb is both.
    rws = bulb_row_index(rows)
    up = np.linspace(1.0, _LIT_FLOOR, rows)
    ghost = np.where(active[None, :], roll[:, col_band], np.float32(0.0))
    sub_c, sub_x = codes[rws], cidx[rws]
    show = (ghost > 0.06) & (sub_c == _OFF)
    sub_c[show] = _HALF_DOWN
    sub_x[show] = ctx.ramp(up[rws][:, None] * ghost)[show]
    codes[rws], cidx[rws] = sub_c, sub_x
    return codes, cidx


@mode("JP Drift", group="jp",
      blurb="the meter as a sandpile — bulbs stack up and avalanche into their neighbours")
def jp_drift(ctx: Ctx):
    """The meter, blended with the sandpile out of ``Dune``.

    Every other mode in this family draws the level: read it, light that many
    bulbs. Here the bulbs are a running *sum*. Sand rains into each band in
    proportion to its level, a column only ever grows, and the sole way down
    is a collapse — height crossing the angle of repose, dumping most of
    itself and pushing the excess sideways into the two bands next to it.
    A loud passage therefore keeps toppling columns for seconds afterwards,
    because what decides that is the pile, not the spectrum.

    The law is ``Dune``'s — inflow with the square of the level, collapse at
    the angle of repose, a third of the excess to each side — but it is drawn
    on the family's ladder rather than as dot-resolution grit, and that
    changes two constants that are worth spelling out, because both were
    found by the audit rather than by taste.

    **Inflow is faster here, and has to be.** ``Dune`` fills a 160-row dot
    grid, so a hundredth of a unit of sand still moves the top of the heap by
    a dot. A twenty-bulb ladder is an order of magnitude coarser: at Dune's
    rate nothing crosses a bulb boundary for a quarter of a second at a time
    and the panel reads as *frozen*, which is what the animation check called
    it. The rate buys back the resolution the ladder gives up.

    **The pile starts at the spectrum, not at zero.** From flat it would spend
    the first seconds as an empty panel — the mode would have nothing to say
    when you switched to it, and its whole colour range would be missing until
    the heap got tall enough to reach it. Seeded, it opens as a bar graph and
    then starts drifting, which is also the honest picture: the sand has been
    falling all along, you just arrived.

    A collapse then needs help to be visible at all. Grit shows an avalanche
    as a texture sliding; sixteen bulbs cannot, so a toppling column and the
    two it feeds flash to the top of the ramp and fade, and that flash is how
    you see where the sand went.

    This replaced a scan-light mode that was the fourth JP. It was cut
    because ``Sonar`` already owns the travelling sweep with fading returns,
    and a JP version of it was a bar chart with a line moving over it —
    a clone of ``Bars`` wearing a different mechanic rather than a blend with
    one. Sideways transfer between bands is something no other mode does.
    """
    w, h = ctx.w, ctx.h
    if w < 10 or h < 4:
        return empty(w, h)

    n = int(np.clip(ctx.n_display, 6, 40))
    lv = ctx.display_bands(n)

    st = ctx.scratch("jp_drift", lambda: {
        "h": lv.astype(np.float64).copy(),
        "flash": np.zeros(n, dtype=np.float64),
        "rng": np.random.default_rng(53),
    })
    if st["h"].shape[0] != n:
        st["h"] = lv.astype(np.float64).copy()
        st["flash"] = np.zeros(n, dtype=np.float64)
    pile, flash, rng = st["h"], st["flash"], st["rng"]

    pile += lv * lv * _DRIFT_FEED * ctx.dt
    np.clip(pile, 0.0, 1.3, out=pile)

    spill = pile > 1.0
    if spill.any():
        idx = np.flatnonzero(spill)
        excess = pile[idx] - 0.55
        pile[idx] = 0.55 + rng.uniform(-0.03, 0.03, idx.size)
        left = np.clip(idx - 1, 0, n - 1)
        right = np.clip(idx + 1, 0, n - 1)
        np.add.at(pile, left, excess * 0.35)
        np.add.at(pile, right, excess * 0.35)
        np.clip(pile, 0.0, 1.3, out=pile)
        # the column that went, and the two it went into
        flash[idx] = 1.0
        np.maximum.at(flash, left, 0.7)
        np.maximum.at(flash, right, 0.7)

    st["flash"] = flash * float(np.exp(-max(ctx.dt, 0.0) / 0.22))

    col_band, active = band_columns(w, n)
    levels = np.where(active, np.clip(pile, 0.0, 1.0)[col_band], 0.0)
    # ``heat`` above 1 is fine: ctx.ramp clips, so a collapsing column simply
    # pins to the top of the ramp for as long as the flash lasts.
    heat = np.where(active, 1.0 + st["flash"][col_band] * 1.6, 0.0)
    return bar_panel(ctx, h, levels, active, heat=heat)


@mode("JP Pulse", group="jp",
      blurb="the meter bent into a ring — a spoke of bulbs per band, chased on the beat")
def jp_pulse(ctx: Ctx):
    """The same ladder, once round the circle: one spoke of bulbs per band.

    ``Radial`` fills a smooth wedge out from the centre and ``Pulse`` blobs a
    disc. This is the bar graph in polar coordinates and nothing else: each
    band gets a spoke with a dark gutter either side of it, the spoke carries
    the family's ladder of discrete bulbs growing outward from a hub, and the
    peak bulb hangs out past the level exactly as it does on the flat panel.
    A band is therefore still readable as a bar — which angle it is at, how
    far out it reaches — rather than as a brightness somewhere on a ring.

    The ring turns on an energy-driven clock accumulated in scratch, and an
    onset launches a chaser: a bright band position that runs once round and
    fades, flaring each spoke it passes. The chaser is added to the *band*
    table before the gather, so it costs a table of ``n`` entries and no extra
    pass over the dot grid — the same reasoning as :func:`angular_lut`.

    **Where the frame goes.** The first version did all of this per dot, and
    at 400x100 that is 320,000 elements times about thirty passes: it benched
    at 13.2 ms, second-costliest mode in the app, for a picture that only ever
    takes ``n * _RING_SEGS`` distinct forms. So the same split the rest of the
    file's polar code uses applies here twice over.

    * The *radial* half — which bulb a dot belongs to, whether it is on a bulb
      rather than in the gap between two — depends only on the geometry, so it
      is built once per size and cached. That is why the dial's outer radius
      is fixed rather than breathing with :attr:`Ctx.pulse`: a breathing
      radius makes every one of those arrays frame-dependent, and the beat is
      already carried by the chaser and by the whole-dial lift on ``top``.
    * The *angular* half becomes a 512-entry table, gathered with an index
      that spin advances by integer addition. Rotation is then quantised to
      1/512 of a turn, which is 0.7 degrees and below what the dot grid can
      show anyway.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 16 or dc < 16:
        return empty(ctx.w, ctx.h)

    dist, turn, max_r = _polar(ctx)
    n = int(min(24, ctx.n_display))
    steps = 512

    st = ctx.scratch("jp_pulse", lambda: {
        "spin": 0.0,
        "born": np.full(_KW_WAVES, -99.0),
        "amp": np.zeros(_KW_WAVES),
        "acc": 0.0,
    })
    st["spin"] = (st["spin"] + (0.04 + ctx.energy * 0.09) * ctx.dt) % 1.0

    # chaser slots — same policy as particles.pulse, which explains the clock
    st["acc"] += (0.8 + ctx.energy * 5.0) * ctx.dt
    due = st["acc"] >= 1.0 or st["born"].max() < 0.0
    if due:
        st["acc"] = max(0.0, st["acc"] - 1.0)
    if (ctx.onsets or due) and (ctx.t - st["born"].max()) > 0.12:
        slot = int(np.argmin(st["born"]))
        st["born"][slot] = ctx.t
        strength = ctx.onset_strength if ctx.onsets else min(1.0, ctx.energy * 1.4)
        st["amp"][slot] = float(np.clip(0.35 + strength * 0.9, 0.0, 1.0))

    lap = 0.85
    band_i = np.arange(n, dtype=np.float32)
    chase = np.zeros(n, dtype=np.float32)
    for k in range(_KW_WAVES):
        age = ctx.t - st["born"][k]
        if not (0.0 <= age < lap):
            continue
        at = (age / lap) * n
        d = np.abs(((band_i - at + n * 0.5) % n) - n * 0.5)
        amp = float(st["amp"][k]) * (1.0 - age / lap)
        chase = np.maximum(chase, np.float32(amp) * np.clip(1.0 - d / 1.6, 0.0, 1.0))

    # ── the radial half, built once per size ─────────────────────────────────
    # Which bulb each dot is part of, and whether it is on one at all. Nothing
    # here depends on the audio or the spin, so it is geometry, not a frame.
    def radial():
        segs = float(_RING_SEGS)
        r0 = max_r * 0.26
        r1 = max_r * 0.94
        s = (dist - np.float32(r0)) * np.float32(segs / (r1 - r0))
        floor_s = np.floor(s)
        ok = ((s - floor_s) < np.float32(0.60)) & (s >= np.float32(0.0)) & (s < np.float32(segs))
        idx0 = (turn * np.float32(steps)).astype(np.int32) & (steps - 1)
        return {
            "bulb": np.where(ok, floor_s, np.float32(-1.0)).astype(np.int16),
            "ok": ok,
            # hot at the outer end of the spoke, which is the ring's version
            # of the ladder being hot at the top
            "warm": (np.float32(_LIT_FLOOR) + np.clip(s * np.float32(1.0 / segs), 0.0, 1.0)
                     * np.float32(1.0 - _LIT_FLOOR)).astype(np.float32),
            "idx0": idx0,
        }

    geo = ctx.scratch("jp_dial", radial)

    # ── the angular half, as a 512-entry table ───────────────────────────────
    # ``-1`` marks the gutter between two spokes, so one gather answers both
    # "is this dot inside a spoke" and "how far up that spoke is lit".
    lv = ctx.display_bands(n).astype(np.float32)
    drive = np.clip(lv + chase * np.float32(0.55) + np.float32(ctx.pulse * 0.06), 0.0, 1.0)
    top_b = np.floor(drive * _RING_SEGS).astype(np.int16)
    pk_b = np.floor(np.clip(ctx.display_peaks(n).astype(np.float32), 0.0, 0.999)
                    * _RING_SEGS).astype(np.int16)

    pos = np.arange(steps, dtype=np.float32) * np.float32(n / steps)
    band = pos.astype(np.int32)
    gut = pos - band
    spoke = (gut > np.float32(0.16)) & (gut < np.float32(0.84))
    top_a = np.where(spoke, top_b[band], -1).astype(np.int16)
    pk_a = np.where(spoke, pk_b[band], -1).astype(np.int16)

    shift = int(st["spin"] * steps) & (steps - 1)
    idx = (geo["idx0"] + shift) & (steps - 1)
    reach = top_a[idx]

    # The whole ladder is drawn, unlit bulbs included, exactly as on the flat
    # panel — that is what makes this read as the same instrument bent round
    # rather than as another ring mode. The dots are the dial; the rest only
    # decides how hot each one is.
    dial = geo["ok"] & (reach >= 0)
    codes = pack_braille(dial)

    # Filled rather than selected: an unlit bulb and an empty cell both want
    # the ramp floor, so the ``where`` that used to pick between them was
    # 320,000 elements of branching to produce a constant — 0.48 ms of this
    # mode's 4.7. A cell with no dots in it paints nothing whatever its index
    # says (see tests/test_blank_runs.py), so filling the whole grid is not
    # merely cheaper, it is the same picture.
    bulb = geo["bulb"]
    heat = np.full(dial.shape, np.float32(_IDLE), dtype=np.float32)
    np.copyto(heat, geo["warm"], where=dial & (bulb < reach))
    np.copyto(heat, np.float32(1.0), where=dial & (bulb == pk_a[idx]))
    cidx = ctx.ramp(cell_max(heat))
    return codes, cidx
