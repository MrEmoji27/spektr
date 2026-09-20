"""The JP family — one bar graph, three machines around it.

Inspired by Japanese car audio hardware: the segmented LED level meters on the
faceplates of in-dash head units and equalisers, where every bulb of every
column is printed on the panel whether it is lit or not. The family takes that
panel, not any one maker's design.

Every mode here is the *same* segmented LED meter: a column per band, a fixed
ladder of bulbs per column, the unlit bulbs drawn rather than left blank. That
ladder is :func:`bar_panel` and all three modes draw it. What differs is the
mode each one is *blended with*: ``Keys``' note roll, ``Dune``'s sandpile, and
``Radial``'s circle.

Three rules keep this a family rather than a shelf of near-duplicates, and all
three were learned by breaking them.

**The bars come first.** The mechanic is layered onto a ladder that stays
legible — the first version of this file lost the bars entirely in three of
the four and had to be rewritten.

**The partner has to be a real mechanic, not a decoration.** A scan light
sweeping the panel was cut because ``Sonar`` already owns the travelling sweep,
and what was left was a bar chart with a line moving over it. A Keys mode was
cut for the same reason from the other side: its partner was a piano keyboard,
which is not a mode, and the keys implied note names that 32 log-spaced FFT
bands cannot give. If a new JP mode cannot be named as "``Bars`` ×
*something that already exists*", it is a clone of ``Bars`` and it should not
be added.

**Hierarchy is carried by ink, not by colour.** Everything on a panel is one
of four weights, heaviest first: a lit bulb ``▆``, a peak ``▂``, a trail
``▁``, an unlit bulb ``·``. Colour cannot do this job, because a theme's ramp
is a hue gradient and not a brightness one — on ``gruvbox`` the bottom of the
ramp is the *brightest* colour on it, so the old unlit bulbs, parked at index
0, were the loudest thing on screen and drew a ruled grid over the bars. See
:func:`recede_index` for the one colour decision that is made on measurement.

Against the flat spectrum group they sit next to: ``Bars`` draws smooth
fractional blocks and ``Ladder`` a contiguous stack, both of which show only
the lit part of a column. These draw the unlit part too, which is the whole
difference between a bar chart with gaps in it and a panel of hardware.
"""

from __future__ import annotations

import numpy as np

from ..render import SPACE, cell_max, pack_braille
from . import (
    Ctx,
    band_columns,
    bg_contrast,
    empty,
    mode,
)
from . import (
    polar_grid as _polar,
)

# The four ink weights, heaviest first. See the module docstring.
_LED = ord("▆")          # a lit bulb
_PEAK = ord("▂")         # where the peak is holding
_TRAIL = ord("▁")        # Bars' shed trail
_OFF = ord("·")          # an unlit bulb

#: Where the foot of a lit ladder sits on the ramp. The ladder is coloured by
#: height, hot at the top, and this keeps its lowest bulb a step along the
#: gradient rather than at the very end of it.
_LIT_FLOOR = 0.22

#: Below this a level lights nothing and a peak is not drawn. Bands decay
#: toward zero without reaching it, and without a floor the bottom bulb of
#: every column stayed lit for as long as the app was open after the music
#: stopped — a threshold of ``0`` is crossed by any positive float.
_QUIET = 0.03

#: The WCAG contrast an unlit bulb aims for against the theme's background.
#: See :func:`recede_index`.
_RECEDE_CONTRAST = 2.2

#: Concurrent chaser slots on the ring. Two is enough to overlap a fast pair
#: of kicks (see particles.pulse for the same policy).
_KW_WAVES = 2

#: Bulbs along one spoke of the ring.
_RING_SEGS = 12

#: Sand per second into a band at full level, for Drift. Inflow goes with the
#: square of the level, as in ``Dune``, so a loud band gains far more than
#: twice what a band half as loud does. See :func:`jp_drift`.
_DRIFT_FEED = 2.6

#: Drift's drainage, per second: ``_DRIFT_DRAIN_LIN · h + _DRIFT_DRAIN_SQ · h²``
#: for a pile of height ``h``.
#:
#: Height-dependent rather than a fixed rate. A fixed rate is a cutoff: it
#: took 0.35 of a unit a second out of every band, so anything steadily under
#: a level of about 0.47 sank to nothing and the quiet half of every mix was
#: an empty panel. A drain that grows with the pile has no cutoff — every band
#: settles *somewhere* (steady 0.2 shows about 0.11, 0.35 about 0.32, 0.5
#: about 0.60) — and still cannot accumulate, because the taller a pile gets
#: the faster it loses sand.
#:
#: The linear term sets how fast silence clears: the tail of an emptying pile
#: decays at ``_DRIFT_DRAIN_LIN``, which empties a full panel in about three
#: and a half seconds. The square term sets where avalanches begin: inflow and
#: drainage balance at the angle of repose for a steady level of
#: ``sqrt((0.9 + 0.3) / 2.6)`` ≈ 0.68, so from about there up a band keeps
#: toppling, and below it a band holds a height.
_DRIFT_DRAIN_LIN = 0.9
_DRIFT_DRAIN_SQ = 0.3

#: Seconds a band that has just *received* spill must wait before it can
#: collapse itself. Under loud input every band sits at the angle of repose,
#: so without this a collapse tipped its neighbours over on the very next
#: frame, they tipped it back, and two thirds of all collapses were that
#: ping-pong: measured on sustained loud input, 68–77% of collapses followed a
#: neighbour's collapse one frame earlier, and the tops of the columns jumped
#: every one or two frames.
_DRIFT_SPILL_REST = 0.2

#: How long a column may stand past the angle of repose waiting for a beat.
#: A pile over the edge topples on a beat — so on bass-heavy music the
#: avalanches land on the kicks, and a cascade steps outward a column per beat
#: — and if no beat comes it topples anyway after this long, so a sustained
#: loud pad cannot leave a column standing full. Longer than a beat at 100 bpm,
#: or the clock beats the music to it.
_DRIFT_HOLD = 0.6

#: Onset strength that can topple a column, and how long after it lands a
#: column may still topple on it. Below the strength (a hat, a ghost note) the
#: pile keeps waiting. The window exists because the onset arrives before the
#: sand does: a kick is detected on its first frame, and the pile it feeds
#: crosses the edge a moment later — tested on the onset frame alone, bass
#: music produced no avalanches at all.
_DRIFT_TRIGGER = 0.5
_DRIFT_ARMED_S = 0.3

#: Seconds an avalanche takes to pour out. The sand leaves the collapsing
#: column and lands on its neighbours over this long, so the top of the column
#: visibly slides down and the columns beside it visibly rise. It used to be
#: moved in a single frame: the column teleported down, and what made it an
#: avalanche was left to a colour flash.
_DRIFT_POUR_S = 0.18

#: How long a column rests after an avalanche lands before it takes sand
#: again — long enough to read as the landing, short enough not to read as a
#: pause. See the feed in :func:`jp_drift`.
_DRIFT_SETTLE_S = 0.08

#: How fast the tint on moving sand fades once the sand stops moving. The tint
#: marks sand that is *actually* moving — a column pouring out and the columns
#: it pours into — rather than a flash fired at the moment of collapse, which
#: under loud input was retriggered faster than it faded and kept the panel red.
_DRIFT_FLASH_TAU = 0.12


def _drift_rest(levels: np.ndarray) -> np.ndarray:
    """The height a Drift pile settles at under a steady level.

    Inflow ``F·level²`` balances drainage ``a·h + b·h²`` where
    ``h = (sqrt(a² + 4bF·level²) - a) / 2b``. Capped just under the angle of
    repose, so switching to the mode mid-song does not open on every loud
    column collapsing at once.
    """
    a, b, f = _DRIFT_DRAIN_LIN, _DRIFT_DRAIN_SQ, _DRIFT_FEED
    lv = np.asarray(levels, dtype=np.float64)
    if b <= 0.0:
        rest = f * lv * lv / a
    else:
        rest = (np.sqrt(a * a + 4.0 * b * f * lv * lv) - a) / (2.0 * b)
    return np.minimum(rest, 0.98)


def recede_index(palette) -> int:
    """The ramp index an unlit element should be drawn in, measured.

    "Draw it at the bottom of the ramp" is the obvious answer and it is wrong
    in both directions. On about half the built-ins the low anchor is the
    *most* visible colour on the ramp — gruvbox's index 0 is a contrast of 7.1
    against its background and index 63 is 4.3 — so the thing meant to recede
    is the loudest thing on screen. On most of the rest index 0 is nearly the
    background itself (plasma 1.34, sapphire 1.19), and a small glyph in it
    simply is not there.

    So pick by measurement: the index whose contrast against ``theme.bg`` is
    closest to :data:`_RECEDE_CONTRAST`. That is the quietest available colour
    on a theme whose whole ramp is loud, and a colour that is still visible on
    a theme whose ramp fades into its background. Recomputed per call, because
    an animated theme's ramp moves every frame; it is 64 entries.
    """
    return int(np.argmin(np.abs(bg_contrast(palette) - _RECEDE_CONTRAST)))


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


def bulb_thresholds(rows: int) -> np.ndarray:
    """The level each bulb lights above, bottom bulb first.

    Bulb *k* lights when ``level > 2k / rows`` — and never below
    :data:`_QUIET`, which is what lets the bottom bulb go dark. The one place
    this is decided: :func:`bar_panel` and :func:`crest_bulb` both read it, and
    they have to agree to the ulp (see :func:`crest_bulb`).
    """
    return np.maximum(2 * np.arange(bulb_count(rows)) / rows, _QUIET)


def crest_bulb(levels: np.ndarray, rows: int) -> np.ndarray:
    """Index of the topmost lit bulb for each level, or -1 for none.

    This counts the same comparison :func:`bar_panel` makes rather than
    inverting it into ``ceil(level * rows / 2) - 1``, which is the closed form
    and is wrong about one level in forty. The two arithmetic paths disagree
    at exactly the boundaries: a level that is a couple of ulps above ``34/40``
    lights the bulb, while the closed form rounds ``level * 40 / 2`` back down
    to ``17.0`` and reports the bulb below it. Half a screen away that is a
    trail detaching one bulb inside its own bar, which looks like a rendering
    glitch and is impossible to find by reading either function on its own.
    """
    thresh = bulb_thresholds(rows)
    return (np.asarray(levels)[:, None] > thresh[None, :]).sum(axis=1).astype(np.int32) - 1


def ladder_colours(ctx: Ctx, rows: int) -> np.ndarray:
    """Ramp index of each screen row of a ladder: hot at the top."""
    return ctx.ramp(np.linspace(1.0, _LIT_FLOOR, rows))


def bar_panel(ctx: Ctx, rows: int, levels: np.ndarray, active: np.ndarray,
              heat: np.ndarray | None = None, unlit: bool = True):
    """The ladder itself — ``(codes, cidx)`` of shape ``(rows, ctx.w)``.

    ``levels`` and ``active`` are per *column*, already gathered through
    :func:`band_columns`, because every caller here needs the column mapping
    for something else as well and gathering it twice would be waste.

    ``heat`` is an optional per-column multiplier on where the lit bulbs land
    on the ramp — Drift pins a collapsing column to the top of it. Values above
    1 are fine, since :meth:`Ctx.ramp` clips. Leave it out and the ladder is
    coloured by height alone.
    """
    w = ctx.w
    on_row = bulb_rows(rows)
    panel = on_row[:, None] & active[None, :]

    # Row r of a bulb row holds bulb (rows - 1 - r) / 2; the rows between
    # bulbs never light, so what they are compared against does not matter.
    k = np.maximum(rows - 1 - np.arange(rows), 0) // 2
    thresh = bulb_thresholds(rows)[np.minimum(k, bulb_count(rows) - 1)][:, None]
    lit = panel & (levels[None, :] > thresh)

    if heat is None:
        hot = np.repeat(ladder_colours(ctx, rows)[:, None], w, axis=1)
    else:
        up = np.linspace(1.0, _LIT_FLOOR, rows)
        hot = ctx.ramp(up[:, None] * heat[None, :])

    # ``unlit`` draws the bulbs above the level as dots, which is what makes
    # this a panel with the power on rather than a bar chart. Drift turns it
    # off: its columns are a sandpile rather than a reading, and a full
    # lattice of dots behind falling sand reads as a grid over the picture.
    off = _OFF if unlit else SPACE
    codes = np.where(lit, _LED, np.where(panel, off, SPACE)).astype(np.int32)
    cidx = np.where(lit, hot, recede_index(ctx.palette)).astype(np.int32)
    return codes, cidx


def peak_bulbs(ctx: Ctx, codes, cidx, peaks: np.ndarray, levels: np.ndarray,
               rows: int) -> None:
    """Mark the bulb the peak is holding at, one weight under a lit bulb.

    Only where the peak has come clear of its bar. A peak level with the bar
    snaps to the bulb directly above the crest, and drawing that put a cap on
    every column all the time — a second outline of the bars that said nothing
    the bars were not already saying. So a peak shows once it is at least two
    bulbs above the crest, which is when a bar has *fallen away* from it.

    The marker takes the ladder's own colour at that height rather than a
    highlight, so it reads as *where that bar was*, not as a second thing
    competing with it.

    The row is snapped **down** onto the nearest bulb, never up. The bottom
    row is always a bulb row and bulb rows alternate, so ``r + 1`` is a bulb
    row whenever ``r`` is not and is always still on the panel — snapping up
    is what walked off the top of a full-scale column.
    """
    on_row = bulb_rows(rows)
    r = rows - 1 - np.floor(np.clip(peaks, 0.0, 0.999) * rows).astype(np.int32)
    r = np.where(on_row[r], r, r + 1)
    above = (rows - 1 - r) // 2 > crest_bulb(levels, rows) + 1
    cols = np.flatnonzero((peaks > _QUIET) & above)
    if cols.size == 0:
        return
    rr = r[cols]
    free = codes[rr, cols] == _OFF
    rr, cols = rr[free], cols[free]
    codes[rr, cols] = _PEAK
    cidx[rr, cols] = ladder_colours(ctx, rows)[rr]


@mode("JP Bars", group="jp",
      blurb="a segmented LED meter whose bars peel off and rise as they fall away")
def jp_bars(ctx: Ctx):
    """The meter, blended with the note roll out of ``Keys``.

    ``Bars`` shows a level as a height of ink and nothing else. A graphic
    equalizer's panel is fully there at all times — the bulbs above the level
    are unlit, not absent — and on top of that this keeps what the bar *did*:
    when a bar falls, the bulbs it is leaving keep their shape, detach, and
    climb the ladder as a thin trail, the way a struck note leaves the
    keyboard and travels up the roll in ``Keys``.

    The trail is the lightest lit thing on the panel, and it is short: it
    stops being drawn once it has decayed to a quarter of the level that shed
    it. A longer trail was the first thing that made this mode unreadable —
    on busy material every column above its bar was full of history, and the
    bar was one stack of ink among several.

    **Not VFD, and the difference is the whole point.** ``VFD`` is the other
    Japanese hi-fi display and it also puts something above the bar, but its
    phosphor decays *in place* — the trail marks where the bar was and stays
    there while it dims. These trails *travel*.
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
        "lv": np.zeros(n, dtype=np.float32),
        "acc": 0.0,
    })
    if st["roll"].shape != (nb, n):
        st["roll"] = np.zeros((nb, n), dtype=np.float32)
        st["was"] = np.full(n, -1, dtype=np.int32)
        st["lv"] = np.zeros(n, dtype=np.float32)
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

    # Emit only the bulbs a *rise* actually added. Emitting at the crest every
    # frame filled the whole panel with trail: the crest wanders, so a
    # continuous emitter smears across every bulb it visits. A held note sheds
    # nothing, and that is correct; it is not doing anything.
    crest = np.clip(crest_bulb(lv, rows), -1, nb - 1)
    kk = np.arange(nb, dtype=np.int32)[:, None]
    # The bulbs a *falling* bar has just vacated are the ones that peel off:
    # the top of the bar keeps its shape, detaches, and climbs the ladder
    # while the bar drops away underneath it. Seeding on the rise instead put
    # the trail above a bar that was still growing into it, so the two were
    # never apart and the peeling never read.
    released = (kk <= st["was"][None, :]) & (kk > crest[None, :])
    np.maximum(roll, released * st["lv"][None, :], out=roll)
    st["was"] = crest
    st["lv"] = lv.astype(np.float32)

    codes, cidx = bar_panel(ctx, rows, levels, active)
    peak_bulbs(ctx, codes, cidx, peaks, levels, rows)

    # Trails go in only on unlit bulbs, so the live bar and its peak always
    # win their own cells. Colour is the ladder's at that height — the fade
    # is carried by the cutoff, since a hue ramp has no "dimmer" to fade to.
    rws = bulb_row_index(rows)
    ghost = np.where(active[None, :], roll[:, col_band], np.float32(0.0))
    sub_c, sub_x = codes[rws], cidx[rws]
    show = (ghost > 0.25) & (sub_c == _OFF)
    sub_c[show] = _TRAIL
    sub_x[show] = np.broadcast_to(ladder_colours(ctx, rows)[rws][:, None], show.shape)[show]
    codes[rws], cidx[rws] = sub_c, sub_x
    return codes, cidx


@mode("JP Drift", group="jp",
      blurb="the meter as a sandpile — bulbs stack up, and pour into their neighbours on the beat")
def jp_drift(ctx: Ctx):
    """The meter, blended with the sandpile out of ``Dune``.

    The other modes in this family draw the level: read it, light that many
    bulbs. Here the bulbs are a running *balance*. Sand rains into each band in
    proportion to the square of its level and drains out faster the taller the
    pile is. A column that crosses the angle of repose waits there for a beat,
    then pours most of itself out sideways into the two bands next to it. A
    band steadily above about 0.68 keeps toppling; below that it settles at a
    height, so what you see is a meter for the quiet and medium part of a mix
    and a sandpile, keeping time, for the loud part.

    **Avalanches land on the beat.** A column past the edge topples on the next
    hit (or within a few frames after one, since the sand a kick feeds arrives
    a moment after the kick is detected), so on bass-heavy music the columns
    come down with the kicks, and a spill that tips a neighbour over makes it
    topple on a *later* beat — a cascade steps across the spectrum in time. A
    column no beat comes for topples anyway after :data:`_DRIFT_HOLD`, so a
    loud sustained pad cannot leave anything standing full.

    **An avalanche pours.** The sand leaves over :data:`_DRIFT_POUR_S`, so the
    column's top slides down bulb by bulb while its neighbours rise; the
    column takes no new sand while it pours and holds still for a moment when
    it lands. Only moving sand is tinted toward the top of the ramp — hardest on
    the column pouring out, lightly on the ones it lands on.

    **Drainage is what keeps it a meter.** Without it the pile only ever grew:
    measured on bass-heavy input, after thirty seconds the *emptiest* column
    was 55% lit, and once the music stopped the panel froze, full, forever. A
    fixed drain fixed that and broke the other end — it was a cutoff, and every
    band steadily under ≈ 0.47 sank to an empty column. The drain now grows
    with the height of the pile (see :data:`_DRIFT_DRAIN_LIN`), which leaves a
    small response to quiet input and still clears a full panel in about three
    and a half seconds of silence.

    **The pile opens at rest.** It is seeded at the height each band would
    settle at for the current level (:func:`_drift_rest`), so switching to the
    mode shows the steady picture at once instead of the whole panel climbing
    or sinking toward it — and a frozen spectrum below the avalanche level is a
    frozen picture, as a meter's should be.

    Before this, a collapse moved all its sand in one frame and a flash fired
    on the collapsing column and its neighbours. Under loud input two thirds of
    collapses were neighbours tipping each other over on consecutive frames,
    the tops of the columns jumped every frame or two, and the flash was
    retriggered faster than it faded, so the panel sat red. Sand that pours off
    either end of the spectrum is lost; the first and last bands used to spill
    into *themselves*.
    """
    w, h = ctx.w, ctx.h
    if w < 10 or h < 4:
        return empty(w, h)

    n = int(np.clip(ctx.n_display, 6, 40))
    lv = ctx.display_bands(n)

    st = ctx.scratch("jp_drift", lambda: {
        "h": _drift_rest(lv),
        "flash": np.zeros(n, dtype=np.float64),
        "rest": np.zeros(n, dtype=np.float64),
        "over": np.zeros(n, dtype=np.float64),
        "pour": np.zeros(n, dtype=np.float64),
        "rate": np.zeros(n, dtype=np.float64),
        "settle": np.zeros(n, dtype=np.float64),
        "beat_t": -9.0,
        "rng": np.random.default_rng(53),
    })
    if st["h"].shape[0] != n:
        st["h"] = _drift_rest(lv)
        for key in ("flash", "rest", "over", "pour", "rate", "settle"):
            st[key] = np.zeros(n, dtype=np.float64)
    pile, flash, rng = st["h"], st["flash"], st["rng"]

    dt = max(ctx.dt, 0.0)
    # A column that is pouring out takes no new sand until it has finished.
    # Fed while it poured, a loud band gained almost as fast as it lost — its
    # inflow is about the pour rate — so the avalanche barely descended, ended
    # well short of the rest height, and bounced back across a bulb boundary
    # the moment it stopped. It also holds still for a moment after landing —
    # neither fed nor drained: resumed on the next frame, the feed pushed the
    # last bulb the pour had just emptied straight back on, and left draining
    # it could slip one bulb and come straight back, either way a one-frame
    # blink at the bottom of an avalanche.
    settling = st["settle"] > 0.0
    feed = np.where((st["pour"] > 0.0) | settling, 0.0, lv * lv * _DRIFT_FEED)
    drain = np.where(settling, 0.0, (_DRIFT_DRAIN_LIN + _DRIFT_DRAIN_SQ * pile) * pile)
    pile += (feed - drain) * dt
    np.clip(pile, 0.0, 1.3, out=pile)

    rest, over, pour, rate = st["rest"], st["over"], st["pour"], st["rate"]
    np.maximum(rest - dt, 0.0, out=rest)
    np.maximum(st["settle"] - dt, 0.0, out=st["settle"])

    # ── toppling: past the angle of repose, on the beat ──
    edge = (pile > 1.0) & (pour <= 0.0)
    over[:] = np.where(edge, over + dt, 0.0)
    if ctx.onsets and ctx.onset_strength >= _DRIFT_TRIGGER:
        st["beat_t"] = ctx.t
    armed = ctx.t - st["beat_t"] <= _DRIFT_ARMED_S
    go = edge & (rest <= 0.0) & (armed | (over >= _DRIFT_HOLD))
    if go.any():
        idx = np.flatnonzero(go)
        pour[idx] = pile[idx] - 0.55 + rng.uniform(-0.03, 0.03, idx.size)
        rate[idx] = pour[idx] / _DRIFT_POUR_S
        over[idx] = 0.0

    # ── pouring: the sand slides out over a fifth of a second ──
    flowing = pour > 0.0
    moving = np.zeros(n, dtype=np.float64)
    if flowing.any():
        idx = np.flatnonzero(flowing)
        step = np.minimum(pour[idx], rate[idx] * dt)
        pour[idx] -= step
        landed = idx[pour[idx] <= 0.0]
        st["settle"][landed] = _DRIFT_SETTLE_S
        pile[idx] -= step
        moving[idx] = 1.0
        for side, ok in ((idx - 1, idx > 0), (idx + 1, idx < n - 1)):
            np.add.at(pile, side[ok], step[ok] * 0.35)
            # the columns it lands on warm only a little: the one pouring out
            # is the event, and tinting its neighbours as hard spread the
            # colour across most of a loud panel
            np.maximum.at(moving, side[ok], 0.25)
            np.maximum.at(rest, side[ok], _DRIFT_SPILL_REST)
        np.clip(pile, 0.0, 1.3, out=pile)
    np.maximum(flash * float(np.exp(-dt / _DRIFT_FLASH_TAU)), moving, out=flash)

    col_band, active = band_columns(w, n)
    levels = np.where(active, np.clip(pile, 0.0, 1.0)[col_band], 0.0)
    # ``heat`` above 1 is fine: ctx.ramp clips. Moving sand warms toward the
    # top of the ramp; still sand is coloured by height alone.
    heat = np.where(active, 1.0 + flash[col_band] * 0.9, 0.0)
    return bar_panel(ctx, h, levels, active, heat=heat, unlit=False)


@mode("JP Pulse", group="jp",
      blurb="the meter bent into a ring — a spoke of bulbs per band, chased on the beat")
def jp_pulse(ctx: Ctx):
    """The same ladder, once round the circle: one spoke of bulbs per band.

    ``Radial`` fills a smooth wedge out from the centre and ``Pulse`` blobs a
    disc. This is the bar graph in polar coordinates: each band gets a spoke
    with a dark gutter either side of it, the spoke carries the family's
    ladder of discrete bulbs growing outward from a hub, and the peak hangs
    out past the level exactly as it does on the flat panel.

    The ink weights are the flat panel's, in dots. A lit bulb is a solid arc
    ``_RING_SEGS`` of the spoke deep; the peak is a thin arc; an unlit bulb is
    a few dots on the spoke's centreline. The first version drew unlit bulbs
    with every dot set and told them apart from lit ones by colour alone — so
    on a hue ramp the dial read as one solid ring and the spokes that were
    actually lit were close to invisible.

    An onset launches a chaser: a bright band position that runs once round
    and fades, flaring each spoke it passes. *Only* an onset — the chaser
    used to be launched on a timer as well, which fired it in silence.

    **Where the frame goes.** The *radial* half — which bulb a dot belongs to,
    and whether it is on the solid, thin or dotted part of it — depends only
    on the geometry, so it is built once per size and cached. The *angular*
    half is a 512-entry table gathered with an index that spin advances by
    integer addition, so rotation is quantised to 0.7 degrees, below what the
    dot grid can show anyway.
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
    })
    st["spin"] = (st["spin"] + (0.04 + ctx.energy * 0.09) * ctx.dt) % 1.0

    if ctx.onsets and (ctx.t - st["born"].max()) > 0.12:
        slot = int(np.argmin(st["born"]))
        st["born"][slot] = ctx.t
        st["amp"][slot] = float(np.clip(0.35 + ctx.onset_strength * 0.9, 0.0, 1.0))

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
    def radial():
        segs = float(_RING_SEGS)
        r0 = max_r * 0.26
        r1 = max_r * 0.94
        pitch = (r1 - r0) / segs                       # dots per bulb
        s = (dist - np.float32(r0)) * np.float32(segs / (r1 - r0))
        floor_s = np.floor(s)
        f = s - floor_s
        inside = (s >= np.float32(0.0)) & (s < np.float32(segs))
        ok = inside & (f < np.float32(0.60))
        # at least a dot deep, or a small terminal loses the thin arcs entirely
        thin = inside & (f < np.float32(min(0.6, max(0.22, 1.05 / max(pitch, 1e-6)))))
        yy, xx = np.indices(dist.shape)
        # Every other dot, on a ladder deep enough for a thin arc to be thinner
        # than a lit one. Under three dots a bulb it is not — thin and solid
        # are the same depth — so the unlit arc thins by dropping to one dot in
        # four instead, or at 60x20 it was half as dense as a lit bulb and the
        # lit spokes did not stand out.
        if pitch >= 3.0:
            sparse = (yy + xx) % 2 == 0
        else:
            sparse = (yy % 2 == 0) & (xx % 2 == 0)
        return {
            "bulb": np.where(ok, floor_s, np.float32(-1.0)).astype(np.int16),
            "ok": ok,
            "thin": thin,
            "dotted": thin & sparse,
            # hot at the outer end of the spoke, which is the ring's version
            # of the ladder being hot at the top
            "warm": (np.float32(_LIT_FLOOR) + np.clip(s * np.float32(1.0 / segs), 0.0, 1.0)
                     * np.float32(1.0 - _LIT_FLOOR)).astype(np.float32),
            "idx0": (turn * np.float32(steps)).astype(np.int32) & (steps - 1),
        }

    geo = ctx.scratch("jp_dial", radial)

    # ── the angular half, as a 512-entry table ───────────────────────────────
    # ``-1`` marks the gutter between two spokes, so one gather answers both
    # "is this dot inside a spoke" and "how far up that spoke is lit". A peak
    # at or under the quiet floor is ``-1`` too, so silence draws no peak arcs
    # — it used to light a full hot ring at the hub.
    lv = ctx.display_bands(n).astype(np.float32)
    drive = np.clip(lv + chase * np.float32(0.55) + np.float32(ctx.pulse * 0.06), 0.0, 1.0)
    top_b = np.where(drive > _QUIET, np.floor(drive * _RING_SEGS), 0).astype(np.int16)
    pk = ctx.display_peaks(n).astype(np.float32)
    pk_b = np.where(pk > _QUIET,
                    np.floor(np.clip(pk, 0.0, 0.999) * _RING_SEGS), -1).astype(np.int16)

    pos = np.arange(steps, dtype=np.float32) * np.float32(n / steps)
    band = pos.astype(np.int32)
    gut = pos - band
    spoke = (gut > np.float32(0.16)) & (gut < np.float32(0.84))
    top_a = np.where(spoke, top_b[band], -1).astype(np.int16)
    # a peak in the same bulb as the level is part of the bar, not above it
    pk_a = np.where(spoke & (pk_b[band] > top_b[band]), pk_b[band], -1).astype(np.int16)

    shift = int(st["spin"] * steps) & (steps - 1)
    idx = (geo["idx0"] + shift) & (steps - 1)
    reach = top_a[idx]

    bulb = geo["bulb"]
    lit = geo["ok"] & (bulb < reach)
    peak = geo["thin"] & (bulb == pk_a[idx])
    # Unlit bulbs are a dotted centreline down each spoke, at every size. A
    # full-width dotted arc per bulb was a dot field the size of the dial, and
    # live in the terminal a bass hit's two or three active spokes did not
    # stand out of it — first at 60x20 on every theme, then at 120x40 on
    # flexoki-light, where the lit and unlit colours are both olive on cream.
    # A centreline still shows where every spoke and bulb is, and is too
    # little ink to compete with a spoke that is lit, whatever the ramp.
    centre = (np.abs(gut - np.float32(0.5)) < np.float32(0.06))[idx]
    unlit = geo["dotted"] & centre & (reach >= 0) & ~(bulb < reach)
    codes = pack_braille(lit | peak | unlit)

    # A cell holding any lit or peak dot takes that colour; a cell holding
    # only unlit dots takes the recede colour. A cell with no dots paints
    # nothing whatever its index says (see tests/test_blank_runs.py).
    heat = np.full(bulb.shape, np.float32(-1.0), dtype=np.float32)
    np.copyto(heat, geo["warm"], where=lit | peak)
    hot = cell_max(heat)
    cidx = np.where(hot < 0.0, recede_index(ctx.palette), ctx.ramp(hot)).astype(np.int32)
    return codes, cidx
