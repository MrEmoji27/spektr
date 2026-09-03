"""The JP family — one bar graph, four machines around it.

Every mode here is the *same* segmented LED meter: a column per band, a fixed
ladder of bulbs per column, the unlit bulbs drawn rather than left blank. That
ladder is :func:`bar_panel` and all four modes draw it. What differs is the
machine wrapped around it — a peak-hold meter, a travelling scan, a piano, a
ring — so the family is an experiment in what a bar graph can be *put inside*
rather than four unrelated pictures that happen to share a prefix.

That is the rule to keep if these are edited: the bars come first and stay
legible, and the mechanic is layered onto them. A JP mode where you can
no longer see a bar per band has stopped being one — the first version of this
file lost the bars in three of the four and had to be rewritten.

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
_FULL = ord("█")
_HALF_DOWN = ord("▄")
_RULE = ord("│")

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

#: After which white key of each seven-key octave a black key sits: C# after C,
#: D# after D, F# after F, G# after G, A# after A. There is deliberately no
#: entry for 2 (E) or 6 (B) — that is the point of the pattern, and the first
#: version of this file had ``[0, 2, 4, 5, 6]`` here, which put black keys
#: between E–F and B–C and none between C–D.
_BLACK_AFTER = np.array([0, 1, 3, 4, 5], dtype=np.int32)

#: Concurrent chaser slots on the ring. Two is enough to overlap a fast pair
#: of kicks (see particles.pulse for the same policy).
_KW_WAVES = 2

#: Bulbs along one spoke of the ring.
_RING_SEGS = 12


def bulb_rows(rows: int) -> np.ndarray:
    """Which of ``rows`` rows carry bulbs: the bottom one and every other up.

    The unlit rows between them are what make a column read as a stack of
    discrete elements instead of a solid bar, so they are structural rather
    than decorative — a ladder drawn on every row is ``Ladder``.
    """
    return ((rows - 1) - np.arange(rows)) % 2 == 0


def bar_panel(ctx: Ctx, rows: int, levels: np.ndarray, active: np.ndarray,
              heat: np.ndarray | None = None):
    """The ladder itself — ``(codes, cidx)`` of shape ``(rows, ctx.w)``.

    ``levels`` and ``active`` are per *column*, already gathered through
    :func:`band_columns`, because every caller here needs the column mapping
    for something else as well and gathering it twice would be waste.

    ``heat`` is an optional per-column 0..1 multiplier on where the lit bulbs
    land on the ramp. It exists for Sweep, which dims a column as the scan
    light moves away from it; leave it out and the ladder is coloured by
    height alone, which is the green-amber-red an equalizer is expected to be.
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
      blurb="a segmented LED meter, unlit bulbs and all, with peak bulbs that hang and fall")
def jp_bars(ctx: Ctx):
    """The family's plain meter, and the reference for the other three.

    ``Bars`` shows a level as a height of ink and nothing else, so a quiet
    passage is a nearly empty screen. A graphic equalizer's panel is fully
    there at all times — the bulbs above the level are unlit, not absent —
    and that is what is drawn here, with the peak flag lighting a bulb of its
    own that hangs where the level got to and falls back after it.
    """
    n = ctx.n_display
    col_band, active = band_columns(ctx.w, n)
    levels = np.where(active, ctx.display_bands(n)[col_band], 0.0)
    peaks = np.where(active, ctx.display_peaks(n)[col_band], 0.0)

    codes, cidx = bar_panel(ctx, ctx.h, levels, active)
    peak_bulbs(ctx, codes, cidx, peaks, ctx.h)
    return codes, cidx


@mode("JP Sweep", group="jp",
      blurb="the same meter, refreshed one band at a time by a travelling scan")
def jp_sweep(ctx: Ctx):
    """A sample-and-hold meter: the scan light is what updates a bar.

    Every other bar mode redraws every column from the current level every
    frame. Here a column only takes a new reading when the scan light passes
    over it, and then *holds* that height until the light comes back round —
    so what is on screen is the spectrum smeared across one lap of the sweep,
    oldest just ahead of the light and newest just behind it. The colour
    carries the age: a freshly-read column sits at the top of the ramp and
    cools as the light walks away from it, which is the comet the old
    swept-analyser displays drew.

    The head is a rule drawn down the dark rows *between* the bulbs and
    nowhere else, so the light passes behind the ladder. Brightening the
    bulbs in that column as well was the obvious thing and it was wrong: it
    lit the unlit ones too, and the head read as a full-height bar rather
    than as a light.
    """
    w, h = ctx.w, ctx.h
    if w < 10 or h < 4:
        return empty(w, h)

    n = ctx.n_display
    col_band, active = band_columns(w, n)
    lv = ctx.display_bands(n)

    st = ctx.scratch("jp_sweep", lambda: {
        "pos": 0.0,
        "hold": np.zeros(n, dtype=np.float64),
        "age": np.ones(n, dtype=np.float64),
    })
    if st["hold"].shape[0] != n:
        st["hold"] = np.zeros(n, dtype=np.float64)
        st["age"] = np.ones(n, dtype=np.float64)

    # Position is accumulated from dt rather than computed as t * rate: the
    # rate varies with the music, and multiplying a varying rate by the whole
    # elapsed time retroactively rewrites where the light has already been.
    # scenes._tunnel has the long version of why that bug is not theoretical.
    rate = 0.30 + ctx.drive * 0.45 + ctx.energy * 0.30      # laps per second
    prev = st["pos"]
    st["pos"] = (prev + rate * ctx.dt) % 1.0
    span = ((st["pos"] - prev) % 1.0) * n

    # Which bands the head crossed this frame, measured at the band's centre
    # and in band units ahead of where the light was — one modular subtraction
    # instead of a wrapped pair of slices.
    ahead = ((np.arange(n) + 0.5) - prev * n) % n
    crossed = ahead <= span
    st["hold"][crossed] = lv[crossed]
    st["age"][crossed] = 0.0
    st["age"] = np.minimum(st["age"] + ctx.dt / 1.7, 1.0)

    fresh = 0.25 + 0.75 * (1.0 - st["age"])
    levels = np.where(active, st["hold"][col_band], 0.0)
    codes, cidx = bar_panel(ctx, h, levels, active,
                            heat=np.where(active, fresh[col_band], 0.0))

    head = int(np.clip(st["pos"] * w, 0, w - 1))
    gap = ~bulb_rows(h)
    codes[gap, head] = _RULE
    cidx[gap, head] = RAMP_STEPS - 1
    return codes, cidx


@mode("JP Keys", group="jp",
      blurb="the meter standing on a piano — each bar rises out of the key it belongs to")
def jp_keys(ctx: Ctx):
    """A bar graph whose axis is a keyboard, not a ruler.

    ``Keys`` (scenes.py) is a note roll: a struck band spawns something that
    scrolls away and the keyboard is a two-row strip with no black keys. This
    is the other half of that idea — the keyboard is a real one, with the
    5-black/7-white octave pattern and the blacks set into the back of the
    whites, and what stands on it is the family's ladder, so a bar is read as
    "how hard is this key being held" rather than as a bar at position *i*.

    The keys are driven by a level with a release tail rather than by the raw
    band, so a key that is let go decays over about a third of a second
    instead of snapping shut on the next quiet frame. On a held note the tail
    has already reached the level, which is why this mode is not in the
    audit's self-animating set: it is a bar chart, and a bar chart of a
    constant signal is supposed to be constant.
    """
    w, h = ctx.w, ctx.h
    if w < 16 or h < 9:
        return empty(w, h)

    n = min(ctx.n_display, max(4, w // 3))
    col_band, active = band_columns(w, n)
    lv = ctx.display_bands(n)

    st = ctx.scratch("jp_keys", lambda: {"press": np.zeros(n, dtype=np.float64)})
    if st["press"].shape[0] != n:
        st["press"] = np.zeros(n, dtype=np.float64)
    st["press"] = np.maximum(st["press"] * float(np.exp(-ctx.dt / 0.30)), lv)
    press = st["press"]

    # Three rows of keyboard: the black keys occupy the back two, the whites
    # all three, and the blacks are punched into the whites rather than drawn
    # above them, which is what sets them behind the front edge of the keys.
    top = h - 3
    codes = np.full((h, w), SPACE, dtype=np.int32)
    cidx = np.zeros((h, w), dtype=np.int32)

    bars, bar_cidx = bar_panel(ctx, top, np.where(active, press[col_band], 0.0), active)
    codes[:top] = bars
    cidx[:top] = bar_cidx

    white = ctx.ramp(np.clip(press[col_band], 0.0, 1.0))
    idle = ctx.palette.index(0.34)
    down = active & (press[col_band] > 0.16)
    for row in (h - 3, h - 2, h - 1):
        codes[row, active] = _FULL
        cidx[row, active] = idle
        cidx[row, down] = white[down]

    # A black key sits over the boundary between two white keys, and only
    # between the pairs that have one. Its level is the mean of the two it
    # divides, so it lights with the region rather than being decoration.
    run0 = np.concatenate(([True], col_band[1:] != col_band[:-1])) & active
    starts = np.flatnonzero(run0)
    if starts.size >= 2:
        band_at = col_band[starts]
        adjacent = band_at[1:] == band_at[:-1] + 1
        pair = adjacent & np.isin(band_at[:-1] % 7, _BLACK_AFTER)
        edge = starts[1:][pair] - 1                  # the gutter column
        left = band_at[:-1][pair]
        if edge.size:
            span = max(1, min(3, int(np.min(np.diff(starts))) - 1))
            off = np.arange(span) - (span - 1) // 2
            cols = np.clip((edge[:, None] + off[None, :]).ravel(), 0, w - 1)
            blv = np.repeat((press[left] + press[left + 1]) * 0.5, span)
            bhot = np.where(blv > 0.16, ctx.ramp(np.clip(blv, 0.0, 1.0)),
                            ctx.palette.index(_IDLE))
            codes[h - 3, cols] = _FULL
            codes[h - 2, cols] = _HALF_DOWN
            cidx[h - 3, cols] = bhot
            cidx[h - 2, cols] = bhot
    return codes, cidx


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
