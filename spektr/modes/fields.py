"""Full-field modes: waterfall, plasma, and level meters."""

from __future__ import annotations

import math

import numpy as np

from ..analysis import resample_bands
from ..palette import RAMP_STEPS
from ..render import SHADES, SPACE, cell_max, cell_mean, pack_braille
from . import Ctx, empty, mode, spread

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


@mode("Plasma", group="fields", blurb="solid colour field, warped by the spectrum")
def plasma(ctx: Ctx):
    """Drawn with ``▀`` so each cell carries two colours — foreground for the
    top half, background for the bottom. That doubles the vertical resolution
    for free, which matters a lot for a smooth gradient field.

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

    rows2 = h * 2

    def geo():
        y = np.arange(rows2, dtype=np.float64)[:, None] / max(1, rows2 - 1)
        x = np.arange(w, dtype=np.float64)[None, :] / max(1, w - 1)
        return y, x

    y, x = ctx.scratch("plasma", geo)

    t = ctx.t
    # fractions rather than fixed indices — this mode has no business knowing
    # how many bands the analyser happens to produce, and docs/plugins.md tells
    # plugin authors exactly this
    lows = ctx.range(0.00, 0.20)
    mids = ctx.range(0.25, 0.62)
    highs = ctx.range(0.70, 1.00)

    v = (
        np.sin((x * 3.0 + t * 0.7) * (1.0 + lows * 1.4))
        + np.sin((y * 2.5 - t * 0.5) * (1.0 + mids * 1.2))
        + np.sin(((x + y) * 2.0 + t * 0.9))
        + np.sin(np.sqrt((x - 0.5) ** 2 * 2.2 + (y - 0.5) ** 2) * 7.0 - t * 2.2 * (0.4 + highs * 2.0))
    )
    field = (v + 4.0) / 8.0
    field = np.clip(field * (0.35 + ctx.energy * 1.5), 0.0, 1.0)

    idx = ctx.ramp(field)
    codes = np.full((h, w), _UPPER_HALF, dtype=np.int32)
    return codes, idx[0::2], idx[1::2]


@mode("Chladni", group="fields", blurb="nodal interference pattern, plate modes set by the dominant pitch")
def chladni(ctx: Ctx):
    """A vibrating-plate figure, not a warped colour field.

    Plasma is a smooth continuous field; this is the opposite kind of
    pattern — an interference figure with hard nodal lines, the shape sand
    takes on a real Chladni plate, where it collects at the nodes (zero
    motion) and gets shaken off everywhere else. The two integer mode
    numbers that decide the figure's shape track the spectrum's centre of
    mass, so a bass-heavy passage gives a coarse few-line figure and a
    bright treble-heavy one gives a dense, many-celled figure — the same way
    sweeping a real plate's drive frequency snaps it between resonant
    figures. Loudness sharpens the lines rather than moving them, the way a
    harder-driven plate throws sand into tighter bands.

    The calmest of the three. This one is discrete and physical: it holds a
    figure and jumps to the next. ``Chladni Flow`` sweeps continuously between
    them instead, and ``Chladni Extreme`` does that *and* escalates as a track
    sustains energy.

    Two things here exist only to keep make_strips cheap, the same lesson
    Plasma's docstring already tells: this uses the same two-colour ``▀``
    trick, and unlike Plasma's smooth field, ``nodal`` is a sharp,
    non-monotonic difference-of-products — measured at ~50 distinct ramp
    values per row at 400x100, versus Plasma's handful, because the
    interference pattern has many local extrema even at low mode numbers.
    ``m``/``n`` are capped lower than a "real" Chladni figure would use
    (halving them only cut cost by ~15%, so frequency wasn't the main
    driver), and ``nodal`` is quantised to 10 buckets before ramping so
    neighbouring pixels collapse into the same colour index more often. That
    combination measured worst-case ~7ms at 400x100, down from ~9-14ms and
    frequently over the 11ms slow-mode threshold — which read as exactly the
    "pops up, lags, then catches up" pattern it was reported as, since
    crossing that threshold makes the widget reuse every other frame.
    """
    w, h = ctx.w, ctx.h
    if w < 4 or h < 4:
        return empty(w, h)

    rows2 = h * 2

    def geo():
        # Cell *centres*, not corners. Sampling the exact boundary hits
        # sin(k*pi*0) and sin(k*pi*1), both identically zero for integer modes,
        # so the whole outer row and column come out as a perfect node and the
        # figure gets a solid lit frame around it that reads as a UI border
        # rather than as part of the plate.
        y = (np.arange(rows2, dtype=np.float64)[:, None] + 0.5) / rows2
        x = (np.arange(w, dtype=np.float64)[None, :] + 0.5) / w
        return y, x

    y, x = ctx.scratch("chladni_geo", geo)

    bands8 = ctx.display_bands(8).astype(np.float64)
    total = float(bands8.sum())
    centroid = float((bands8 * np.arange(8)).sum() / total / 7.0) if total > 1e-9 else 0.0
    highs = ctx.range(0.6, 1.0)

    # Seeded from the audio that is actually playing, not from a fixed 0.3/0.5.
    # A hardcoded start means switching to this mode shows the figure sliding
    # from someone else's default to the right one over the first half second —
    # and with integer modes that slide is a visible snap or two through
    # figures the music never asked for. It also made the mode read as
    # self-animating to the audit, because a settling ease keeps changing the
    # picture on a frozen spectrum.
    st = ctx.scratch("chladni_ease", lambda: {"c": centroid, "e": highs})
    st["c"] += (centroid - st["c"]) * min(1.0, ctx.dt / 0.35)
    st["e"] += (highs - st["e"]) * min(1.0, ctx.dt / 0.35)

    # Integer mode numbers. These are the physically real ones — a plate only
    # resonates at whole mode counts — so the figure holds a shape and *snaps*
    # to the next one when the spectrum moves far enough, the way a real plate
    # jumps between figures as you sweep the drive frequency. ``Chladni
    # Extreme`` is the version that morphs continuously through the fractional
    # values in between; this one keeps the discrete behaviour, which is the
    # whole reason a Chladni figure looks like a resonance and not like a
    # warped field.
    #
    # The figure being discrete is also why the plate has to stay alive some
    # other way: the *sharpness* below is continuous, so loudness keeps
    # changing the picture between snaps rather than leaving it frozen until
    # the next one.
    m = float(2 + round(st["c"] * 4.0))    # 2..6
    n = float(3 + round(st["e"] * 4.0))    # 3..7
    if n == m:                             # m == n cancels the figure to zero
        n += 1.0

    xr = x
    yr = y

    z = np.sin(m * math.pi * xr) * np.sin(n * math.pi * yr) - np.sin(
        n * math.pi * xr
    ) * np.sin(m * math.pi * yr)

    sharpness = 1.4 + ctx.energy * 3.2
    nodal = np.clip(1.0 - np.abs(z) * sharpness, 0.0, 1.0)
    nodal *= nodal

    # Quantised before ramping, for the ``make_strips`` reason in the
    # docstring above. Twelve buckets rather than ten: the figure is smooth
    # curves and the extra steps visibly soften the banding across a broad
    # nodal region, at no measurable cost.
    #
    # Deliberately *not* dithered into a sand texture, which is the obvious
    # thing to try given what a Chladni plate physically is. It was tried:
    # at terminal resolution a nodal band is only a few cells across, so
    # thresholding the field against a noise mask leaves isolated speckle
    # with no curve left to read — and restricting the dither to the fringe
    # while keeping the ridge solid still broke the thin parts of the
    # figure, which is most of it. The clean field is the better picture.
    nodal = np.round(nodal * 12.0) * (1.0 / 12.0)

    idx = ctx.ramp(nodal)
    codes = np.full((h, w), _UPPER_HALF, dtype=np.int32)
    return codes, idx[0::2], idx[1::2]


@mode(
    "Chladni Flow",
    group="fields",
    blurb="a plate figure melting continuously from one resonance into the next",
)
def chladni_flow(ctx: Ctx):
    """A vibrating-plate figure that morphs rather than snapping.

    The middle of the three. ``Chladni`` uses integer mode numbers, so it
    holds a figure and jumps to the next one, which is what a real plate
    does. ``Chladni Extreme`` sweeps continuously *and* escalates as a
    track sustains energy. This one is the plain continuous sweep: no
    snapping, no charge, no harmonic — one figure melting into the next at
    a fixed slow spin. It is the original, restored unchanged from 966aefb.

    Plasma is a smooth continuous field; this is the opposite kind of
    pattern — an interference figure with hard nodal lines, the shape sand
    takes on a real Chladni plate, where it collects at the nodes (zero
    motion) and gets shaken off everywhere else. The two integer mode
    numbers that decide the figure's shape track the spectrum's centre of
    mass, so a bass-heavy passage gives a coarse few-line figure and a
    bright treble-heavy one gives a dense, many-celled figure — the same way
    sweeping a real plate's drive frequency snaps it between resonant
    figures. Loudness sharpens the lines rather than moving them, the way a
    harder-driven plate throws sand into tighter bands.

    Two things here exist only to keep make_strips cheap, the same lesson
    Plasma's docstring already tells: this uses the same two-colour ``▀``
    trick, and unlike Plasma's smooth field, ``nodal`` is a sharp,
    non-monotonic difference-of-products — measured at ~50 distinct ramp
    values per row at 400x100, versus Plasma's handful, because the
    interference pattern has many local extrema even at low mode numbers.
    ``m``/``n`` are capped lower than a "real" Chladni figure would use
    (halving them only cut cost by ~15%, so frequency wasn't the main
    driver), and ``nodal`` is quantised to 10 buckets before ramping so
    neighbouring pixels collapse into the same colour index more often. That
    combination measured worst-case ~7ms at 400x100, down from ~9-14ms and
    frequently over the 11ms slow-mode threshold — which read as exactly the
    "pops up, lags, then catches up" pattern it was reported as, since
    crossing that threshold makes the widget reuse every other frame.
    """
    w, h = ctx.w, ctx.h
    if w < 4 or h < 4:
        return empty(w, h)

    rows2 = h * 2

    def geo():
        y = np.arange(rows2, dtype=np.float64)[:, None] / max(1, rows2 - 1)
        x = np.arange(w, dtype=np.float64)[None, :] / max(1, w - 1)
        return y, x

    y, x = ctx.scratch("chladni_flow_geo", geo)

    bands8 = ctx.display_bands(8).astype(np.float64)
    total = float(bands8.sum())
    centroid = float((bands8 * np.arange(8)).sum() / total / 7.0) if total > 1e-9 else 0.0
    highs = ctx.range(0.6, 1.0)

    st = ctx.scratch("chladni_flow_ease", lambda: {"c": 0.3, "e": 0.5})
    st["c"] += (centroid - st["c"]) * min(1.0, ctx.dt / 0.35)
    st["e"] += (highs - st["e"]) * min(1.0, ctx.dt / 0.35)

    # Mode numbers are continuous, not snapped to integers. Integer modes are
    # the physically real ones, but stepping between them makes the whole
    # figure change shape between one frame and the next — a hard cut, on a
    # mode whose appeal is watching the pattern reorganise. Sweeping through
    # the fractional values in between morphs one figure into the next, and
    # the interference pattern stays a plausible plate figure throughout.
    m = 2.0 + st["c"] * 4.4    # 2.0 .. 6.4
    n = 3.2 + st["e"] * 4.4    # 3.2 .. 7.6

    # a slow spin keeps a held tone's figure visibly alive rather than frozen
    ang = ctx.t * 0.06
    cs, sn = math.cos(ang), math.sin(ang)
    xr = (x - 0.5) * cs - (y - 0.5) * sn + 0.5
    yr = (x - 0.5) * sn + (y - 0.5) * cs + 0.5

    z = np.sin(m * math.pi * xr) * np.sin(n * math.pi * yr) - np.sin(
        n * math.pi * xr
    ) * np.sin(m * math.pi * yr)

    sharpness = 1.4 + ctx.energy * 3.2
    nodal = np.clip(1.0 - np.abs(z) * sharpness, 0.0, 1.0) ** 2

    # Quantised before ramping, for the ``make_strips`` reason in the
    # docstring above. Twelve buckets rather than ten: the figure is smooth
    # curves and the extra steps visibly soften the banding across a broad
    # nodal region, at no measurable cost.
    #
    # Deliberately *not* dithered into a sand texture, which is the obvious
    # thing to try given what a Chladni plate physically is. It was tried:
    # at terminal resolution a nodal band is only a few cells across, so
    # thresholding the field against a noise mask leaves isolated speckle
    # with no curve left to read — and restricting the dither to the fringe
    # while keeping the ridge solid still broke the thin parts of the
    # figure, which is most of it. The clean field is the better picture.
    nodal = np.round(nodal * 12.0) * (1.0 / 12.0)

    idx = ctx.ramp(nodal)
    codes = np.full((h, w), _UPPER_HALF, dtype=np.int32)
    return codes, idx[0::2], idx[1::2]


def _rot_sines(k: float, cs: float, sn: float, ax: np.ndarray, by: np.ndarray):
    """``sin(k*pi*xr)`` and ``sin(k*pi*yr)`` without a transcendental over the field.

    The rotated coordinates separate. With ``ax = x - 0.5`` a row vector and
    ``by = y - 0.5`` a column vector::

        k*pi*xr = (k*pi*cs)*ax + k*pi/2  +  (-k*pi*sn)*by  =  U(x) + V(y)
        k*pi*yr = (k*pi*sn)*ax + k*pi/2  +  ( k*pi*cs)*by  =  P(x) + Q(y)

    so ``sin(U + V) = sin U cos V + cos U sin V`` turns one sine over
    ``rows2 x w`` cells into four over ``w`` plus four over ``rows2``, and two
    broadcast multiplies. Same identity ``Flame`` uses on its wobble, and the
    reason this mode can afford a harmonic layered on top: at 400x100 the
    field is 80k cells and the two vectors are 400 and 200.
    """
    kp = k * math.pi
    u = kp * cs * ax + kp * 0.5
    v = -kp * sn * by
    pp = kp * sn * ax + kp * 0.5
    q = kp * cs * by
    return (
        np.sin(u) * np.cos(v) + np.cos(u) * np.sin(v),
        np.sin(pp) * np.cos(q) + np.cos(pp) * np.sin(q),
    )


@mode(
    "Chladni Extreme",
    group="fields",
    blurb="a plate driven past its modes - morphs, escalates, and snaps on the beat",
)
def chladni_extreme(ctx: Ctx):
    """``Chladni``, driven far past what a real plate would survive.

    Four differences from its siblings, in order of how much they matter.

    **It snaps on the beat.** This is the one built for four-on-the-floor and
    for funk. A kick drives a ``punch`` that decays over about 160 ms, and
    while it is up the nodal lines tighten hard and the plate lurches round -
    so a hit lands as a visible crack through the pattern rather than as a
    slightly brighter frame. ``Chladni Flow`` has no concept of a beat at all,
    which is the main reason to reach for this one.

    Punch deliberately does not touch the mode numbers. Bumping them per kick
    draws a *different* figure each time and the plate reads as scrambling
    rather than as being struck; shape belongs to the spectrum and to charge,
    and the beat gets crispness and rotation applied to whatever is there.

    How hard the punch hits is gated on ``ctx.flatness``. A drum-led groove is
    spectrally noise-like and scores high; a sustained pad scores low, so the
    same kick energy under a pad produces a fraction of the snap. Without that
    gate the mode reads as twitchy on anything with a bassline rather than as
    percussive on anything with drums.

    **The modes are continuous.** ``Chladni`` snaps between whole mode numbers
    because that is what a plate physically does. This sweeps the fractional
    values between them, so one figure melts into the next.

    **It escalates.** A ``charge`` builds while the track stays energetic and
    bleeds away when it doesn't, over about eight seconds either way - slow
    enough that no single hit buys it and no quiet bar loses it. Charge folds
    in a second interference term at higher mode numbers, speeds the spin and
    tightens the lines. Punch is the bar; charge is the track.

    **It reacts harder.** Easing is 0.35 s -> 0.12 s, so the figure chases the
    spectrum instead of drifting after it.

    Cost. Trig is the expensive part of any Chladni and it is separable under
    rotation - see ``_rot_sines``. Mode numbers are bounded by what the grid
    can resolve: a figure with k cycles across n cells needs several cells per
    cycle to read as a curve, and an early cut that doubled them for the
    harmonic rendered a fully charged plate as pure speckle. Same lesson
    Plasma's docstring records about its swirl frequencies.
    """
    w, h = ctx.w, ctx.h
    if w < 4 or h < 4:
        return empty(w, h)

    rows2 = h * 2

    def geo():
        # Centred once: every use of the grid here is relative to the middle.
        by = (np.arange(rows2, dtype=np.float64)[:, None] / max(1, rows2 - 1)) - 0.5
        ax = (np.arange(w, dtype=np.float64)[None, :] / max(1, w - 1)) - 0.5
        return by, ax

    by, ax = ctx.scratch("chladni_x_geo", geo)

    bands8 = ctx.display_bands(8).astype(np.float64)
    total = float(bands8.sum())
    centroid = float((bands8 * np.arange(8)).sum() / total / 7.0) if total > 1e-9 else 0.0
    highs = ctx.range(0.6, 1.0)

    st = ctx.scratch(
        "chladni_x",
        lambda: {
            "c": centroid, "e": highs, "charge": 0.0, "spin": 0.0,
            "punch": 0.0, "hit_t": -99.0,
            "level": ctx.energy,
        },
    )
    # Chases rather than drifts: a third of Chladni's time constant.
    st["c"] += (centroid - st["c"]) * min(1.0, ctx.dt / 0.12)
    st["e"] += (highs - st["e"]) * min(1.0, ctx.dt / 0.12)

    # -- the beat --
    # A detected onset is the hit. The analyser picks peaks in the spectral
    # flux across the whole band plan, so a snare snaps the plate as well as
    # a kick, and a swell in level — the thing the old fast/slow envelope
    # pair over the bass band fired on — does not. The refractory keeps a
    # single hit from paying for the plate twice at a low frame rate.
    # Percussion is spectrally flat, a pad is not. The same kick energy under
    # a sustained chord should not throw the plate around.
    groove = float(np.clip((ctx.flatness - 0.35) / 0.45, 0.0, 1.0))
    if ctx.onsets and (ctx.t - st["hit_t"]) > 0.09:
        st["hit_t"] = ctx.t
        st["punch"] = min(
            1.4, st["punch"] + (0.45 + ctx.onset_strength * 1.6) * (0.35 + groove)
        )
    # Decay in seconds. 160 ms clears well before the next sixteenth at any
    # tempo worth watching, so hits read as separate cracks rather than smear.
    st["punch"] *= math.exp(-max(ctx.dt, 0.0) / 0.16)
    punch = st["punch"]

    # Charge is driven by a *smoothed* level, not the instantaneous one, and
    # that is not a refinement — without it the feature is dead on exactly the
    # music this mode is for. Energy on a four-on-the-floor track swings across
    # any fixed threshold once per kick (measured 0.21 to 0.32 on a 128 BPM
    # loop against a 0.22 line), so the charge was pushed down as often as up
    # and sat at 0.00 forever. A 1.5 s envelope asks "is this track busy",
    # which is the question, instead of "is this instant loud".
    st["level"] += (ctx.energy - st["level"]) * min(1.0, ctx.dt / 1.5)
    drive = 1.0 if st["level"] > 0.22 else -1.0
    st["charge"] = min(1.0, max(0.0, st["charge"] + drive * ctx.dt / 8.0))
    charge = st["charge"]

    lim_m = max(2.0, w / 9.0)
    lim_n = max(2.0, rows2 / 9.0)
    # Punch deliberately does *not* reach the mode numbers. Bumping them on
    # every kick redraws a different figure each time, and at 0.7-0.9 of a
    # mode that is a big enough jump that the plate reads as scrambling rather
    # than as being struck. Shape is the spectrum's and charge's; the beat gets
    # crispness and rotation, which land on the figure that is already there.
    m = min(1.8 + st["c"] * 5.2 + charge * 0.8, lim_m)
    n = min(2.6 + st["e"] * 5.2 + charge * 1.0, lim_n)

    # Spin accumulates through dt rather than reading ctx.t * rate, because the
    # rate is audio-driven: against ctx.t a change in speed retroactively
    # rewrites the whole history and the plate teleports. Same trap Tunnel's
    # docstring documents. The punch term is what makes it lurch on the beat.
    st["spin"] = (
        st["spin"] + (0.06 + charge * 0.55 + punch * 1.9) * max(ctx.dt, 0.0)
    ) % (2 * math.pi)
    cs, sn = math.cos(st["spin"]), math.sin(st["spin"])

    mx, my = _rot_sines(m, cs, sn, ax, by)
    nx, ny = _rot_sines(n, cs, sn, ax, by)
    z = mx * ny - nx * my

    # Only paid for when it shows. Below this the term contributes less than
    # one ramp bucket, so computing it would be more work over the whole field
    # for a picture nobody can tell apart.
    if charge > 0.05:
        # 1.7x rather than 2x, and still clamped to the resolvable limit: the
        # harmonic puts fine structure *inside* the figure's cells, and it can
        # only do that while it is still a figure itself.
        hm = min(m * 1.7, lim_m * 1.35)
        hn = min(n * 1.7, lim_n * 1.35)
        hmx, hmy = _rot_sines(hm, cs, sn, ax, by)
        hnx, hny = _rot_sines(hn, cs, sn, ax, by)
        # Mixed in at 0.22 rather than 0.4. Measured as the fraction of
        # horizontally adjacent cells landing in the same ramp bucket - how
        # followable the curves are - a fully charged plate scores 0.757 with
        # no harmonic, 0.748 at 0.22 and 0.701 at 0.38. The last is a visible
        # slide toward speckle for 0.007 of reactivity.
        mix = 0.22 * charge
        z = z * (1.0 - mix) + (hmx * hny - hnx * hmy) * mix

    # Punch is worth more here than anywhere else: tightening the lines is what
    # turns a hit into a crack across the figure rather than a flash.
    # Charge tightens the lines, punch cracks them. Both are modest on purpose:
    # at charge*2.4 a fully built-up plate covered 25% of the screen and every
    # kick thinned it further, which reads as the picture dropping out rather
    # than as being struck. At 1.0 the charged figure stays legible (33%) and
    # the punch has somewhere to swing from -- a kick takes it to 25%, a
    # visible snap tighter without a blackout.
    sharpness = 1.4 + ctx.energy * 3.2 + charge * 1.0 + punch * 2.2
    nodal = np.clip(1.0 - np.abs(z) * sharpness, 0.0, 1.0)
    nodal *= nodal
    # Eight buckets, not the twelve its two siblings use, and done in place:
    # this is the heaviest mode in the app and over half its frame is the strip
    # builder, which pays per colour boundary. Eight costs almost nothing
    # visually because sharpness runs to 7 and beyond, and a sharp nodal field
    # is already nearly bimodal - most cells sit hard against 0 or 1 rather
    # than in the midtones the extra buckets would resolve.
    nodal *= 8.0
    np.round(nodal, 0, out=nodal)
    nodal *= 1.0 / 8.0

    idx = ctx.ramp(nodal)
    codes = np.full((h, w), _UPPER_HALF, dtype=np.int32)
    return codes, idx[0::2], idx[1::2]


@mode("VFD", group="fields", blurb="vacuum-fluorescent bargraph with phosphor afterglow")
def vfd(ctx: Ctx):
    """The other Japanese hi-fi display technology, next to Kenwood's LEDs.

    A vacuum-fluorescent display doesn't turn off instantly — the phosphor
    keeps glowing for a beat after the drive current cuts, so a fast
    transient leaves a fading trail above where the bar now sits instead of a
    hard edge. Reproduced literally: each frame draws a crisp new bar into a
    dot-resolution persistence buffer that decays exponentially, and dots
    stay lit — at fading brightness — until the decay carries them below the
    draw threshold.
    """
    w, h = ctx.w, ctx.h
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 4 or dc < 4:
        return empty(w, h)

    buf = ctx.scratch("vfd", lambda: np.zeros((dr, dc), dtype=np.float32))

    level = spread(ctx.display_bands(), dc)
    rows = np.arange(dr - 1, -1, -1, dtype=np.float32) / dr    # row 0 (top) -> ~1
    bar = (level[None, :] > rows[:, None]).astype(np.float32)

    decay = float(np.exp(-max(ctx.dt, 0.0) / 0.12))
    buf *= decay
    np.maximum(buf, bar, out=buf)

    dots = buf > 0.04
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(np.where(dots, buf, 0.0)))
    return codes, cidx


def _vu_level(ctx: Ctx) -> np.ndarray:
    """Per-channel loudness, weighted like a real VU rather than flat.

    Treble weighs more than bass, so the reading tracks perceived loudness
    instead of whatever the kick drum is doing. Shared by both meter modes so
    they can never disagree about how loud the signal is.
    """
    weight = np.linspace(0.55, 1.25, len(ctx.bands_l))
    return np.array(
        [
            float(np.clip((ctx.bands_l * weight).mean() * 1.5, 0.0, 1.0)),
            float(np.clip((ctx.bands_r * weight).mean() * 1.5, 0.0, 1.0)),
        ]
    )


def _peak_hold(st: dict, lvl: np.ndarray, dt: float) -> np.ndarray:
    """Instant attack, held, then a slow fall — the mechanical meter's ballistic."""
    rise = lvl >= st["peak"]
    st["peak"][rise] = lvl[rise]
    st["hold"][rise] = 0.9
    st["hold"][~rise] -= dt
    falling = st["hold"] <= 0
    st["peak"][falling] = np.maximum(lvl[falling], st["peak"][falling] - 0.7 * dt)
    return st["peak"]


@mode("Needle", group="stereo", blurb="analogue VU — one sweeping needle, one red zone")
def needle(ctx: Ctx):
    """A hardware VU meter: pivot, arc, red zone, and a peak flag.

    The other meters are bar graphs. This is the dial — the level is an angle,
    not a length, and the red zone sits where it does on the glass rather than
    being implied by colour. Geometry is cached per size, so a frame is three
    comparisons over the (small) text grid.
    """
    w, h = ctx.w, ctx.h
    if w < 20 or h < 6:
        return empty(w, h)

    def geo():
        cx = (w - 1) / 2.0
        cy = h - 1.0
        # halve x: a text cell is about twice as tall as it is wide, and
        # without this the arc comes out as an ellipse
        dx = (np.arange(w, dtype=np.float64) - cx) * 0.5
        dy = cy - np.arange(h, dtype=np.float64)
        dist = np.sqrt(dx[None, :] ** 2 + dy[:, None] ** 2)
        ang = np.arctan2(dy[:, None] + np.zeros_like(dx)[None, :], dx[None, :] + np.zeros_like(dy)[:, None])
        radius = max(2.0, min(h - 1.5, (w * 0.5) * 0.5 - 1.0))
        return dist, ang, radius

    dist, ang, radius = ctx.scratch("needle", geo)
    st = ctx.scratch("needle_peak", lambda: {"peak": np.zeros(2), "hold": np.zeros(2)})

    lvl = _vu_level(ctx)
    peak = _peak_hold(st, lvl, ctx.dt)
    level = float(lvl.mean())
    peak_level = float(peak.max())

    lo, hi = math.radians(160.0), math.radians(20.0)   # left = empty, right = full
    sweep = lo + (hi - lo) * level
    peak_sweep = lo + (hi - lo) * min(1.0, peak_level)
    red_from = lo + (hi - lo) * 0.78

    on_arc = (np.abs(dist - radius) < 0.75) & (ang <= lo) & (ang >= hi)
    red = on_arc & (ang <= red_from)

    codes, cidx = empty(w, h)
    codes[on_arc] = _DOT
    cidx[on_arc] = ctx.palette.index(0.2)
    codes[red] = _BOLD_DOT
    cidx[red] = RAMP_STEPS - 1

    # scale ticks every 20% of the sweep, just inside the arc
    tol_tick = 1.0 / np.maximum(dist, 1.0)
    for frac in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        a = lo + (hi - lo) * frac
        tick = (np.abs(ang - a) < tol_tick) & (np.abs(dist - radius * 0.86) < 0.8)
        codes[tick] = _TICK
        cidx[tick] = ctx.palette.index(0.3 + 0.5 * frac)

    # the needle: thin at the tip because the angular tolerance shrinks with radius
    tol = 1.1 / np.maximum(dist, 1.0)
    shaft = (np.abs(ang - sweep) < tol) & (dist <= radius * 0.92)
    codes[shaft] = _FULL
    cidx[shaft] = ctx.ramp(np.clip(0.25 + level, 0.0, 1.0))

    flag = (np.abs(ang - peak_sweep) < tol_tick * 0.8) & (np.abs(dist - radius * 1.04) < 0.8)
    codes[flag] = _FLAG
    cidx[flag] = RAMP_STEPS - 1

    pivot = dist < 1.6
    codes[pivot] = _FULL
    cidx[pivot] = ctx.palette.index(0.45)
    return codes, cidx


@mode("VU", group="stereo", blurb="big L/R LED meters with peak hold")
def vu(ctx: Ctx):
    w, h = ctx.w, ctx.h
    if w < 12 or h < 4:
        return empty(w, h)

    def init():
        return {"peak": np.zeros(2), "hold": np.zeros(2)}

    st = ctx.scratch("vu", init)

    lvl = _vu_level(ctx)
    _peak_hold(st, lvl, ctx.dt)

    codes, cidx = empty(w, h)

    gap = 1 if h >= 6 else 0
    band_h = max(1, (h - gap) // 2)
    top = 0
    bottom = band_h + gap

    xs = np.arange(w)
    heat = ctx.ramp(xs / max(1, w - 1))
    gutter = (xs % 4) == 3          # LED segmentation

    for ch, y0 in ((0, top), (1, bottom)):
        n_lit = int(round(lvl[ch] * w))
        lit = (xs < n_lit) & ~gutter
        y1 = min(h, y0 + band_h)
        if y1 <= y0:
            continue
        rows = slice(y0, y1)
        codes[rows, lit] = _FULL
        cidx[rows] = heat

        px = int(round(st["peak"][ch] * (w - 1)))
        if px > 0:
            codes[rows, px] = _TICK
            cidx[rows, px] = RAMP_STEPS - 1

    if gap:
        codes[band_h] = SPACE
    return codes, cidx


#: Resolution of the Kaleidoscope source patch — the little chamber of glass
#: the mirrors look at, sampled as a (radius, wedge-angle) table.
#:
#: ``_KAL_NU`` is the angular resolution and has to stay a power of two: the
#: per-dot gather masks with ``& (nu - 1)`` rather than taking a modulus. It
#: is sized against the fraction of the chamber one wedge actually shows
#: (:data:`_KAL_SECTOR`) rather than against the dot grid: a wedge spans
#: ``_KAL_SECTOR`` of the patch, so it gets ``_KAL_NU * _KAL_SECTOR`` = 85
#: samples, which is the number that has to beat the dots competing for them.
_KAL_NU, _KAL_NR = 512, 128

#: How much of the chamber's circumference one mirror wedge looks at.
#:
#: Getting this wrong is what a first attempt at the rebuild did, and the
#: symptom was unmistakable: mapping the *whole* source disc onto one wedge
#: means a screen wedge sweeps all 2*pi of the glass, so every fragment is
#: compressed into a thin radial sliver and the rosette reads as a starburst
#: of rays rather than as pieces of anything. Fragments need an aspect ratio
#: close to one to read as fragments, and that is set here.
#:
#: A sixth, fixed, rather than the physically exact ``1 / k``. The real object
#: has the wedge angle and the chamber sector locked together, so a 20-mirror
#: tube shows a twentieth of the glass; honouring that would make the
#: fragments visibly thinner every time the spectrum pushed the mirror count
#: up, which is backwards — more mirrors should mean more repeats of the same
#: pretty thing, not a worse thing repeated more often. Holding the sector
#: fixed keeps fragment shape constant and lets the count follow the music
#: freely. It is the same trade the shard widths made in the version before
#: this one, for the same reason.
_KAL_SECTOR = 1.0 / 6.0

#: How many straight cuts shatter the glass, and how many different chambers
#: are kept ready to swap between.
#:
#: The cuts are what make this a kaleidoscope rather than an iris. See
#: :func:`_kal_glass` for the construction.
#:
#: The count is set by the *worst* wedge, not the average one. What a wedge
#: shows is a 60-degree slice of the chamber, and with too few cuts the odds
#: are decent that a given slice at a given rotation falls almost entirely
#: inside one fragment — at which point the frame is a flat wash with a thin
#: figure in it. Swept over six chambers and twelve rotations, measuring how
#: much of the wedge the largest visible fragment takes:
#:
#:     cuts   fragments   visible/wedge   largest    worst case
#:       12          58            10.8       42%           89%
#:       16          98            16.1       33%           80%
#:       22         180            25.5       25%           48%
#:       32         368            45.9       17%           37%
#:
#: 22 is where the worst case stops being a flat frame. Past it the average
#: fragment drops below a few cells at ordinary terminal sizes, where the
#: reduction to half-blocks starts dropping pieces rather than drawing them,
#: and every extra fragment is another colour boundary for the strip builder.
_KAL_CUTS = 22
_KAL_CHAMBERS = 4

#: Screen radius that maps to the rim of the chamber.
#:
#: The dot geometry normalises so ``r == 1`` at the nearer edge, which puts
#: the corners at ``sqrt(2)``. Mapping 0..1 onto the patch therefore clamped
#: everything outside the inscribed circle to the outermost ring of samples —
#: one fragment smeared across all four corners, which at 78x11 was most of
#: the frame and read as a flat surround with a small figure in it. Dividing
#: by the corner distance puts the whole terminal inside the glass at every
#: aspect ratio.
_KAL_RIM = 1.45

#: Where in the chamber the screen centre sits, as a source radius.
#:
#: Nonzero, and this is the last thing standing between the rebuild and the
#: iris it replaced. Sampling a pie slice that runs from the chamber's own
#: centre out to its rim means screen radius *is* source radius, so any
#: fragment spanning a range of source radii lands as an arc band and the
#: rosette organises itself into concentric rings — the exact read the
#: gaussians used to produce, arrived at from a different direction.
#:
#: Looking at an annular sector instead removes the shared centre: the glass
#: in view has no radial structure relative to the screen, because the point
#: everything is radial about is not in the picture. It also squares the
#: fragments up. The visible region spans 0.45..1.0 in radius against a 60
#: degree arc, which at mid-radius is 0.55 by 0.73 — near enough one to one
#: that a fragment reads as a piece rather than as a band.
_KAL_CORE = 0.45

def _kal_glass(seed: int) -> tuple[np.ndarray, int]:
    """One chamber of shattered glass, as a fragment index per patch sample.

    **This is the whole difference between a kaleidoscope and an iris**, and
    the previous version of this mode was the iris. It drew twelve soft 2-D
    gaussians at staggered radii inside the wedge, which has two consequences
    that between them name the shape: the blobs have no edges, so nothing
    reads as a *piece* of anything; and mirroring a radial chain of blobs
    around a circle stacks them into concentric rings, which is an aperture.
    No amount of retuning widths or radii escapes that — the arrangement is
    the problem.

    Real stained glass in a mirror tube is the opposite on both counts. The
    fragments are hard-edged, flat in colour, and they *tile*: they meet each
    other along straight seams and cover the whole field, with no privileged
    centre and no radial banding. So build exactly that.

    Seven random lines are drawn across the source disc. Every sample records
    which side of each line it fell on, giving a 7-bit code; samples sharing a
    code are on the same side of every cut, which is precisely the definition
    of one convex cell of the arrangement. Those cells are the fragments. The
    edges are hard because the code changes discontinuously at a line, which
    is free — there is no anti-aliasing to switch off and no width to tune.

    The cuts are placed in the *Cartesian* plane of the patch, not in
    ``(radius, angle)``, and that matters for a reason easy to miss: the
    sampler wraps the angular axis with ``& (nu - 1)`` once the spin is added,
    so the patch has to be periodic in u or there is a visible seam at the
    wrap. Laid out as chords of a disc, periodicity is automatic — a line in
    the plane is a closed curve in ``(r, phi)`` — where any pattern authored
    directly in u would have to be made periodic by hand.

    Returns the fragment index per sample and the fragment count. Both are
    functions of the seed alone, so chambers are built once at import and
    cost nothing at any terminal size.
    """
    rng = np.random.default_rng(seed)
    u = (np.arange(_KAL_NU, dtype=np.float32) + 0.5) * np.float32(1.0 / _KAL_NU)
    r = (np.arange(_KAL_NR, dtype=np.float32) + 0.5) * np.float32(1.0 / _KAL_NR)
    phi = u * np.float32(2.0 * math.pi)
    px = r[:, None] * np.cos(phi)[None, :]
    py = r[:, None] * np.sin(phi)[None, :]

    code = np.zeros((_KAL_NR, _KAL_NU), dtype=np.int32)
    for _ in range(_KAL_CUTS):
        th = float(rng.uniform(0.0, math.pi))
        # Offsets kept well inside the disc: a chord that grazes the rim
        # splits off a sliver too thin to survive the reduction to half-block
        # cells, and spends a fragment index on something invisible.
        d = float(rng.uniform(-0.62, 0.62))
        side = (px * np.float32(math.cos(th))
                + py * np.float32(math.sin(th))) > np.float32(d)
        code = (code << 1) | side

    # Dense-remap the sparse sign codes to 0..M-1 so the per-frame value array
    # is M long rather than 128 long with holes.
    uniq, flat = np.unique(code.ravel(), return_inverse=True)
    return flat.reshape(_KAL_NR, _KAL_NU).astype(np.int32), int(uniq.size)


#: The chambers, built once at import. Size-independent by construction, so
#: this is not scratch and never rebuilds — the whole point of authoring the
#: glass in patch space rather than on the dot grid.
_KAL_GLASS = [_kal_glass(0x6C1A55 + i) for i in range(_KAL_CHAMBERS)]

#: Per-fragment constants: which band lights a fragment, how much light it
#: passes, and where it sits in its own slow shimmer. Fixed, because a piece
#: of glass does not change colour — the light behind it changes.
#:
#: Sized from the chambers actually built rather than from ``1 << _KAL_CUTS``.
#: Twenty-two cuts have an upper bound of four million sign codes and produce
#: about a hundred and eighty regions; allocating for the bound would be four
#: million entries per array to use a hundred and eighty of them.
_KAL_MAX_CELLS = max(n for _, n in _KAL_GLASS)
_KAL_RNG = np.random.default_rng(0x91A55)
_KAL_FRAG_BAND = _KAL_RNG.integers(0, 8, _KAL_MAX_CELLS).astype(np.int32)
_KAL_FRAG_TONE = _KAL_RNG.uniform(0.30, 1.0, _KAL_MAX_CELLS).astype(np.float32)
_KAL_FRAG_PHASE = _KAL_RNG.uniform(0.0, 2.0 * math.pi, _KAL_MAX_CELLS).astype(np.float32)


@mode("Kaleidoscope", group="fields",
      blurb="a mirrored tube of stained glass — the wedge count follows the spectrum, beats shake the chamber")
def kaleidoscope(ctx: Ctx):
    """A mirrored tube looking at a chamber of broken glass.

    Three parts, and they map one-to-one onto the physical object: a chamber
    of coloured fragments (:func:`_kal_glass`), a ring of mirrors around it,
    and the fact that turning the tube rotates the glass behind fixed mirror
    seams.

    **The mirrors.** A ring of 8, 12, 16 or 20 of them, eased by the spectral
    centroid and snapped on a beat, each showing the same source slice
    reflected left-right alternately. Every dot's angle is wrapped into its
    sector, then even sectors read the source forward and odd sectors read it
    reversed, so adjacent sectors mirror across the shared boundary and the
    picture is symmetric about every mirror line. Only a multiple of four puts
    a mirror line on the vertical axis, which is why the count snaps rather
    than running through every integer.

    The mirror lines are fixed in screen space; the spin rotates the *source*
    inside the wedge. The fold coordinate stays put and only the lookup shifts
    by the accumulated phase, so the picture stays bilaterally symmetric at
    every angle of rotation — a property that falls out of the construction
    rather than being close enough for the eye.

    That construction is the important part. The fold runs on a grid of
    absolute columns: the geometry is evaluated on ``|x|`` measured from the
    centre line between the two middle dot columns, so a dot and its L/R
    mirror compute exactly the same angle, radius and folded coordinate, and
    any function of those coordinates is bit-for-bit identical between the
    two. No averaging, no tolerance: the rendered frame is symmetric by
    construction. The gather then runs on the left half only and is mirrored
    for the right, which halves the per-dot path without changing a bit.

    **The glass.** Hard-edged convex fragments tiling the whole source disc,
    cut once at import and never rebuilt. What changes per frame is only how
    brightly each fragment is lit: one value per fragment, forty-odd of them,
    then a single gather turns that into the source table. That is a much
    smaller per-frame job than the twelve gaussians this used to evaluate over
    a 128x512 table, and it is the reason the mode now costs less than the
    version that looked worse.

    Each fragment is lit by one band, scaled by its own fixed transmittance
    and a slow shimmer on its own phase — a piece of glass does not change
    colour, the light behind it does. Flat colour within a fragment is not a
    simplification: it is what glass looks like, and it is also what the
    run-length encoder in the strip builder wants, so the look and the render
    cost agree for once.

    **Shaking the tube.** A beat swaps the whole chamber for a different one,
    round-robin through four cut at import. That is the one gesture a real
    kaleidoscope has that rotation cannot give you — the fragments tumble into
    a new arrangement — and swapping a precomputed index array costs nothing,
    where re-cutting the glass would cost a sort over the patch. The same beat
    kicks the rotation and snaps the sector count.

    The rotation rate is re-integrated through ``ctx.dt`` rather than read
    from ``ctx.t``, so pausing the audio locks the rotation in place without
    disturbing the phase. ``beat_phase`` drives a gentle brightening between
    hits — gated on ``tempo_bpm``, which is 0.0 until a tempo is established
    and takes the phase to 0.0 with it, i.e. a permanent on-the-beat swell if
    read ungated.

    Colour: every cell renders as a solid two-colour ``▀`` pair — the top half
    from the max over the cell's top two dot rows, the bottom half from its
    bottom two — quantised to eight buckets before ramping. Quantisation is
    the Chladni Flow lesson: the two-colour strip builder pays per colour
    boundary. Flat fragments make that cheap here rather than merely bearable.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 8 or dc < 8:
        return empty(ctx.w, ctx.h)

    from ..render import frac

    # Geometry on the |x|-folded grid: dot (x, y) and its L/R mirror
    # (dc - 1 - x, y) compute identical turn and radius, so any function of
    # them is bit-for-bit symmetric. The pole sits on the boundary between the
    # two centre dot columns, so no dot straddles the mirror axis.
    #
    # Only the left half is kept — everything downstream reads the half and
    # mirrors the result — and the full-width brightness buffer rides in the
    # same entry, since it is per-size and has exactly this lifetime. The
    # audit caps a mode at four scratch keys.
    def geo():
        cx, cy = (dc - 1) / 2.0, (dr - 1) / 2.0
        hw = dc // 2
        x_scale = cy / max(cx, 1.0)
        ax = np.abs(np.arange(hw, dtype=np.float32) - np.float32(cx)) * np.float32(x_scale)
        ys = np.arange(dr, dtype=np.float32) - np.float32(cy)
        dx = ax[None, :]
        dy = ys[:, None]
        dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        ang = np.arctan2(dy, dx).astype(np.float32)
        ang = np.where(ang < 0, ang + np.float32(2 * math.pi), ang).astype(np.float32)
        turn = (ang / np.float32(2 * math.pi)).astype(np.float32)
        r = (dist / max(cy - 1.0, 1.0)).astype(np.float32)
        # Radius index into the source patch, fixed for the size. Clamped
        # rather than wrapped: past the rim of the glass there is no glass.
        rr = np.minimum(r * np.float32(1.0 / _KAL_RIM), np.float32(1.0))
        rr = rr * np.float32(1.0 - _KAL_CORE) + np.float32(_KAL_CORE)
        ir = np.minimum((rr * np.float32(_KAL_NR)).astype(np.int32), _KAL_NR - 1)
        return turn, ir, np.empty((dr, dc), dtype=np.float32)

    turn, ir, bright = ctx.scratch("kaleido_geo", geo)
    hw = dc // 2

    bands8 = ctx.display_bands(8).astype(np.float32)
    total = float(bands8.sum())
    centroid = float((bands8 * np.arange(8)).sum() / total / 7.0) if total > 1e-9 else 0.0
    bass = ctx.range(0.0, 0.18)

    # ``beat_phase`` is continuous and available between hits, where the onset
    # effects are not, so it carries the pulse on material the detector is
    # sparse about. Gated on the tempo, never on the phase.
    breathe = (1.0 - float(ctx.beat_phase)) ** 2 if ctx.tempo_bpm > 0.0 else 0.0

    st = ctx.scratch("kaleido", lambda: {
        "kc": 8.0, "spin": 0.0, "k": None, "folded": None,
        "chamber": 0, "shimmer": 0.0,
    })
    # ctx.onsets, not a private difference of ctx.onset_seq. Scratch survives
    # a mode switch, so differencing here would replay every beat that played
    # while the mode was not drawing, all in a single frame.
    onsets = ctx.onsets

    # The mirror count eases toward the centroid and snaps on a beat. It only
    # ever lands on a multiple of four — that is what keeps a mirror line on
    # the vertical axis — and the reachable set is 8, 12, 16, 20.
    st["kc"] += (8.0 + centroid * 10.0 - st["kc"]) * min(1.0, ctx.dt / 1.2)
    if onsets:
        st["kc"] = float(8.0 + 10.0 * min(1.0, bass * 1.5))
        # Shake the tube: the glass tumbles into a different arrangement.
        st["chamber"] = (st["chamber"] + onsets) % _KAL_CHAMBERS
    k = 4 * int(round(st["kc"] / 4.0))
    k = max(8, min(k, 20))

    # Spin moves the SOURCE inside the wedge; the mirror lines never move.
    st["spin"] = (st["spin"] + (0.22 + ctx.energy * 0.8) * max(ctx.dt, 0.0)) % (2 * math.pi)
    if onsets:
        st["spin"] = (st["spin"] + 0.3 * min(onsets, 3)) % (2 * math.pi)
    spin = st["spin"]
    st["shimmer"] = (st["shimmer"] + 1.7 * max(ctx.dt, 0.0)) % (2 * math.pi)

    # The fold: wrap the angle into its sector, then read even sectors forward
    # and odd sectors backward, so the array alternates orientation around the
    # ring exactly like physical mirror tubes. It depends only on the sector
    # count, which takes four values, so it is cached; the per-frame path is
    # just the phase shift and the gather below.
    if st["k"] != k:
        wedge = turn * np.float32(k)
        m = np.floor(wedge).astype(np.int32)
        u = wedge - np.float32(m)
        st["folded"] = np.where((m & 1) == 0, u, np.float32(1.0) - u)
        st["k"] = k
    folded = st["folded"]

    # ── how brightly each fragment is lit ──
    # One value per fragment — forty-odd numbers — then a single gather builds
    # the whole source table. The old build evaluated twelve 2-D gaussians
    # over the table every frame; this is the same table for a fraction of the
    # arithmetic, and it comes out with hard edges instead of soft ones.
    cell_id, n_cells = _KAL_GLASS[st["chamber"]]
    band = _KAL_FRAG_BAND[:n_cells]
    tone = _KAL_FRAG_TONE[:n_cells]
    shim = 0.80 + 0.20 * np.sin(_KAL_FRAG_PHASE[:n_cells] + np.float32(st["shimmer"]))
    lit = bands8[band] * shim * tone
    # Floor so unlit glass still reads as glass rather than as a hole: a dark
    # fragment is a dark fragment, not an absence of one, and the seams have
    # to stay visible for the tiling to be legible at all.
    val = (np.float32(0.12) + np.float32(0.88) * lit) * np.float32(
        (0.62 + 0.55 * ctx.energy) * (1.0 + 0.14 * breathe))
    np.clip(val, 0.0, 1.0, out=val)

    table = val[cell_id]

    # The source is sampled on the LEFT half of the |x|-symmetric grid and
    # mirrored: dot (x, y) and its mirror gather the same table cell
    # bit-for-bit, so the right half is a reversed copy of the left. That
    # halves the frac, the index arithmetic and the gather without averaging,
    # and the rendered frame stays symmetric by construction.
    src = frac(folded * np.float32(_KAL_SECTOR) + np.float32(spin / (2 * math.pi)))
    iu = (src * np.float32(_KAL_NU)).astype(np.int32) & (_KAL_NU - 1)
    b_half = table.ravel()[ir * _KAL_NU + iu]
    bright[:, :hw] = b_half
    bright[:, hw:] = b_half[:, ::-1]

    # Two colour samples per text cell, one per half-row: the top half is the
    # max over the cell's top two dot rows, the bottom half over its bottom
    # two. The bright grid is symmetric about the middle column, so these
    # reductions are too — every half-row equals its mirror, bit for bit.
    top = np.maximum(bright[0::4, 0::2], bright[1::4, 0::2])
    top = np.maximum(top, bright[0::4, 1::2])
    top = np.maximum(top, bright[1::4, 1::2])
    bot = np.maximum(bright[2::4, 0::2], bright[3::4, 0::2])
    bot = np.maximum(bot, bright[2::4, 1::2])
    bot = np.maximum(bot, bright[3::4, 1::2])

    field = np.empty((2 * top.shape[0], top.shape[1]), dtype=np.float32)
    field[0::2] = top
    field[1::2] = bot

    # Quantised to eight buckets before ramping: the two-colour ▀ strip
    # builder pays per colour boundary, and an unquantised field with this
    # many local extrema pushed Chladni Flow to 9-14 ms a frame (its docstring
    # records the numbers). Flat fragments already produce long runs, so this
    # mostly guards the shimmer against splitting them.
    #
    # No radial vignette. The old version blended the cell radius in at half
    # weight, which is a concentric gradient laid over everything and one of
    # the two things making the mode read as an iris; the other was the
    # gaussians. Both are gone.
    field *= 8.0
    np.round(field, 0, out=field)
    field *= 1.0 / 8.0

    idx = ctx.ramp(field)
    codes = np.full((ctx.h, ctx.w), _UPPER_HALF, dtype=np.int32)
    return codes, idx[0::2], idx[1::2]


@mode("Dither", group="fields", blurb="the spectrum printed as a newspaper halftone, in one-bit crosshatch")
def dither(ctx: Ctx):
    """The spectrum as a continuous two-dimensional field, thresholded to one
    bit by an ordered dither — so the whole frame is one textured surface,
    not a row of bars.

    The field is the sum of eight directional plane waves, one per band, at
    angles spreading around the circle. Each wave's amplitude is that band's
    level, so the *shape* of the spectrum steers the texture: heavy bass
    swells the low-frequency waves into broad slow undulations while the
    high bands draw fine grain on top, and the sum is normalized by the band
    total so the mix rotates with the music instead of saturating. The
    wavelengths run from a twentieth of the width up to a quarter of it, so
    every braille cell sees a different slice of the wave mix and the
    cross-hatch pattern varies across the frame instead of repeating. The
    level drives two scalars — a baseline lift and the texture depth — so a
    louder signal reads as a denser, deeper field and a quiet one as a flat,
    even grey. There is no clock in here and no state beyond the cached
    geometry: a frozen spectrum is a frozen frame, exactly like the bars
    modes.

    The threshold is an s x s Bayer matrix tiled by *absolute* dot position
    on both axes. The tile alignment is what makes this a texture rather
    than stripes: because the threshold varies with the column inside every
    row, a row whose field is uniform still breaks into the cross-hatch
    pattern of the matrix, and because the field varies along x as well as
    y, no row ever resolves to a single solid line. Wherever the field is
    near the threshold the matrix turns the gradient into structured dots
    and wedges — the ordered-dither effect — and the density of the pattern
    stands in for the field's value: a grey-scale rendering of a surface
    that is only ever lit or unlit.

    One guard on top of that: a row whose field rides high across its whole
    width would saturate every dot. Each row's maximum threshold is known,
    so the top threshold cell of any such row is forced dark — a uniformly
    solid row becomes structurally impossible without dimming anything
    else, and the baseline keeps every row lit somewhere.

    There is deliberately no left-right mirroring here. A mirror line
    through the centre is exactly the seam the eye locks onto, splitting
    one surface into two panels; a dithered field should read as a single
    continuous skin, and the absolute tiling keeps it that way across the
    whole frame.

    Colour walks the ramp by the lit density of each cell, so brighter
    patches of the texture sit higher in the ramp.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 8 or dc < 8:
        return empty(ctx.w, ctx.h)

    def grid():
        # Eight directional plane waves, one per band. Band q runs at angle
        # pi*q/8; its wavelength goes from a twentieth of the width (fine
        # grain, high bands) up to a quarter of the width (broad swells,
        # low bands), scaled to the terminal so the texture keeps its
        # structure at any size.
        x = (np.arange(dc, dtype=np.float32) - np.float32((dc - 1) / 2.0)) / np.float32(max(dc - 1, 1))
        y = (np.arange(dr, dtype=np.float32) - np.float32((dr - 1) / 2.0)) / np.float32(max(dr - 1, 1))
        twopi = np.float32(2 * math.pi)
        ph = []
        for q in range(8):
            thq = np.float32(math.pi * q / 8.0)
            # Cycles across the field, not a fraction of the dot count.
            #
            # This read ``dc * (0.06 + 0.28 * q/7)``, which at a 400x100
            # terminal is 48 cycles for the lowest band and 272 for the
            # highest -- three dots per cycle. Nothing at that frequency is
            # visible as structure; it is the speckle failure recorded in
            # Plasma's docstring, and it is why the mode read as static.
            # The comment above it described the intent correctly ("a
            # twentieth of the width up to a quarter") and the arithmetic
            # did something else entirely.
            #
            # Three to seventeen cycles is broad swells for the bass through
            # fine grain for the treble, all of it coarse enough to see, and
            # it is resolution-independent because x and y are normalised.
            kq = np.float32(1.4 + 5.2 * (q / 7.0))
            wx = np.cos(thq) * twopi * kq
            wy = np.sin(thq) * twopi * kq
            # Stored already evaluated. The wave's geometry depends only on
            # the grid, so sin() of it is a constant for the whole life of
            # this size — but it was being recomputed every frame, eight
            # times over the dot grid. At 400x100 that is 2.56 million
            # transcendentals a frame and it was most of the mode's cost.
            # Only the band weights change, so the per-frame work is a
            # weighted sum of fixed fields, with no sin() in it at all.
            ph.append(np.sin(wx * x[None, :] + wy * y[:, None]).astype(np.float32))
        # The ordered-dither threshold: an s x s Bayer matrix (the standard
        # recursion, normalized to [0, 1)) tiled by ABSOLUTE dot position on
        # both axes, so the cross-hatch runs continuously across the whole
        # grid instead of restarting at each row or folding at the centre.
        s = 8 if (dr >= 16 and dc >= 16) else 4
        b = np.zeros((1, 1), dtype=np.float32)
        size = 1
        while size < s:
            b = np.block([[4 * b, 4 * b + 2], [4 * b + 3, 4 * b + 1]])
            size *= 2
        b = b / np.float32(size * size)
        th = b[np.arange(dr, dtype=np.int32) % s][:, np.arange(dc, dtype=np.int32) % s]
        # One contiguous (8, dr, dc) block so the per-frame combine is a
        # single tensordot rather than eight separate multiply-adds each
        # walking the whole grid.
        driftx = (np.arange(dc, dtype=np.float32) / np.float32(max(dc - 1, 1)))[None, :] * np.float32(1.5)
        return {"sinph": np.ascontiguousarray(np.stack(ph)), "driftx": driftx,
                "th": th, "th_rowmax": th.max(axis=1).astype(np.float32),
                "th_argmax": np.argmax(th, axis=1).astype(np.int32)}

    g = ctx.scratch("dither_grid", grid)

    # The field is the spectrum's own silhouette, not a wave interference
    # pattern.
    #
    # This mode used to sum eight directional plane waves. That produced a
    # texture, and the texture was the problem: an interference field is what
    # Plasma, Chladni and Maelstrom already draw, so dithering one only
    # changed how it was shaded, not what it was. A halftone is a printing
    # technique, and a printing technique needs a subject.
    #
    # The subject here is the spectrum, drawn the way a newspaper would print
    # it: the band profile as a filled silhouette, given a vertical tone ramp
    # so it is dense along the floor and thins toward its own upper edge, and
    # then thresholded to pure black and white. What survives is a picture of
    # the spectrum made entirely of crosshatch, which nothing else in the set
    # does.
    prof = resample_bands(ctx.bands, dc).astype(np.float32)
    # A little horizontal smoothing so band boundaries do not read as steps;
    # the halftone exaggerates any hard vertical edge into a visible seam.
    prof = (prof + np.roll(prof, 1) + np.roll(prof, -1)) * np.float32(1.0 / 3.0)

    v = (np.arange(dr - 1, -1, -1, dtype=np.float32) / np.float32(max(dr - 1, 1)))[:, None]
    # Height above the floor, in the same 0..1 units as the profile. Soft
    # rather than a hard cut, so the top edge of the silhouette dissolves
    # into dots instead of terminating in a line -- that dissolve is the
    # halftone's signature and the reason the edge reads as tone.
    edge = np.clip((prof[None, :] * np.float32(1.05) - v) * np.float32(6.0), 0.0, 1.0)

    # Tone inside the silhouette: densest at the floor, lighter toward the
    # top, so the fill carries a gradient for the dither to resolve rather
    # than being one flat grey.
    shade = np.float32(0.30) + np.float32(0.62) * (np.float32(1.0) - v)

    # A slow drift keeps the crosshatch from looking like a frozen print, and
    # gives quiet passages something to do. Integrated through ctx.dt, and
    # the wave that carries it is a single low-frequency swell rather than a
    # field of them -- it is lighting, not subject.
    ph = ctx.scratch("dither_drift", lambda: {"v": 0.0})
    ph["v"] += (0.05 + ctx.energy * 0.20) * max(ctx.dt, 0.0)
    swell = np.sin((g["driftx"] + np.float32(ph["v"])) * np.float32(2 * math.pi)) * np.float32(0.10)

    field = (edge * (shade + swell)).astype(np.float32)
    lit = field > g["th"]
    # guard: a row whose field stays above its own row-max threshold would
    # saturate every dot. The top threshold cell of such a row is forced
    # dark, which makes a uniformly solid row structurally impossible
    # without dimming anything else.
    hot = field.max(axis=1) > g["th_rowmax"]
    if bool(hot.any()):
        cols = g["th_argmax"]
        lit[hot, cols[hot]] = False
    codes = pack_braille(lit)

    # Colour follows the underlying field, not the dithered result.
    #
    # Reading it off the lit dots looks equivalent and is not, for two
    # reasons. Visually, the dither pattern is already carrying the texture;
    # colouring by lit density paints per-cell noise on top of it and fights
    # the halftone the mode exists to draw. And structurally it is the worst
    # possible input to the strip builder: neighbouring cells almost never
    # agree, so run-length encoding finds no runs and emits a segment per
    # cell. Measured at 400x100 that was 24.61 ms in make_strips alone,
    # against 15.79 ms to build the frame -- the mode was over the 16.7 ms
    # budget almost entirely on colour it did not need.
    #
    # Sampling the smooth field instead gives neighbours the same ramp index
    # over stretches, which is what run-length encoding is for.
    # Quantised to a few buckets before ramping, the same lesson Chladni's
    # docstring records. The field is a sum of eight waves, so even sampled
    # smoothly it crosses ramp buckets almost every cell, and neighbours that
    # never agree are neighbours run-length encoding cannot merge. Eight
    # levels is invisible against a one-bit texture -- the dither is already
    # doing the shading -- and collapses the segment count enough to bring
    # make_strips from 15.4 ms back to something ordinary.
    col = cell_mean(np.clip(field, 0.0, 1.0))
    np.rint(col * 8.0, out=col)
    idx = ctx.ramp(np.clip(col * 0.125, 0.0, 1.0))
    return codes, idx



#: Most hearts a Valentine frame will carry. Bounded so a dense passage
#: cannot grow the particle arrays without limit.
_VAL_HEARTS = 24


@mode("Valentine", group="fields", blurb="a heart that beats with the track, trailing smaller ones upward")
def valentine(ctx: Ctx):
    """A heart that actually beats, rather than a heart that pulses.

    The distinction is the whole mode. A shape scaled by ``ctx.energy`` swells
    and sags with the music's loudness, which is a throb, not a heartbeat — a
    real one is two strokes, a loud *lub* and a softer *dub* about a sixth of
    a second behind it, and then stillness until the next beat. So an onset
    fires the first stroke and schedules the second, and between beats the
    heart is still. Recognising it as a heartbeat depends entirely on that
    second stroke and on the silence after it.

    The shape is the standard implicit heart, ``(x^2 + y^2 - 1)^3 <= x^2 y^3``,
    which is bilaterally symmetric by construction — its own mirror, with no
    folding needed. Tested against a radius table built from the curve once
    per grid size, so beating is a matter of scaling the radius rather than
    redrawing anything.

    Smaller hearts rise from it, spawned on beats and released with a sideways
    drift, shrinking as they climb. They are also what keeps the mode alive
    when the detector is quiet: a slow rate spawns them under a drone, the
    same reasoning as Pulse's clock. Neither the rise nor the sway is a
    function of ``ctx.t`` scaled by anything audio-driven; both integrate
    ``ctx.dt``, so the drift is the same speed at 30 fps and at 144.

    Colour walks the ramp by depth inside the shape, so the heart reads as
    solid with a brighter rim rather than as a flat silhouette.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 12 or dc < 12:
        return empty(ctx.w, ctx.h)

    def geo():
        # Aspect-corrected so the heart is a heart and not an oval: braille
        # dots are about twice as tall as they are wide, and the cell grid is
        # itself wider than it is tall.
        cx, cy = (dc - 1) / 2.0, (dr - 1) / 2.0
        sx = max(cx, 1.0)
        sy = max(cy, 1.0)
        x = (np.arange(dc, dtype=np.float32) - np.float32(cx)) / np.float32(sx)
        y = (np.float32(cy) - np.arange(dr, dtype=np.float32)) / np.float32(sy)
        gx, gy = x[None, :], y[:, None]

        # Outline radius per ray of the unit heart, for the depth inside the
        # main heart. The heart is star-shaped about the origin, so along any
        # ray the outline sits at one radius r_h(theta), and scaling the heart
        # by s scales that radius by exactly s -- which makes 1 - rad/r_h a
        # clean interior depth. Built in the offset space (u = gx, w = gy +
        # 0.05), where the heart is the fixed curve (u^2 + (1.15 w)^2 - 1)^3 -
        # u^2 (1.15 w)^3 <= 0: the per-frame scale divides BOTH coordinates,
        # so the scaled heart is exactly the unit curve scaled by s.
        na = 1024
        th = np.linspace(-math.pi, math.pi, na, endpoint=False)
        rs = np.linspace(0.02, 1.60, 640)[:, None]
        u = rs * np.cos(th)[None, :]
        v = rs * np.sin(th)[None, :] * np.float32(1.15)
        q = u * u + v * v - np.float32(1.0)
        ins = (q * q * q - u * u * v * v * v) <= 0.0
        # Outermost radius still inside along each ray, not the first
        # crossing: the convention Locket uses, and the safe one if a ray
        # ever grazes the notch at the top of the heart.
        idx = ins.shape[0] - 1 - np.argmax(ins[::-1], axis=0)
        r_h = np.where(ins.any(axis=0), rs[:, 0][idx], 0.02)

        w = gy + np.float32(0.05)
        rad = np.sqrt(gx * gx + w * w)
        ai = ((np.arctan2(w, gx) + math.pi) / (2.0 * math.pi) * na).astype(np.int32) % na
        return gx, gy, r_h, rad, ai

    gx, gy, r_h, rad, ai = ctx.scratch("valentine_geo", geo)

    st = ctx.scratch("valentine", lambda: {
        "beat": 0.0, "dub": -1.0, "acc": 0.0,
        "hx": np.zeros(_VAL_HEARTS, dtype=np.float32),
        "hy": np.full(_VAL_HEARTS, 9.0, dtype=np.float32),   # 9 == dead
        "hs": np.zeros(_VAL_HEARTS, dtype=np.float32),
        "hv": np.zeros(_VAL_HEARTS, dtype=np.float32),
        "rng": np.random.default_rng(214),
    })
    rng = st["rng"]
    # ctx.onsets, not a private difference of ctx.onset_seq. Scratch survives
    # a mode switch, so differencing here would replay every beat that played
    # while the mode was not drawing, all in a single frame.
    onsets = ctx.onsets

    bass = ctx.range(0.0, 0.22)

    # ── the two strokes ──
    # An onset is the lub; the dub is scheduled a sixth of a second later and
    # lands at a bit over half the amplitude. Both decay on the same short
    # time constant, integrated in seconds so the beat has the same shape at
    # any frame rate.
    if onsets:
        st["beat"] = min(1.6, st["beat"] + 0.85 + 0.5 * ctx.onset_strength)
        st["dub"] = ctx.t + 0.17
    if st["dub"] > 0.0 and ctx.t >= st["dub"]:
        st["beat"] = min(1.6, st["beat"] + 0.42)
        st["dub"] = -1.0
    st["beat"] *= math.exp(-max(ctx.dt, 0.0) / 0.11)
    beat = st["beat"]

    # Size: a resting heart that swells a little with the track's body, plus
    # the beat on top. The resting term is deliberately gentle — if loudness
    # moved the heart much, the beat would stop reading as a beat.
    scale = 0.44 + 0.07 * bass + 0.15 * beat

    # ── the main heart ──
    # (x^2 + y^2 - 1)^3 - x^2 y^3 <= 0, on coordinates divided by the size so
    # a bigger scale means a bigger heart. The y offset lifts it slightly:
    # the curve's own centroid sits below the origin and it looks dropped
    # without it.
    #
    # Depth inside the shape is measured along rays from the origin, not by
    # rescaling the polynomial: its magnitude is not a distance, it collapses
    # toward zero down the seam where the lobes meet and explodes away from
    # it, so a rescaled reading streaks the middle of the heart with a colour
    # band of its own. The heart is star-shaped about the origin, so each ray
    # crosses the outline once, at a radius r_h(theta) built in geo() against
    # these constants, and 1 - rad / (scale * r_h) is smooth everywhere and
    # zero exactly on the rim. So the middle is solid and the rim recedes
    # rather than the whole silhouette being one flat value.
    inv = np.float32(1.0) / np.float32(scale)
    t = rad * inv / r_h[ai]
    depth = np.clip(np.float32(1.0) - t, np.float32(0.0), np.float32(1.0)).astype(np.float32)
    field = np.where(t <= np.float32(1.0), np.float32(0.45) + np.float32(0.55) * depth, np.float32(0.0))

    # ── rising hearts ──
    # Spawned on a beat, and on a slow clock so the frame is never empty on
    # material the detector reads poorly.
    st["acc"] += (0.25 + ctx.energy * 1.1) * max(ctx.dt, 0.0)
    want = onsets
    if st["acc"] >= 1.0:
        st["acc"] -= 1.0
        want += 1
    # Never let the sky above the heart go completely empty. At a moderate
    # level the clock alone takes over a second to release the first one, so
    # the mode would open on a motionless heart and read as frozen -- and on
    # a drone, where no onsets ever arrive, it would stay that way.
    if not want and not (st["hy"] <= 1.6).any():
        want = 1
    if want:
        free = np.flatnonzero(st["hy"] > 1.6)[:want]
        for i in free:
            # Spread wider than the main heart so some rise clear of its
            # silhouette; the ones launched from inside read as escaping it.
            st["hx"][i] = np.float32(rng.uniform(-0.95, 0.95))
            st["hy"][i] = np.float32(-0.25)
            st["hs"][i] = np.float32(rng.uniform(0.10, 0.19))
            st["hv"][i] = np.float32(rng.uniform(0.30, 0.62))

    alive = st["hy"] <= 1.6
    if alive.any():
        dt = np.float32(max(ctx.dt, 0.0))
        st["hy"][alive] += st["hv"][alive] * dt
        # Sway is a function of the heart's own height, not of wall time, so
        # it traces a fixed path upward instead of shimmying in place.
        st["hx"][alive] += np.sin(st["hy"][alive] * np.float32(5.0)) * dt * np.float32(0.09)
        st["hs"][alive] *= np.float32(math.exp(-max(ctx.dt, 0.0) / 1.9))

    for i in np.flatnonzero(alive):
        s = float(st["hs"][i])
        if s < 0.035:
            st["hy"][i] = 9.0
            continue
        px, py = float(st["hx"][i]), float(st["hy"][i])
        # Bound the work to the heart's own box: at 400x100 a small heart is
        # a few percent of the grid, and evaluating the implicit curve over
        # the whole field for each of two dozen of them is most of a frame.
        r = s * 1.6
        c0 = int(max(0, (px - r + 1.0) * 0.5 * (dc - 1)))
        c1 = int(min(dc, (px + r + 1.0) * 0.5 * (dc - 1) + 2))
        r0 = int(max(0, (1.0 - (py + r)) * 0.5 * (dr - 1)))
        r1 = int(min(dr, (1.0 - (py - r)) * 0.5 * (dr - 1) + 2))
        if c1 <= c0 or r1 <= r0:
            continue
        sx = (gx[:, c0:c1] - np.float32(px)) / np.float32(s)
        sy = (gy[r0:r1, :] - np.float32(py)) / np.float32(s) * np.float32(1.15)
        qq = sx * sx + sy * sy - np.float32(1.0)
        ff = qq * qq * qq - sx * sx * sy * sy * sy
        sub = field[r0:r1, c0:c1]
        np.maximum(sub, np.where(ff <= 0.0, np.float32(0.9), np.float32(0.0)), out=sub)

    codes = pack_braille(field > 0.0)
    idx = ctx.ramp(np.clip(cell_max(field), 0.0, 1.0))
    return codes, idx


#: Concurrent pulses in Locket.
_LOCKET_RINGS = 12


@mode("Locket", group="fields", blurb="an outlined heart, pulsing rings of hearts outward on the beat")
def locket(ctx: Ctx):
    """Nothing but hearts.

    A single outlined heart sits at the centre and is always drawn, so the
    mode has a subject in silence. On an onset it pulses: a ring leaves from
    *inside* it, passes out through the outline, and expands away — so the
    resting heart reads as the source of everything rather than as a frame
    the pulses happen to start near.

    There is no corridor, no spokes, no ribs. An earlier version had all
    three, and they fought the pulses: the ribs receded toward the centre
    while the hearts expanded away from it, so two conflicting flows shared
    one frame and neither read cleanly. Removing them leaves the pulses as
    the only motion, which is what makes the direction legible.

    Expansion is ``k / z``, the same perspective Tunnel In uses: a pulse
    accelerates as it grows, so it reads as something travelling outward past
    you rather than a shape inflating in place. Constant growth loses that.

    **``z`` decays geometrically, and the birth radius sits on the outline.**
    Both were wrong together, and the symptom was the mode's whole premise
    failing to land: you did not see hearts shooting out, you saw a long
    nothing and then a blur.

    ``z`` fell *linearly* before, which is literally correct perspective for
    something approaching at constant speed — and exactly why it does not work
    here, because ``r0 / z`` diverges as ``z`` approaches zero. Measured on
    the old constants: a pulse was born at ``0.30 * core``, one third of the
    way out to a resting outline it then had to grow past, and it crossed the
    visible band in the last 0.37 s of a 1.78 s life. 73% of every pulse was
    smaller than the heart it came from, i.e. hidden inside it, and the part
    you could see went by at up to 46x the speed it started at.

    Geometric decay — ``z *= exp(-lambda dt)`` — makes the *fractional* growth
    per second constant. That still accelerates in absolute terms, by about
    4.5x across the journey, because the same fraction of a larger radius is a
    larger step; it just cannot run away. Combined with a birth radius at
    ``0.80 * core`` the pulse starts just inside the outline, crosses it within
    about a fifth of a second, and is visible for the rest of its life.

    Each pulse carries its own birth radius, fixed at release. Deriving it from
    the heart's current size every frame instead — which is what this did — ties
    every ring in flight to a heart that breathes with the bass and the beat,
    so a beat jerks the entire field outward at once and a shrinking heart
    drags the rings back in. A ring's path has to be its own.

    Sizing a heart outline anywhere on the grid would normally cost an
    implicit curve evaluation per pulse per frame. It does not, because the
    heart is star-shaped about its own centre: along any ray the outline sits
    at one radius ``r_h(theta)``, and scaling the heart by ``s`` scales that
    radius by exactly ``s``. A dot at radius ``R`` on ray ``theta`` is
    therefore on the outline of the heart scaled to ``R / r_h(theta)`` —
    computed once per grid size, after which every heart here, resting or
    travelling, is one comparison against that field.

    The ``r_h`` table keeps the *outermost* radius still inside the curve
    along each ray. The first crossing would trace the notch at the top of
    the heart, where a ray leaves the shape and re-enters, and every
    silhouette would have a bite out of it.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 12 or dc < 16:
        return empty(ctx.w, ctx.h)

    def geo():
        cx, cy = (dc - 1) / 2.0, (dr - 1) / 2.0
        x = (np.arange(dc, dtype=np.float64) - cx) / max(cx, 1.0)
        y = (cy - np.arange(dr, dtype=np.float64)) / max(cy, 1.0)
        gx, gy = x[None, :], y[:, None]
        ang = np.arctan2(gy, gx)
        rad = np.sqrt(gx * gx + gy * gy)

        na = 1024
        th = np.linspace(-math.pi, math.pi, na, endpoint=False)
        rs = np.linspace(0.02, 1.60, 320)[:, None]
        u = (rs * np.cos(th)[None, :]) / 0.92
        v = (rs * np.sin(th)[None, :]) / 0.92 * 1.18 + 0.06
        q = u * u + v * v - 1.0
        ins = (q * q * q - u * u * v * v * v) <= 0.0
        idx = ins.shape[0] - 1 - np.argmax(ins[::-1], axis=0)
        r_h = np.where(ins.any(axis=0), rs[:, 0][idx], 0.02)
        ai = ((ang + math.pi) / (2 * math.pi) * na).astype(np.int32) % na
        # Turn: 0..1 once around, measured from straight down so band 0 sits
        # at the heart's point and the spectrum climbs each side symmetrically
        # rather than splitting across an arbitrary seam.
        turn = ((ang + math.pi * 0.5) / (2 * math.pi)) % 1.0
        # The band lookup for the rim is a pure function of position, so the
        # index pair and the blend weight are constants for this grid. Only
        # the two gathers below survive into the frame path; computing the
        # fold, the indices and a cosine over the whole dot grid every frame
        # cost 11.7 ms at 400x100 against 3.2 ms for the rest of the mode.
        # One index per dot into a small angular table, rather than a pair of
        # band indices and a blend weight. The rim's radius and thickness are
        # functions of angle alone, so they can be built as a 256-entry table
        # each frame -- 256 elements of arithmetic -- and read with a single
        # gather. Two full-grid gathers plus a multiply-add per dot cost
        # 10.4 ms at 400x100; one gather is most of that back.
        nt = 256
        # Distance from the *point*, so band 0 lands on the heart's tip as the
        # note above says. Measured from the cleft instead — which is what
        # ``abs(turn * 2 - 1)`` gives — the bass sits on the notch at the top,
        # and the bass is both the loudest band and the one that swings most.
        # The notch is the only feature that makes the silhouette read as a
        # heart rather than as a blob, so it is the last part of the outline
        # that should be pushed around.
        fold = 1.0 - np.abs(turn * 2.0 - 1.0)
        aidx = np.clip((fold * (nt - 1)).astype(np.int32), 0, nt - 1)
        scale = (rad / np.maximum(r_h[ai], 1e-3)).astype(np.float32)
        # The widest heart the grid can still show any part of. A pulse past
        # it cannot light a single dot, so it is finished — see the retirement
        # below. Measured rather than guessed: the coordinates are normalised,
        # so this is 2.024 at every terminal size.
        return {"scale": scale, "aidx": aidx, "nt": nt,
                "rmax": float(scale.max())}

    _g = ctx.scratch("locket_geo", geo)
    sfield = _g["scale"]

    st = ctx.scratch("locket", lambda: {
        "z": np.zeros(_LOCKET_RINGS, dtype=np.float32),
        # Birth radius, held per pulse. See the release below for why this
        # cannot be recomputed from the current heart.
        "r0": np.zeros(_LOCKET_RINGS, dtype=np.float32),
        "amp": np.zeros(_LOCKET_RINGS, dtype=np.float32),
        "spd": np.ones(_LOCKET_RINGS, dtype=np.float32),
        "beat": 0.0, "acc": 0.0, "since": 0.0,
        "rng": np.random.default_rng(214),
    })
    # ctx.onsets, not a private difference of ctx.onset_seq. Scratch survives a
    # mode switch, so a mode that keeps its own ``last_seq`` comes back from a
    # minute away holding every beat that played while it was not drawing and
    # releases them in one frame. Measured here before the fix: ninety-one
    # onsets on the first frame back, against the one that had just happened.
    onsets = ctx.onsets
    bass = ctx.range(0.0, 0.22)

    # ── the resting heart ──
    # Outline, not a filled shape: the pulses are outlines too, so a solid
    # centre would read as a different object rather than as their source.
    st["beat"] *= math.exp(-max(ctx.dt, 0.0) / 0.14)
    if onsets:
        st["beat"] = min(1.5, st["beat"] + 0.8 + 0.5 * ctx.onset_strength)
    core = float(0.30 + 0.05 * bass + 0.07 * st["beat"])

    # The spectrum, read around the rim.
    #
    # Before this the heart knew only three numbers -- bass, energy and the
    # onset count -- so the whole mode reacted to *how much* was playing and
    # nothing about *what*. Mapping the bands around the outline, mirrored so
    # the two halves match, gives the edge a shape that follows the music:
    # the rim swells and brightens where its band is loud. Mirrored rather
    # than wrapped because the heart is symmetric and a seam running up one
    # side would be the only asymmetric thing on screen.
    nt = _g["nt"]
    lv = resample_bands(ctx.bands, 8).astype(np.float32)
    # Cosine-blended between neighbouring bands, the same easing the shared
    # _angular_bands helper uses, so the rim has no visible band steps. Built
    # over 256 entries, not over the grid.
    tpos = np.linspace(0.0, 7.0, nt, dtype=np.float32)
    t0 = tpos.astype(np.int32)
    t1 = np.minimum(t0 + 1, 7)
    tf = tpos - t0
    tf = (np.float32(1.0) - np.cos(tf * np.float32(math.pi))) * np.float32(0.5)
    band_t = lv[t0] * (np.float32(1.0) - tf) + lv[t1] * tf

    # Radius and thickness both follow it, so a loud band pushes its part of
    # the outline outward as well as lighting it.
    rim_r_t = np.float32(core) * (np.float32(0.94) + np.float32(0.16) * band_t)
    rim_w_t = np.float32(0.020 + 0.018 * st["beat"]) + np.float32(0.016) * band_t
    val_t = (np.float32(0.45) + np.float32(0.30) * band_t
             + np.float32(0.25) * st["beat"]).astype(np.float32)
    ai = _g["aidx"]
    rim = np.abs(sfield - rim_r_t[ai]) < rim_w_t[ai]
    glow = np.where(rim, val_t[ai], np.float32(0.0)).astype(np.float32)

    # No interior fill, and that is a decision rather than an omission. A
    # flat wash was tried and turned the heart into a silhouette; a sparse
    # stipple was tried after it and still read as fill, because braille
    # packs eight dots to a cell and even a few percent of the dots lights
    # most of the cells. The outline is the only continuous line here and the
    # pulses are the only things crossing it, which is what keeps the motion
    # legible.

    # ── pulses, released from inside it ──
    # The free-running release is a fallback for music with no attack to find
    # — a pad, a held chord — not a second source of pulses running alongside
    # the beats. It only accumulates once nothing has hit for a while, because
    # a stream of rings that owe nothing to the music is exactly what stops
    # the rings that do from reading as the beat.
    st["since"] = 0.0 if onsets else st["since"] + max(ctx.dt, 0.0)
    if onsets:
        st["acc"] = 0.0
    elif st["since"] > 1.2:
        st["acc"] += (0.25 + ctx.energy * 0.75) * max(ctx.dt, 0.0)
    want = onsets
    if st["acc"] >= 1.0:
        st["acc"] -= 1.0
        want += 1
    if not want and not (st["z"] > 0.0).any():
        want = 1
    if want:
        slots = list(np.flatnonzero(st["z"] <= 0.0)[:want])
        # A beat with no free slot takes the ring nearest the edge rather than
        # being dropped. Twelve slots sounds generous and is not: a pulse lives
        # about two seconds, so a fast enough beat fills them and the rest draw
        # nothing at all — four of forty-eight at 6 Hz before this, and the gap
        # widens with the rate. The mode's one promise is that a beat produces
        # a pulse, and a beat that silently produces nothing is worse than one
        # that cuts a ring already on its way out of frame. Smallest z is
        # furthest out. Retiring off-screen rings below covers the ordinary
        # rates on its own; this never fires under 6 Hz and carries drum rolls.
        short = want - len(slots)
        if short > 0:
            busy = np.flatnonzero(st["z"] > 0.0)
            if busy.size:
                slots.extend(busy[np.argsort(st["z"][busy])][:short])
        # Born just inside the resting outline, so a pulse starts within the
        # heart and crosses out through it in its first fifth of a second
        # rather than spending most of its life hidden in there. 0.30 put it
        # a third of the way out and cost 73% of every pulse — see the
        # docstring for the measurement.
        #
        # Held per pulse from here on: ``core`` breathes with the bass and
        # the beat, and re-deriving the birth radius every frame moved every
        # ring in flight with it. Each onset teleported the whole field
        # outward — a step 10.6x a normal frame's on average, and backwards
        # whenever the heart shrank.
        born = np.float32(core * 0.80)
        for i in slots:
            st["z"][i] = np.float32(1.0)
            st["r0"][i] = born
            st["amp"][i] = np.float32(0.55 + 0.45 * min(1.0, ctx.onset_strength))
            # A little spread in speed, so two pulses released close together
            # separate as they travel instead of moving as one thick ring.
            st["spd"][i] = np.float32(st["rng"].uniform(0.82, 1.22))

    live = st["z"] > 0.0
    if live.any():
        # Geometric, not linear. ``sc = r0 / z``, so a constant *rate of
        # decay* in z is a constant fractional growth in screen radius: the
        # pulse still accelerates outward, by roughly 4.5x over its life,
        # but the rate cannot diverge the way a linear z does as it
        # approaches zero. The old law spent 73% of a pulse inside the heart
        # and the rest at up to 46x its starting speed.
        #
        # Integrated through dt as a decay factor rather than subtracted,
        # which keeps it frame-rate independent for the same reason the
        # springs in spektr.motion are: exp(-k*dt) composes over substeps
        # where a per-frame multiply does not.
        lam = np.float32(0.95 + ctx.energy * 0.90)
        st["z"][live] *= np.exp(-lam * st["spd"][live] * np.float32(max(ctx.dt, 0.0)))
        st["z"][st["z"] <= 0.02] = 0.0
        # Retire anything wider than the grid instead of holding its slot until
        # z runs out. ``r0 / z > rmax`` is the same test as "off screen",
        # written without the divide.
        gone = (st["z"] > 0.0) & (st["r0"] > st["z"] * np.float32(_g["rmax"]))
        st["z"][gone] = np.float32(0.0)

    for i in np.flatnonzero(st["z"] > 0.0):
        z = float(st["z"][i])
        sc = float(st["r0"][i]) / z
        w = 0.014 + 0.055 * (1.0 - z)
        band = np.abs(sfield - np.float32(sc)) < np.float32(w)
        if not band.any():
            continue
        # Fade as it goes, so the ring dissolves outward instead of hitting
        # the frame edge at full strength.
        a = float(st["amp"][i]) * (0.30 + 0.70 * z)
        np.maximum(glow, np.where(band, np.float32(a), np.float32(0.0)), out=glow)

    lit = glow > np.float32(0.10)
    codes = pack_braille(lit)
    idx = ctx.ramp(np.clip(cell_max(glow), 0.0, 1.0))
    return codes, idx
