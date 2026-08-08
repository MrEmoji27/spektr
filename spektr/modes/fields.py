"""Full-field modes: waterfall, plasma, and level meters."""

from __future__ import annotations

import math

import numpy as np

from ..analysis import resample_bands
from ..palette import RAMP_STEPS
from ..render import SHADES, SPACE, cell_max, pack_braille
from . import Ctx, empty, mode, spread
from .particles import _angular_bands, _polar

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
    bass = ctx.range(0.0, 0.18)

    st = ctx.scratch(
        "chladni_x",
        lambda: {
            "c": centroid, "e": highs, "charge": 0.0, "spin": 0.0,
            "fast": bass, "slow": bass, "punch": 0.0, "hit_t": -99.0,
            "level": ctx.energy,
        },
    )
    # Chases rather than drifts: a third of Chladni's time constant.
    st["c"] += (centroid - st["c"]) * min(1.0, ctx.dt / 0.12)
    st["e"] += (highs - st["e"]) * min(1.0, ctx.dt / 0.12)

    # -- the beat --
    # A 20 ms attack against a 300 ms reference. Faster than the 30 ms the rest
    # of the codebase uses, because at 188 analyses/sec there is the resolution
    # for it, and on this mode a kick that lands late reads as lag.
    st["fast"] += (bass - st["fast"]) * min(1.0, ctx.dt / 0.02)
    st["slow"] += (bass - st["slow"]) * min(1.0, ctx.dt / 0.30)
    onset = st["fast"] - st["slow"]
    # Percussion is spectrally flat, a pad is not. The same kick energy under a
    # sustained chord should not throw the plate around.
    groove = float(np.clip((ctx.flatness - 0.35) / 0.45, 0.0, 1.0))
    if onset > 0.09 and (ctx.t - st["hit_t"]) > 0.09:
        st["hit_t"] = ctx.t
        st["punch"] = min(1.4, st["punch"] + (0.45 + onset * 1.6) * (0.35 + groove))
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

    Mids weigh more than bass, so the reading tracks perceived loudness instead
    of whatever the kick drum is doing. Shared by both meter modes so they can
    never disagree about how loud the signal is.
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


# ── rhythm helpers ───────────────────────────────────────────────────────────

def _onsets_since(ctx: Ctx, st: dict) -> int:
    """Onsets since the previous frame, by differencing the monotonic count.

    The rhythm contract for :attr:`Ctx.onset_seq` is explicit: key on it
    *changing*, never on ``onset_strength > 0`` — the analyser drops two
    analyses in three at 60 fps and a boolean would be missed most of the
    times it was true, while a counter cannot be. The count also survives
    silence, so a pause cannot masquerade as a burst of beats. A negative
    delta is a detector bug, not a beat; it reads as nothing.
    """
    n = ctx.onset_seq - st["last_seq"]
    st["last_seq"] = ctx.onset_seq
    return max(n, 0)


@mode("Kaleidoscope", group="fields", blurb="radial mirror symmetry — the wedge count follows the spectrum")
def kaleidoscope(ctx: Ctx):
    """A real mirror array, not a spun copy of a picture.

    Every dot is folded into a single angular wedge — take its angle, wrap
    it into one sector, then mirror it inside that sector — so the picture is
    genuinely symmetric: each sector is the same slice reflected, the way
    the tubes of a real kaleidoscope bounce one source image around a ring.

    The wedge count is the musical part. It eases toward a figure set by the
    spectral centroid over a 1.2 s constant, and the *discrete* count only
    changes when the eased value crosses an integer — never per frame, or
    the figure strobes. A beat snaps it straight to a bass-derived count:
    that one instant of abrupt change reads as intent, and it is the reason
    the rhythm fields exist. The same beat also kicks the rotation.

    Rotation integrates through ``ctx.dt`` because its rate is audio-driven.
    Read against ``ctx.t``, a change of rate would retroactively rewrite the
    whole rotation history and the picture would teleport — the exact trap
    Retro's scroll and Chladni Extreme's spin document, and the audit's
    frame-rate-independence check is built to catch it.

    Colour walks the ramp by radius, with the band level at the dot's folded
    angle acting as brightness — so every sector shows the same frequency
    profile, and the whole ring flickers with the spectrum.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 8 or dc < 8:
        return empty(ctx.w, ctx.h)

    # VU's sweep already owns the name at module level — import locally so
    # this mode can share the render helper without shadowing it.
    from ..render import frac

    dist, turn, rmax = _polar(ctx)
    r = dist / max(rmax, 1.0)

    bands8 = ctx.display_bands(8).astype(np.float64)
    total = float(bands8.sum())
    centroid = float((bands8 * np.arange(8)).sum() / total / 7.0) if total > 1e-9 else 0.0
    bass = ctx.range(0.0, 0.18)

    st = ctx.scratch("kaleido", lambda: {"kc": 4.0, "spin": 0.0, "last_seq": ctx.onset_seq})
    onsets = _onsets_since(ctx, st)

    # the eased count, and the discrete wedge count it rounds to. The snap on
    # a beat bypasses the ease entirely — that is what makes hits land.
    st["kc"] += (4.0 + centroid * 6.0 - st["kc"]) * min(1.0, ctx.dt / 1.2)
    if onsets:
        st["kc"] = float(4 + min(6, round(bass * 6.0)))
    k = int(round(st["kc"]))
    k = max(3, min(k, 10))

    # spin accumulates through dt (audio-driven rate — see the docstring),
    # plus an angular kick per beat so hits land as a visible lurch.
    st["spin"] = (st["spin"] + (0.22 + ctx.energy * 0.8) * max(ctx.dt, 0.0)) % (2 * math.pi)
    if onsets:
        st["spin"] = (st["spin"] + 0.3 * min(onsets, 3)) % (2 * math.pi)
    spin = st["spin"]

    # The pattern is smooth — a handful of cycles across the whole field —
    # so it is drawn on a half-resolution grid and upsampled: a quarter of
    # the sin() and gather work per frame, with no visible change to shapes
    # many dots wide.
    ph = dr % 2
    pw = dc % 2
    if ph or pw:
        turn = np.pad(turn, ((0, ph), (0, pw)), mode="edge")
        r = np.pad(r, ((0, ph), (0, pw)), mode="edge")
    turn = turn[::2, ::2]
    r = r[::2, ::2]

    # fold: turn*k wraps into k sectors; the min() mirrors each sector's own
    # half, so every wedge is bilaterally symmetric about its centre line.
    wedge = turn * np.float32(k) + np.float32(spin * k / (2 * math.pi))
    folded = np.minimum(frac(wedge), 1.0 - frac(wedge)) * 2.0

    # rings compressed by the centroid, stripes per wedge by the bass; the
    # second term adds a slowly precessing radial ring that keeps the spin
    # readable in the fold (a pure rotation of a perfectly symmetric figure
    # is invisible, and this is what makes it visibly turn).
    f_r = 2.0 + centroid * 2.5
    f_a = 1.5 + bass * 2.5
    v = np.sin((folded * f_a + r * f_r) * np.float32(math.pi) + np.float32(spin * 2.0))
    v += np.sin(r * np.float32(math.pi) * 3.0 - np.float32(spin)) * 0.5

    band_lvl = _angular_bands(ctx, folded * np.float32(0.5), 8, np.float32(0.0))
    bright = np.clip(
        (0.5 + 0.5 * v) * (0.4 + ctx.energy * 1.2) * (0.55 + band_lvl * 0.9),
        0.0, 1.0,
    )
    bright = np.repeat(np.repeat(bright, 2, axis=0), 2, axis=1)[:dr, :dc]
    dots = bright > 0.45
    codes = pack_braille(dots)

    # colour from radius and band level: the ramp position walks outward
    # with the radius, the brightness rides the frequency profile above;
    # the cell-max reduction is folded into the cache so the per-frame path
    # never re-reduces a static grid
    def cell_r():
        cx, cy = dc / 2.0, dr / 2.0
        x_scale = cy / max(cx, 1.0)
        xs = (np.arange(dc, dtype=np.float32) - cx) * x_scale
        ys = np.arange(dr, dtype=np.float32) - cy
        r = (np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2) / max(1.0, cy - 1.0)).astype(np.float32)
        return cell_max(r)

    cr = ctx.scratch("kaleido_r", cell_r)
    col = cell_max(np.where(dots, bright, 0.0))
    idx = ctx.ramp(np.clip(col * 0.5 + cr * 0.5, 0.0, 1.0))
    return codes, idx


@mode("Sterling", group="fields", blurb="engraved silver — a mirrored gothic emblem with a hard specular edge")
def sterling(ctx: Ctx):
    """Polished metal, not a soft gradient: a hard specular ramp on a dark ground.

    The picture is an ornamental emblem — dagger, cross, and slowly turning
    fleur-de-lis scrollwork — drawn in a mirror: every feature is a function
    of ``|x|``, so the composition is bilaterally symmetric by construction
    rather than by accident of layout. The aesthetic is heavy jewellery:
    deep black ground, the metal body a dark silver, and a narrow band of
    near-white along the edge that faces the light.

    The silver is the point, and it is a shading model, not a colour choice.
    The ornament is an implicit surface — each shape is a Gaussian ridge
    whose height falls to zero away from the centre line — and the frame
    computes the surface's gradient by central differences, then lights it
    from the upper left. Squaring the clamped light term is what makes the
    highlight *hard*: most of each ridge's slope is dark, and only the
    steepest gradient facing the light escapes the square, so the picture
    reads as polished sterling with a single sharp glint rather than as a
    smoothed glow. A beat flash multiplies the same term, so a kick glints
    off the metal instead of brightening it evenly.

    The emblem grows and retreats with the smoothed level — the whole
    composition scales from the centre, so a loud bar swells the silverwork
    and a quiet one lets it recede — and the two base curls turn slowly on
    their anchors in seconds, which is what keeps a held tone alive without
    breaking the frame-rate rule (a fixed rate read against ``ctx.t`` is
    fine; only audio-driven rates need to accumulate through ``ctx.dt``).
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 16 or dc < 16:
        return empty(ctx.w, ctx.h)

    def geo():
        cx, cy = dc / 2.0, dr / 2.0
        x_scale = cy / max(cx, 1.0)      # braille dots are ~2x taller than wide
        a = (np.abs(np.arange(dc, dtype=np.float32) - cx) * x_scale)[None, :] / cy
        b = (np.arange(dr, dtype=np.float32) - cy)[:, None] / cy
        return a, b

    a, b = ctx.scratch("sterling_geo", geo)

    bass = ctx.range(0.0, 0.18)

    st = ctx.scratch("sterling", lambda: {
        "growth": 0.6, "last_seq": ctx.onset_seq,
        "fast": bass, "slow": bass, "hit_t": -99.0, "punch": 0.0,
    })
    onsets = _onsets_since(ctx, st)

    # A beat punch on the same fast/slow bass envelope as Chladni Extreme,
    # OR'd with the rhythm counter so it also fires once the real detector
    # lands; decays over ~180 ms so hits read as separate glints.
    st["fast"] += (bass - st["fast"]) * min(1.0, ctx.dt / 0.02)
    st["slow"] += (bass - st["slow"]) * min(1.0, ctx.dt / 0.30)
    if (st["fast"] - st["slow"] > 0.09 and (ctx.t - st["hit_t"]) > 0.09) or onsets:
        st["hit_t"] = ctx.t
        st["punch"] = min(1.4, st["punch"] + 0.45 + 0.25 * min(onsets, 3))
    st["punch"] *= math.exp(-max(ctx.dt, 0.0) / 0.18)

    # growth swells the whole composition from the centre. Dividing the
    # coordinates by the eased level scales the shapes outward; the range is
    # tuned so the base bar stays on screen at full loudness.
    st["growth"] += (0.55 + ctx.energy * 0.45 - st["growth"]) * min(1.0, ctx.dt / 0.6)
    g = st["growth"]
    aa = a / g
    bb = b / g

    # Every ridge is several dots wide, so the surface is drawn on a
    # half-resolution grid and upsampled before lighting: a quarter of the
    # exp()/hypot() work per frame, and the bilinear upsample keeps the
    # specular edge where the full-res surface would put it.
    ph = dr % 2
    pw = dc % 2
    if ph or pw:
        aa = np.pad(aa, ((0, ph), (0, pw)), mode="edge")
        bb = np.pad(bb, ((0, ph), (0, pw)), mode="edge")
    aa = aa[::2, ::2]
    bb = bb[::2, ::2]

    def ridge(t, w):
        return np.exp(-(t / w) ** 2)

    # a ring, not a filled disc: the Gaussian peaks at radius r0, but the
    # squared argument would also light the interior, so a ramp clips the
    # disc out — the annulus keeps a soft inner edge that the light catches.
    def ring(t, r0, w):
        return ridge(t - r0, w) * np.clip((t - 0.5 * r0) * (2.0 / r0), 0.0, 1.0)

    # the dagger stands on the base bar, reading top to bottom: an orb
    # finial, the blade tapering from its base to a point, a small ring low
    # on the blade, the long crossguard, a curl wrapped under each guard
    # end, the pommel spike, and the base. The whole composition scales by
    # g, so a loud bar swells the metal toward the screen edges.
    base = ridge(bb - 0.85, 0.028) * ridge(aa, 0.26)
    pommel = ridge(aa, 0.03) * np.clip((0.78 - bb) * 4.0, 0.0, 1.0) * np.clip((bb - 0.50) * 4.0, 0.0, 1.0)
    guard = ridge(bb - 0.34, 0.045) * ridge(aa, 0.20)
    blade_w = 0.02 + 0.075 * (bb + 0.78) / 1.08
    blade = ridge(aa / np.maximum(blade_w, 0.005), 1.0) * np.clip(0.30 - bb, 0.0, 1.0) * np.clip((bb + 0.78) * 2.0, 0.0, 1.0)
    finial = ring(np.hypot(aa, bb + 0.86), 0.06, 0.026)

    # fleur-de-lis scrollwork: a curl below each guard end and a small ring
    # low on the blade, the curls turning slowly on their anchors; the
    # mirror does the other side
    turn = ctx.t * 0.5
    ca = 0.28 + 0.05 * math.cos(turn)
    cb = 0.62 + 0.05 * math.sin(turn)
    curl = ring(np.hypot(aa - np.float32(ca), bb - np.float32(cb)), 0.075, 0.026)
    ringlet = ring(np.hypot(aa - 0.13, bb + 0.02), 0.05, 0.02)

    h = np.maximum.reduce([base, pommel, guard, blade, finial, curl, ringlet])

    # bilinear 2x upsample back to the dot grid, edge-clamped: even rows
    # hold the surface, odd rows and columns are the linear middles, so the
    # gradient below is a smooth function of the true surface
    h0 = h
    h = np.empty((2 * h0.shape[0], 2 * h0.shape[1]), dtype=np.float32)
    h[0::2, 0::2] = h0
    h[0::2, 1:-1:2] = 0.5 * (h0[:, :-1] + h0[:, 1:])
    h[0::2, -1] = h0[:, -1]
    h[1:-1:2, :] = 0.5 * (h[0:-2:2, :] + h[2::2, :])
    h[-1, :] = h[-2, :]
    h = h[:dr, :dc]

    # gradient by central differences — the surface the light hits
    hx = np.empty_like(h)
    hx[:, :-1] = h[:, 1:] - h[:, :-1]
    hx[:, -1] = hx[:, -2]
    hy = np.empty_like(h)
    hy[:-1, :] = h[1:, :] - h[:-1, :]
    hy[-1, :] = hy[-2, :]

    # one hard light from the upper left: the surfaces that rise toward it
    # (hx, hy > 0) catch it, and the square is what makes the specular
    # narrow — see the docstring. The body is a dim silver so the metal
    # reads as metal even where the light misses it.
    spec = np.clip((hx + hy) * (1.0 + st["punch"] * 1.6), 0.0, 1.0) ** 2.0
    body = np.clip((h - 0.10) * 2.4, 0.0, 1.0) * 0.42
    v = np.clip(body + spec, 0.0, 1.0)

    dots = v > 0.045
    codes = pack_braille(dots)
    idx = ctx.ramp(cell_max(np.where(dots, v, 0.0)))
    return codes, idx
