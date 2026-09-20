"""Full-field modes: waterfall, plasma, and level meters."""

from __future__ import annotations

import math

import numpy as np

from ..render import (
    shade_cells,
    subcell_rows,
)
from . import Ctx, empty, mode

_UPPER_HALF = ord("▀")


def _chladni(ctx: Ctx, cells: str):
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
    driver), and ``nodal`` is quantised to 12 buckets before ramping so
    neighbouring pixels collapse into the same colour index more often. That
    combination measured worst-case ~7ms at 400x100, down from ~9-14ms and
    frequently over the 11ms slow-mode threshold — which read as exactly the
    "pops up, lags, then catches up" pattern it was reported as, since
    crossing that threshold makes the widget reuse every other frame.
    """
    w, h = ctx.w, ctx.h
    if w < 4 or h < 4:
        return empty(w, h)

    octant = cells == "octant"
    rows2 = h * subcell_rows() if octant else h * 2
    cols = w * 2 if octant else w

    def geo():
        # Cell *centres*, not corners. Sampling the exact boundary hits
        # sin(k*pi*0) and sin(k*pi*1), both identically zero for integer modes,
        # so the whole outer row and column come out as a perfect node and the
        # figure gets a solid lit frame around it that reads as a UI border
        # rather than as part of the plate.
        y = (np.arange(rows2, dtype=np.float32)[:, None] + np.float32(0.5)) / np.float32(rows2)
        x = (np.arange(cols, dtype=np.float32)[None, :] + np.float32(0.5)) / np.float32(cols)
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

    # Separable: each term is an outer product of two 1-D sines, so evaluating
    # this on the 4x2 subcell grid rather than the 1x2 half-row one costs four
    # times the combination and nothing extra in the trig — which is what makes
    # the octant variant affordable at all.
    sx_m = np.sin(np.float32(m * math.pi) * x)
    sy_n = np.sin(np.float32(n * math.pi) * y)
    sx_n = np.sin(np.float32(n * math.pi) * x)
    sy_m = np.sin(np.float32(m * math.pi) * y)
    z = sx_m * sy_n - sx_n * sy_m

    sharpness = np.float32(1.4 + ctx.energy * 3.2)
    nodal = np.clip(np.float32(1.0) - np.abs(z) * sharpness, 0.0, 1.0)
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
    if octant:
        # The field goes in raw, *not* through the 12-bucket quantisation
        # below. A cut needs the gradient to find the midpoint with, and
        # quantising first flattens exactly the information it works from.
        #
        # Shaded rather than masked, and this mode is the one that settled how.
        # A nodal field is two different problems at once: smooth almost
        # everywhere, with thin sharp curves through it. shade_cells splits it
        # the same way — the smooth part is drawn as a half-block with two exact
        # colours, and only a cell the curve passes through spends its glyph on
        # the shape. Dithering was tried for both halves and was wrong for both:
        # it turned the smooth part into an all-over stipple, and then, once
        # that was fixed, it turned the curves themselves into a halftone
        # screen. It also grades itself, so the hand-tuned bucket count that
        # used to live here is gone.
        return shade_cells(nodal)

    nodal = np.round(nodal * np.float32(12.0)) * np.float32(1.0 / 12.0)

    idx = ctx.ramp(nodal)
    codes = np.full((h, w), _UPPER_HALF, dtype=np.int32)
    return codes, idx[0::2], idx[1::2]


@mode("Chladni", group="fields",
      blurb="nodal interference pattern, plate modes set by the dominant pitch")
def chladni(ctx: Ctx):
    return _chladni(ctx, "half")


@mode("Chladni (o)", hidden=True, after="Chladni", group="fields",
      blurb="the same plate at eight samples a cell, nodal lines antialiased — needs Unicode 16 octants")
def chladni_fine(ctx: Ctx):
    """Chladni on octant cells.

    The mode the rendering audit was written about. A nodal line is a thin
    curve through a field with many local extrema, which is the worst case for
    a renderer that can only place an edge on a cell boundary: at 1x2 samples
    a cell the curve is a staircase, and the fix is not more colours but more
    places to put the edge. Eight samples a cell, cut against the cell's own
    midpoint, put it between them.
    """
    return _chladni(ctx, "octant")


def _chladni_flow(ctx: Ctx, cells: str):
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
    driver), and ``nodal`` is quantised to 12 buckets before ramping so
    neighbouring pixels collapse into the same colour index more often. That
    combination measured worst-case ~7ms at 400x100, down from ~9-14ms and
    frequently over the 11ms slow-mode threshold — which read as exactly the
    "pops up, lags, then catches up" pattern it was reported as, since
    crossing that threshold makes the widget reuse every other frame.
    """
    w, h = ctx.w, ctx.h
    if w < 4 or h < 4:
        return empty(w, h)

    octant = cells == "octant"
    rows2 = h * subcell_rows() if octant else h * 2
    # Vertical only, the same trade Chladni Extreme documents at length and for
    # the same reason: the rotation makes every sample cost four broadcast
    # multiplies over the whole field, and at 4x2 a cell that was 3.0 ms of a
    # 12.4 ms frame in `_rot_sines` alone. At 4x1 it is half of that, and the
    # resolution it gives up is the horizontal one — a text cell is about twice
    # as tall as it is wide, so the coarseness worth spending on is vertical.
    # The two subcell columns of a cell then share a value, which costs
    # something on a near-vertical nodal line and nothing on the horizontal
    # ones the figure is mostly made of.
    cols = w

    def geo():
        y = np.arange(rows2, dtype=np.float32)[:, None] / np.float32(max(1, rows2 - 1))
        x = np.arange(cols, dtype=np.float32)[None, :] / np.float32(max(1, cols - 1))
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

    # Separated, the way Chladni Extreme has always done it. This built two
    # rotated 2-D grids and took four sines over the whole field — the one
    # thing _rot_sines exists to avoid, sitting next to it unused. Four
    # transcendentals over 80,000 cells became four over a 400-long row and a
    # 200-long column, which is most of why the octant variant fits at all.
    ax = x - np.float32(0.5)
    by = y - np.float32(0.5)
    sx_m, sy_m = _rot_sines(m, cs, sn, ax, by)
    sx_n, sy_n = _rot_sines(n, cs, sn, ax, by)
    z = sx_m * sy_n - sx_n * sy_m

    # In place through ``z``, which nothing else holds: at the octant grid this
    # is 320,000 elements and the operator form built four temporaries of it.
    np.abs(z, out=z)
    z *= np.float32(1.4 + ctx.energy * 3.2)
    np.subtract(np.float32(1.0), z, out=z)
    np.clip(z, 0.0, 1.0, out=z)
    z *= z
    nodal = z

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
    if octant:
        # Shaded rather than masked — see Chladni (o).
        #
        # A coarser colour block than the default, the same as Chladni Extreme
        # and for the same reason: this figure sweeps continuously, so a broad
        # nodal region is crossed by many cells each picking a slightly
        # different colour pair, and every distinct pair is a run boundary
        # make_strips has to pay for. At 400x100 that measured 14,721 runs
        # against 9,919, worth 1.3 ms of the frame — which is the difference
        # between sitting on the 16.7 ms budget and sitting clear of it. It
        # costs nothing visible: the block moves the two colours, never the
        # threshold, so the nodal lines land in exactly the same place.
        #
        # One repeat rather than twice the trig, exactly as Chladni Extreme
        # does it — the field is built at one sample per cell column and the
        # cell's two subcell columns take the same value.
        #
        # Block 12 rather than 8: measured 8,190 colour runs against 10,701 at
        # 400x100, worth 2 ms of the frame, and rasterising both to pixels and
        # looking showed no difference at all. Chladni Extreme cannot take the
        # same widening — its field is a broad smooth wash rather than a
        # plateau with lines through it, so at 16 the contours start to band
        # and at 24 it is obvious.
        return shade_cells(np.repeat(nodal, 2, axis=1), block=12)

    nodal = np.round(nodal * np.float32(12.0)) * np.float32(1.0 / 12.0)

    idx = ctx.ramp(nodal)
    codes = np.full((h, w), _UPPER_HALF, dtype=np.int32)
    return codes, idx[0::2], idx[1::2]


@mode(
    "Chladni Flow",
    group="fields",
    blurb="a plate figure melting continuously from one resonance into the next",
)
def chladni_flow(ctx: Ctx):
    return _chladni_flow(ctx, "half")


@mode("Chladni Flow (o)", hidden=True, after="Chladni Flow", group="fields",
      blurb="the melting plate at eight samples a cell — needs Unicode 16 octants")
def chladni_flow_fine(ctx: Ctx):
    return _chladni_flow(ctx, "octant")


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


def _chladni_extreme(ctx: Ctx, cells: str):
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

    octant = cells == "octant"
    rows2 = h * subcell_rows() if octant else h * 2
    # Vertical only, unlike its three siblings: this is the heaviest field in
    # the app -- two figures, a harmonic and a rotation -- and sampling it at
    # 4x2 a cell put the frame over the 16.7 ms budget outright on a loaded
    # machine. At 4x1 it costs half that and still quadruples the vertical
    # resolution, which is the axis that matters: a text cell is about twice as
    # tall as it is wide, so the half-block renderer's coarseness is vertical.
    # The two subcell columns of a cell then share a value, which is a real
    # loss on a near-vertical nodal line and no loss at all on the horizontal
    # ones the figure is mostly made of.
    cols = w

    def geo():
        # Centred once: every use of the grid here is relative to the middle.
        by = (np.arange(rows2, dtype=np.float32)[:, None] / np.float32(max(1, rows2 - 1))) - np.float32(0.5)
        ax = (np.arange(cols, dtype=np.float32)[None, :] / np.float32(max(1, cols - 1))) - np.float32(0.5)
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
    # In place through ``z``, which nothing else holds — see Chladni Flow.
    np.abs(z, out=z)
    z *= np.float32(1.4 + ctx.energy * 3.2 + charge * 1.0 + punch * 2.2)
    np.subtract(np.float32(1.0), z, out=z)
    np.clip(z, 0.0, 1.0, out=z)
    z *= z
    nodal = z
    # Eight buckets, not the twelve its two siblings use, and done in place:
    # this is the heaviest mode in the app and over half its frame is the strip
    # builder, which pays per colour boundary. Eight costs almost nothing
    # visually because sharpness runs to 7 and beyond, and a sharp nodal field
    # is already nearly bimodal - most cells sit hard against 0 or 1 rather
    # than in the midtones the extra buckets would resolve.
    if octant:
        # Shaded rather than masked — see Chladni (o).
        #
        # The field is one sample per subcell *row* here, so each cell's two
        # subcell columns take the same value — one repeat rather than twice
        # the trig, sines included.
        return shade_cells(np.repeat(nodal, 2, axis=1), block=8)

    nodal *= np.float32(8.0)
    np.round(nodal, 0, out=nodal)
    nodal *= np.float32(1.0 / 8.0)



    idx = ctx.ramp(nodal)
    codes = np.full((h, w), _UPPER_HALF, dtype=np.int32)
    return codes, idx[0::2], idx[1::2]


@mode(
    "Chladni Extreme",
    group="fields",
    blurb="a plate driven past its modes - morphs, escalates, and snaps on the beat",
)
def chladni_extreme(ctx: Ctx):
    return _chladni_extreme(ctx, "half")


@mode("Chladni Extreme (o)", hidden=True, after="Chladni Extreme", group="fields",
      blurb="the overdriven plate at eight samples a cell — needs Unicode 16 octants")
def chladni_extreme_fine(ctx: Ctx):
    return _chladni_extreme(ctx, "octant")
