"""Full-field modes: waterfall, plasma, and level meters."""

from __future__ import annotations

import math

import numpy as np

from ..analysis import resample_bands
from ..audio.drums import named
from ..render import (
    cell_max,
    pack_braille,
    pack_octant_bits,
)
from . import Ctx, empty, mode

#: Most hearts a Valentine frame will carry. Bounded so a dense passage
#: cannot grow the particle arrays without limit.
_VAL_HEARTS = 24


def _valentine(ctx: Ctx, octant: bool):
    """A heart that actually beats, rather than a heart that pulses.

    Two modes share this body. ``octant=False`` packs the lit dots into
    braille; ``octant=True`` packs the identical dot set into Unicode 16
    octant glyphs. Nothing else differs — same shape, same beat, same colour.

    That one substitution is worth a mode here because this is a silhouette.
    Braille draws eight separated round dots per cell, so a filled heart is a
    field of stipple with a ragged edge; an octant cell is a solid block
    mosaic at the same resolution, so the same dots become a surface with a
    clean rim. The foreground is the only colour either version sets, which
    leaves the unlit subcells showing the terminal's own background — a
    background index would paint the space around the heart opaque, which is
    exactly what a shape drawn against empty space must not do.

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

    lit = field > 0.0
    codes = pack_octant_bits(lit) if octant else pack_braille(lit)
    idx = ctx.ramp(np.clip(cell_max(field), 0.0, 1.0))
    return codes, idx


@mode("Valentine", group="fields",
      blurb="a heart that beats with the track, trailing smaller ones upward")
def valentine(ctx: Ctx):
    return _valentine(ctx, octant=False)


@mode("Valentine (o)", hidden=True, after="Valentine", group="fields",
      blurb="the same heart drawn solid instead of stippled — needs a terminal that draws Unicode 16 octants")
def valentine_fine(ctx: Ctx):
    """Valentine on octant cells.

    Separate mode rather than a switch on the original, for the same reason
    Kaleidoscope (o) is: octants are Unicode 16 and an older terminal or font
    draws a grid of tofu. That is a thing to opt into, not to discover when a
    mode you liked stops working.
    """
    return _valentine(ctx, octant=True)


#: Concurrent pulses in Locket.
_LOCKET_RINGS = 12


def _locket_geo(dr: int, dc: int) -> dict:
    """Locket's grid, built once per size. See :func:`locket`."""
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
    rmax = float(scale.max())

    # Every pulse is a function of ``scale`` alone — a band around one
    # value of it — so the whole set of them can be answered by a table
    # over ``scale`` and one gather, exactly as the rim is answered by a
    # table over angle. Without this, a soft-edged ring costs four passes
    # over the dot grid *per pulse* and twelve can be in flight: 13 ms of
    # build at 400x100 against 2 ms for a table 1024 long.
    #
    # 1024 buckets puts 7 to 35 of them across a ring, whose width runs
    # 0.014 to 0.069 of the same scale — fine enough that the falloff
    # arrives graded rather than stepped.
    # int16, not int32: the index only has to reach 1023, and the gather
    # below is a random read over 320k dots, where halving the index
    # traffic is worth more than the cast costs once per size.
    nsc = 1024
    sidx = np.clip((scale * np.float32((nsc - 1) / max(rmax, 1e-6))
                    ).astype(np.int32), 0, nsc - 1).astype(np.int16)
    sc_at = np.linspace(0.0, rmax, nsc, dtype=np.float32)

    return {"scale": scale, "aidx": aidx, "nt": nt, "rmax": rmax,
            "sidx": sidx, "sc_at": sc_at, "nsc": nsc}



def _band_table(ctx: Ctx, nt: int, bands: int = 8) -> np.ndarray:
    """The spectrum as a table over the heart's angle, ``nt`` entries long.

    Read through ``_locket_geo``'s ``aidx``: entry 0 is the heart's point and
    the table climbs each side symmetrically, so a band is mirrored left and
    right. Cosine-blended between neighbouring bands, the same easing the
    shared ``_angular_bands`` helper uses, so there are no visible band steps.
    """
    lv = resample_bands(ctx.bands, bands).astype(np.float32)
    top = bands - 1
    tpos = np.linspace(0.0, float(top), nt, dtype=np.float32)
    t0 = tpos.astype(np.int32)
    t1 = np.minimum(t0 + 1, top)
    tf = tpos - t0
    tf = (np.float32(1.0) - np.cos(tf * np.float32(math.pi))) * np.float32(0.5)
    return lv[t0] * (np.float32(1.0) - tf) + lv[t1] * tf


def _locket_rim(ctx: Ctx, _g: dict, core: float, beat: float) -> np.ndarray:
    """The resting heart's outline, swelling where its band is loud. See
    :func:`locket` for why the spectrum is read around the rim."""
    sfield = _g["scale"]
    band_t = _band_table(ctx, _g["nt"])

    # Radius and thickness both follow it, so a loud band pushes its part of
    # the outline outward as well as lighting it.
    rim_r_t = np.float32(core) * (np.float32(0.94) + np.float32(0.16) * band_t)
    rim_w_t = np.float32(0.020 + 0.018 * beat) + np.float32(0.016) * band_t
    val_t = (np.float32(0.45) + np.float32(0.30) * band_t
             + np.float32(0.25) * beat).astype(np.float32)
    ai = _g["aidx"]
    rim = np.abs(sfield - rim_r_t[ai]) < rim_w_t[ai]
    # No astype here. ``val_t`` is float32 and a float32 times a bool is
    # float32, so the cast was a second full-size copy of the dot grid that
    # changed nothing — 320k floats a frame at 400x100.
    glow = val_t[ai] * rim
    return glow


def _locket_out(ctx: Ctx, glow: np.ndarray):
    """Dots and colours for a heart field. See :func:`locket`."""
    lit = glow > np.float32(0.10)
    codes = pack_braille(lit)

    # Graded values are what the soft edge is for, but the two-colour strip
    # builder pays per colour boundary and a continuous falloff hands it one
    # per cell: strips went from 0.8 ms to 4.2 ms at 400x100 when the rings
    # stopped being flat. Rounding to twelve levels — done on the *cell* grid,
    # which is an eighth the size of the dot grid — gives most of that back and
    # is not visible: twelve steps across a ring three to eight cells wide is
    # finer than the ramp itself resolves.
    cm = np.clip(cell_max(glow), 0.0, 1.0)
    np.multiply(cm, np.float32(10.0), out=cm)
    np.round(cm, 0, out=cm)
    np.multiply(cm, np.float32(1.0 / 10.0), out=cm)
    idx = ctx.ramp(cm)
    return codes, idx


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

    **The rings are soft-edged, and they are all drawn by one gather.** A pulse
    is a band around one value of ``scale``, and it used to be exactly that: a
    single brightness inside the band and nothing outside it. At braille
    resolution that is a stack of hard steps which shears as the radius moves
    between frames, so the pulses read as chunky arcs rather than as travelling
    light. They now carry a squared falloff across the same width — bright in
    the middle, dissolving at both edges — and they fade *in* over the first
    tenth of their journey, so a ring emerges from the heart instead of
    appearing at full strength on top of it.

    Both would be expensive done per pulse over the dot grid: four passes each,
    twelve pulses in flight, 13 ms of build at 400x100. Since every pulse is a
    function of ``scale`` alone, the whole set of them is resolved on a
    1024-entry table over ``scale`` and read back with one gather — the same
    shape of trick the rim uses over angle — which puts it back at 8 ms.

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

    _g = ctx.scratch("locket_geo", lambda: _locket_geo(dr, dc))

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
    # 0.22 rather than 0.30. The heart is the *source*, not the subject: at
    # 0.30 it took a third of the height and a pulse spent its first half
    # crossing it, so the frame read as one big heart with rings stuck to it.
    # Smaller leaves the travel visible, which is the motion the mode is about,
    # and the outline is no less legible for it — the notch that makes the
    # silhouette a heart survives down to about a sixth of the height.
    core = float(0.22 + 0.04 * bass + 0.05 * st["beat"])

    # The spectrum, read around the rim.
    #
    # Before this the heart knew only three numbers -- bass, energy and the
    # onset count -- so the whole mode reacted to *how much* was playing and
    # nothing about *what*. Mapping the bands around the outline, mirrored so
    # the two halves match, gives the edge a shape that follows the music:
    # the rim swells and brightens where its band is loud. Mirrored rather
    # than wrapped because the heart is symmetric and a seam running up one
    # side would be the only asymmetric thing on screen.
    glow = _locket_rim(ctx, _g, core, st["beat"])

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

    # Every live pulse, resolved on a 1024-entry table over ``scale`` and read
    # back with one gather. Soft edges, not a hard band: the ring used to be
    # one value across its whole width and nothing outside it, which at braille
    # resolution is a stack of hard steps that shears as the radius moves
    # between frames — the pulses read as chunky arcs rather than as travelling
    # light. A squared falloff across the same width grades the ramp instead,
    # so a ring has a bright middle and dissolves at both edges.
    sc_at = _g["sc_at"]
    lut = None
    for i in np.flatnonzero(st["z"] > 0.0):
        z = float(st["z"][i])
        sc = float(st["r0"][i]) / z
        w = 0.014 + 0.055 * (1.0 - z)
        d = np.abs(sc_at - np.float32(sc))
        band = d < np.float32(w)
        if not band.any():
            continue
        # Fade as it goes, so the ring dissolves outward instead of hitting the
        # frame edge at full strength, and fade *in* over the first tenth of
        # the journey so it emerges from the heart rather than appearing at
        # full strength on top of it. ``1 - z`` is how far along the pulse is
        # and is already integrated through dt, so the ramp costs nothing and
        # stays frame-rate independent.
        a = float(st["amp"][i]) * (0.30 + 0.70 * z) * min(1.0, (1.0 - z) / 0.10)
        dm = d * np.float32(1.0 / w)
        soft = np.where(band, np.float32(a) * (np.float32(1.0) - dm * dm),
                        np.float32(0.0))
        lut = soft if lut is None else np.maximum(lut, soft, out=lut)

    if lut is not None:
        # ``lut[idx]`` rather than ``np.take(lut, idx, out=buf)``. The out=
        # form was tried to save the 1.3 MB allocation and measured slower over
        # three runs each — 10.1-10.8 ms of build against 8.7-9.0 — because
        # take's bounds-checked path over 320k dots costs more than the
        # allocation it avoids.
        np.maximum(glow, lut[_g["sidx"]], out=glow)

    return _locket_out(ctx, glow)










#: Rings in flight at once. A new ring with no free slot takes the oldest's.
#: Enough for a loud stream: four to a beat, flying for two beats.
_SHOT_RINGS = 12

#: How each drum's ring is drawn: width against a plain beat, and brightness.
#: A kick is a thick heavy ring, a hat a thin one. A beat whose hit the drums
#: cannot name is drawn as a plain one.
_SHOT = {"kick": (1.8, 1.0), "snare": (1.2, 0.9), "hat": (0.6, 0.7)}

#: How many beats a ring takes to cross from the heart to the edge of the frame.
_SHOT_BEATS = 2.0

#: The flight time in seconds is held to this, whatever the tempo, and is the
#: flight time when there is no tempo at all.
_SHOT_LIFE_RANGE = (0.7, 1.8)
_SHOT_LIFE_S = 1.1

#: A plain ring's width, as a share of the heart-scale. Thin, with a crisp
#: edge, so rings in flight together stay separate rings.
_SHOT_WIDTH = 0.024

#: Where a ring's flight ends, on the heart-scale: just past the edge of the
#: frame. Measured, and the same at every size because the scale is
#: normalised: the frame's border runs from 0.95 to 2.02 with a median of
#: 1.22. The flight used to end at 2.02, the far corner, so a ring reached the
#: edge a quarter of the way through its life, left in a flash, and spent the
#: rest as arcs in the corners -- which is what read as too fast, and as a gap.
_RING_END = 1.3

#: A hit this close to a beat, as a share of the beat, is that beat's hit: it
#: sets how the beat's ring is drawn rather than throwing one of its own.
_ON_BEAT = 0.2

#: The stream between the beats: how many rings a beat is divided into, by how
#: loud the music is. Two at an ordinary level, four when it is loud, and none
#: below the quiet floor -- so the stream thickens and thins with the track.
_STREAM_QUIET = 0.06
_STREAM_LOUD = 0.35

#: With no tempo, the stream runs on its own clock: a ring this often, in
#: seconds, at an ordinary level, and twice as often when it is loud.
_STREAM_FREE_S = 0.28


@mode("Locket Beat", group="fields", after="Locket",
      blurb="the locket heart, sending out a steady stream of rings, bright on every beat")
def locket_beat(ctx: Ctx):
    """``Locket`` sending out a steady stream of rings, in time with the music.

    ``Locket`` keeps its sky full whatever is playing: a free-running release
    fills in when nothing hits, and its rings read as a cascade. Here the rings
    are a stream kept in time. When the analyser has a tempo, a bright ring
    leaves the heart on every beat and the heart pumps as it goes; the hit that
    lands on that beat decides how it looks -- a kick sends a thick heavy ring,
    a hat a thin one, a hard hit a bright one. Between the beats, lighter
    rings leave on the beat's subdivisions: two to a beat at an ordinary
    level, four when the music is loud, none when it is quiet. So the stream
    never stops while the music plays, thickens and brightens as it gets
    louder, and every ring in it is on the grid. A strong hit well off the
    beat still sends a ring of its own. With no tempo, the stream runs on a
    clock of its own and every hit sends a ring.

    **On the beat, not after it.** A ring used to wait for the detector to
    confirm the hit -- about 30 ms late on the test tracks -- and was then born
    inside the heart and faded in, so it appeared later still. The tempo's
    beat clock runs ahead of the detector and lands within 10 ms of the true
    beat, so the rings are timed from that, and born on the heart's outline,
    where each is visible from its first frame.

    Each ring keeps the radius it was born at. Taking its start from the heart
    as it is now -- which swells on every beat -- jerked every ring in flight
    outward on the next hit, which is the mess ``Locket``'s own notes warn
    about.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 12 or dc < 16:
        return empty(ctx.w, ctx.h)
    _g = ctx.scratch("locket_geo", lambda: _locket_geo(dr, dc))
    st = ctx.scratch("locket_beat", lambda: {
        "born": np.full(_SHOT_RINGS, -99.0), "r0": np.zeros(_SHOT_RINGS),
        "amp": np.zeros(_SHOT_RINGS), "wide": np.ones(_SHOT_RINGS),
        "life": np.full(_SHOT_RINGS, _SHOT_LIFE_S), "beat": 0.0,
        "phase": 0.0, "sub": 0, "last": -1, "free": 0.0, "level": 0.0,
        "heard": -99.0,
    })

    dt = max(ctx.dt, 0.0)
    st["beat"] *= math.exp(-dt / 0.14)
    bass = ctx.range(0.0, 0.22)
    core = float(0.22 + 0.04 * bass + 0.05 * st["beat"])
    # How loud the music is, held: it rises at once and falls away over half
    # a second or so. The level right now dips before every hit on sparse
    # drums -- which is exactly when the beat clock ticks -- and a stream
    # judged on it switched off and on between the hits and dropped the beat
    # rings. Held, it answers "is the music loud" rather than "is this frame".
    now_level = 0.0 if ctx.silent else float(ctx.energy)
    if not ctx.silent and now_level > 0.0:
        st["heard"] = ctx.t
    st["level"] = max(now_level, st["level"] * math.exp(-dt / 0.6))
    level = st["level"]
    playing = ctx.t - st["heard"] < 1.0
    tempo = float(ctx.tempo_bpm)
    period = 60.0 / tempo if tempo > 0.0 else 0.0
    lo, hi = _SHOT_LIFE_RANGE
    life = min(hi, max(lo, period * _SHOT_BEATS)) if period else _SHOT_LIFE_S

    def send(amp: float, wide: float, pump: float) -> int:
        slot = int(np.argmin(st["born"]))
        st["born"][slot] = ctx.t
        st["r0"][slot] = core
        st["amp"][slot] = amp
        st["wide"][slot] = wide
        st["life"][slot] = life
        st["beat"] = min(1.5, st["beat"] + pump)
        return slot

    def stream_ring() -> None:
        # lighter and thinner than a beat's, and as bright as the music is loud
        send(0.3 + 0.6 * min(1.0, level * 2.0), 0.7, 0.0)

    def weight(hit: list[str], strength: float) -> tuple[float, float]:
        wide, bright = _SHOT[hit[0]] if hit else (1.0, 0.85)
        return bright * (0.55 + 0.45 * strength), wide

    per_beat = 0 if level < _STREAM_QUIET else (4 if level >= _STREAM_LOUD else 2)

    # ── on the beat clock, when there is one ─────────────────────────────────
    phase = float(ctx.beat_phase)
    if period:
        if phase < st["phase"] - 0.5:
            # A beat: a bright ring now, weighted by its hit when it arrives.
            if playing or ctx.onsets:
                st["last"] = send(0.75, 1.0, 1.0)
            st["sub"] = 0
        elif per_beat:
            # The stream between beats, on the beat's own subdivisions.
            sub = int(phase * per_beat)
            if sub > st["sub"]:
                stream_ring()
            st["sub"] = sub
        st["phase"] = phase
    elif per_beat:
        # No tempo: the stream keeps its own time, quicker when it is loud.
        st["free"] += dt
        every = _STREAM_FREE_S * (0.5 if per_beat == 4 else 1.0)
        if st["free"] >= every:
            st["free"] = 0.0
            stream_ring()

    if ctx.onsets:
        strength = min(1.0, float(ctx.onset_strength))
        amp, wide = weight(named(ctx.drums or {}), strength)
        near = min(phase, 1.0 - phase) if period else 1.0
        newest = st["last"]
        if period and near <= _ON_BEAT and newest >= 0 \
                and ctx.t - st["born"][newest] <= _ON_BEAT * period:
            # The hit of the beat just sent: it decides how that ring looks.
            st["amp"][newest] = max(st["amp"][newest], amp)
            st["wide"][newest] = wide
            st["beat"] = min(1.5, st["beat"] + 0.3 * strength)
        elif not period or near > _ON_BEAT:
            # No beat clock, or a strong hit well off the beat: a ring of its
            # own.
            st["last"] = send(amp * (1.0 if not period else 0.8), wide, 0.6 + 0.5 * amp)

    glow = _locket_rim(ctx, _g, core, st["beat"])
    sc_at = _g["sc_at"]
    lut = None
    for i in range(_SHOT_RINGS):
        age = ctx.t - st["born"][i]
        span = float(st["life"][i])
        if not (0.0 <= age < span):
            continue
        u = age / span
        # From the outline it was born on to just past the edge of the frame,
        # on a gentle ease: it leaves the heart briskly and slows as it goes.
        start = float(st["r0"][i])
        sc = start + (_RING_END - start) * (1.0 - (1.0 - u) ** 2.0)
        width = _SHOT_WIDTH * float(st["wide"][i]) * (1.0 + 0.5 * u)
        a = float(st["amp"][i]) * (1.0 - u) * min(1.0, age / 0.04)
        d = np.abs(sc_at - np.float32(sc)) / np.float32(width)
        ring = np.where(d < 1.0, np.float32(a) * (np.float32(1.0) - d * d * d),
                        np.float32(0.0))
        lut = ring if lut is None else np.maximum(lut, ring)
    if lut is not None:
        np.maximum(glow, lut[_g["sidx"]].astype(np.float32), out=glow)
    return _locket_out(ctx, glow)
