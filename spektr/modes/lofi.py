"""Lofi modes — warm, textured scenes that still move with the music.

These share an *aesthetic* — a record, rain on glass, a coal bed, a string of
bulbs, a tape deck, a mug going cold — not a reactivity budget. An earlier
version of this file deliberately damped everything through ~1s smoothing so
the scenes would stay calm, and only ever let loudness change brightness,
never geometry. That was a design mistake: it read as wallpaper rather than
as a visualiser, because nothing on screen was actually *showing* you the
spectrum.

Every mode here now maps real band data into its geometry — a groove's
brightness, a bulb's size, a tendril's height, a tape strand's displacement —
and responds on the ~0.1s timescale the rest of the codebase uses rather
than on a one-second lag. What carries over from the original design is the
*look*: warm objects, soft edges, nothing strobing. The reactivity lives in
what the object is doing, not in how hard the whole picture flashes.

Anything whose *rate* the audio controls — the record's spin, the tape
reels — accumulates phase through ``ctx.dt`` instead of multiplying ``ctx.t``
by a varying speed; see ``Tunnel``'s docstring in ``scenes.py`` for why that
distinction matters. A genuinely constant rate times ``ctx.t`` is still fine
and is used where the rate never changes.

Modes that read a fixed number of bands (rather than ``ctx.n_display``) do so
deliberately: several cache a full-grid radius→band index map keyed on
terminal size, and ``n_display`` can change from the settings panel *without*
a resize, which would leave that cache indexing past the end of a shorter
band array.
"""

from __future__ import annotations

import math

import numpy as np

from ..render import cell_max, frac, noise, pack_braille
from . import Ctx, empty, mode, spread
from . import polar_grid as _polar

_ARM_TURN = 0.08   # fixed tonearm rest angle, as a fraction of a full turn
_VINYL_BANDS = 14
#: groove spacing, in dots. A module constant rather than a local because the
#: static cache divides the radius by it once instead of every frame.
_VINYL_PERIOD = 2.6


def _vinyl_static(dist: np.ndarray, turn: np.ndarray, max_r: float, dr: int, dc: int) -> dict:
    """Everything about the disc that depends only on geometry, not time or audio.

    The first cut of this mode recomputed all of it — the disc/label/hole
    masks, the tonearm, the dust speckle — as fresh full-grid passes every
    frame, none of which change between frames. Measured at 400x100 that was
    ~10.7ms, over half the frame budget. None of it depends on ``ctx.t`` or
    the spectrum, so it belongs in the same once-per-resize cache ``_polar``
    already uses.

    ``band_at_r`` is the piece that makes the grooves a spectrum: it maps
    every dot's radius onto a band index once, so the per-frame cost of
    lighting the grooves by band level is a single gather.
    """
    disc_r = max_r * 0.94
    label_r = max_r * 0.24
    hole_r = max(1.0, max_r * 0.035)
    on_disc = dist <= disc_r

    ang_d = np.abs(turn - _ARM_TURN)
    ang_d = np.minimum(ang_d, 1.0 - ang_d)

    span = max(disc_r - label_r, 1e-6)
    band_at_r = np.clip(
        ((dist - label_r) / span * _VINYL_BANDS).astype(np.int32), 0, _VINYL_BANDS - 1
    )

    return {
        "disc_r": disc_r, "label_r": label_r, "hole_r": hole_r,
        # Two rescalings of the radius the groove pass would otherwise redo
        # every frame. At 400x100 the dot grid is 320,000 wide and this mode is
        # memory-bound, so a pass saved is worth more than an operation saved:
        # these two cost nothing here and remove two full traversals from the
        # frame.
        "dist035": (dist * np.float32(0.35)).astype(np.float32),
        "dist_over_period": (dist / np.float32(_VINYL_PERIOD)).astype(np.float32),
        "label_over_period": np.float32(label_r / _VINYL_PERIOD),
        "on_disc": on_disc,
        "groove_zone": on_disc & (dist > label_r),
        "label": on_disc & (dist <= label_r) & (dist > hole_r),
        "glint_zone": on_disc & (dist > hole_r),
        "tonearm": (ang_d < 0.006) & (dist > label_r * 0.55) & (dist < disc_r * 1.03),
        # dust flecks: a fixed seed, not ``ctx.frame``. ``Dune``'s grain
        # texture shimmered every frame before that same fix; dust sitting on
        # a record doesn't rearrange itself 60 times a second.
        "dust": on_disc & (noise((dr, dc), 31) < 0.0035),
        "band_at_r": band_at_r,
    }


@mode("Vinyl", group="lofi", blurb="a record whose grooves light up as a radial spectrum")
def vinyl(ctx: Ctx):
    """A turntable whose grooves *are* the spectrum, read from the label outward.

    Bass sits near the label and treble at the rim, so a bass-heavy track
    lights the disc from the inside and a bright one rings the edge. The
    spin rate follows the music through a ``ctx.dt`` phase accumulator — not
    ``ctx.t * speed``, which would teleport the glint every time the energy
    changed — and a hard bass onset makes the needle skip, jogging that
    phase forward.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 16 or dc < 16:
        return empty(ctx.w, ctx.h)

    dist, turn, max_r = _polar(ctx)
    sv = ctx.scratch("vinyl_static", lambda: _vinyl_static(dist, turn, max_r, dr, dc))

    st = ctx.scratch("vinyl", lambda: {"warm": 0.0, "skip_t": -99.0})
    st["warm"] += (ctx.energy - st["warm"]) * min(1.0, ctx.dt / 0.12)

    bass = ctx.range(0.0, 0.2)
    # A real onset is the needle skip. The old fast/slow envelope over the
    # bass band could not tell a kick from the groove simply swelling; the
    # analyser's spectral-flux detector can. The 1.2 s refractory stays — a
    # skip is a rare jog, not a per-beat habit, and one per bar at most is
    # what makes it read as the needle catching.
    if ctx.onsets and (ctx.t - st["skip_t"]) > 1.2:
        st["skip_t"] = ctx.t
    skip = (ctx.t - st["skip_t"]) < 0.10

    sp = ctx.scratch("vinyl_spin", lambda: {"v": 0.0})
    sp["v"] += (0.12 + ctx.energy * 0.85) * max(ctx.dt, 0.0)
    if skip:
        sp["v"] += 1.4 * max(ctx.dt, 0.0)

    lv = ctx.display_bands(_VINYL_BANDS).astype(np.float32)
    band_at_r = sv["band_at_r"]

    # float32 throughout the heat pipeline, the same deal Flame's docstring
    # makes: at 400x100 this is several passes over a 320k-cell grid, and
    # every float64 one moves twice the memory of a float32 one. Which also
    # means the thing worth counting here is *passes*, not operations — this
    # loop is memory-bound, and the sine below is not the expensive part of it.
    #
    # groove rings ripple outward; the ripple rides the smoothed level so the
    # whole surface breathes rather than jittering ring to ring. Built in place
    # from the pre-divided radius, so what used to be
    # ``frac((dist + k*sin(dist*0.35 - b) - label_r) / period)`` — nine
    # traversals of the dot grid, four of them building temporaries that were
    # read once and thrown away — is the same arithmetic in six.
    groove = np.subtract(sv["dist035"], np.float32(sp["v"] * 3.0))
    np.sin(groove, out=groove)
    groove *= np.float32(st["warm"] * 2.2 / _VINYL_PERIOD)
    groove += sv["dist_over_period"]
    groove -= sv["label_over_period"]
    groove -= np.floor(groove)                       # frac, in place

    # One gather, not two, and none of it repeated on a boolean subset. Both
    # numbers the groove needs — how wide the lit part is and how bright it
    # burns — are affine in the band level, so the *level* is gathered once and
    # the two are derived from it. The previous form gathered a width over the
    # whole grid, then gathered the band index again through the lit mask, then
    # gathered a heat through that: three irregular passes where one regular
    # one and two multiply-adds do the same job.
    lvg = np.take(lv, band_at_r)
    groove_lit = sv["groove_zone"] & (groove < np.float32(0.16) + np.float32(0.34) * lvg)
    heat = np.where(groove_lit, np.float32(0.12) + np.float32(0.80) * lvg, np.float32(0.0))

    # a narrow catch of light, not a wedge: 0.035 of a turn is 25 degrees of
    # solid fill sweeping the disc, which reads as a slab rather than a glint
    #
    # Tested as a window on ``turn`` rather than by rotating the grid into the
    # window's frame: ``frac(turn - v)`` is three traversals and an absolute
    # value to find a band two percent of a turn wide, where two comparisons
    # against a pair of scalars answer the same question. ``turn`` is in
    # [0, 1), so the window wraps at most once and the wrapped case is an OR.
    lo, hi = (sp["v"] + 0.49) % 1.0, (sp["v"] + 0.51) % 1.0
    if lo < hi:
        in_glint = (turn >= np.float32(lo)) & (turn < np.float32(hi))
    else:
        in_glint = (turn >= np.float32(lo)) | (turn < np.float32(hi))
    glint = sv["glint_zone"] & in_glint

    heat[sv["label"]] = 0.35 + 0.5 * bass
    heat[sv["dust"]] = np.maximum(heat[sv["dust"]], 0.55)
    heat[glint] = np.maximum(heat[glint], 0.80 + 0.20 * skip)
    heat[sv["tonearm"]] = np.maximum(heat[sv["tonearm"]], 0.45 + 0.4 * st["warm"])

    np.clip(heat, 0.0, 1.0, out=heat)
    lit = heat > 0.03
    codes = pack_braille(lit)
    cidx = ctx.ramp(cell_max(heat))
    return codes, cidx


_RAIN_CAP = 260
_RAIN_GLOWS = 6
_SPLASH_CAP = 40


def _rain_state() -> dict:
    rng = np.random.default_rng(211)
    return {
        "y": np.full(_RAIN_CAP, -1.0), "x": np.zeros(_RAIN_CAP),
        "spd": np.zeros(_RAIN_CAP), "drift": np.zeros(_RAIN_CAP), "len": np.zeros(_RAIN_CAP),
        "acc": 0.0,
        "sx": np.zeros(_SPLASH_CAP), "sage": np.full(_SPLASH_CAP, -1.0),
        "glow_x": rng.uniform(0.08, 0.92, _RAIN_GLOWS),
        "glow_y": rng.uniform(0.05, 0.5, _RAIN_GLOWS),
        "glow_r": rng.uniform(0.05, 0.10, _RAIN_GLOWS),
        "peak": np.zeros(_RAIN_GLOWS),
        "rng": rng,
    }


@mode("Rain", group="lofi", blurb="rain on the glass, falling harder when it's loud")
def rain(ctx: Ctx):
    """Weather that actually tracks the track — density *and* fall speed.

    The first version held fall speed constant on purpose and only let
    loudness thicken the drizzle, which meant a quiet passage and a loud one
    looked nearly identical. Both now scale, and drops splash when they land
    so the bottom edge carries the beat too. The bokeh lights behind the
    glass are one band each, peak-held so they glow and fade rather than
    flickering, and each is evaluated in a cropped box around its own centre
    so cost tracks blob size rather than terminal size.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 16 or dc < 20:
        return empty(ctx.w, ctx.h)

    st = ctx.scratch("rain", _rain_state)
    rng = st["rng"]

    mid = ctx.range(0.15, 0.7)
    bass = ctx.range(0.0, 0.2)
    # Rain falls harder on percussive material, not merely on loud material —
    # ``ctx.drive`` is continuous, so a busy passage the onset detector is
    # conservative about still drives it.
    fall = 0.55 + ctx.energy * 2.0 + ctx.drive * 1.0

    st["acc"] += (8.0 + mid * 55.0) * ctx.dt
    want = int(st["acc"])
    if want:
        st["acc"] -= want
        free = np.flatnonzero(st["y"] < 0.0)[:want]
        if free.size:
            k = free.size
            # y < 0 doubles as the "slot is free" sentinel (same convention as
            # Bubbles/Fireworks/Ember), so a spawned drop must start at >= 0
            # even though visually it enters from just off the top — starting
            # it negative made it read as an empty slot again the very next
            # frame and it never actually fell.
            st["y"][free] = rng.uniform(0.0, dr * 0.05, k)
            st["x"][free] = rng.uniform(0.0, dc - 1.0, k)
            st["spd"][free] = rng.uniform(0.35, 0.65, k) * dr
            st["drift"][free] = rng.uniform(0.03, 0.09, k) * dr
            st["len"][free] = rng.uniform(2.5, 5.5, k)

    alive = st["y"] >= 0.0
    st["y"][alive] += st["spd"][alive] * fall * ctx.dt
    st["x"][alive] += st["drift"][alive] * fall * ctx.dt

    landed = alive & (st["y"] > dr - 1)
    if landed.any():
        hit_x = st["x"][landed]
        sfree = np.flatnonzero(st["sage"] < 0.0)[: hit_x.size]
        if sfree.size:
            st["sx"][sfree] = hit_x[: sfree.size]
            st["sage"][sfree] = 0.0
    st["y"][landed | (alive & (st["x"] > dc + 4))] = -1.0

    salive = st["sage"] >= 0.0
    st["sage"][salive] += ctx.dt
    st["sage"][salive & (st["sage"] > 0.35)] = -1.0

    field = np.zeros((dr, dc), dtype=np.float64)

    lv = ctx.display_bands(_RAIN_GLOWS)
    st["peak"] = np.maximum(st["peak"] - ctx.dt * 1.5, lv)
    for i in range(_RAIN_GLOWS):
        cx, cy = st["glow_x"][i] * dc, st["glow_y"][i] * dr
        lvl = float(st["peak"][i])
        r = st["glow_r"][i] * min(dr, dc) * (0.75 + lvl * 0.9)
        y0, y1 = max(0, int(cy - r * 2)), min(dr, int(cy + r * 2) + 1)
        x0, x1 = max(0, int(cx - r * 2)), min(dc, int(cx + r * 2) + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy = np.arange(y0, y1)[:, None]
        xx = np.arange(x0, x1)[None, :]
        d2 = ((yy - cy) ** 2 + (xx - cx) ** 2) / max(r * r, 1e-6)
        blob = np.exp(-d2) * (0.14 + 0.55 * lvl)
        np.maximum(field[y0:y1, x0:x1], blob, out=field[y0:y1, x0:x1])

    live = np.flatnonzero(st["y"] >= 0.0)
    if live.size:
        y0 = st["y"][live]
        x0 = st["x"][live]
        length = st["len"][live] * (0.6 + ctx.energy * 1.4)
        steps = 4
        for s in range(steps):
            f = s / (steps - 1)
            drop_y = y0 - length * f
            py = np.clip(np.rint(drop_y).astype(np.int32), 0, dr - 1)
            px = np.clip(np.rint(x0 - length * f * 0.35).astype(np.int32), 0, dc - 1)
            ok = (drop_y >= 0) & (drop_y < dr)
            if ok.any():
                np.maximum.at(field, (py[ok], px[ok]), (1.0 - f) * (0.55 + 0.45 * bass))

    sl = np.flatnonzero(st["sage"] >= 0.0)
    if sl.size:
        age = st["sage"][sl]
        bright = np.clip(1.0 - age / 0.35, 0.0, 1.0) * 0.75
        spread_px = np.rint(age * dc * 0.06).astype(np.int32) + 1
        base_x = np.rint(st["sx"][sl]).astype(np.int32)
        row = dr - 1
        for off in (-1, 0, 1):
            px = np.clip(base_x + off * spread_px, 0, dc - 1)
            np.maximum.at(field, (np.full(px.size, row), px), bright)

    np.clip(field, 0.0, 1.0, out=field)
    dots = field > 0.05
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(field))
    return codes, cidx


#: More flakes than Rain has drops, because snow reads as snow by being
#: *many*: a dozen streaks is rain, a dozen dots is dirt on the lens.
_SNOW_CAP = 520

#: Three depths, drawn as three sizes. The near plane is a five-dot crystal,
#: the middle a three-dot one, the far plane a single dot — and each falls and
#: sways more slowly than the one in front of it. That parallax is the whole
#: reason a field of dots reads as weather with depth rather than as static.
#: The middle plane is a diagonal pair rather than a horizontal one on
#: purpose. A three-dot bar reads as a dash, and a dash is rain's vocabulary —
#: a field of them looks like drizzle blown sideways. A diagonal has no strong
#: axis, so it reads as a speck of ice catching the light.
_SNOW_ARMS = (
    ((0, 0),),                                            # far: one dot
    ((0, 0), (-1, 1)),                                    # mid: a diagonal pair
    ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)),           # near: a crystal
)


def _snow_state() -> dict:
    rng = np.random.default_rng(29)
    return {
        # y < 0 is the free-slot sentinel, same convention as Rain, Bubbles,
        # Fireworks and Ember.
        "y": np.full(_SNOW_CAP, -1.0),
        "x": np.zeros(_SNOW_CAP),
        "spd": np.zeros(_SNOW_CAP),
        "amp": np.zeros(_SNOW_CAP),
        "freq": np.zeros(_SNOW_CAP),
        "phase": np.zeros(_SNOW_CAP),
        "plane": np.zeros(_SNOW_CAP, dtype=np.int32),
        #: Per-flake brightness. Three depth planes give three brightnesses,
        #: which is three ramp steps out of sixty-four — the theme's gradient
        #: may as well not exist. A continuous spread per flake, plus a slow
        #: twinkle, is what makes a snowfield show the palette rather than
        #: three flat tones of it.
        "shine": np.zeros(_SNOW_CAP),
        "acc": 0.0,
        "wind": 0.0,
        "settle": None,          # per-column depth of lying snow, sized on first use
        "rng": rng,
    }


@mode("Snow", group="lofi", after="Rain",
      blurb="snowfall in three planes, thickening and gusting with the track")
def snow(ctx: Ctx):
    """Rain's sibling, and deliberately its opposite in how it moves.

    A raindrop is a streak: it falls fast enough that its own motion is the
    shape, which is why Rain draws each drop as a short trail and lets the
    bottom edge splash. A snowflake has almost no terminal velocity and a
    great deal of air resistance, so it does the opposite — it hangs, sways,
    and arrives. Drawing snow as short vertical streaks would just be quiet
    rain, so the geometry here is a crystal that drifts sideways on its own
    sine and is pushed around by a shared wind.

    Three depth planes rather than one field of identical dots. The near
    plane is a five-dot crystal falling fastest and swaying widest, the far
    plane a single dim dot barely moving. Without that parallax a screen of
    white dots reads as noise; with it, it reads as depth — and it costs one
    extra array and a lookup, because all three planes are drawn from the
    same arrays with a different stamp.

    No bokeh behind the glass. Rain's blurred circles are lights seen through
    a wet window, which is a thing that happens indoors looking out; snow is
    the weather itself, and putting glass in front of it makes it someone
    else's snow.

    What the music does: the mid band sets how thickly it falls, energy sets
    how fast, and ``ctx.drive`` gusts the wind sideways — percussive material
    blows the fall about rather than merely thickening it. Snow also lies:
    flakes that reach the bottom add to a per-column depth that melts back
    slowly, so a loud passage leaves drifts along the floor for a while after
    it has passed.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 16 or dc < 20:
        return empty(ctx.w, ctx.h)

    st = ctx.scratch("snow", _snow_state)
    rng = st["rng"]
    if st["settle"] is None or st["settle"].size != dc:
        st["settle"] = np.zeros(dc, dtype=np.float64)

    mid = ctx.range(0.15, 0.7)
    # Slower than Rain's 0.55 + energy*2 — snow that falls at rain speed is
    # rain drawn with the wrong glyph — but not *much* slower. The first
    # version took between eleven and thirty-seven seconds to cross the
    # screen, so the bottom half was permanently empty and the picture read
    # as a few specks near the ceiling rather than as weather. The near plane
    # now crosses in about three seconds and the far plane in about eight,
    # which is the parallax doing the work instead of the clock.
    fall = 0.7 + ctx.energy * 1.1

    # One wind for the whole field, easing toward a target rather than
    # jumping: a gust that arrives in a single frame teleports every flake
    # sideways, which reads as a glitch and not as weather.
    gust = math.sin(ctx.t * 0.23) * 0.35 + (ctx.drive - 0.5) * 1.6
    st["wind"] += (gust - st["wind"]) * min(1.0, ctx.dt * 1.8)

    # Flakes per second, scaled down on a small window.
    #
    # The rate is otherwise absolute, and a flake's lifetime is roughly
    # constant — fall speed scales with the height it has to cross — so the
    # same rate fills a small terminal to the same *count* as a large one and
    # therefore to a far greater density. Measured at 40x12 it lit 48% of the
    # cells, which is not snowfall, it is static.
    #
    # Only ever scaled down. Growing it on a large terminal would be the
    # honest completion of the idea, but it costs a scatter per flake per
    # frame and the tablet is already the slowest thing that runs this.
    area = min(1.0, max(0.3, (dr * dc) / (120.0 * 240.0)))
    st["acc"] += (22.0 + mid * 90.0) * area * ctx.dt
    want = int(st["acc"])
    if want:
        st["acc"] -= want
        free = np.flatnonzero(st["y"] < 0.0)[:want]
        if free.size:
            k = free.size
            plane = rng.integers(0, 3, k)
            st["plane"][free] = plane
            # Depth sets everything: nearer is faster, wider-swaying, bigger.
            near = plane.astype(np.float64) / 2.0
            st["y"][free] = rng.uniform(0.0, dr * 0.04, k)
            st["x"][free] = rng.uniform(0.0, dc - 1.0, k)
            st["spd"][free] = (0.15 + near * 0.22) * dr * rng.uniform(0.8, 1.25, k)
            # ``amp`` is a sideways *speed*, in dot columns per second, not a
            # displacement — see where it is integrated.
            st["amp"][free] = (0.02 + near * 0.03) * dc * rng.uniform(0.6, 1.4, k)
            st["freq"][free] = rng.uniform(0.5, 1.4, k)
            st["phase"][free] = rng.uniform(0.0, 2 * math.pi, k)
            st["shine"][free] = rng.uniform(0.45, 1.0, k)

    alive = st["y"] >= 0.0
    if alive.any():
        st["y"][alive] += st["spd"][alive] * fall * ctx.dt
        # The sway is a velocity, not a position: setting x from a sine of t
        # would snap every flake back to its spawn column whenever the wind
        # moved it, because the two would be fighting over the same value.
        # ``amp`` is already that velocity's amplitude in columns per second,
        # so the cosine is the only thing multiplying it — an earlier version
        # multiplied by ``freq`` as well, which is the derivative arriving
        # twice and made the fastest-swaying flakes skate sideways.
        sway = np.cos(ctx.t * st["freq"][alive] + st["phase"][alive])
        near = st["plane"][alive].astype(np.float64) / 2.0
        st["x"][alive] += (
            sway * st["amp"][alive] + st["wind"] * (0.3 + near) * dc * 0.02
        ) * ctx.dt
        # Wrap rather than kill: a flake blown off the right edge is still
        # falling, and respawning it at the top would thin the field exactly
        # when the wind is most interesting.
        st["x"][alive] %= dc

    landed = alive & (st["y"] > dr - 1)
    if landed.any():
        cols = np.clip(np.rint(st["x"][landed]).astype(np.int32), 0, dc - 1)
        # Weight by depth so near flakes build the drift faster than far ones.
        np.add.at(st["settle"], cols, 0.35 + st["plane"][landed] * 0.25)
        st["y"][landed] = -1.0

    # Melt, and slump sideways so drifts round off instead of standing as
    # single-column towers.
    st["settle"] *= max(0.0, 1.0 - ctx.dt * 0.22)
    if dc > 2:
        st["settle"][1:-1] += (
            st["settle"][:-2] + st["settle"][2:] - 2.0 * st["settle"][1:-1]
        ) * 0.12

    field = np.zeros((dr, dc), dtype=np.float64)

    live = np.flatnonzero(st["y"] >= 0.0)
    if live.size:
        py = np.rint(st["y"][live]).astype(np.int32)
        px = np.rint(st["x"][live]).astype(np.int32)
        plane = st["plane"][live]
        # A flake catches the light as it turns. Slow, and never all the way
        # off, so it reads as tumbling rather than as flickering.
        twinkle = 0.82 + 0.18 * np.sin(ctx.t * st["freq"][live] * 1.7 + st["phase"][live])
        # The planes are spread nearly the whole ramp rather than sitting in
        # the middle third of it. A snowfield is one of the few pictures where
        # almost every cell is at its own depth, so the depth *is* the
        # gradient — bunching the three planes close together threw that away
        # and left the theme showing as a single colour with texture.
        centre = np.clip(
            (0.22 + plane * 0.34) * st["shine"][live] * twinkle * (0.70 + ctx.energy * 0.9),
            0.06, 1.0,
        )
        for p in range(3):
            sel = plane == p
            if not sel.any():
                continue
            vals = centre[sel]
            for oy, ox in _SNOW_ARMS[p]:
                yy = py[sel] + oy
                xx = (px[sel] + ox) % dc
                ok = (yy >= 0) & (yy < dr)
                if ok.any():
                    # Arms dimmer than the centre, so a crystal reads as a
                    # crystal rather than as a solid block of five dots.
                    arm = 1.0 if (oy == 0 and ox == 0) else 0.55
                    np.maximum.at(field, (yy[ok], xx[ok]), vals[ok] * arm)

    lying = st["settle"]
    if lying.max() > 0.02:
        depth = np.clip(lying, 0.0, 6.0)
        rows = np.arange(dr, dtype=np.float64)[:, None]
        # Lying snow is measured up from the bottom row, and shaded by how
        # deep the drift is — a flat fill would be one more ramp step doing
        # the work of a gradient.
        mask = rows >= (dr - depth[None, :])
        shade = (0.22 + 0.5 * np.clip(depth / 5.0, 0.0, 1.0))[None, :]
        np.maximum(field, mask * shade, out=field)

    np.clip(field, 0.0, 1.0, out=field)
    dots = field > 0.05
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(field))
    return codes, cidx


_EMBER_CAP = 150


def _ember_state() -> dict:
    return {
        "y": np.full(_EMBER_CAP, -1.0), "x": np.zeros(_EMBER_CAP),
        "spd": np.zeros(_EMBER_CAP), "wfreq": np.zeros(_EMBER_CAP), "wph": np.zeros(_EMBER_CAP),
        "wamp": np.zeros(_EMBER_CAP), "age": np.zeros(_EMBER_CAP), "life": np.zeros(_EMBER_CAP),
        "acc": 0.0, "pop_t": -99.0,
        "rng": np.random.default_rng(89),
    }


@mode("Ember", group="lofi", blurb="a coal bed burning by band, sparks rising off the hot spots")
def ember(ctx: Ctx):
    """A coal bed whose heat *is* the spectrum, spanning the width band by band.

    ``Flame`` draws licking tongues upward from each band and redraws them
    every frame. This keeps the fire in the bed: each column's glow is its
    band level, and sparks lift off where the coals are actually hot —
    spawn positions are drawn weighted by the spectrum, so a bass-heavy
    track throws sparks from the left and a bright one from the right.
    Sparks then rise and cool on their own timers, which is why the mode
    keeps moving after the audio stops.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 12 or dc < 16:
        return empty(ctx.w, ctx.h)

    st = ctx.scratch("ember", _ember_state)
    rng = st["rng"]

    # A real onset pops a handful of sparks off the coals. The old fast/slow
    # envelope over the bass band fired on any rise in level, so a pad swell
    # showered embers as hard as a drum hit; the analyser's detector answers
    # "did a transient actually land". The refractory stays so a fast roll
    # pops once rather than emptying the cap in a frame.
    pop = bool(ctx.onsets) and (ctx.t - st["pop_t"]) > 0.4

    bed_h = max(3, int(dr * 0.20))
    bed_top = dr - bed_h
    # float32, and the same deal Flame's and Vinyl's docstrings make: at
    # 400x100 this is a 320,000-cell grid and every float64 pass over it moves
    # twice the memory of a float32 one, for a value that ends up quantised to
    # 64 ramp steps.
    field = np.zeros((dr, dc), dtype=np.float32)

    lv_cols = spread(ctx.display_bands(), dc).astype(np.float32)

    # The bed is the bottom fifth of the screen and the haze is what is above
    # it, so neither is a full-grid job. Both used to be built over every row
    # and then masked, which is four fifths of the work thrown away in one
    # case and one fifth in the other.
    bed_rows = np.arange(bed_top, dr, dtype=np.float32)[:, None]
    depth = np.clip((bed_rows - bed_top) / max(bed_h - 1, 1), 0.0, 1.0)
    field[bed_top:] = (0.18 + 0.80 * lv_cols)[None, :] * (0.55 + 0.45 * depth)

    # heat haze standing over the coals: each column glows as far up as its
    # own band reaches. Without this the only band-driven area was the bed
    # itself — about a fifth of the height — so the picture barely changed
    # between a quiet passage and a loud one even though the bed underneath
    # was tracking the spectrum correctly the whole time.
    if bed_top > 0:
        above = (bed_top - np.arange(bed_top, dtype=np.float32))[:, None]
        glow_h = np.maximum(lv_cols * dr * 0.55, 1.0)[None, :]
        haze = np.clip(1.0 - above / glow_h, 0.0, 1.0) * lv_cols[None, :]
        # Dithered against a *fixed* noise field, not drawn solid. A continuous
        # haze value lights every dot inside the envelope the moment it clears
        # the 0.04 threshold, which rendered as a hard triangular slab of ⣿
        # rather than anything resembling heat. Thresholding turns the same
        # envelope into a sparse speckle that thickens as the band rises, and a
        # fixed seed keeps the speckle from strobing (see ``Dune``'s grain).
        #
        # Fixed means fixed: seed 7 every frame produced the same 320,000
        # hashes every frame, at 2 ms a time. It is scratch, not per-frame
        # work.
        grain = ctx.scratch("ember_grain", lambda: noise((dr, dc), 7))
        lit_haze = grain[:bed_top] < (haze * 0.7)
        np.maximum(field[:bed_top], (0.14 + 0.5 * haze) * lit_haze, out=field[:bed_top])

    st["acc"] += (2.0 + ctx.energy * 55.0) * ctx.dt
    want = int(st["acc"])
    if want:
        st["acc"] -= want
    if pop:
        want += int(rng.integers(6, 14))
        st["pop_t"] = ctx.t
    if want:
        free = np.flatnonzero(st["y"] < 0.0)[:want]
        if free.size:
            k = free.size
            total = lv_cols.sum()
            if total > 1e-6:
                xs = rng.choice(dc, size=k, p=lv_cols / total)
            else:
                xs = rng.integers(0, dc, k)
            st["x"][free] = xs.astype(np.float64)
            st["y"][free] = dr - bed_h - rng.uniform(0.0, 1.5, k)
            st["spd"][free] = rng.uniform(0.10, 0.30, k) * dr * (0.6 + ctx.energy * 1.6)
            st["wfreq"][free] = rng.uniform(0.5, 1.4, k)
            st["wph"][free] = rng.uniform(0.0, 2 * math.pi, k)
            st["wamp"][free] = rng.uniform(0.01, 0.035, k) * dc
            st["age"][free] = 0.0
            st["life"][free] = rng.uniform(1.4, 3.0, k)

    alive = st["y"] >= 0.0
    st["y"][alive] -= st["spd"][alive] * ctx.dt
    st["age"][alive] += ctx.dt
    dead = alive & ((st["age"] > st["life"]) | (st["y"] < -2))
    st["y"][dead] = -1.0

    live = np.flatnonzero(st["y"] >= 0.0)
    if live.size:
        wobble = np.sin(ctx.t * st["wfreq"][live] + st["wph"][live]) * st["wamp"][live]
        py = np.clip(np.rint(st["y"][live]).astype(np.int32), 0, dr - 1)
        px = np.clip(np.rint(st["x"][live] + wobble).astype(np.int32), 0, dc - 1)
        cool = np.clip(1.0 - st["age"][live] / st["life"][live], 0.0, 1.0)
        np.maximum.at(field, (py, px), 0.35 + 0.60 * cool)

    np.clip(field, 0.0, 1.0, out=field)
    dots = field > 0.04
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(field))
    return codes, cidx


_LIGHTS_N = 18
#: Three strands, not one. A single strand of small bulbs leaves almost the
#: whole frame static no matter how hard each bulb reacts, which measured as
#: the least audio-responsive mode in the group despite every bulb being
#: wired directly to a band. Hanging several strands multiplies the reactive
#: area without changing the idea.
_LIGHTS_STRANDS = 3
#: Sag depth and vertical offset per strand, as fractions of the dot height.
_LIGHTS_ROWS = ((0.09, 0.13), (0.38, 0.10), (0.66, 0.15))


def _strand_y(dr: int, top_f: float, sag_f: float, xs_norm: np.ndarray) -> np.ndarray:
    t = (xs_norm - 0.5) * 2.0
    return top_f * dr + sag_f * dr * (t * t)


def _lights_state(dr: int, dc: int) -> dict:
    rng = np.random.default_rng(151)
    n = _LIGHTS_N * _LIGHTS_STRANDS
    xs = np.empty(n)
    ys = np.empty(n)
    for s, (top_f, sag_f) in enumerate(_LIGHTS_ROWS):
        # stagger alternate strands so bulbs don't line up in vertical columns
        off = 0.5 / _LIGHTS_N if s % 2 else 0.0
        xn = np.clip(np.linspace(0.05, 0.95, _LIGHTS_N) + off, 0.02, 0.98)
        sl = slice(s * _LIGHTS_N, (s + 1) * _LIGHTS_N)
        xs[sl] = xn * dc
        ys[sl] = _strand_y(dr, top_f, sag_f, xn)
    return {
        "x": xs, "y": ys,
        "band": np.tile(np.arange(_LIGHTS_N), _LIGHTS_STRANDS),
        "freq": rng.uniform(0.4, 1.1, n),
        "ph": rng.uniform(0.0, 2 * math.pi, n),
        "r0": rng.uniform(0.010, 0.016, n) * min(dr, dc),
        "peak": np.zeros(_LIGHTS_N),
        "rng": rng,
    }


_CASSETTE_BANDS = 8


