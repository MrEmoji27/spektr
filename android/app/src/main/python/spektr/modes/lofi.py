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


