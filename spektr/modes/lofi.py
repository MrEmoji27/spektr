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
from .particles import _polar

_ARM_TURN = 0.08   # fixed tonearm rest angle, as a fraction of a full turn
_VINYL_BANDS = 14


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

    st = ctx.scratch("vinyl", lambda: {"warm": 0.0, "fast": 0.0, "slow": 0.0, "skip_t": -99.0})
    st["warm"] += (ctx.energy - st["warm"]) * min(1.0, ctx.dt / 0.12)

    bass = ctx.range(0.0, 0.2)
    st["fast"] += (bass - st["fast"]) * min(1.0, ctx.dt / 0.03)
    st["slow"] += (bass - st["slow"]) * min(1.0, ctx.dt / 0.4)
    if (st["fast"] - st["slow"]) > 0.15 and (ctx.t - st["skip_t"]) > 1.2:
        st["skip_t"] = ctx.t
    skip = (ctx.t - st["skip_t"]) < 0.10

    sp = ctx.scratch("vinyl_spin", lambda: {"v": 0.0})
    sp["v"] += (0.12 + ctx.energy * 0.85) * max(ctx.dt, 0.0)
    if skip:
        sp["v"] += 1.4 * max(ctx.dt, 0.0)

    lv = ctx.display_bands(_VINYL_BANDS)
    groove_level = lv[sv["band_at_r"]]

    # groove rings ripple outward; the ripple rides the smoothed level so the
    # whole surface breathes rather than jittering ring to ring
    period = 2.6
    ripple = dist + st["warm"] * 2.2 * np.sin(dist * 0.35 - sp["v"] * 3.0)
    groove = frac((ripple - sv["label_r"]) / period)
    groove_lit = sv["groove_zone"] & (groove < (0.16 + 0.34 * groove_level))

    # a narrow catch of light, not a wedge: 0.035 of a turn is 25 degrees of
    # solid fill sweeping the disc, which reads as a slab rather than a glint
    spin = frac(turn - sp["v"])
    glint = sv["glint_zone"] & (np.abs(spin - 0.5) < 0.010)

    heat = np.zeros((dr, dc), dtype=np.float64)
    heat[groove_lit] = (0.12 + 0.80 * groove_level)[groove_lit]
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
    fall = 0.55 + ctx.energy * 2.0

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
        "acc": 0.0, "fast": 0.0, "slow": 0.0, "pop_t": -99.0,
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

    bass = ctx.range(0.0, 0.2)
    st["fast"] += (bass - st["fast"]) * min(1.0, ctx.dt / 0.03)
    st["slow"] += (bass - st["slow"]) * min(1.0, ctx.dt / 0.45)
    pop = (st["fast"] - st["slow"]) > 0.16 and (ctx.t - st["pop_t"]) > 0.4

    bed_h = max(3, int(dr * 0.20))
    field = np.zeros((dr, dc), dtype=np.float64)

    lv_cols = spread(ctx.display_bands(), dc)
    rows = np.arange(dr)[:, None]
    bed_top = dr - bed_h
    bed_rows = rows >= bed_top
    depth = np.clip((rows - bed_top) / max(bed_h - 1, 1), 0.0, 1.0)
    bed_heat = (0.18 + 0.80 * lv_cols)[None, :] * (0.55 + 0.45 * depth)
    field = np.where(bed_rows, bed_heat, field)

    # heat haze standing over the coals: each column glows as far up as its
    # own band reaches. Without this the only band-driven area was the bed
    # itself — about a fifth of the height — so the picture barely changed
    # between a quiet passage and a loud one even though the bed underneath
    # was tracking the spectrum correctly the whole time.
    above = bed_top - rows
    glow_h = np.maximum(lv_cols * dr * 0.55, 1.0)[None, :]
    haze = np.clip(1.0 - above / glow_h, 0.0, 1.0) * lv_cols[None, :]
    # Dithered against a *fixed* noise field, not drawn solid. A continuous
    # haze value lights every dot inside the envelope the moment it clears
    # the 0.04 threshold, which rendered as a hard triangular slab of ⣿
    # rather than anything resembling heat. Thresholding turns the same
    # envelope into a sparse speckle that thickens as the band rises, and a
    # fixed seed keeps the speckle from strobing (see ``Dune``'s grain).
    lit_haze = noise((dr, dc), 7) < (haze * 0.7)
    field = np.maximum(field, np.where((above > 0) & lit_haze, 0.14 + 0.5 * haze, 0.0))

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


@mode("Fairylights", group="lofi", blurb="a string of bulbs, one per band, lighting to the music")
def fairylights(ctx: Ctx):
    """A spectrum analyser wearing a string of bulbs — one bulb per band.

    Left to right along the sagging strand is low to high, and each bulb's
    brightness *and* size follow its own band, peak-held so it flares on a
    transient and eases back down rather than flickering frame to frame.
    That peak-hold is what keeps it readable as lights: a bare band level
    would strobe. A small fixed per-bulb twinkle stays underneath so the
    strand still looks alive during silence, but it's a garnish on the band
    level now rather than the whole signal.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 14 or dc < 30:
        return empty(ctx.w, ctx.h)

    st = ctx.scratch("lights", lambda: _lights_state(dr, dc))

    lv = ctx.display_bands(_LIGHTS_N)
    st["peak"] = np.maximum(st["peak"] - ctx.dt * 1.8, lv)

    field = np.zeros((dr, dc), dtype=np.float64)

    # the strands: a shallow sag between two fixed posts each, one run per row
    # so they cost almost nothing against the frame budget
    xs_i = np.arange(dc)
    xn_i = xs_i / max(dc - 1, 1)
    for top_f, sag_f in _LIGHTS_ROWS:
        sy = np.clip(np.rint(_strand_y(dr, top_f, sag_f, xn_i)).astype(np.int32), 0, dr - 1)
        field[sy, xs_i] = np.maximum(field[sy, xs_i], 0.10)

    for i in range(st["x"].size):
        lvl = float(st["peak"][st["band"][i]])
        cx, cy = st["x"][i], st["y"][i]
        # Elliptical, not circular: bulbs sit only ~8 dots apart, so a
        # circular glow big enough to be worth looking at merges the strand
        # into one lit bar — but shrinking it to stay separable left almost
        # nothing on screen reacting (measured as the least responsive mode
        # in the group). Stretching the glow *vertically* buys back the area
        # in the empty space under each strand, which is also where light
        # from a hanging bulb would actually fall.
        rx = st["r0"][i] * (0.45 + lvl * 1.6)
        ry = rx * 2.4
        twinkle = 0.85 + 0.15 * math.sin(ctx.t * st["freq"][i] + st["ph"][i])
        bright = min(1.15, (0.16 + 0.95 * lvl) * twinkle)

        y0, y1 = max(0, int(cy - ry * 2)), min(dr, int(cy + ry * 2) + 1)
        x0, x1 = max(0, int(cx - rx * 2)), min(dc, int(cx + rx * 2) + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy = np.arange(y0, y1)[:, None]
        xx = np.arange(x0, x1)[None, :]
        d2 = ((yy - cy) / max(ry, 1e-6)) ** 2 + ((xx - cx) / max(rx, 1e-6)) ** 2
        np.maximum(field[y0:y1, x0:x1], np.exp(-d2) * bright, out=field[y0:y1, x0:x1])

    np.clip(field, 0.0, 1.0, out=field)
    dots = field > 0.05
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(field))
    return codes, cidx


_CASSETTE_SPOKES = 5
_CASSETTE_BANDS = 8


def _cassette_static(dr: int, dc: int) -> dict:
    """Fixed tape-deck geometry, cached once — see ``_vinyl_static`` for why.

    Two reels need two local coordinate systems rather than the single
    grid-centred one ``_polar`` gives every other mode in this file. Each is
    built here with the same aspect-ratio correction ``_polar`` uses (dot
    cells aren't square), just re-centred per reel.
    """
    x_scale = dr / dc
    cy = dr * 0.52
    cx1 = dc * 0.30
    cx2 = dc * 0.70
    reel_r = min(dr, dc) * 0.15
    hub_r = reel_r * 0.28

    def local(cx: float):
        y = np.arange(dr, dtype=np.float64)[:, None] - cy
        x = (np.arange(dc, dtype=np.float64)[None, :] - cx) * x_scale
        return np.sqrt(x * x + y * y), np.arctan2(y, x)

    d1, a1 = local(cx1)
    d2, a2 = local(cx2)
    hub1 = d1 <= hub_r
    hub2 = d2 <= hub_r

    # radius -> band index inside each reel, so the wound tape reads as a
    # spectrum the way ``Vinyl``'s grooves do. The reels are the largest
    # shapes on screen; leaving them a flat fill was what kept this the
    # least audio-responsive mode in the group even after the strand and
    # meter were wired up.
    span_r = max(reel_r - hub_r, 1e-6)

    def band_of(d):
        return np.clip(
            ((d - hub_r) / span_r * _CASSETTE_BANDS).astype(np.int32), 0, _CASSETTE_BANDS - 1
        )

    # Wound-tape rings, as a phase the per-frame code thresholds against the
    # band level. Filling the reels solid and only varying their *colour* was
    # measurably weak: a solid fill lights the same braille dots at every
    # level, so the glyph layer never changes and only the ramp index moves.
    # A mask that gains and loses coverage is what makes ``Vinyl``'s grooves
    # the most audio-responsive thing in this file.
    ring_period = max(1.5, reel_r * 0.16)

    body_x0, body_x1 = int(dc * 0.06), int(dc * 0.94)
    body_y0, body_y1 = int(dr * 0.18), int(dr * 0.88)
    rows = np.arange(dr)[:, None]
    cols = np.arange(dc)[None, :]
    in_body = (rows >= body_y0) & (rows < body_y1) & (cols >= body_x0) & (cols < body_x1)
    body_border = in_body & (
        (rows == body_y0) | (rows == body_y1 - 1) | (cols == body_x0) | (cols == body_x1 - 1)
    )

    xs = np.arange(dc, dtype=np.float64)
    span = max(cx2 - cx1, 1.0)
    tt = np.clip((xs - cx1) / span, 0.0, 1.0)
    sag_y = cy - reel_r * 0.55 + reel_r * 0.55 * 4.0 * tt * (1.0 - tt)

    return {
        "reel_r": reel_r, "hub_r": hub_r,
        "reel1": (d1 <= reel_r) & ~hub1, "reel2": (d2 <= reel_r) & ~hub2,
        "rim1": (d1 <= reel_r) & (d1 >= reel_r * 0.86),
        "rim2": (d2 <= reel_r) & (d2 >= reel_r * 0.86),
        "hub1": hub1, "hub2": hub2,
        "a1": a1, "a2": a2,
        "band1": band_of(d1), "band2": band_of(d2),
        "ringf1": frac(d1 / ring_period), "ringf2": frac(d2 / ring_period),
        "body_border": body_border,
        "sag_y": sag_y, "on_tape_zone": (xs >= cx1) & (xs <= cx2),
        "meter_y0": int(dr * 0.21), "meter_y1": int(dr * 0.37),
        "meter_x0": int(dc * 0.10), "meter_x1": int(dc * 0.90),
    }


def _spoke_mask(ang: np.ndarray, offset: float, n: int) -> np.ndarray:
    """Angular distance to the *nearest* of ``n`` evenly-spaced spokes, in one pass.

    The first cut looped once per spoke, wrapping each with the plain ``%``
    operator — the exact operation ``render.py``'s ``frac`` docstring warns
    is ~13x slower than the fast path, done five times per reel. Measured at
    400x100: ~33ms for one reel's spokes, pushing the whole mode to ~23ms,
    well over budget. Rescaling into "spoke-index space" turns five wrapped
    comparisons into one ``frac`` call regardless of spoke count — down to
    ~3.5ms, confirmed bit-identical to the old loop's output.
    """
    x = (ang - offset) * (n / (2 * math.pi))
    f = frac(x)
    nearest = np.minimum(f, 1.0 - f)
    return (nearest * (2 * math.pi / n)) < 0.05


@mode("Cassette", group="lofi", blurb="a tape deck; the strand carries the waveform, reels chase it")
def cassette(ctx: Ctx):
    """A deck where the tape strand *is* the waveform, stretched between the reels.

    This is the one mode in the group driven by ``ctx.wave`` rather than the
    band levels: the strand between the two hubs is displaced sample by
    sample, so it reads as tape physically vibrating with what's on it. The
    reels spin through a ``ctx.dt`` phase accumulator so their rate can
    follow the music without the glint-teleport failure ``Tunnel``'s
    docstring describes, and a small level meter across the deck body gives
    the spectrum a second, steadier readout.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 20 or dc < 40:
        return empty(ctx.w, ctx.h)

    sv = ctx.scratch("cassette_static", lambda: _cassette_static(dr, dc))
    st = ctx.scratch("cassette", lambda: {"warm": 0.0, "counter": 0.0, "spin": 0.0})
    st["warm"] += (ctx.energy - st["warm"]) * min(1.0, ctx.dt / 0.12)
    st["counter"] += (4.0 + st["warm"] * 26.0) * ctx.dt
    st["spin"] += (0.5 + ctx.energy * 5.5) * max(ctx.dt, 0.0)

    spoke1 = sv["reel1"] & _spoke_mask(sv["a1"], st["spin"], _CASSETTE_SPOKES)
    spoke2 = sv["reel2"] & _spoke_mask(sv["a2"], st["spin"], _CASSETTE_SPOKES)

    lvb = ctx.display_bands(_CASSETTE_BANDS)
    heat = np.zeros((dr, dc), dtype=np.float64)
    for reel, bkey, rkey in (("reel1", "band1", "ringf1"), ("reel2", "band2", "ringf2")):
        lvl = lvb[sv[bkey]]
        lit = sv[reel] & (sv[rkey] < (0.16 + 0.64 * lvl))
        heat[lit] = (0.14 + 0.74 * lvl)[lit]
    heat[sv["rim1"] | sv["rim2"]] = 0.38 + 0.3 * st["warm"]
    heat[sv["hub1"] | sv["hub2"]] = 0.5 + 0.4 * st["warm"]
    heat[spoke1 | spoke2] = 0.45 + 0.45 * st["warm"]
    heat[sv["body_border"]] = 0.34 + 0.2 * st["warm"]

    # the strand: sag plus the live waveform, so the tape shakes with the audio
    on = sv["on_tape_zone"]
    cols = np.arange(dc)[on]
    if cols.size:
        wave = ctx.wave
        if wave.size:
            idx = np.linspace(0.0, wave.size - 1.0, cols.size)
            wv = np.interp(idx, np.arange(wave.size, dtype=np.float64), wave)
        else:
            wv = np.zeros(cols.size)
        amp = sv["reel_r"] * 0.85
        base = sv["sag_y"][on] + wv * amp
        # drawn three dots thick: a one-dot strand is nearly invisible against
        # the deck body once the waveform starts moving it around
        for off, w_off in ((-1, 0.5), (0, 1.0), (1, 0.5)):
            ty = np.clip(np.rint(base + off).astype(np.int32), 0, dr - 1)
            np.maximum.at(heat, (ty, cols), (0.55 + 0.45 * st["warm"]) * w_off)

    # a small band meter across the deck face, bass at the left
    my0, my1 = sv["meter_y0"], sv["meter_y1"]
    mx0, mx1 = sv["meter_x0"], sv["meter_x1"]
    mw = mx1 - mx0
    if mw > 8 and my1 > my0:
        lv_cols = spread(ctx.display_bands(), mw)
        mh = my1 - my0
        top = my1 - np.rint(np.clip(lv_cols, 0.0, 1.0) * mh).astype(np.int32)
        mrows = np.arange(my0, my1)[:, None]
        bar = mrows >= top[None, :]
        sub = heat[my0:my1, mx0:mx1]
        np.maximum(sub, np.where(bar, 0.30 + 0.6 * lv_cols[None, :], 0.0), out=sub)

    np.clip(heat, 0.0, 1.0, out=heat)
    dots = heat > 0.03
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(heat))

    # tape counter, drawn onto the text-cell output pack_braille already
    # produced — the one place in this file that isn't pure dot-grid
    w, h = ctx.w, ctx.h
    label = f"{int(st['counter']) % 10000:04d}"
    col0 = max(0, w - len(label) - 1)
    row0 = h - 1
    if row0 >= 0 and col0 + len(label) <= w:
        codes[row0, col0 : col0 + len(label)] = [ord(c) for c in label]
        cidx[row0, col0 : col0 + len(label)] = ctx.ramp(
            np.full(len(label), 0.5 + 0.45 * st["warm"])
        )
    return codes, cidx


_STEAM_TENDRILS = 7


def _steam_state() -> dict:
    rng = np.random.default_rng(173)
    return {
        "phase": rng.uniform(0.0, 2 * math.pi, _STEAM_TENDRILS),
        "freq": rng.uniform(0.5, 1.1, _STEAM_TENDRILS),
        "peak": np.zeros(_STEAM_TENDRILS),
        "rng": rng,
    }


@mode("Steam", group="lofi", blurb="a mug whose steam rises band by band")
def steam(ctx: Ctx):
    """Steam off a cup, where each tendril's height is its own band.

    ``Flame`` and ``Ember`` own the licking-upward fire silhouette; this
    curls and dissipates instead. Tendrils are laid out low to high across
    the cup's mouth and each rises as far as its band tells it to,
    peak-held so a transient throws a plume that eases back down. Curl rate
    and phase are fixed per tendril (drawn once at spawn, not per-frame
    noise — the same reasoning as ``Vinyl``'s dust), so what the audio
    changes is the shape of the steam, not the texture of it.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 20 or dc < 24:
        return empty(ctx.w, ctx.h)

    st = ctx.scratch("steam", _steam_state)

    lv = ctx.display_bands(_STEAM_TENDRILS)
    st["peak"] = np.maximum(st["peak"] - ctx.dt * 1.1, lv)
    treble = ctx.range(0.6, 1.0)

    field = np.zeros((dr, dc), dtype=np.float64)

    # cup silhouette: a tapered body, bottom-centred, plus a rim glint
    cup_y0 = int(dr * 0.78)
    cup_h = max(1, dr - cup_y0)
    cup_cx = dc * 0.5
    cup_hw = dc * 0.22
    for i, r in enumerate(range(cup_y0, dr)):
        f = i / max(1, cup_h - 1)
        half_w = cup_hw * (0.72 + 0.28 * f)
        x0, x1 = max(0, int(cup_cx - half_w)), min(dc, int(cup_cx + half_w))
        if x1 > x0:
            field[r, x0:x1] = np.maximum(field[r, x0:x1], 0.28)
    rim_x0, rim_x1 = max(0, int(cup_cx - cup_hw)), min(dc, int(cup_cx + cup_hw))
    if rim_x1 > rim_x0:
        field[cup_y0, rim_x0:rim_x1] = 0.5 + 0.4 * ctx.energy

    max_h = cup_y0 * 0.92
    for i in range(_STEAM_TENDRILS):
        lvl = float(st["peak"][i])
        h_i = int(max_h * (0.12 + 0.88 * lvl))
        if h_i < 2:
            continue
        frac_x = (i + 0.5) / _STEAM_TENDRILS
        x_base = cup_cx + (frac_x - 0.5) * cup_hw * 1.7
        ys = np.arange(cup_y0 - h_i, cup_y0)
        prog = (cup_y0 - ys) / max(h_i, 1)
        curl_amp = dc * (0.035 + treble * 0.06)
        curl = np.sin(prog * 5.0 + ctx.t * st["freq"][i] + st["phase"][i]) * curl_amp * prog
        xs = np.clip(np.rint(x_base + curl).astype(np.int32), 0, dc - 1)
        bright = (1.0 - prog * 0.75) * (0.30 + 0.65 * lvl)
        ok = (ys >= 0) & (ys < dr)
        if not ok.any():
            continue
        # a plume, not a hairline: steam broadens as it rises and a louder
        # band pushes a thicker column. Two-dot-wide tendrils were the reason
        # this mode barely registered as reactive — the geometry tracked the
        # band correctly, there just wasn't enough of it on screen to see.
        half = np.maximum(1, np.rint((0.6 + prog * 2.2) * (0.5 + lvl * 1.6)).astype(np.int32))
        max_half = int(half.max())
        yy, xx0, bb, hh = ys[ok], xs[ok], bright[ok], half[ok]
        for off in range(-max_half, max_half + 1):
            sel = np.abs(off) <= hh
            if not sel.any():
                continue
            falloff = 1.0 - (abs(off) / (max_half + 1.0)) * 0.7
            px = np.clip(xx0[sel] + off, 0, dc - 1)
            np.maximum.at(field, (yy[sel], px), bb[sel] * falloff)

    np.clip(field, 0.0, 1.0, out=field)
    dots = field > 0.04
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(field))
    return codes, cidx
