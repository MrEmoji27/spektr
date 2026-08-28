"""Modes built for the Android terrain view.

These are ordinary registered modes — they return the standard glyph contract
and render flat on the desktop like any half-block picture. What makes them
different is the bargain they strike with the port's GLES renderer: each one
computes its picture as a continuous height field and hands that field to
``ctx.ramp`` whole, in a single call, at full field resolution. The bridge
records exactly those pre-quantisation floats (see ``spektr_android``), so on
Android the terrain view displaces and lights the mode's own numbers rather
than a reconstruction of them. Zero precision lost between mode and mountain.

The craft rules that fall out of that:

* half-block geometry — two field rows per cell, so the height map is twice
  the screen's cell resolution and slopes stay smooth;
* continuous values everywhere, never binary dots — braille silhouettes make
  terrible mountains;
* content that means something as elevation: swells, ridges, craters — not
  pictures flattened into relief.

On a desktop terminal they read as moody shaded landscapes. On the tablet,
lit and displaced, they read as weather.
"""

from __future__ import annotations

import math

import numpy as np

from ..render import cell_max  # noqa: F401 — kept for parity with siblings
from . import Ctx, empty, mode


def _aspect(w: int, h: int) -> float:
    """Field-pixel aspect: a cell is twice as tall as wide, and the field is
    two rows per cell, so field pixels are square — scale x by width/height."""
    return w / max(1, h)


def _soft(field: np.ndarray, knee: float = 0.18) -> np.ndarray:
    """Fold a field into 0..1 without ever landing flat on either end.

    ``np.clip`` was what these modes used, and on a flat picture it is
    invisible. Under the terrain view it is not: a clipped region is a plateau
    with a crease around it, and — worse — a trough clipped to exactly 0.0 is
    indistinguishable from a pixel the mode never drew. Swell at any real
    listening level spent most of its surface pinned at one end or the other,
    and the sea came out with holes in it.

    So the outer ``knee`` of the range is exponential instead of a wall. The
    middle is untouched and keeps its contrast; past the knee the curve eases
    over and approaches the limit without ever arriving, for any input however
    far outside. It matches value and slope at the join, so no crease appears
    where the two halves meet.
    """
    k = np.float32(knee)
    out = np.asarray(field, dtype=np.float32)
    out = np.where(out < k, k * np.exp((out - k) / k), out)
    out = np.where(out > 1.0 - k, 1.0 - k * np.exp((1.0 - k - out) / k), out)
    return out.astype(np.float32)


# ── Swell ────────────────────────────────────────────────────────────────────
#
# An ocean surface: three travelling swells whose character comes from the
# spectrum, plus impact ripples thrown by onsets. Sum-of-sines was chosen
# over noise deliberately — analytic waves are perfectly smooth under
# lighting, cost almost nothing, and never alias however the camera drifts.

#: Ripple slots. Splashes are rare by design — one per strong hit — and old
#: ones fade fast, so six covers any honest shower without growing forever.
_SWELL_RIPPLES = 6
_SWELL_RIPPLE_LIFE = 2.6


def _swell_state(w: int, fh: int) -> dict:
    rng = np.random.default_rng(211)
    # Coordinate lattices, built once per shape: x spans ±aspect so rings
    # stay round, y spans ±1 down the screen.
    xs = np.linspace(-_aspect(w, fh // 2), _aspect(w, fh // 2), w, dtype=np.float32)[None, :]
    ys = np.linspace(-1.0, 1.0, fh, dtype=np.float32)[:, None]
    return {
        "xs": np.broadcast_to(xs, (fh, w)).copy(),
        "ys": np.broadcast_to(ys, (fh, w)).copy(),
        # cx, cy, born, strength per ripple; age < 0 is the free sentinel.
        "cx": np.zeros(_SWELL_RIPPLES),
        "cy": np.zeros(_SWELL_RIPPLES),
        "born": np.full(_SWELL_RIPPLES, -1e9),
        "power": np.zeros(_SWELL_RIPPLES),
        "rng": rng,
    }


@mode("Swell", group="terrain",
      blurb="an open sea in half-tones — built for the Android terrain view")
def swell(ctx: Ctx):
    """A sea whose swell is the spectrum and whose splashes are the beat.

    Three travelling waves give the water its idle motion — long, slow, and
    always there, because a frozen sea reads as a painting. The spectrum
    then shapes it: bass drives the long heavy swell, mids add chop, energy
    raises the whole surface. Onsets throw stones: each hit lands a ripple
    somewhere on the water, an expanding ring that fades as it runs, sized
    by ``onset_strength`` — a hard snare reaches the horizon, a soft one
    dies mid-field.

    Heights cross to the renderer untouched: one ``ramp`` call over the full
    field, which is the arrangement this family exists to make.
    """
    w, h = ctx.w, ctx.h
    if h < 6 or w < 16:
        return empty(w, h)

    fh = h * 2
    st = ctx.scratch("swell", lambda: _swell_state(w, fh))
    rng = st["rng"]
    t = ctx.t

    bass = ctx.range(0.0, 0.2)
    mid = ctx.range(0.15, 0.7)

    # ── the travelling swells ──
    # Wavelengths and speeds are fixed; amplitudes come from the music. The
    # phases drift at different rates so the interference pattern never
    # repeats visibly.
    # The three amplitudes are a budget, not three free choices: they sum,
    # and the sum is the excursion either side of the waterline. At the old
    # figures three crests in phase reached 1.48 — three times the room there
    # is — so the surface lived at its limits and the mode read as terraces.
    swell_long = (0.16 + bass * 0.28) * np.sin(
        st["xs"] * 2.1 - st["ys"] * 1.1 + t * 0.9)
    swell_mid = (0.05 + mid * 0.15) * np.sin(
        st["xs"] * 4.7 + st["ys"] * 2.9 - t * 1.7 + 1.3)
    chop = (0.025 + mid * 0.09) * np.sin(
        st["xs"] * 9.3 - st["ys"] * 5.1 + t * 2.9 + 0.5)
    field = 0.5 + swell_long + swell_mid + chop

    # ── impact ripples ──
    if ctx.onsets:
        hard = float(np.clip(ctx.onset_strength, 0.0, 1.0))
        slot = int(np.argmin(st["born"]))
        st["cx"][slot] = rng.uniform(-0.8, 0.8)
        st["cy"][slot] = rng.uniform(-0.7, 0.7)
        st["born"][slot] = t
        st["power"][slot] = 0.16 + 0.26 * hard

    alive = (t - st["born"]) < _SWELL_RIPPLE_LIFE
    if alive.any():
        age = (t - st["born"])[alive]
        dx = st["xs"] - st["cx"][alive][:, None, None]
        dy = st["ys"] - st["cy"][alive][:, None, None]
        dist = np.sqrt(dx * dx + dy * dy)
        front = age[:, None, None] * 1.4
        band = np.exp(-((dist - front) ** 2) / 0.02)
        ring = np.sin((dist - front) * 14.0)
        fade = np.exp(-age[:, None, None] * 1.6) * st["power"][alive][:, None, None]
        field += np.sum(band * ring * fade, axis=0)

    field = _soft(field)

    # One ramp call over the whole field — the contract this family exists
    # for. Top and bottom halves take their colours from the same array.
    ramped = ctx.ramp(field)
    codes = np.full((h, w), 0x2580, dtype=np.int32)
    return codes, ramped[0::2].astype(np.int32), ramped[1::2].astype(np.int32)


# ── Terra ────────────────────────────────────────────────────────────────────
#
# A ridged landscape. Value-noise fBm would be the obvious build; this uses
# three octaves of sine-hash noise instead — cheaper than a permutation
# table, stable across resizes, and smooth enough that the lit slopes show
# no facets. The music owns the relief: bass lifts the ranges toward real
# mountains, and every onset raises a hill somewhere that subsides slowly.

_TERRA_BUMPS = 5
_TERRA_BUMP_LIFE = 7.0
_TERRA_OCTAVES = ((3.0, 0.55), (6.5, 0.30), (13.0, 0.15))


def _terra_state(w: int, fh: int) -> dict:
    rng = np.random.default_rng(97)
    xs = np.linspace(0.0, _aspect(w, fh // 2), w, dtype=np.float32)[None, :]
    ys = np.linspace(0.0, 1.0, fh, dtype=np.float32)[:, None]
    return {
        "xs": np.broadcast_to(xs, (fh, w)).copy(),
        "ys": np.broadcast_to(ys, (fh, w)).copy(),
        "bx": np.zeros(_TERRA_BUMPS),
        "by": np.zeros(_TERRA_BUMPS),
        "born": np.full(_TERRA_BUMPS, -1e9),
        "power": np.zeros(_TERRA_BUMPS),
        "drift": rng.uniform(0.0, 40.0, (2,)),
        "rng": rng,
    }


def _sine_noise(u: np.ndarray, v: np.ndarray, freq: float, seed: float) -> np.ndarray:
    """Hash-free value noise: sines of lattice coordinates, smoothed.

    Not real gradient noise, and does not need to be — the ridged transform
    downstream only asks for smooth hills and hollows at a known scale, and
    this delivers them for four multiplies and two trig calls per octave.
    """
    su = u * freq + seed
    sv = v * freq * 0.62 + seed * 1.7
    iu, iv = np.floor(su), np.floor(sv)
    fu, fv = su - iu, sv - iv
    fu = fu * fu * (3.0 - 2.0 * fu)          # smoothstep the lattice crossings
    fv = fv * fv * (3.0 - 2.0 * fv)
    n00 = np.sin(iu * 12.9898 + iv * 78.233 + seed)
    n10 = np.sin((iu + 1) * 12.9898 + iv * 78.233 + seed)
    n01 = np.sin(iu * 12.9898 + (iv + 1) * 78.233 + seed)
    n11 = np.sin((iu + 1) * 12.9898 + (iv + 1) * 78.233 + seed)
    top = n00 * (1 - fu) + n10 * fu
    bot = n01 * (1 - fu) + n11 * fu
    return (top * (1 - fv) + bot * fv).astype(np.float32)


@mode("Terra", group="terrain",
      blurb="a bass-lifted mountain range, hills raised by hits — for terrain view")
def terra(ctx: Ctx):
    """Mountains the mix raises.

    The base landscape is ridged fractal noise drifting slowly eastward, so
    the range evolves even in silence — geology on a clock nobody hears. The
    music then acts on it the way weather acts on land: sustained bass lifts
    the whole massif (the kick drum literally moves mountains here), while
    each onset raises a discrete hill that erodes back over about seven
    seconds. Loud passages are alpine; quiet ones are high steppe.

    Same single-ramp contract as :func:`swell`: the field crosses to the
    renderer as raw floats, one call, no reconstruction.
    """
    w, h = ctx.w, ctx.h
    if h < 6 or w < 16:
        return empty(w, h)

    fh = h * 2
    st = ctx.scratch("terra", lambda: _terra_state(w, fh))
    t = ctx.t

    bass = ctx.range(0.0, 0.2)

    # Drifting sample origin: the landscape slides slowly east. Two stored
    # drift rates keep consecutive sessions from starting identically.
    u = st["xs"] * 0.35 + t * 0.014 + st["drift"][0]
    v = st["ys"] * 0.35 + t * 0.004 + st["drift"][1]

    acc = np.zeros((fh, w), dtype=np.float32)
    norm = 0.0
    for freq, amp in _TERRA_OCTAVES:
        n = _sine_noise(u, v, freq, seed=freq * 3.77)
        acc += n * amp
        norm += amp
    ridge = 1.0 - np.abs(acc / norm) * 2.0        # sharp crests, soft valleys
    field = np.clip(ridge, 0.0, 1.0) ** 1.35

    lift = 0.22 + bass * 0.72
    field *= lift

    # ── onset hills ──
    if ctx.onsets:
        hard = float(np.clip(ctx.onset_strength, 0.0, 1.0))
        slot = int(np.argmin(st["born"]))
        st["bx"][slot] = st["rng"].uniform(0.15, 0.85)
        st["by"][slot] = st["rng"].uniform(0.2, 0.9)
        st["born"][slot] = t
        st["power"][slot] = 0.30 + 0.50 * hard

    alive = (t - st["born"]) < _TERRA_BUMP_LIFE
    if alive.any():
        age = (t - st["born"])[alive]
        px = st["bx"][alive] * w
        py = st["by"][alive] * fh
        dx = st["xs"] * w - px[:, None, None]
        dy = st["ys"] * fh - py[:, None, None]
        rad = 0.09 * min(fh, w)
        hill = np.exp(-(dx * dx + dy * dy) / (2.0 * rad * rad))
        grow = np.minimum(age[:, None, None] * 3.0, 1.0)      # rise fast…
        erode = np.exp(-np.maximum(age[:, None, None] - 1.2, 0.0) * 0.55)  # …erode slow
        field += np.sum(hill * grow * erode * st["power"][alive][:, None, None], axis=0)

    field = _soft(field)

    ramped = ctx.ramp(field)
    codes = np.full((h, w), 0x2580, dtype=np.int32)
    return codes, ramped[0::2].astype(np.int32), ramped[1::2].astype(np.int32)