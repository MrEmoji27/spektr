"""Dot-field modes: sparkle, fire, and polar geometry."""

from __future__ import annotations

import math

import numpy as np

from ..render import cell_max, noise, pack_braille
from . import Ctx, empty, mode, spread


def _polar(ctx: Ctx):
    """Cached distance/angle grids. Only rebuilt when the terminal resizes."""
    dr, dc = ctx.dot_rows, ctx.dot_cols

    def build():
        cx, cy = dc / 2.0, dr / 2.0
        x_scale = cy / max(cx, 1.0)          # braille cells are ~2x taller than wide
        xs = (np.arange(dc, dtype=np.float32) - cx) * x_scale
        ys = np.arange(dr, dtype=np.float32) - cy
        dx = xs[None, :]
        dy = ys[:, None]
        dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        ang = np.arctan2(dy + np.zeros_like(dx), dx + np.zeros_like(dy))
        ang = np.where(ang < 0, ang + 2 * math.pi, ang).astype(np.float32)
        # angle expressed as a 0..1 turn, so the per-frame path avoids a divide
        turn = (ang / np.float32(2 * math.pi)).astype(np.float32)
        return dist, turn, max(1.0, cy - 1.0)

    return ctx.scratch("polar", build)


def _angular_bands(ctx: Ctx, turn: np.ndarray, n: int, spin: float) -> np.ndarray:
    """Map every dot's angle onto the band set, blended between neighbours.

    ``turn`` is the angle as a fraction of a full turn. Doing the lookup as a
    single table index into a pre-blended ramp costs one gather instead of the
    two gathers, a cosine and three multiplies the per-dot blend needed — worth
    it when this runs over 100k dots a frame.
    """
    steps = 512
    bands = ctx.display_bands(n).astype(np.float32)
    pos = np.linspace(0.0, n, steps, endpoint=False, dtype=np.float32)
    bi = pos.astype(np.int32) % n
    frac = pos - np.floor(pos)
    tm = (1.0 - np.cos(frac * np.float32(math.pi))) * np.float32(0.5)
    lut = bands[bi] * (1.0 - tm) + bands[(bi + 1) % n] * tm

    # keep the spin bounded so float32 precision doesn't drift over a long session
    offset = np.float32(float(spin) % 1.0)
    idx = ((turn + offset) * np.float32(steps)).astype(np.int32) & (steps - 1)
    return lut[idx]


@mode("Scatter", group="particles", blurb="density sparkle, thicker where it's loud")
def scatter(ctx: Ctx):
    dr, dc = ctx.dot_rows, ctx.dot_cols
    level = spread(ctx.display_bands(), dc)[None, :]
    fade = (0.5 + 0.5 * np.arange(dr) / max(1, dr - 1))[:, None]
    density = level * level * fade
    dots = noise((dr, dc), ctx.frame) < density
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(np.where(dots, density, 0.0)))
    return codes, cidx


@mode("Flame", group="particles", blurb="fire, licking upward from each band")
def flame(ctx: Ctx):
    dr, dc = ctx.dot_rows, ctx.dot_cols
    n = ctx.n_display
    band_of = np.minimum((np.arange(dc) * n) // dc, n - 1)
    lvl = ctx.display_bands(n)[band_of][None, :]

    y = ((dr - 1 - np.arange(dr)) / max(1, dr - 1))[:, None]
    alive = y <= lvl

    seg = max(1.0, dc / n)
    centre = (band_of * seg + seg / 2.0)[None, :]

    # sin(a + b) expanded so the trig runs over a column vector and a row
    # vector rather than the full 100k-dot grid — same wobble, a fraction of
    # the cost, and this is the hottest line in the mode.
    a = (ctx.t * 9.0 + y * 6.0).astype(np.float64)
    b = (band_of * 2.1)[None, :]
    wobble = (np.sin(a) * np.cos(b) + np.cos(a) * np.sin(b)) * 1.5
    tip = 1.0 - y / np.maximum(lvl, 0.01)
    fw = (0.3 + 0.7 * tip) * (seg / 2.0)
    dist = np.abs(np.arange(dc)[None, :] - centre + 0.5 - wobble)

    edge = dist / np.maximum(fw, 1e-6)
    dots = alive & (dist < fw) & ((edge < 0.7) | (noise((dr, dc), ctx.frame + 31) < 0.6))

    # hot at the base, cooling toward the tip — the opposite of the bar modes
    heat = np.where(dots, np.clip(1.0 - y / np.maximum(lvl, 0.01), 0.0, 1.0), 0.0)
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(heat))
    return codes, cidx


@mode("Pulse", group="particles", blurb="radial pulse with shockwaves")
def pulse(ctx: Ctx):
    dr, dc = ctx.dot_rows, ctx.dot_cols
    dist, turn, max_r = _polar(ctx)
    n = min(16, ctx.n_display)

    avg = ctx.energy
    nrg = _angular_bands(ctx, turn, n, ctx.t * (0.10 + avg * 0.30))

    r = max_r * (0.1 + 0.9 * nrg * nrg)
    nz = noise((dr, dc), ctx.frame)

    core = dist < 1.0
    inside = (dist <= r) & (r >= 1.0)
    prox = np.where(inside, dist / np.maximum(r, 1e-6), 0.0)
    lit = inside & ((prox > 0.45) | (nz < 0.3 + prox * 0.7))

    halo = (~inside) & (dist < r + 4.0) & (nrg > 0.15)
    ov = np.clip((dist - r) / 4.0, 0.0, 1.0)
    lit |= halo & (nz < nrg * (1.0 - ov) * 0.4)

    phase = math.fmod(ctx.t * 3.6, 1.0)
    strength = avg * (1.0 - phase)
    if strength > 0.1:
        band = 1.0 + strength * 2.0
        edge = np.abs(dist - max_r * phase)
        near = edge < band
        fade = 1.0 - edge / band
        lit |= near & (noise((dr, dc), ctx.frame + 7) < fade * strength)

    lit |= core
    codes = pack_braille(lit)
    heat = np.maximum(prox, np.where(core, 0.2, 0.0))
    cidx = ctx.ramp(cell_max(np.where(lit, heat, 0.0)))
    return codes, cidx


@mode("Arcs", group="particles", blurb="hollow rings, one per band, pushed out by level")
def arcs(ctx: Ctx):
    """Concentric rings whose radius is a band's level.

    ``Radial`` fills wedges out from the centre; this lights only a thin
    annulus per band, so a kick reads as a ring expanding outward rather than
    as a slab growing.

    Ring membership depends only on the radius, so the entire per-band loop
    collapses into a 512-entry lookup table indexed by distance — one gather
    over the dot grid instead of one full-grid comparison per ring. That is the
    same trick :func:`_angular_bands` plays with the angle, and it is why the
    ring count can be raised without touching the frame cost.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 8 or dc < 8:
        return empty(ctx.w, ctx.h)

    dist, turn, max_r = _polar(ctx)
    n = int(min(12, max(4, ctx.n_display // 2)))
    lv = ctx.display_bands(n).astype(np.float32)

    steps = 512
    u = np.linspace(0.0, 1.0, steps, endpoint=False, dtype=np.float32)
    lut = np.zeros(steps, dtype=np.float32)
    for j in range(n):
        level = float(lv[j])
        # a slow breath per ring so a held note still has life in it
        breath = 0.012 * math.sin(ctx.t * 1.7 + j * 0.8)
        r = 0.12 + 0.84 * level + breath
        width = 0.010 + 0.028 * level
        near = np.abs(u - r)
        np.maximum(
            lut,
            np.where(near < width, (1.0 - near / width) * (0.30 + 0.70 * level), 0.0),
            out=lut,
        )

    idx = np.clip((dist * np.float32(steps / max_r)).astype(np.int32), 0, steps - 1)
    heat = lut[idx]

    # the spectrum also modulates brightness around the ring, so the rings
    # shimmer where the energy is rather than glowing evenly
    nrg = _angular_bands(ctx, turn, min(12, n), ctx.t * 0.05)
    heat *= 0.55 + 0.45 * nrg

    lit = heat > 0.06
    codes = pack_braille(lit)
    cidx = ctx.ramp(cell_max(np.where(lit, heat, 0.0)))
    return codes, cidx


#: Ring outlines for bubble radii 1..4, as (dy, dx) offset pairs. Built once at
#: import rather than per frame — a bubble is always one of four sizes.
def _ring_offsets(radius: int) -> tuple[np.ndarray, np.ndarray]:
    # x runs twice as far as y and is halved in the distance test, because a
    # braille dot is about twice as tall as it is wide. Without both halves of
    # that the "circle" comes out as a pair of horizontal dashes with no sides.
    ys = np.arange(-radius, radius + 1)
    xs = np.arange(-2 * radius, 2 * radius + 1)
    dy, dx = np.meshgrid(ys, xs, indexing="ij")
    d = np.sqrt(dy * dy + (dx * 0.5) ** 2)
    ring = (np.abs(d - radius) < 0.55) if radius > 1 else (d <= radius)
    return dy[ring].astype(np.int32), dx[ring].astype(np.int32)


_BUBBLE_RINGS = {r: _ring_offsets(r) for r in (1, 2, 3, 4)}


@mode("Bubbles", group="particles", blurb="bubbles from the low end, popping at the top")
def bubbles(ctx: Ctx):
    """Rising outlines that burst at the surface, fed by the bass.

    Distinct from ``Flame``, which grows upward from a fixed base: these are
    discrete objects that travel, and the spawn rate — not the height — is what
    the audio drives. Spawning uses an accumulator rather than a per-frame
    probability so the rate is bubbles per second at any frame rate.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 12 or dc < 12:
        return empty(ctx.w, ctx.h)

    cap = int(np.clip(dc // 6, 12, 140))

    def spawn_state():
        return {
            "y": np.full(cap, -1.0),          # < 0 means the slot is free
            "x": np.zeros(cap),
            "r": np.zeros(cap),
            "spd": np.zeros(cap),
            "pop": np.zeros(cap),             # seconds left in the burst
            "acc": 0.0,
            "rng": np.random.default_rng(23),
        }

    st = ctx.scratch("bubbles", spawn_state)
    rng = st["rng"]
    alive = st["y"] >= 0.0

    rising = alive & (st["pop"] <= 0.0)
    st["y"][rising] -= st["spd"][rising] * ctx.dt

    # reaching the surface starts a burst; the burst is what kills the bubble
    burst = rising & (st["y"] <= st["r"] + 1.0)
    st["pop"][burst] = 0.16
    popping = alive & (st["pop"] > 0.0)
    st["pop"][popping] -= ctx.dt
    st["y"][alive & (st["pop"] <= 0.0) & ~rising] = -1.0

    lows = ctx.range(0.0, 0.18)
    st["acc"] += lows * lows * 90.0 * ctx.dt
    want = int(st["acc"])
    if want:
        st["acc"] -= want
        free = np.flatnonzero(st["y"] < 0.0)[:want]
        if free.size:
            k = free.size
            st["y"][free] = dr - 1.0 - rng.uniform(0.0, dr * 0.05, k)
            st["x"][free] = rng.uniform(0.0, dc - 1.0, k)
            st["r"][free] = rng.integers(1, 5, k)
            st["spd"][free] = rng.uniform(0.18, 0.42, k) * dr * (0.6 + lows)

    field = np.zeros((dr, dc), dtype=np.float64)
    live = np.flatnonzero(st["y"] >= 0.0)
    if live.size:
        ys = np.rint(st["y"][live]).astype(np.int32)
        xs = np.rint(st["x"][live]).astype(np.int32)
        radii = st["r"][live].astype(np.int32)
        # a popping bubble expands and fades over its last frames
        grow = np.where(st["pop"][live] > 0.0, 1.0 + (0.16 - st["pop"][live]) * 9.0, 1.0)
        bright = np.where(
            st["pop"][live] > 0.0,
            st["pop"][live] / 0.16,
            0.35 + 0.65 * (1.0 - st["y"][live] / max(1.0, dr - 1.0)),
        )
        drawn = np.clip(np.rint(radii * grow).astype(np.int32), 1, 4)

        for size, (dy, dx) in _BUBBLE_RINGS.items():
            sel = drawn == size
            if not sel.any():
                continue
            py = ys[sel][:, None] + dy[None, :]
            px = xs[sel][:, None] + dx[None, :]
            ok = (py >= 0) & (py < dr) & (px >= 0) & (px < dc)
            vals = np.repeat(bright[sel][:, None], dy.size, axis=1)
            np.maximum.at(field, (py[ok], px[ok]), vals[ok])

    # the surface the bubbles are heading for. Dotted and shallow on purpose —
    # a full-amplitude line across the top reads as a waveform, which is the
    # one thing this mode should not look like.
    sx = np.arange(0, dc, 3)
    ripple = np.sin(sx * 0.07 + ctx.t * 1.4) * (0.4 + ctx.range(0.25, 0.60) * 1.1)
    srow = np.clip(np.rint(1.5 + ripple).astype(np.int32), 0, dr - 1)
    field[srow, sx] = np.maximum(field[srow, sx], 0.22)

    np.clip(field, 0.0, 1.0, out=field)
    dots = field > 0.05
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(field))
    return codes, cidx


@mode("Radial", group="particles", blurb="the spectrum wrapped into a circle")
def radial(ctx: Ctx):
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 8 or dc < 8:
        return empty(ctx.w, ctx.h)

    dist, turn, max_r = _polar(ctx)
    n = min(24, max(8, ctx.n_display))
    nrg = _angular_bands(ctx, turn, n, ctx.t * 0.04)

    inner = max_r * 0.18
    outer = inner + (max_r - inner) * nrg

    # thin radial gutters so the wedges read as separate bars
    wedge = np.mod(turn * n, 1.0)
    gutter = (wedge < 0.06) | (wedge > 0.94)

    lit = (dist >= inner) & (dist <= outer) & ~gutter
    ring = np.abs(dist - inner) < 0.9
    lit |= ring

    heat = np.where(lit, np.clip((dist - inner) / np.maximum(outer - inner, 1e-6), 0, 1), 0.0)
    codes = pack_braille(lit)
    cidx = ctx.ramp(cell_max(heat))
    return codes, cidx
