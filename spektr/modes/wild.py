"""The wild family: four modes as wild as Crosscurrent, without the tunnel.

Crosscurrent's charge comes from four things at once: two streams running
against each other, a motion that speeds up with the music, sparks where the
streams meet, and light that stutters on the beat. Each mode here keeps that
and drops the tunnel:

- ``Riptide``: two currents of wave lines sweeping across a flat screen in
  opposite directions, sparking where they cross, bent by a shockwave on
  every hit;
- ``Twin Storms``: two whirlpools of particles spinning opposite ways and
  trading streams, colliding in the middle;
- ``Arc Storm``: lightning across the screen, an arc per slice of the
  spectrum, forking on hits;
- ``Shatter``: a pane of glass in shards that fill with the music and blow
  apart where a hit lands.
"""
from __future__ import annotations

import math

import numpy as np

from ..render import cell_max, pack_braille
from . import Ctx, empty, mode


def _plane(ctx: Ctx) -> dict:
    """The dot grid as a flat plane: ``y`` runs -1..1 down the frame, ``x``
    the same scale across, so a unit is the same length both ways."""
    dr, dc = ctx.dot_rows, ctx.dot_cols

    def build():
        half = dr / 2
        y = ((np.arange(dr, dtype=np.float32) - half + 0.5) / half)[:, None]
        x = ((np.arange(dc, dtype=np.float32) - dc / 2 + 0.5) / half)[None, :]
        return {"x": np.broadcast_to(x, (dr, dc)).copy(),
                "y": np.broadcast_to(y, (dr, dc)).copy(), "aspect": dc / dr}

    return ctx.scratch("wild_plane", build)


def _hit(ctx: Ctx) -> float:
    """How hard this frame's hit is, 0..1, or 0.0 with none."""
    return min(1.0, float(ctx.onset_strength)) if ctx.onsets else 0.0


def _decay(value: float, dt: float, tau: float) -> float:
    return value * math.exp(-dt / tau)


# ── Riptide ──────────────────────────────────────────────────────────────────

#: Lines of each current across the height of the frame.
_RT_LINES = 7.0

#: Each current's angle from the vertical, in radians; the two lean opposite ways.
_RT_ANGLE = 0.55

#: A line's half-width, as a share of the gap between lines.
_RT_WIDTH = 0.11

#: A hit's shockwave: how fast it spreads (plane units a second), how long it
#: lasts, how far it bends the lines, and how many run at once.
_RT_SHOCK_SPEED = 2.4
_RT_SHOCK_LIFE = 0.7
_RT_SHOCK_BEND = 0.3
_RT_SHOCKS = 6


@mode("Riptide", group="wild", after="Boot",
      blurb="two currents of wave lines sweeping across each other, sparking where they cross, bent by every hit")
def riptide(ctx: Ctx):
    """Two currents of wave lines running against each other across the screen.

    One current leans one way and runs on the bass, the other leans the other
    way and runs on the top end, each faster the louder its half of the
    spectrum. The lines wave harder as the music gets louder, and where two
    lines cross they spark. A hit bends every line with a shockwave rolling
    out from where it lands and kicks both currents forward, and while the
    music is hitting the lines stutter into dashes.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 8 or dc < 8:
        return empty(ctx.w, ctx.h)
    p = _plane(ctx)
    x, y = p["x"], p["y"]
    def build():
        ca, sa = math.cos(_RT_ANGLE), math.sin(_RT_ANGLE)
        va, vb = -x * sa + y * ca, x * sa + y * ca
        # the waves' sines, once: sin(a + b) = sin a cos b + cos a sin b
        return {"ua": x * ca + y * sa, "ub": x * ca - y * sa, "va": va, "vb": vb,
                "sa": np.sin(va * 3.1), "ca": np.cos(va * 3.1),
                "sb": np.sin(vb * 2.7), "cb": np.cos(vb * 2.7)}

    geo = ctx.scratch("riptide_geo", build)
    st = ctx.scratch("riptide", lambda: {
        "a": 0.0, "b": 0.0, "flash": 0.0, "shocks": [], "rng": np.random.default_rng(3),
    })
    dt = max(ctx.dt, 0.0)
    quiet = ctx.silent
    bass = 0.0 if quiet else ctx.range(0.0, 0.25)
    treble = 0.0 if quiet else ctx.range(0.5, 1.0)
    level = 0.0 if quiet else float(ctx.energy)

    st["a"] += dt * (0.12 + 2.4 * bass)
    st["b"] += dt * (0.12 + 2.4 * treble)
    st["flash"] = _decay(st["flash"], dt, 0.12)
    s = _hit(ctx)
    if s:
        st["flash"] = max(st["flash"], 0.5 + 0.5 * s)
        st["a"] += 0.3 * s
        st["b"] += 0.3 * s
        rng = st["rng"]
        ox = float(rng.uniform(-p["aspect"] * 0.8, p["aspect"] * 0.8))
        oy = float(rng.uniform(-0.8, 0.8))
        # its distance field, once, for as long as it runs
        r = np.hypot(x - np.float32(ox), y - np.float32(oy))
        st["shocks"] = (st["shocks"] + [(r, ctx.t, s)])[-_RT_SHOCKS:]

    # the lines wave harder as the music gets louder, and every live shock
    # bends them
    wave = np.float32(0.05 + 0.25 * level)
    bend = np.zeros_like(x)
    live = []
    for shock in st["shocks"]:
        r, born, amp = shock
        age = ctx.t - born
        if age > _RT_SHOCK_LIFE:
            continue
        live.append(shock)
        front = age * _RT_SHOCK_SPEED
        fade = amp * _RT_SHOCK_BEND * (1.0 - age / _RT_SHOCK_LIFE)
        # a tent round the front: a ridge the lines ride over
        ridge = np.float32(1.0) - np.abs(r - np.float32(front)) * np.float32(1 / 0.2)
        np.maximum(ridge, 0.0, out=ridge)
        bend += np.float32(fade) * ridge
    st["shocks"] = live

    spacing = _RT_LINES / 2.0
    pa, pb = ctx.t * 1.3, -ctx.t * 1.1
    sin_a = geo["sa"] * np.float32(math.cos(pa)) + geo["ca"] * np.float32(math.sin(pa))
    sin_b = geo["sb"] * np.float32(math.cos(pb)) + geo["cb"] * np.float32(math.sin(pb))
    ua = (geo["ua"] + wave * sin_a + bend) * np.float32(spacing) - np.float32(st["a"])
    ub = (geo["ub"] + wave * sin_b - bend) * np.float32(spacing) + np.float32(st["b"])
    # distance to the nearest line of each current, in gaps, worked in place
    da = ua - np.rint(ua)
    np.abs(da, out=da)
    db = ub - np.rint(ub)
    np.abs(db, out=db)
    on_a, on_b = da < 0.7 * _RT_WIDTH, db < 0.7 * _RT_WIDTH
    if st["flash"] > 0.3:
        # stutter: the lines break into dashes whose gaps jump every strobe
        step = int(ctx.t * 14)
        sa = geo["va"] * np.float32(5.0) + np.float32(step * 0.37)
        sb = geo["vb"] * np.float32(5.0) + np.float32(step * 0.61)
        on_a &= sa - np.floor(sa) > 0.3
        on_b &= sb - np.floor(sb) > 0.3
    dots = on_a | on_b
    # the bass current low on the ramp, the top end's higher, sparks hot:
    # a line's colour falls off from its centre, and only lit dots are read
    ka = 0.25 + 0.3 * bass + 0.2 * st["flash"]
    kb = 0.5 + 0.3 * treble + 0.2 * st["flash"]
    da *= np.float32(-ka / _RT_WIDTH)
    da += np.float32(ka)
    db *= np.float32(-kb / _RT_WIDTH)
    db += np.float32(kb)
    heat = np.maximum(da, db, out=da)
    heat[on_a & on_b] = 1.0
    return pack_braille(dots), ctx.ramp(cell_max(heat))


# ── Twin Storms ──────────────────────────────────────────────────────────────

#: Particles across both storms.
_TS_N = 520

#: The storms' eyes, as a share of the half-width out from the middle.
_TS_EYE = 0.55

#: How long a particle's trail lingers, in seconds.
_TS_TRAIL = 0.09

#: The chance a second, scaled by how loud it is, that a particle on the
#: side facing the other storm is pulled across into it.
_TS_TRADE = 2.5


@mode("Twin Storms", group="wild", after="Riptide",
      blurb="two whirlpools of particles spinning opposite ways, trading streams and colliding in the middle")
def twin_storms(ctx: Ctx):
    """Two whirlpools spinning against each other, and the storm between them.

    The left storm spins on the bass, the right one the other way on the top
    end, each faster the louder its half of the spectrum, the inside of each
    faster than its rim. Particles on the rim facing the other storm are
    pulled across into it, more of them the louder it is, so streams cross the
    middle and spark where they meet. A hit flings every particle outward and
    the storms draw them back in.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 8 or dc < 8:
        return empty(ctx.w, ctx.h)
    p = _plane(ctx)
    aspect = p["aspect"]
    eye = _TS_EYE * aspect

    def seed():
        rng = np.random.default_rng(5)
        rest = rng.uniform(0.12, min(0.95, eye * 0.95), _TS_N).astype(np.float32)
        return {
            "rng": rng, "rest": rest, "rad": rest.copy(),
            "th": rng.uniform(0, 2 * np.pi, _TS_N).astype(np.float32),
            "rv": np.zeros(_TS_N, np.float32),
            "side": (np.arange(_TS_N) % 2).astype(np.int8),
            "spin": np.array([1.0, -1.0]), "trail": np.zeros((dr, dc), np.float32),
            "flash": 0.0,
        }

    st = ctx.scratch("twin_storms", seed)
    dt = max(ctx.dt, 0.0)
    rng = st["rng"]
    quiet = ctx.silent
    bass = 0.0 if quiet else ctx.range(0.0, 0.25)
    treble = 0.0 if quiet else ctx.range(0.5, 1.0)
    level = 0.0 if quiet else float(ctx.energy)

    # each storm's spin eases toward what its half of the spectrum asks for
    target = np.array([0.8 + 9.0 * bass, -(0.8 + 9.0 * treble)])
    st["spin"] += (target - st["spin"]) * (1.0 - math.exp(-dt / 0.3))
    st["flash"] = _decay(st["flash"], dt, 0.1)
    s = _hit(ctx)
    if s:
        st["rv"] += (s * rng.uniform(1.2, 3.2, _TS_N)).astype(np.float32)
        st["spin"] *= 1.0 + 0.5 * s
        st["flash"] = max(st["flash"], s)

    side, rad = st["side"], st["rad"]
    spin = st["spin"][side].astype(np.float32)
    st["th"] += np.float32(dt) * spin * np.float32(0.35) / (rad + np.float32(0.12))
    # a spring back to each particle's own orbit, damped
    st["rv"] += (st["rest"] - rad) * np.float32(10.0 * dt)
    st["rv"] *= np.float32(math.exp(-dt / 0.3))
    rad += st["rv"] * np.float32(dt)
    np.maximum(rad, 0.02, out=rad)

    # the rim facing the other storm pulls particles across
    facing = np.cos(st["th"]) * np.where(side == 0, 1.0, -1.0) > 0.92
    trade = facing & (rad > eye * 0.7) & (rng.random(_TS_N) < _TS_TRADE * (0.2 + level) * dt)
    if trade.any():
        side[trade] = 1 - side[trade]
        st["th"][trade] += np.float32(np.pi)

    cx = np.where(side == 0, -eye, eye).astype(np.float32)
    px = cx + rad * np.cos(st["th"])
    py = rad * np.sin(st["th"]) * np.float32(0.9)

    st["trail"] *= np.float32(math.exp(-dt / _TS_TRAIL))
    col = np.round(px * (dr / 2) + dc / 2 - 0.5).astype(np.int32)
    row = np.round(py * (dr / 2) + dr / 2 - 0.5).astype(np.int32)
    ok = (row >= 0) & (row < dr) & (col >= 0) & (col < dc)
    # the left storm low on the ramp, the right one high, the middle sparking
    heat = np.where(side == 0, 0.3 + 0.35 * bass, 0.55 + 0.3 * treble).astype(np.float32)
    heat = np.where(np.abs(px) < 0.12 * aspect, np.float32(1.0), heat)
    heat += np.float32(0.2) * np.minimum(np.abs(st["rv"]), 1.0) + np.float32(0.35 * st["flash"])
    np.maximum.at(st["trail"], (row[ok], col[ok]), np.minimum(heat[ok], 1.0))

    trail = st["trail"]
    return pack_braille(trail > 0.12), ctx.ramp(cell_max(trail))


# ── Arc Storm ────────────────────────────────────────────────────────────────

#: Arcs across the screen, one per slice of the spectrum, bass at the bottom.
_AS_ARCS = 7

#: Below this a slice's arc is not drawn.
_AS_FLOOR = 0.07

#: How long an arc holds one shape before it jumps to another, in seconds:
#: lightning holds for a moment, it does not boil.
_AS_HOLD_S = 0.14

#: How long the glow of an arc lingers after it jumps, in seconds.
_AS_GLOW = 0.07

#: A fork's life, in seconds.
_AS_FORK_S = 0.25


def _arc_path(rng, dc: int, base: float, reach: float) -> np.ndarray:
    """A jagged path across ``dc`` columns, pinned to ``base`` at both ends."""
    walk = np.cumsum(rng.normal(0.0, 1.0, dc))
    walk -= np.linspace(0.0, walk[-1], dc)
    peak = max(float(np.abs(walk).max()), 1e-6)
    return base + walk / peak * reach


def _raster(path: np.ndarray, dr: int, rows: np.ndarray) -> np.ndarray:
    """The dots of a path drawn as a connected line, one column at a time."""
    y = np.clip(np.round(path), 0, dr - 1)
    nxt = np.concatenate((y[1:], y[-1:]))
    lo, hi = np.minimum(y, nxt), np.maximum(y, nxt)
    return (rows >= lo[None, :]) & (rows <= hi[None, :])


@mode("Arc Storm", group="wild", after="Twin Storms",
      blurb="lightning across the screen, an arc per slice of the spectrum, forking on every hit")
def arc_storm(ctx: Ctx):
    """Lightning across the screen: an arc for each slice of the spectrum.

    Each slice of the spectrum, bass at the bottom and the top end at the
    top, is an arc of lightning pinned to both edges of the frame. The louder
    its slice, the brighter it burns and the wilder it zig-zags, and a quiet
    slice goes dark, so a loud passage fills the screen with lightning and a
    quiet one leaves a line or two. Arcs hold a shape for a moment and jump to
    another, the way lightning does. A hit reshapes every arc at once,
    flares them, and throws forks off them.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 8 or dc < 8:
        return empty(ctx.w, ctx.h)
    st = ctx.scratch("arc_storm", lambda: {
        "glow": np.zeros((dr, dc), np.float32), "paths": [None] * _AS_ARCS,
        "shaped": [-99.0] * _AS_ARCS, "forks": [], "flash": 0.0,
        "rng": np.random.default_rng(11),
        "rows": np.arange(dr, dtype=np.float64)[:, None],
    })
    dt = max(ctx.dt, 0.0)
    rng, glow = st["rng"], st["glow"]
    glow *= np.float32(math.exp(-dt / _AS_GLOW))
    st["flash"] = _decay(st["flash"], dt, 0.1)
    s = _hit(ctx)
    if s:
        st["flash"] = max(st["flash"], 0.5 + 0.5 * s)

    gap = dr / (_AS_ARCS + 1)
    for k in range(_AS_ARCS):
        lv = 0.0 if ctx.silent else ctx.range(k / _AS_ARCS, (k + 1) / _AS_ARCS)
        if lv < _AS_FLOOR:
            st["paths"][k] = None
            continue
        if s or st["paths"][k] is None or ctx.t - st["shaped"][k] >= _AS_HOLD_S:
            base = dr - gap * (k + 1)
            reach = gap * (0.25 + 1.1 * lv + 0.7 * st["flash"])
            st["paths"][k] = _arc_path(rng, dc, base, reach)
            st["shaped"][k] = ctx.t
        heat = np.float32(min(1.0, 0.25 + 0.7 * lv + 0.35 * st["flash"]))
        line = _raster(st["paths"][k], dr, st["rows"])
        np.maximum(glow, np.where(line, heat, np.float32(0.0)), out=glow)
        if s and rng.random() < 0.5 + 0.5 * s:
            # a fork off this arc, at a random point, heading up or down
            x0 = int(rng.integers(0, dc))
            y0 = float(st["paths"][k][x0])
            n = int(rng.integers(dr // 4, dr // 2 + 2))
            xs = x0 + np.cumsum(rng.integers(-2, 3, n) + rng.choice((-1, 1)))
            ys = y0 + np.arange(1, n + 1) * rng.choice((-1.0, 1.0))
            st["forks"].append((ys, xs, ctx.t))

    live = []
    for ys, xs, born in st["forks"]:
        age = ctx.t - born
        if age > _AS_FORK_S:
            continue
        live.append((ys, xs, born))
        r = np.round(ys).astype(np.int32)
        c = xs.astype(np.int32)
        ok = (r >= 0) & (r < dr) & (c >= 0) & (c < dc)
        glow[r[ok], c[ok]] = np.maximum(glow[r[ok], c[ok]],
                                        np.float32(1.0 - 0.6 * age / _AS_FORK_S))
    st["forks"] = live[-24:]
    return pack_braille(glow > 0.1), ctx.ramp(cell_max(glow))


# ── Shatter ──────────────────────────────────────────────────────────────────

#: Shards across and down the pane.
_SH_ACROSS, _SH_DOWN = 7, 4

#: How hard a hit blows the shards apart, and how fast they spring back.
_SH_BLAST = 1.6
_SH_SPRING = 28.0
_SH_DAMP = 5.0

#: A hit's cracks: how many run out from where it lands, and how long they last.
_SH_CRACKS = 7
_SH_CRACK_S = 0.35


def _pane(dr: int, dc: int) -> dict:
    """The pane cut into shards: each dot's shard, and the dots each draws."""
    rng = np.random.default_rng(17)
    gy, gx = np.mgrid[0:_SH_DOWN, 0:_SH_ACROSS]
    sy = ((gy + 0.5 + rng.uniform(-0.35, 0.35, gy.shape)) * dr / _SH_DOWN).ravel()
    sx = ((gx + 0.5 + rng.uniform(-0.35, 0.35, gx.shape)) * dc / _SH_ACROSS).ravel()
    rows = np.arange(dr, dtype=np.float32)[:, None]
    cols = np.arange(dc, dtype=np.float32)[None, :]
    best = np.full((dr, dc), np.inf, np.float32)
    label = np.zeros((dr, dc), np.int32)
    for i in range(sy.size):
        d = (rows - sy[i]) ** 2 + (cols - sx[i]) ** 2
        closer = d < best
        best[closer] = d[closer]
        label[closer] = i
    edge = np.zeros((dr, dc), bool)
    edge[:, :-1] |= label[:, :-1] != label[:, 1:]
    edge[:-1, :] |= label[:-1, :] != label[1:, :]
    ey, ex = np.nonzero(edge)
    # the inside of each shard as a stipple, every other dot each way, each
    # with its own threshold so a shard fills in grain by grain as it gets loud
    iy, ix = np.nonzero(~edge & ((rows % 2 == 0) & (cols % 2 == 0)))
    return {
        "sy": sy, "sx": sx, "ey": ey, "ex": ex, "el": label[ey, ex],
        "iy": iy, "ix": ix, "il": label[iy, ix],
        "grain": rng.random(iy.size).astype(np.float32),
        "order": np.argsort(np.argsort(sx)),
    }


def _splat(field, ys, xs, value, dr, dc):
    r = np.round(ys).astype(np.int32)
    c = np.round(xs).astype(np.int32)
    ok = (r >= 0) & (r < dr) & (c >= 0) & (c < dc)
    v = value[ok] if np.ndim(value) else value
    field[r[ok], c[ok]] = np.maximum(field[r[ok], c[ok]], v)


@mode("Shatter", group="wild", after="Arc Storm",
      blurb="a pane of glass in shards that fill with the music and blow apart where a hit lands")
def shatter(ctx: Ctx):
    """A pane of glass in shards, and every hit breaking it.

    The pane is cut into shards, the bass on the left and the top end on the
    right, and each fills with grain as its part of the spectrum gets loud. A
    hit lands somewhere on the pane: cracks run out from the point, it
    flashes, and the shards near it are blown apart, the nearest hardest.
    They spring back together between the hits, so a steady beat keeps the
    pane breaking and mending, and a quiet passage lets it close up.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 8 or dc < 8:
        return empty(ctx.w, ctx.h)
    pane = ctx.scratch("shatter_pane", lambda: _pane(dr, dc))
    k = pane["sy"].size
    st = ctx.scratch("shatter", lambda: {
        "off": np.zeros((k, 2)), "vel": np.zeros((k, 2)), "glow": np.zeros(k),
        "cracks": [], "rng": np.random.default_rng(23),
    })
    dt = max(ctx.dt, 0.0)
    rng = st["rng"]
    st["vel"] += (-st["off"] * _SH_SPRING - st["vel"] * _SH_DAMP) * dt
    st["off"] += st["vel"] * dt
    st["glow"] *= math.exp(-dt / 0.2)

    s = _hit(ctx)
    if s:
        hy, hx = float(rng.uniform(0.15, 0.85) * dr), float(rng.uniform(0.1, 0.9) * dc)
        dy, dx = pane["sy"] - hy, pane["sx"] - hx
        d = np.hypot(dy, dx) + 1e-6
        push = s * _SH_BLAST * dr * np.exp(-d / (0.45 * dr))
        st["vel"][:, 0] += push * dy / d * 6.0
        st["vel"][:, 1] += push * dx / d * 6.0
        st["glow"] = np.maximum(st["glow"], s * np.exp(-d / (0.35 * dr)))
        # each crack's jag is drawn once, so it grows along the same line
        cracks = [(hy, hx, float(a), ctx.t, s,
                   np.cumsum(rng.normal(0.0, 0.35, dr)).astype(np.float32))
                  for a in rng.uniform(0, 2 * np.pi, _SH_CRACKS)]
        st["cracks"] = (st["cracks"] + cracks)[-4 * _SH_CRACKS:]

    bands = np.asarray(ctx.display_bands(k), np.float32)
    level = np.zeros(k, np.float32) if ctx.silent else bands[pane["order"]]
    field = np.zeros((dr, dc), np.float32)
    oy, ox = st["off"][:, 0], st["off"][:, 1]

    # the grain inside each shard, as much of it as its band is loud
    il = pane["il"]
    fill = pane["grain"] < level[il] * 1.3
    _splat(field, pane["iy"][fill] + oy[il[fill]], pane["ix"][fill] + ox[il[fill]],
           (0.2 + 0.5 * level[il[fill]] + 0.4 * st["glow"][il[fill]]).astype(np.float32), dr, dc)
    # the edges, always there, brighter where a hit just landed
    el = pane["el"]
    _splat(field, pane["ey"] + oy[el], pane["ex"] + ox[el],
           (0.45 + 0.55 * st["glow"][el]).astype(np.float32), dr, dc)

    # the cracks: jagged lines running out from where the hit landed
    live = []
    for crack in st["cracks"]:
        hy, hx, a, born, amp, jag = crack
        age = ctx.t - born
        if age > _SH_CRACK_S:
            continue
        live.append(crack)
        n = min(jag.size, int(dr * (0.3 + 0.5 * amp) * min(1.0, age / 0.08)))
        if n < 2:
            continue
        steps = np.arange(n, dtype=np.float32)
        wobble = jag[:n]
        ys = hy + steps * math.sin(a) + wobble * math.cos(a)
        xs = hx + steps * math.cos(a) * 2.0 - wobble * math.sin(a)
        _splat(field, ys, xs, np.float32(1.0 - 0.5 * age / _SH_CRACK_S), dr, dc)
    st["cracks"] = live
    return pack_braille(field > 0.1), ctx.ramp(cell_max(field))
