"""Dot-field modes: sparkle, fire, and polar geometry."""

from __future__ import annotations

import math

import numpy as np

from ..render import cell_max, frac, noise, pack_braille
from . import Ctx, band_columns, empty, mode, spread


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
        # reuses nz rather than a second noise() call, same as halo above does
        # against inside's — a different threshold on the same random field
        # reads as independently random, and a full-grid hash is expensive
        # enough (see render.frac's docstring for the same lesson re: np.mod)
        # that this mode was calling it up to twice a frame for no visible gain.
        band = 1.0 + strength * 2.0
        edge = np.abs(dist - max_r * phase)
        near = edge < band
        fade = 1.0 - edge / band
        lit |= near & (nz < fade * strength)

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
            "x": np.zeros(cap),               # centreline; wobble is added at render time
            "r": np.zeros(cap),
            "spd": np.zeros(cap),
            "pop": np.zeros(cap),             # seconds left in the burst
            "wfreq": np.zeros(cap),           # per-bubble wobble rate, constant once spawned
            "wph": np.zeros(cap),
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
            st["wfreq"][free] = rng.uniform(1.4, 3.2, k)
            st["wph"][free] = rng.uniform(0.0, 2 * math.pi, k)

    field = np.zeros((dr, dc), dtype=np.float64)
    live = np.flatnonzero(st["y"] >= 0.0)
    if live.size:
        ys = np.rint(st["y"][live]).astype(np.int32)
        radii = st["r"][live].astype(np.int32)
        # small bubbles get buffeted more than large ones on the way up —
        # a real bubble doesn't rise in a straight line, and a fixed wobble
        # amplitude for every size reads as the whole column swaying in
        # unison rather than each bubble jittering on its own. Frequency and
        # phase are drawn once at spawn and multiplied by ``ctx.t`` directly,
        # not accumulated — the rate is constant per bubble, not audio-driven,
        # so this is the safe half of the ``ctx.t * rate`` pattern (see
        # ``Tunnel``'s docstring for the unsafe half).
        wobble_amp = (dr * 0.018) / np.maximum(radii.astype(np.float64), 1.0)
        wobble = np.sin(ctx.t * st["wfreq"][live] + st["wph"][live]) * wobble_amp
        xs = np.rint(st["x"][live] + wobble).astype(np.int32)
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
    wedge = frac(turn * n)
    gutter = (wedge < 0.06) | (wedge > 0.94)

    lit = (dist >= inner) & (dist <= outer) & ~gutter
    ring = np.abs(dist - inner) < 0.9
    lit |= ring

    heat = np.where(lit, np.clip((dist - inner) / np.maximum(outer - inner, 1e-6), 0, 1), 0.0)
    codes = pack_braille(lit)
    cidx = ctx.ramp(cell_max(heat))
    return codes, cidx


#: Beam thickness, in braille dots, held constant at every radius.
_BEAM_DOTS = 1.7

#: How brightly the bare sweep line is written into the phosphor buffer.
#:
#: This is a tuned value, not a taste one. Dots are drawn while the buffer is
#: above 0.05 and it decays on a 0.9 s constant, so the intensity written here
#: decides how far the beam smears before it drops out: 0.17 survives 1.1 s,
#: which at 0.15 turns/sec is a 59-degree wedge, and because the sweep line
#: runs the full radius that reads as a solid filled sector rather than a
#: beam. 0.07 lasts ~0.3 s, or about 16 degrees — a crisp line with just
#: enough trail to show which way it is going. Contacts are written far
#: brighter and so persist ~146 degrees, which is what makes them read as
#: lingering returns against a moving beam.
_SWEEP_GLOW = 0.07


def _sonar_circ(ctx: Ctx, dist: np.ndarray) -> np.ndarray:
    """Arc length per turn of angular offset — cached, it only depends on radius."""
    return ctx.scratch(
        "sonar_circ",
        lambda: (np.float32(2.0 * math.pi) * np.maximum(dist, np.float32(1.0))).astype(
            np.float32
        ),
    )


@mode("Sonar", group="particles", blurb="one sweep, not the whole spectrum — returns fade like CRT phosphor")
def sonar(ctx: Ctx):
    """A ship's sonar scope, not another way to draw a spectrum.

    Radial already shows every band at once, wrapped into a static ring —
    the whole picture, all the time. This shows one bearing at a time: a
    beam sweeps continuously, and only the band it's currently crossing gets
    painted, as a return reaching out from the centre in proportion to that
    band's level. Everywhere else on the screen is memory, not signal — each
    return decays exponentially once the beam moves on, the way a real
    CRT's phosphor keeps glowing after the electron gun passes, so what's on
    screen at any instant is a fading fan of *recent* returns trailing the
    beam around rather than a live readout. Band-to-bearing assignment is
    fixed (``spin=0`` into ``_angular_bands``) like a real plot, where a
    contact stays at its bearing and only the sweep moves over it.

    Three things about the beam are load-bearing and each has a comment at
    the line that does it: it holds a constant *screen* width instead of a
    constant angle, it has a soft edge instead of a hard threshold, and it
    is drawn faintly even where there is no return. A contact is painted at
    its range rather than filled in from the centre, which is the difference
    between a sonar plot and a polar bar chart.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 8 or dc < 8:
        return empty(ctx.w, ctx.h)

    dist, turn, max_r = _polar(ctx)
    n = min(24, max(8, ctx.n_display))
    nrg = _angular_bands(ctx, turn, n, 0.0)

    buf = ctx.scratch("sonar_buf", lambda: np.zeros((dr, dc), dtype=np.float32))

    # SWEEP_TURNS_PER_SEC=0.15 is one full rotation every ~6.7s — slow enough
    # that a band's return is still clearly a fading trail behind the beam
    # rather than a blur, at the frame rates this runs at. Deliberately not
    # audio-driven: a real scope sweeps at a constant rate, and the returns
    # are what carry the music.
    sweep = frac(ctx.t * 0.15)
    # shortest signed angular distance from the beam, in turns: wrapping
    # turn-sweep through +0.5 before taking frac() folds the far side of the
    # wrap back onto [-0.5, 0.5) instead of leaving a 1.0-turn discontinuity
    # at the 0/1 seam that a plain subtraction would show as a false return.
    dtheta = np.abs(frac(turn - sweep + 0.5) - 0.5)

    # Angular distance converted to arc length in dots, so the beam keeps a
    # constant on-screen thickness. Thresholding the angle directly — which
    # this did originally — makes a wedge rather than a beam: a fixed 0.004
    # turns is 0.1 dots across near the centre, where it falls between dots
    # and disappears, against 20 at the rim of a large terminal.
    arc = dtheta * _sonar_circ(ctx, dist)

    decay = float(np.exp(-max(ctx.dt, 0.0) / 0.9))
    buf *= decay

    # Everything below runs only on the dots the beam actually covers. A
    # constant-width beam is a ~2-dot line reaching one radius across a grid
    # of up to 320k cells — around 0.2% of it — so evaluating the contact
    # maths densely spent essentially the whole frame on cells guaranteed to
    # paint nothing: measured 17.6 ms that way against ~2 ms here at 400x100,
    # which is the difference between missing the 60 fps budget and sitting
    # comfortably inside it.
    sel = (arc < _BEAM_DOTS) & (dist <= max_r)
    if sel.any():
        a = arc[sel]
        d = dist[sel]
        e = nrg[sel]

        # Soft edge rather than a hard cutoff, for the same reason
        # ``_animate_ramp`` rounds instead of flooring: a binary edge visibly
        # jitters between dot columns as it rotates, where a ramp lets
        # intensity carry the sub-dot position and the sweep reads as smooth.
        beam = 1.0 - a * np.float32(1.0 / _BEAM_DOTS)

        # A contact sits *at* its range rather than smearing back to the
        # centre. Filling from the origin out to the level, as this used to,
        # is a polar bar chart with a sweep over it; a return is an echo from
        # one distance, and keeping it that way is also what leaves the middle
        # of the scope open instead of permanently lit.
        blip = np.clip(
            1.0 - np.abs(d - e * np.float32(max_r)) * np.float32(1.0 / 2.6), 0.0, 1.0
        )
        # gate out silent bands, whose target radius would otherwise be ~0 and
        # park a permanent false contact on top of the origin
        live = np.clip((e - np.float32(0.05)) * np.float32(12.0), 0.0, 1.0)

        # the floor keeps the sweep line itself drawn even with no return
        # under it — without it the scope goes blank through a quiet passage
        # and reads as switched off rather than as showing no contacts
        vals = beam * np.maximum(
            np.float32(_SWEEP_GLOW), blip * live * (np.float32(0.45) + np.float32(0.55) * e)
        )
        buf[sel] = np.maximum(buf[sel], vals)

    dots = buf > 0.05
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(np.where(dots, buf, 0.0)))
    return codes, cidx


@mode("Orbit", group="particles", blurb="one dot per band, actually revolving")
def orbit(ctx: Ctx):
    """Bodies in continuous motion, not a shape sampled from angle.

    ``Radial`` maps the spectrum onto a static ring and ``Sonar`` sweeps one
    beam over it, but neither has anything that actually travels frame to
    frame. Here every band is a body with real angular velocity — faster for
    higher bands — carried in scratch and integrated by ``dt``, with a
    phosphor-style trail so the motion reads clearly rather than teleporting.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 10 or dc < 10:
        return empty(ctx.w, ctx.h)

    n = min(16, max(4, ctx.n_display))
    lv = ctx.display_bands(n)

    cx, cy = dc / 2.0, dr / 2.0
    x_scale = cy / max(cx, 1.0)          # braille cells are ~2x taller than wide
    max_r = max(1.0, cy - 1.0)

    st = ctx.scratch("orbit", lambda: {"angle": np.linspace(0.0, 2 * math.pi, n, endpoint=False)})
    if len(st["angle"]) != n:
        st["angle"] = np.linspace(0.0, 2 * math.pi, n, endpoint=False)

    # higher band index -> faster revolution; a loud band also speeds up
    # rather than just sitting farther out, so energy reads as motion too
    speed = 0.6 + (np.arange(n) / max(n - 1, 1)) * 2.2 + lv * 1.5
    st["angle"] = (st["angle"] + speed * ctx.dt) % (2 * math.pi)

    radius = max_r * (0.15 + 0.8 * lv)
    dx = radius * np.cos(st["angle"])
    dy = radius * np.sin(st["angle"])
    px = np.clip(np.rint(cx + dx / x_scale).astype(np.int32), 0, dc - 1)
    py = np.clip(np.rint(cy + dy).astype(np.int32), 0, dr - 1)

    buf = ctx.scratch("orbit_buf", lambda: np.zeros((dr, dc), dtype=np.float32))
    buf *= float(np.exp(-max(ctx.dt, 0.0) / 0.13))

    field = np.zeros((dr, dc), dtype=np.float32)
    dist, _turn, _mr = _polar(ctx)

    # A spoke from the centre out to each body. The first attempt drew each
    # body's full orbit as a ring instead, which looks right for one body and
    # fails for sixteen: the radii sweep with their bands, so the rings tile
    # the whole disc and the picture saturates into a solid blob. A spoke
    # anchors a body to the centre and shows its radius without claiming any
    # of the area between the orbits.
    steps = max(4, int(max_r))
    tt = np.linspace(0.12, 1.0, steps)[None, :]
    sx = cx + (dx[:, None] / x_scale) * tt
    sy = cy + dy[:, None] * tt
    spx = np.clip(np.rint(sx).astype(np.int32), 0, dc - 1)
    spy = np.clip(np.rint(sy).astype(np.int32), 0, dr - 1)
    spoke_v = (0.06 + 0.20 * lv)[:, None] * (1.0 - 0.45 * tt)
    np.maximum.at(field, (spy.ravel(), spx.ravel()), spoke_v.ravel().astype(np.float32))

    # a central mass, breathing on the low end
    bass = ctx.range(0.0, 0.2)
    core_r = max(1.0, max_r * (0.05 + 0.07 * bass))
    core = dist <= core_r
    np.maximum(field, core * np.float32(0.55 + 0.45 * bass), out=field)

    # Bodies grow with their band instead of being a fixed one-dot ring, but
    # only up to radius 2. Allowing 3 turns each body's trail from a thin arc
    # into a wide swept band, and sixteen of those fill the disc solid.
    for j in range(n):
        size = 1 + int(min(1, lv[j] * 1.6))
        dy_off, dx_off = _BUBBLE_RINGS[size]
        ys = py[j] + dy_off
        xs = px[j] + dx_off
        ok = (ys >= 0) & (ys < dr) & (xs >= 0) & (xs < dc)
        if ok.any():
            np.maximum.at(buf, (ys[ok], xs[ok]), np.float32(0.45 + 0.55 * lv[j]))

    # Spokes and core composite over the trail buffer rather than into it.
    # Writing them into ``buf`` leaves them there permanently — the buffer
    # only decays a few percent a frame and they are redrawn every frame, so
    # they never fade and the disc fills in solid. Only the bodies belong in
    # the buffer; that is what the buffer is for.
    out = np.maximum(buf, field)
    dots = out > 0.04
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(out))
    return codes, cidx


#: burst kinds, chosen per rocket at launch — each biases its shower's count,
#: speed, gravity and lifetime, so a hit-heavy stretch doesn't just replay the
#: same spherical pop over and over.
_FW_SHELL, _FW_WILLOW, _FW_CRACKLE = 0, 1, 2
_FW_KIND_PARAMS = {
    #        count       speed (x dr)   gravity (x dr)  life (s)
    _FW_SHELL:   ((34, 52), (0.28, 0.80), 1.6, (1.1, 1.5)),
    _FW_WILLOW:  ((26, 40), (0.14, 0.34), 0.9, (1.8, 2.4)),
    _FW_CRACKLE: ((48, 72), (0.35, 0.95), 1.9, (0.45, 0.75)),
}

#: A spark is drawn as a small cross rather than a single dot. One dot per
#: spark left the mode covering ~2% of the screen — the sparsest in the file
#: by a wide margin — so a burst barely registered against a dark terminal.
_FW_GLOW = ((0, 0, 1.0), (-1, 0, 0.5), (1, 0, 0.5), (0, -1, 0.5), (0, 1, 0.5))


@mode("Fireworks", group="particles", blurb="beat-triggered launches, bursts, and fall")
def fireworks(ctx: Ctx):
    """Discrete triggered events, not a continuous field.

    Every other mode in this file paints a field that reacts to the current
    level. This is the only one built around one-shot events with their own
    lifetime: a bass transient launches a rocket, the rocket climbs and
    bursts into a shower, and the shower falls under gravity and fades —
    three phases, not a shape redrawn from this frame's energy.

    ``ctx.energy`` is spring-smoothed (see widget.py) and would smear a hit
    across several frames if thresholded directly, so this rolls its own fast
    /slow envelope over the low end instead — a rising edge over the slow
    average is the trigger.

    Three burst kinds (``_FW_KIND_PARAMS``), chosen randomly per launch, give
    the shower real variety: a shell's even spherical pop, a willow's slow
    sparse droop with a long fade, a crackle's dense fast flicker that burns
    out quickly. Rockets ease off their climb speed near the burst height
    instead of popping at a constant velocity, which reads as a brief hang at
    the apex. Both rockets and sparks carry a one-dot motion trail — the same
    "streak the fastest point backward along its own velocity" trick ``Warp``
    uses for its stars, just applied to two more particle systems.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 12 or dc < 12:
        return empty(ctx.w, ctx.h)

    r_cap, s_cap = 16, 900

    def spawn_state():
        return {
            "ry": np.full(r_cap, -1.0), "rx": np.zeros(r_cap),
            "rvy": np.zeros(r_cap), "rtarget": np.zeros(r_cap),
            "rkind": np.zeros(r_cap, dtype=np.int32),
            "sy": np.full(s_cap, -1.0), "sx": np.zeros(s_cap),
            "svy": np.zeros(s_cap), "svx": np.zeros(s_cap), "sage": np.zeros(s_cap),
            "sgrav": np.zeros(s_cap), "slife": np.full(s_cap, 1.1),
            "skind": np.zeros(s_cap, dtype=np.int32),
            "fast": 0.0, "slow": 0.0, "launch_acc": 0.0,
            "rng": np.random.default_rng(41),
        }

    st = ctx.scratch("fireworks", spawn_state)
    rng = st["rng"]

    bass = ctx.range(0.0, 0.2)
    st["fast"] += (bass - st["fast"]) * min(1.0, ctx.dt / 0.03)
    st["slow"] += (bass - st["slow"]) * min(1.0, ctx.dt / 0.5)
    hit = (st["fast"] - st["slow"]) > 0.1

    # Burst kind follows the spectrum's centre of mass instead of a coin
    # flip: a bass-heavy passage throws slow drooping willows, a bright one
    # throws fast crackles. Picking at random meant the audio decided *when*
    # a shell went up but never *what* it was.
    b8 = ctx.display_bands(8).astype(np.float64)
    tot = float(b8.sum())
    centroid = float((b8 * np.arange(8)).sum() / tot / 7.0) if tot > 1e-9 else 0.0

    def pick_kind() -> int:
        r = float(rng.random()) * 0.5 + centroid * 0.75
        if r < 0.42:
            return _FW_WILLOW
        return _FW_SHELL if r < 0.82 else _FW_CRACKLE

    # A barrage, not one shell per onset. Rockets also launch on sustained
    # loudness rather than only on a rising bass edge, so a dense passage
    # keeps the sky busy instead of going dark between transients.
    st["launch_acc"] += (0.35 + ctx.energy * 7.0) * ctx.dt
    want = int(st["launch_acc"])
    if want:
        st["launch_acc"] -= want
    if hit:
        want += 1 + int(min(2, st["fast"] * 4.0))

    # stage 1: rockets climb toward a randomly chosen burst height, easing
    # off their speed over the final stretch — a constant-velocity climb that
    # just stops and pops read as mechanical; slowing into the burst reads as
    # a rocket fighting gravity, cresting, and letting go.
    ralive = st["ry"] >= 0.0
    dist = st["ry"][ralive] - st["rtarget"][ralive]
    ease = np.clip(dist / (dr * 0.3), 0.3, 1.0)
    st["ry"][ralive] -= st["rvy"][ralive] * ease * ctx.dt
    burst = ralive & (st["ry"] <= st["rtarget"])

    if want:
        free = np.flatnonzero(~ralive)[:want]
        for i in free:
            st["ry"][i] = dr - 1.0
            st["rx"][i] = rng.uniform(dc * 0.10, dc * 0.90)
            st["rvy"][i] = dr * rng.uniform(1.3, 1.8)
            st["rtarget"][i] = rng.uniform(dr * 0.12, dr * 0.55)
            st["rkind"][i] = pick_kind()

    # stage 2: a bursting rocket seeds a shower of sparks from its position,
    # shaped by whichever kind it was launched as
    for i in np.flatnonzero(burst):
        kind = int(st["rkind"][i])
        (k_lo, k_hi), (spd_lo, spd_hi), grav_mul, (life_lo, life_hi) = _FW_KIND_PARAMS[kind]
        free = np.flatnonzero(st["sy"] < 0.0)
        k = int(min(free.size, rng.integers(k_lo, k_hi)))
        if k:
            slots = free[:k]
            ang = rng.uniform(0.0, 2 * math.pi, k)
            spd = rng.uniform(dr * spd_lo, dr * spd_hi, k)
            st["sy"][slots] = st["ry"][i]
            st["sx"][slots] = st["rx"][i]
            st["svy"][slots] = -np.sin(ang) * spd * 0.5
            st["svx"][slots] = np.cos(ang) * spd
            st["sage"][slots] = 0.0
            st["sgrav"][slots] = dr * grav_mul
            st["slife"][slots] = rng.uniform(life_lo, life_hi, k)
            st["skind"][slots] = kind
        st["ry"][i] = -1.0

    # stage 3: sparks fall under their own gravity and fade over their own life
    salive = st["sy"] >= 0.0
    st["svy"][salive] += st["sgrav"][salive] * ctx.dt
    st["sy"][salive] += st["svy"][salive] * ctx.dt
    st["sx"][salive] += st["svx"][salive] * ctx.dt
    st["sage"][salive] += ctx.dt
    dead = salive & ((st["sage"] > st["slife"]) | (st["sy"] < -2) | (st["sy"] > dr + 2))
    st["sy"][dead] = -1.0

    field = np.zeros((dr, dc), dtype=np.float64)

    rl = np.flatnonzero(st["ry"] >= 0.0)
    if rl.size:
        py = np.clip(np.rint(st["ry"][rl]).astype(np.int32), 0, dr - 1)
        px = np.clip(np.rint(st["rx"][rl]).astype(np.int32), 0, dc - 1)
        field[py, px] = 1.0
        # a real exhaust trail, not one dot: the rocket is the only thing on
        # screen between bursts and a single lit cell reads as a stuck pixel
        for k in range(1, 5):
            ty = np.clip(py + k, 0, dr - 1)
            np.maximum.at(field, (ty, px), np.float64(0.55) / k)

    sl = np.flatnonzero(st["sy"] >= 0.0)
    if sl.size:
        fy = st["sy"][sl]
        fx = st["sx"][sl]
        life = np.maximum(st["slife"][sl], 1e-3)
        bright = np.clip(1.0 - st["sage"][sl] / life, 0.0, 1.0) ** 1.3

        crackle = st["skind"][sl] == _FW_CRACKLE
        if crackle.any():
            flick = rng.random(sl.size)
            bright = np.where(crackle, bright * np.where(flick < 0.65, 1.0, 0.2), bright)

        for oy, ox, w_off in _FW_GLOW:
            py = np.rint(fy).astype(np.int32) + oy
            px = np.rint(fx).astype(np.int32) + ox
            ok = (py >= 0) & (py < dr) & (px >= 0) & (px < dc)
            if ok.any():
                np.maximum.at(field, (py[ok], px[ok]), bright[ok] * w_off)

        speed = np.hypot(st["svx"][sl], st["svy"][sl])
        moving = speed > dr * 0.10
        if moving.any():
            inv = 1.0 / np.maximum(speed, 1e-6)
            for k in (1.0, 2.0, 3.0):
                tx = np.rint(fx - st["svx"][sl] * inv * k).astype(np.int32)
                ty = np.rint(fy - st["svy"][sl] * inv * k).astype(np.int32)
                ok = moving & (ty >= 0) & (ty < dr) & (tx >= 0) & (tx < dc)
                if ok.any():
                    np.maximum.at(field, (ty[ok], tx[ok]), bright[ok] * (0.55 / k))

    dots = field > 0.04
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(field))
    return codes, cidx


#: Gray-Scott runs on its own fixed lattice and is upsampled to whatever the
#: terminal is, the same split ``Maelstrom`` uses for its fluid grid. The
#: reaction's feature size is set by the diffusion constants, not by the
#: window, so letting the lattice track the terminal would change the size of
#: the spots every time the window was resized.
_GS_H, _GS_W = 88, 150
_GS_DU, _GS_DV = 0.16, 0.08


def _colony_spawn() -> dict:
    rng = np.random.default_rng(101)
    u = np.ones((_GS_H, _GS_W), dtype=np.float32)
    v = np.zeros((_GS_H, _GS_W), dtype=np.float32)
    # Seed blobs; a uniform field is a fixed point of the reaction and would
    # never develop any pattern at all.
    for _ in range(40):
        cy = int(rng.integers(4, _GS_H - 4))
        cx = int(rng.integers(4, _GS_W - 4))
        u[cy - 3 : cy + 3, cx - 3 : cx + 3] = 0.50
        v[cy - 2 : cy + 2, cx - 2 : cx + 2] = 0.25

    # Burn the reaction in before the first frame is drawn. Gray-Scott needs
    # on the order of a hundred iterations to grow structure out of seeds,
    # and at the mode's own step rate that is the better part of ten seconds
    # of watching a nearly empty screen after switching to it. Measured at
    # ~22ms for this many steps — a couple of dropped frames, once, on open
    # or resize, against a mode that is otherwise uninteresting to look at
    # for its first ten seconds.
    for _ in range(150):
        _colony_step(u, v, 0.024, 0.056)

    return {"u": u, "v": v, "acc": 0.0, "fast": 0.0, "slow": 0.0, "hit_t": -99.0, "rng": rng}


def _lap(a: np.ndarray) -> np.ndarray:
    """5-point Laplacian, toroidal."""
    return (
        np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4.0 * a
    )


def _colony_step(u: np.ndarray, v: np.ndarray, feed: float, kill: float) -> None:
    """One Gray-Scott iteration, in place."""
    uvv = u * v * v
    u += _GS_DU * _lap(u) - uvv + feed * (1.0 - u)
    v += _GS_DV * _lap(v) + uvv - (feed + kill) * v
    np.clip(u, 0.0, 1.0, out=u)
    np.clip(v, 0.0, 1.0, out=v)


@mode("Colony", group="particles", blurb="a growing culture — spots, worms and mazes, set by the music")
def colony(ctx: Ctx):
    """A Gray-Scott reaction-diffusion culture, not a cellular automaton.

    This was Conway's Life first, and Life has a problem as a visualiser: at
    any density that fills the screen its population is essentially noise, so
    however faithfully it reacts, what you *see* is static. Gray-Scott is the
    other classic grid system and it fails in the opposite, useful direction
    — two chemicals, one feeding on the other, produce coherent structures
    (drifting spots, branching worms, labyrinths) whose *character* is set by
    just two numbers, the feed and kill rates.

    Those two numbers are what the music drives, which is the whole point of
    choosing this system: the spectrum's centre of mass sets ``kill`` and its
    energy sets ``feed``, so a bass-heavy passage grows fat dividing blobs
    and a bright one etches fine mazes. That is a change in the *kind* of
    pattern on screen, not just its brightness or speed.

    Both rates are clamped to a band that is known to produce structure.
    Outside it the reaction has a uniform fixed point it falls into and never
    leaves — the screen goes blank and stays blank, which as a failure mode
    looks exactly like a crash. The lattice is also seeded with blobs rather
    than left uniform, for the same reason: perfectly uniform is that fixed
    point, so an unseeded grid would never develop anything.

    Iteration count is ``ctx.dt``-accumulated and capped, so the culture
    evolves at a rate in seconds rather than in frames, and a stalled frame
    cannot dump a burst of catch-up steps into it.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 16 or dc < 16:
        return empty(ctx.w, ctx.h)

    st = ctx.scratch("colony", _colony_spawn)
    rng = st["rng"]
    u, v = st["u"], st["v"]

    bass = ctx.range(0.0, 0.2)
    st["fast"] += (bass - st["fast"]) * min(1.0, ctx.dt / 0.03)
    st["slow"] += (bass - st["slow"]) * min(1.0, ctx.dt / 0.45)
    hit = (st["fast"] - st["slow"]) > 0.12

    b8 = ctx.display_bands(8).astype(np.float64)
    tot = float(b8.sum())
    centroid = float((b8 * np.arange(8)).sum() / tot / 7.0) if tot > 1e-9 else 0.3

    sm = ctx.scratch("colony_sm", lambda: {"c": 0.3, "e": 0.2})
    sm["c"] += (centroid - sm["c"]) * min(1.0, ctx.dt / 0.25)
    sm["e"] += (ctx.energy - sm["e"]) * min(1.0, ctx.dt / 0.25)

    # Feed/kill stay inside a band measured to survive, not one taken from
    # the textbook. Swept over this implementation, the living region is
    # feed 0.016-0.032 against kill 0.048-0.061, and it is *not* rectangular:
    # a high feed with a low kill dies, as does anything past kill 0.061. The
    # first cut mapped treble straight onto kill up to 0.066 and a bright
    # passage extinguished the culture — permanently, because an all-zero
    # lattice is an absorbing state with nothing left to react. Both corners
    # of this rectangle are verified alive.
    feed = float(np.clip(0.019 + sm["e"] * 0.011, 0.019, 0.030))
    kill = float(np.clip(0.055 + sm["c"] * 0.003, 0.055, 0.058))

    # a hit inoculates the culture with a fresh colony
    if hit and (ctx.t - st["hit_t"]) > 0.25:
        st["hit_t"] = ctx.t
        cy = int(rng.integers(5, _GS_H - 5))
        cx = int(rng.integers(5, _GS_W - 5))
        u[cy - 4 : cy + 4, cx - 4 : cx + 4] = 0.50
        v[cy - 3 : cy + 3, cx - 3 : cx + 3] = 0.28

    # Watchdog. The clamp above is derived from a sweep, but a sweep only
    # proves the corners it sampled, and an extinguished lattice cannot
    # recover on its own — so guarantee it structurally rather than trusting
    # the parameter range to be exhaustive.
    if float(v.max()) < 0.02:
        for _ in range(10):
            cy = int(rng.integers(4, _GS_H - 4))
            cx = int(rng.integers(4, _GS_W - 4))
            u[cy - 3 : cy + 3, cx - 3 : cx + 3] = 0.50
            v[cy - 2 : cy + 2, cx - 2 : cx + 2] = 0.25

    st["acc"] += (7.0 + sm["e"] * 26.0) * ctx.dt
    steps = min(int(st["acc"]), 8)
    if steps:
        st["acc"] -= steps
        for _ in range(steps):
            _colony_step(u, v, feed, kill)

    # nearest-neighbour upsample from the fixed lattice to the dot grid
    idx = ctx.scratch(
        "colony_idx",
        lambda: (
            (np.arange(dr) * _GS_H // max(dr, 1)).clip(0, _GS_H - 1),
            (np.arange(dc) * _GS_W // max(dc, 1)).clip(0, _GS_W - 1),
        ),
    )
    heat = v[idx[0][:, None], idx[1][None, :]].astype(np.float64) * 3.4
    np.clip(heat, 0.0, 1.0, out=heat)

    # Threshold well above the noise floor of the reaction: the V field has a
    # broad low-level halo around every structure, and lighting that too
    # takes a loud passage to ~97% coverage — a solid screen, where the
    # pattern that is the entire point of the mode stops being visible.
    dots = heat > 0.40
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(np.where(dots, heat, 0.0)))
    return codes, cidx


@mode("Dune", group="particles", blurb="sand piles up by band, avalanching past a threshold")
def dune(ctx: Ctx):
    """Height that accumulates, not a level that's redrawn.

    Every other mode in this group is stateless with respect to loudness —
    read the current level, draw it. This is the only one where the picture
    is a running sum: sand rains in proportional to each band, a column's
    pile only grows, and it takes a genuine collapse — height crossing a
    threshold — to bring it back down, spilling into its neighbours the way
    a real sandpile does. A loud passage can still trigger a chain of
    collapses over the following frames even after the level drops, because
    the state that decides that is height, not the current spectrum.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 12 or dc < 16:
        return empty(ctx.w, ctx.h)

    n = int(np.clip(ctx.n_display, 6, 40))
    lv = ctx.display_bands(n)

    def spawn_state():
        return {"h": np.zeros(n, dtype=np.float64), "rng": np.random.default_rng(53)}

    st = ctx.scratch("dune", spawn_state)
    if len(st["h"]) != n:
        st["h"] = np.zeros(n, dtype=np.float64)
    h = st["h"]
    rng = st["rng"]

    h += lv * lv * 0.55 * ctx.dt
    np.clip(h, 0.0, 1.3, out=h)

    spill = h > 1.0
    if spill.any():
        idx = np.flatnonzero(spill)
        excess = h[idx] - 0.55
        h[idx] = 0.55 + rng.uniform(-0.03, 0.03, idx.size)
        left = np.clip(idx - 1, 0, n - 1)
        right = np.clip(idx + 1, 0, n - 1)
        np.add.at(h, left, excess * 0.35)
        np.add.at(h, right, excess * 0.35)
        np.clip(h, 0.0, 1.3, out=h)

    col_band, active = band_columns(dc, n)
    level = np.where(active, h[col_band], 0.0)
    tops = dr - 1 - np.rint(np.clip(level, 0.0, 1.0) * (dr - 1)).astype(np.int32)

    rows = np.arange(dr)[:, None]
    body = active[None, :] & (rows >= tops[None, :])

    # grit texture: threshold noise against depth-into-the-pile rather than a
    # solid fill, so it reads as packed grains and not a filled bar chart.
    # Seeded with a constant, not ``ctx.frame`` — packed sand doesn't
    # rearrange its own grains every 1/60th of a second. With ``ctx.frame``
    # this was measured shimmering ~40 cells a frame even under near-silent
    # audio with the pile height barely moving; a fixed seed means only
    # ``tops`` crossing a grain boundary changes what's lit.
    depth = (rows - tops[None, :]).astype(np.float32)
    grain = noise((dr, dc), 0) < np.clip(0.35 + depth * 0.05, 0.35, 0.97)
    dots = body & grain

    heat = np.clip(np.where(dots, 0.35 + 0.65 * (1.0 - depth / max(1, dr)), 0.0), 0.0, 1.0)
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(heat))
    return codes, cidx


_MURM_CAP = 240


def _murmuration_state(dr: int, dc: int) -> dict:
    rng = np.random.default_rng(211)
    n = int(np.clip(dc * 0.35, 90, _MURM_CAP))
    cx, cy = dc * 0.5, dr * 0.5
    r0 = min(dr, dc) * 0.2
    ang = rng.uniform(0.0, 2 * math.pi, n)
    rad = rng.uniform(0.3, 1.0, n) * r0
    speed0 = min(dr, dc) * 0.22
    vang = rng.uniform(0.0, 2 * math.pi, n)
    return {
        "n": n,
        "x": cx + np.cos(ang) * rad, "y": cy + np.sin(ang) * rad,
        "vx": np.cos(vang) * speed0, "vy": np.sin(vang) * speed0,
        "wind_dir": float(rng.uniform(0.0, 2 * math.pi)),
        "fast": 0.0, "slow": 0.0, "scatter_t": -99.0,
        "rng": rng,
    }


@mode("Murmuration", group="particles", blurb="a flock wheeling and scattering with the beat")
def murmuration(ctx: Ctx):
    """Real boid flocking — separation, alignment, cohesion — not a projectile system.

    Every other particle mode here tracks agents that ignore each other
    (sparks, bubbles, sand grains) or a field with no individual agents at
    all. This is the only one where each agent's motion depends on where its
    *neighbours* are: steer away from whoever's too close, match the local
    average heading, drift toward the local centre of mass. That needs an
    every-agent-to-every-agent distance check, which sounds like it should be
    the expensive part of this mode — it isn't. At the ~200-agent cap that's
    a ~200x200 pairwise distance matrix, on the order of 40,000 floats,
    computed with plain numpy broadcasting and one matmul per rule for the
    neighbour-average reductions (``mask @ vx`` sums velocity over exactly
    the masked neighbours per agent in one BLAS call). That's a rounding
    error next to the 320,000-cell dot grid every mode in this file already
    processes every frame — measured well under a millisecond of the total.

    Distances use the same x_scale aspect correction ``_polar`` computes for
    the whole dot grid (dot cells aren't square), just applied locally here
    instead of over a cached full-grid array, so the perception radius reads
    as circular and not squashed sideways.

    Audio drives real dynamics, not just brightness — this isn't one of the
    lofi modes: bass adds a wind force in a slowly rotating direction
    (``ctx.t * constant``, the safe half of the pattern ``Tunnel``'s
    docstring warns about), treble adds per-agent jitter (genuinely random
    every frame on purpose — this is stochastic forcing on the physics, not
    a texture that should hold still like ``Dune``'s grain), and a bass onset
    gives the whole flock an outward kick from its own centroid. Nothing
    scripts the flock back together afterward — cohesion is already pulling
    every agent toward its neighbours every frame, so it reforms because
    that's what the same three rules that hold it together always do.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 20 or dc < 30:
        return empty(ctx.w, ctx.h)

    st = ctx.scratch("murmuration", lambda: _murmuration_state(dr, dc))
    n = st["n"]
    rng = st["rng"]
    x, y, vx, vy = st["x"], st["y"], st["vx"], st["vy"]

    bass = ctx.range(0.0, 0.2)
    treble = ctx.range(0.6, 1.0)
    st["fast"] += (bass - st["fast"]) * min(1.0, ctx.dt / 0.03)
    st["slow"] += (bass - st["slow"]) * min(1.0, ctx.dt / 0.4)
    hit = (st["fast"] - st["slow"]) > 0.14 and (ctx.t - st["scatter_t"]) > 0.6

    # The domain is bounded, not periodic. Wrapping positions is the obvious
    # thing to do and it was wrong twice over: the pairwise deltas did not
    # wrap with them, so a pair either side of the seam measured a full width
    # apart and cohesion tore apart any flock that touched an edge — and once
    # that was fixed the flock would sit *across* the seam, drawn as two
    # halves glued to opposite edges, because nothing on a terminal tells the
    # viewer the screen is a torus. Soft walls keep it one readable mass.
    x_scale = dr / dc
    raw_dx = x[:, None] - x[None, :]
    raw_dy = y[:, None] - y[None, :]
    ddx = raw_dx * x_scale
    dist2 = ddx * ddx + raw_dy * raw_dy

    # Perception has to be wide enough that a bird actually has neighbours;
    # too tight and cohesion never binds, so the "flock" is just a scatter of
    # independent particles obeying rules against an empty neighbourhood.
    r_perc = min(dr, dc) * 0.22
    r_sep = r_perc * 0.3
    neigh = (dist2 < r_perc * r_perc) & (dist2 > 1e-9)
    sep_m = neigh & (dist2 < r_sep * r_sep)

    nf = neigh.astype(np.float64)
    cnt = nf.sum(axis=1)
    cnt_safe = np.maximum(cnt, 1.0)

    avg_vx = (nf @ vx) / cnt_safe
    avg_vy = (nf @ vy) / cnt_safe
    # Cohesion from *relative* offsets, not an absolute centre of mass: on a
    # wrapped grid the mean of raw coordinates is not a position the flock is
    # anywhere near. ``raw_dx[i, j]`` is already the wrapped i->j delta, so
    # the pull toward the neighbourhood is just its negated mean.
    coh_x = -(nf * raw_dx).sum(axis=1) / cnt_safe
    coh_y = -(nf * raw_dy).sum(axis=1) / cnt_safe

    sf = sep_m.astype(np.float64)
    inv_d = sf / np.sqrt(np.maximum(dist2, 1e-9))
    sep_cnt = np.maximum(sf.sum(axis=1), 1.0)
    sep_ax = (inv_d * raw_dx).sum(axis=1) / sep_cnt
    sep_ay = (inv_d * raw_dy).sum(axis=1) / sep_cnt

    # The two rules that fight each other are what the music drives, so the
    # flock's *shape* tracks the spectrum: bass tightens cohesion into a
    # dense ball, treble drives separation and blows it out into a scattered
    # cloud. Previously the audio only pushed the flock around as a whole
    # (wind) and jittered it, which moved the birds without ever changing
    # what the flock looked like.
    # Both steering terms are normalised to roughly unit vectors before their
    # gains are applied, because they are not natively on the same scale:
    # ``sep_*`` is a sum of unit offsets (magnitude ~1) while ``coh_*`` is an
    # average *displacement* and runs to the perception radius, ~17 dots
    # here. Applying comparable-looking gains to those directly makes
    # cohesion about seventeen times the stronger rule, and the flock
    # collapses to a single point and stays there.
    scale = float(min(dr, dc))
    inv_perc = 1.0 / max(r_perc, 1e-6)
    coh_gain = (0.30 + bass * 0.85) * scale
    sep_gain = (1.20 + treble * 1.60) * scale
    ax = sep_ax * sep_gain + (avg_vx - vx) * 2.0 + coh_x * inv_perc * coh_gain
    ay = sep_ay * sep_gain + (avg_vy - vy) * 2.0 + coh_y * inv_perc * coh_gain

    # The wind veers a full turn every ~20s. At the original 0.05 rad/s it
    # took two minutes to come round, which is long enough that it acts as a
    # steady push: the flock rides it into a wall and sits there balanced
    # against the boundary force for most of a track.
    wind_ang = st["wind_dir"] + ctx.t * 0.32
    wind_mag = bass * scale * 0.55
    ax += math.cos(wind_ang) * wind_mag
    ay += math.sin(wind_ang) * wind_mag

    # a weak pull toward mid-screen so the flock stays where it can be seen
    ax += ((dc - 1) * 0.5 - x) * (scale * 0.010) / max(dc, 1)
    ay += ((dr - 1) * 0.5 - y) * (scale * 0.010) / max(dr, 1)

    jitter_mag = treble * min(dr, dc) * 0.7
    ax += rng.normal(0.0, 1.0, n) * jitter_mag
    ay += rng.normal(0.0, 1.0, n) * jitter_mag

    # soft walls: a smooth inward push that starts a margin in from each edge
    margin = min(dr, dc) * 0.16
    wall = scale * 2.0
    ax += np.clip((margin - x) / margin, 0.0, 1.0) * wall
    ax -= np.clip((x - (dc - 1 - margin)) / margin, 0.0, 1.0) * wall
    ay += np.clip((margin - y) / margin, 0.0, 1.0) * wall
    ay -= np.clip((y - (dr - 1 - margin)) / margin, 0.0, 1.0) * wall

    vx = vx + ax * ctx.dt
    vy = vy + ay * ctx.dt

    if hit:
        st["scatter_t"] = ctx.t
        ox, oy = x - x.mean(), y - y.mean()
        od = np.hypot(ox, oy) + 1e-6
        kick = min(dr, dc) * 1.5
        vx = vx + (ox / od) * kick
        vy = vy + (oy / od) * kick

    speed = np.hypot(vx, vy)
    max_speed = min(dr, dc) * (0.55 + bass * 0.4)
    min_speed = min(dr, dc) * 0.12
    safe_speed = np.maximum(speed, 1e-6)
    too_fast = speed > max_speed
    vx = np.where(too_fast, vx / safe_speed * max_speed, vx)
    vy = np.where(too_fast, vy / safe_speed * max_speed, vy)
    too_slow = speed < min_speed
    vx = np.where(too_slow, vx / safe_speed * min_speed, vx)
    vy = np.where(too_slow, vy / safe_speed * min_speed, vy)

    x = np.clip(x + vx * ctx.dt, 0.0, dc - 1.0)
    y = np.clip(y + vy * ctx.dt, 0.0, dr - 1.0)
    st["x"], st["y"], st["vx"], st["vy"] = x, y, vx, vy

    field = np.zeros((dr, dc), dtype=np.float64)

    px = np.clip(np.rint(x).astype(np.int32), 0, dc - 1)
    py = np.clip(np.rint(y).astype(np.int32), 0, dr - 1)
    density = np.clip(cnt / 8.0, 0.0, 1.0)
    bright = 0.45 + 0.45 * density

    # A bird is a small cross whose arms grow where the flock is dense, and
    # every bird drags a three-step trail. As single dots with one trailing
    # dot the whole flock covered ~6% of the screen — the sparsest mode in
    # the file after Fireworks — so the emergent shape, which is the entire
    # reason for simulating flocking at all, was invisible at a glance.
    np.maximum.at(field, (py, px), bright)
    for oy, ox in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny = np.clip(py + oy, 0, dr - 1)
        nx = np.clip(px + ox, 0, dc - 1)
        np.maximum.at(field, (ny, nx), bright * (0.30 + 0.45 * density))

    fast = speed > (min(dr, dc) * 0.05)
    if fast.any():
        inv_s = 1.0 / np.maximum(speed, 1e-6)
        for k, w_off in ((1.4, 0.55), (2.8, 0.35), (4.2, 0.20)):
            tx = np.rint(x - vx * inv_s * k).astype(np.int32)
            ty = np.rint(y - vy * inv_s * k).astype(np.int32)
            ok = fast & (ty >= 0) & (ty < dr) & (tx >= 0) & (tx < dc)
            if ok.any():
                np.maximum.at(field, (ty[ok], tx[ok]), bright[ok] * w_off)

    np.clip(field, 0.0, 1.0, out=field)
    dots = field > 0.03
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(field))
    return codes, cidx
