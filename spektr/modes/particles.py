"""Dot-field modes: sparkle, fire, and polar geometry."""

from __future__ import annotations

import math

import numpy as np

from ..render import cell_hilo, cell_max, frac, noise, pack_braille, pack_octant, pack_octant_bits
from . import (
    Ctx,
    angular_bands as _angular_bands,
    angular_lut as _angular_lut,
    band_columns,
    empty,
    mode,
    polar_grid as _polar,
    spread,
)

_INV24 = np.float32(1.0 / 0x1000000)


def _noise_at(rows: np.ndarray, cols: np.ndarray, seed: int) -> np.ndarray:
    """``render.noise`` evaluated at explicit (row, col) positions.

    The hash is positional — ``row * 6271 + col * 3037`` before the mixing
    chain — so a field sampled at a few thousand scattered cells is exactly
    the same field :func:`spektr.render.noise` would produce over the whole
    grid at those positions, at a fraction of the memory traffic. This is
    Flame's flicker: it only touches a thin band of cells, and the full-grid
    hash was over 2 ms of the mode's budget at 400x100.
    """
    with np.errstate(over="ignore"):
        h = rows * np.uint32(6271) + cols * np.uint32(3037)
        h = h + np.uint32((int(seed) * 104729) & 0xFFFFFFFF)
        h ^= h >> np.uint32(16)
        h *= np.uint32(0x45D9F3B)
        h ^= h >> np.uint32(16)
    return (h & np.uint32(0xFFFFFF)).astype(np.float32) * _INV24


@mode("Scatter", group="particles", blurb="density sparkle, thicker where it's loud")
def scatter(ctx: Ctx):
    dr, dc = ctx.dot_rows, ctx.dot_cols
    level = spread(ctx.display_bands(), dc)[None, :]
    fade = (0.5 + 0.5 * np.arange(dr) / max(1, dr - 1))[:, None]
    density = level * level * fade
    dots = noise((dr, dc), ctx.frame) < density
    codes = pack_braille(dots)
    # Coloured from the density field itself, not from ``density * dots``.
    # Zeroing the unlit dots made the dither decide the colour as well as the
    # coverage, so two neighbouring cells with a different number of lit dots
    # got a different colour and the strip builder paid for a run boundary
    # between them -- on a sparkle field, that is most pairs of cells. Density
    # is smooth across the width, so colouring from it directly costs a third
    # of the strips and looks the same: a blank cell has no visible foreground
    # anyway, and a lit one takes the colour of the sparkle it contains.
    return codes, ctx.ramp(cell_max(density))


@mode("Flame", group="particles", blurb="fire, licking upward from each band")
def flame(ctx: Ctx):
    dr, dc = ctx.dot_rows, ctx.dot_cols
    n = ctx.n_display
    band_of = np.minimum((np.arange(dc) * n) // dc, n - 1)
    lvl = ctx.display_bands(n)[band_of][None, :].astype(np.float32)

    # The flame is pinned to the top of the grid — ``alive`` is ``y <= lvl``
    # and every band is quieter than the loudest one, so no row above the
    # loudest band's top row can light. Everything below that row is dead
    # every frame; the whole per-frame pass over those rows (the wobble, the
    # width, the edge, the dither) spends its time writing zeros. Bound the
    # grid to the top rows the flame can actually occupy; the values there
    # are identical to the full-grid ones because every row-level input
    # (``y``) is the same function of the absolute row index.
    maxlvl = float(lvl.max())
    top = int(math.ceil((dr - 1) * (1.0 - maxlvl)))
    y = ((dr - 1 - np.arange(top, dr)) / max(1, dr - 1)).astype(np.float32)[:, None]
    alive = y <= lvl

    seg = max(1.0, dc / n)
    centre = (band_of * seg + seg / 2.0).astype(np.float32)[None, :]

    # sin(a + b) expanded so the trig runs over a column vector and a row
    # vector rather than the full dot grid — same wobble, a fraction of the
    # cost, and this is the hottest line in the mode. float32 throughout: at
    # 400x100 a dot grid is 320k cells and every float64 pass moves twice the
    # memory of a float32 one.
    a = np.float32(ctx.t * 9.0) + y * np.float32(6.0)
    b = (band_of * np.float32(2.1))[None, :]
    wobble = (np.sin(a) * np.cos(b) + np.cos(a) * np.sin(b)) * np.float32(1.5)
    # hot at the base, cooling toward the tip — the opposite of the bar modes.
    # On alive cells this lands in [0, 1], so one pass serves both as the
    # flame's width and as its heat.
    tip = np.float32(1.0) - y / np.maximum(lvl, np.float32(0.01))
    fw = (np.float32(0.3) + np.float32(0.7) * tip) * np.float32(seg / 2.0)

    # ``dist < fw`` is exactly ``edge < 1.0`` wherever fw is positive, which is
    # every alive cell, so the separate distance pass folds into this division.
    # fw is ``>= 0.3 * seg / 2 > 0`` on every alive cell, but a dead cell below
    # a quiet band drives ``tip`` negative and fw through zero — ``fw == 0``
    # lands only on cells whose ``edge`` is never consulted, so the ``where=``
    # guard skips the divide there (divide-by-zero warnings, and a hard
    # FloatingPointError under ``np.seterr``) without touching the visible
    # cells, and the old full-grid ``maximum(fw, 1e-6)`` pass stays gone.
    edge = np.abs(np.arange(dc, dtype=np.float32)[None, :] - centre + np.float32(0.5) - wobble)
    np.divide(edge, fw, out=edge, where=fw != 0)

    # Re-seeded every frame, and this is the one mode in the file where that
    # is right. Pulse, Auroras and Murmuration all moved to fixed grain
    # because their dither is a *texture* on something whose own motion
    # supplies the animation. A flame's edge dither is not texture, it is the
    # flicker — it is what the mode is a picture of. Swapping it for a fixed
    # field was measured at 18.3% of cells changing per frame under frozen
    # audio against 5.9%, a three-fold drop in exactly the property that makes
    # it read as fire, to save about 2 ms in a mode that fits its budget.
    #
    # The dither only needs the hash where it is consulted — the band of
    # cells with ``0.7 <= edge < 1.0``, a thin arc a few percent of the grid
    # at 400x100 — so the full-grid noise() call (several uint32 passes over
    # 320k cells, ~2.4 ms here) becomes a hash over just those cells via
    # :func:`_noise_at`. ``noise < 0.6`` is only ever evaluated there, so the
    # result is bit-identical: solid cells stay lit, cells past the flame's
    # edge stay dark, and the flicker band gets exactly the same field.
    solid = alive & (edge < 0.7)
    flick = alive & (edge >= 0.7) & (edge < 1.0)
    dots = np.zeros((dr, dc), dtype=bool)
    dots[top:] = solid
    fy, fx = np.nonzero(flick)
    if fy.size:
        dots[top + fy, fx] |= _noise_at((fy + top).astype(np.uint32), fx.astype(np.uint32), ctx.frame + 31) < np.float32(0.6)

    codes = pack_braille(dots)
    heat = np.zeros((dr, dc), dtype=np.float32)
    heat[top:] = tip * dots[top:]
    cidx = ctx.ramp(cell_max(heat))
    return codes, cidx


#: Concurrent shockwave slots. Two is enough to overlap a fast pair of kicks
#: without the screen turning into a ripple tank, and each costs a handful of
#: full-grid passes.
_PULSE_WAVES = 2


@mode("Pulse", group="particles", blurb="a radial blob with shockwaves thrown off the beat")
def pulse(ctx: Ctx):
    """A dithered radial blob whose edge is the spectrum, ringing on kicks.

    ``Radial`` maps the spectrum onto a static circle and ``Arcs`` lights a
    thin annulus per band. This is a solid mass whose *boundary* is the
    spectrum — the radius at each angle is that angle's band, squared, so
    loud bands bulge and quiet ones pull the outline in — and the fill is
    dithered so the inside reads as energy rather than as a filled disc.

    The shockwaves are thrown by the music. They used to expand on
    ``fmod(ctx.t * 3.6, 1.0)``, a free-running 3.6 Hz sawtooth that fired at
    exactly the same rate whether the track was a ballad or a drum solo, and
    scaled only its brightness by the overall level. Now a bass onset launches
    one, into whichever of two slots is oldest, and its radius is a function
    of how long ago it was launched. That makes the ring a thing the music
    *did* rather than a thing the clock did.

    Noise is a fixed per-size field rather than a fresh hash every frame. The
    hash was over 2 ms of the mode's budget at 400x100 and bought nothing: the
    blob's edge and the shockwaves both sweep across the field, so a
    stationary grain still animates everywhere it matters.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    dist, turn, max_r = _polar(ctx)
    n = min(16, ctx.n_display)

    avg = ctx.energy
    lut, idx = _angular_lut(ctx, turn, n, ctx.t * (0.10 + avg * 0.30))
    r_lut = max_r * (0.1 + 0.9 * lut * lut)
    halo_lut = lut * np.float32(0.4)

    st = ctx.scratch(
        "pulse",
        lambda: {
            "born": np.full(_PULSE_WAVES, -99.0),
            "amp": np.zeros(_PULSE_WAVES),
            "acc": 0.0,
        },
    )
    # A shockwave per onset, sized by how hard it hit. The analyser detects
    # these properly now — spectral flux across the whole band plan, with an
    # adaptive threshold — so this no longer keeps a pair of envelopes over
    # the bass band and calls their difference an onset. That approximation
    # could not tell a kick from the track simply getting louder, which is
    # why it needed a threshold tuned by hand.
    #
    # The rate term underneath it is not a second detector, and the
    # distinction matters. It is a clock whose speed is the loudness: it
    # cannot tell where a beat is and does not try. It exists because the
    # detector has measured gaps — recall drops to about 0.4 on dense fast
    # drums — and a mode whose entire identity is "radial pulse with
    # shockwaves" must not go still on material the analyser reads poorly. On
    # anything with a clear beat the onsets arrive first and the clock rarely
    # reaches its threshold; on a drone it keeps the mode breathing.
    st["acc"] += (0.8 + ctx.energy * 5.0) * ctx.dt
    # Fire the first one on arrival rather than making the clock earn it.
    # Otherwise the mode opens on an empty screen for most of a second, which
    # reads as broken rather than as anticipation.
    due = st["acc"] >= 1.0 or st["born"].max() < 0.0
    if due:
        st["acc"] = max(0.0, st["acc"] - 1.0)
    if (ctx.onsets or due) and (ctx.t - st["born"].max()) > 0.12:
        slot = int(np.argmin(st["born"]))
        st["born"][slot] = ctx.t
        strength = ctx.onset_strength if ctx.onsets else min(1.0, ctx.energy * 1.4)
        st["amp"][slot] = float(np.clip(0.35 + strength * 0.9, 0.0, 1.0))

    # The blob radius and the halo's brightness are both functions of the band
    # level alone, so they are computed on the 512-entry table and gathered,
    # rather than gathering the level and then running the arithmetic over
    # 320,000 dots. Same numbers, three fewer full-grid passes. See
    # :func:`angular_lut`.
    r = r_lut[idx]
    nz = ctx.scratch(
        "pulse_grain", lambda: np.random.default_rng(419).random((dr, dc)).astype(np.float32)
    )

    core = dist < 1.0
    # ``r >= 0.1 * max_r >= 1`` at every cell, so the second comparison of the
    # original mask never fired and was a full-grid pass for nothing; and
    # ``maximum(r, 1e-6)`` could never fire either. Both drop bit-identically.
    inside = dist <= r
    prox = np.where(inside, dist / r, np.float32(0.0))
    lit = inside & ((prox > 0.45) | (nz < 0.3 + prox * 0.7))

    # Folded into one threshold field rather than a separate mask, a clipped
    # overshoot and a product: each of those is a pass over the whole dot grid
    # and the halo was eleven of them. The 0.4 rides in the table with the
    # band level, which is the same product with one grid pass fewer.
    hv = np.clip(1.0 - (dist - r) * 0.25, 0.0, 1.0)
    hv *= halo_lut[idx]
    lit |= (~inside) & (nz < hv)

    # A wave crosses the radius in ~0.9 s and fades over the same span, so its
    # position is set by seconds since launch and nothing here is per-frame.
    for k in range(_PULSE_WAVES):
        age = ctx.t - st["born"][k]
        if not (0.0 <= age < 0.9):
            continue
        phase = age / 0.9
        strength = float(st["amp"][k]) * (1.0 - phase)
        if strength <= 0.06:
            continue
        band = 1.0 + strength * 3.0
        # The ring is a thin annulus — a few thousand cells on a 320k grid —
        # so the dither math runs only on the cells inside it. The membership
        # mask still needs the full-grid distance pass, but the expensive part
        # — the division, the fade and the threshold — is per-ring-cell, and
        # the per-cell values are the same float operations on the same
        # inputs as the old dense version. Cells outside the ring never had
        # their ``lit`` bit touched (``near & ...`` was false there).
        # A ring of radius R cannot reach a row further than R + band from the
        # centre, because ``dist`` is never smaller than the vertical offset
        # alone. Clipping the rows first means a young wave — a small ring,
        # which is most of a wave's visible life — tests a few dozen rows
        # instead of all four hundred, and the membership pass stops being the
        # expensive part of the mode. Cells outside the slice could not have
        # been ``near``, so nothing is lost.
        radius = max_r * phase
        reach = radius + band
        r0 = max(0, int(dr * 0.5 - reach))
        r1 = min(dr, int(dr * 0.5 + reach) + 2)
        if r1 <= r0:
            continue
        rows = lit[r0:r1]
        edge = np.abs(dist[r0:r1] - radius)
        wy, wx = np.nonzero(edge < band)
        if wy.size:
            rows[wy, wx] |= nz[r0:r1][wy, wx] < (1.0 - edge[wy, wx] / band) * strength

    lit |= core
    codes = pack_braille(lit)
    # The core is a handful of dots; giving it its own full-grid ``where`` and
    # ``maximum`` cost two passes over 320k cells to colour about twelve.
    heat = prox * lit
    heat[core] = 0.2
    cidx = ctx.ramp(cell_max(heat))
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

    # Peak-hold per ring, so a kick actually *pushes the ring out* and it eases
    # back over the following half-second. Without it the radius tracked the
    # level frame for frame: the ring snapped to a new size and snapped back,
    # which is not the expanding ring this mode is named for. The release rate
    # is in units per second, so it is frame-rate independent.
    st = ctx.scratch("arcs_peak", lambda: np.zeros(n, dtype=np.float32))
    if st.shape[0] != n:
        st = np.zeros(n, dtype=np.float32)
        ctx.state[("arcs_peak", ctx.w, ctx.h)] = st
    np.maximum(st - np.float32(1.5 * ctx.dt), lv, out=st)

    steps = 512
    u = np.linspace(0.0, 1.0, steps, endpoint=False, dtype=np.float32)
    lut = np.zeros(steps, dtype=np.float32)
    for j in range(n):
        level = float(st[j])
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

    # ``dist`` never changes, so the whole index conversion — scale, truncate
    # and clamp — is per-size state, not per-frame work. Rebuilding the three
    # passes over 320k cells every frame cost over a millisecond for a table
    # that is the same 400x800 array of small ints all session.
    ridx = ctx.scratch(
        "arcs_ridx",
        lambda: np.clip(
            (dist * np.float32(steps / max_r)).astype(np.int32), 0, steps - 1
        ),
    )
    heat = lut[ridx]

    # The spectrum also modulates brightness around the ring, so the rings
    # shimmer where the energy is rather than glowing evenly. The modulation
    # is folded into the angle table itself — ``0.55 + 0.45 * lut_a`` over the
    # 512-entry table, then one multiply of two gathered fields — instead of
    # gathering ``lut_a`` and building the modulation over the full dot grid.
    # Each angle step's value is the same float expression either way.
    lut_a = _angular_bands(ctx, turn, min(12, n), ctx.t * 0.05)
    mod = lut_a * np.float32(0.45) + np.float32(0.55)
    heat *= mod

    lit = heat > 0.06
    codes = pack_braille(lit)
    # Graded to twelve steps before ramping, the same trick and for the same
    # reason as Chladni and Auroras. A ring is a radial gradient and colour
    # runs are horizontal, so near the top and bottom of one every cell lands
    # in a different ramp bucket from its neighbour: 47 distinct indices and
    # 8,366 runs a frame at 400x100, which cost more than building the picture
    # did. Twelve steps is finer than the eye separates on an annulus this
    # thin and takes it to 6,716.
    shade = cell_max(heat * lit)
    shade *= np.float32(12.0)
    np.round(shade, 0, out=shade)
    shade *= np.float32(1.0 / 12.0)
    return codes, ctx.ramp(shade)


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

    # The surface the bubbles are heading for. Dotted and shallow on purpose —
    # a full-amplitude line across the top reads as a waveform, which is the
    # one thing this mode should not look like.
    #
    # The amplitude cap is what keeps it a *line*. The surface lives inside the
    # first braille cell, which is four dot rows tall, so an amplitude past
    # about 1.0 scatters it across all four and it stops reading as a surface
    # at all — just speckle along the top edge. The old ``0.4 + mids * 1.1``
    # crossed that at mids > 0.55, which was rare while the analyser ran at 94
    # analyses/sec and is constant at 188, where the mid envelope tracks
    # transients twice as sharply and peaks much higher. Capped at 0.95 the
    # ripple can only ever occupy two adjacent dot rows.
    mids = ctx.range(0.25, 0.60)
    # Every other dot column, not every third. At a stride of 3 the lit dots
    # fall 1.5 braille cells apart, so one cell in three is empty and the
    # "surface" is a broken row of specks — legible on a tall window where
    # there is room to read it as a line, easy to lose entirely on a short
    # one. A stride of 2 puts exactly one dot in every cell column.
    sx = np.arange(0, dc, 2)
    # The ripple is a fraction of the width rather than a fixed 0.07 rad/dot,
    # so the surface carries the same two-and-a-bit waves at any size instead
    # of one lazy curve at 60 columns and nine at 400.
    ripple = np.sin(sx * (14.0 / max(dc, 1)) + ctx.t * 1.4) * min(0.95, 0.4 + mids * 1.1)
    srow = np.clip(np.rint(1.5 + ripple).astype(np.int32), 0, dr - 1)
    # Loudness goes into how brightly the surface glows rather than into how
    # far it swings, so the reaction is one the shape survives.
    field[srow, sx] = np.maximum(field[srow, sx], 0.34 + 0.46 * mids)

    np.clip(field, 0.0, 1.0, out=field)
    dots = field > 0.05
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(field))
    return codes, cidx


def _radial(ctx: Ctx, octant: bool):
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 8 or dc < 8:
        return empty(ctx.w, ctx.h)

    dist, turn, max_r = _polar(ctx)
    n = min(24, max(8, ctx.n_display))
    lut, idx = _angular_lut(ctx, turn, n, ctx.t * 0.04)

    inner = max_r * 0.18
    # Everything here that does not move lives in scratch: the gutters between
    # the wedges are a function of the angle and the band count, the inner ring
    # and the "outside the hub" test are functions of the radius, and the
    # radius offset the shading divides is the same array every frame. Five
    # full-grid passes at 400x100, rebuilt sixty times a second for numbers
    # that only change when the terminal is resized or the band count moves.
    # Each version keeps its own entry, so one never sees the other's.
    def build():
        wedge = frac(turn * n)
        return {
            "n": n,
            "outside": (dist >= inner) & ~((wedge < 0.06) | (wedge > 0.94)),
            "ring": np.abs(dist - inner) < 0.9,
            "from_inner": dist - inner,
        }

    key = "radial_geo" if not octant else "radial_geo_fine"
    geo = ctx.scratch(key, build)
    if geo["n"] != n:
        geo = build()
        ctx.state[(key, ctx.w, ctx.h)] = geo

    # The rays breathe on the beat. ``ctx.pulse`` is the gated form of
    # beat_phase — 0.0 until a tempo is established, so silence and the first
    # seconds of a track get no swell rather than a permanent one. Small,
    # because the rays are already the loudest thing here.
    #
    # The ray length is a function of the band level, so it is computed on the
    # 512-entry table and gathered — see angular_lut. Same for the reciprocal
    # the shading needs, which turns a full-grid division into a gather.
    span_lut = (max_r - inner) * lut * np.float32(1.0 + 0.12 * ctx.pulse)
    outer = inner + span_lut[idx]

    lit = (dist <= outer) & geo["outside"]
    lit |= geo["ring"]

    heat = np.where(lit, np.clip(geo["from_inner"] / np.maximum(span_lut, 1e-6)[idx], 0, 1), 0.0)
    if octant:
        # Two colours and a glyph threshold per cell, like Kaleidoscope: the
        # whole viewport is field — wedges, hub and the dark between them —
        # so the cell's range is ramped into background and foreground and
        # the glyph says which of the eight subcells sit on which side of the
        # cell's midpoint. Hard-edged wedges keep a soft rim where the field
        # crosses its own midpoint inside a cell.
        lo_cell, hi_cell = cell_hilo(heat)
        codes = pack_octant(heat, lo_cell, hi_cell)
        return codes, ctx.ramp(hi_cell), ctx.ramp(lo_cell)
    codes = pack_braille(lit)
    cidx = ctx.ramp(cell_max(heat))
    return codes, cidx


@mode("Radial", group="particles", blurb="the spectrum wrapped into a circle")
def radial(ctx: Ctx):
    return _radial(ctx, octant=False)


@mode("Radial Fine", hidden=True, after="Radial", group="particles",
      blurb="the same ring as a solid field at two colours a cell — needs a terminal that draws Unicode 16 octants")
def radial_fine(ctx: Ctx):
    """Radial on octant cells.

    Separate mode rather than a switch on the original, for the same reason
    Kaleidoscope Fine is: octants are Unicode 16 and an older terminal or
    font draws a grid of tofu, which is a thing to opt into rather than to
    discover when a mode you liked stops working.
    """
    return _radial(ctx, octant=True)


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


def _sonar(ctx: Ctx, octant: bool):
    """A ship's sonar scope, not another way to draw a spectrum.

    Two modes share this body. ``octant=False`` is the original: the lit
    dots are packed into braille, so the sweep reads as a chain of dots and
    a return as a clump of stipple. ``octant=True`` packs the identical dot
    set into Unicode 16 octant glyphs — block mosaic at the same 4x2
    resolution — so the same beam and the same returns read as a solid
    stroke on the scope. Nothing else differs: same sweep rate, same
    phosphor decay, same colours. See :func:`render.pack_octant_bits`.

    That substitution is worth a mode here because this one is almost pure
    edge. A sweep line and contact blips against empty space have no
    interior to speak of, so the foreground is the only colour either
    version sets: an octant cell is opaque once it is given a background
    index, and the space around the beam has to stay the terminal's own.

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

    # The phosphor buffer is the mode's one piece of mutable state, and each
    # version keeps its own: sharing one would make the two modes advance
    # each other's decay, so every frame would come out "different" for no
    # reason.
    buf = ctx.scratch(
        "sonar_buf" if not octant else "sonar_buf_fine",
        lambda: np.zeros((dr, dc), dtype=np.float32),
    )

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

    # Persistence, and the beat holds it. A radar screen's phosphor is what
    # carries the last sweep, so lengthening it on the pulse leaves the whole
    # trail hanging a moment longer on the beat and fading back between —
    # visible across the entire swept area rather than only where the beam
    # currently is, which a brightness kick would not be. ``ctx.pulse`` is 0.0
    # until a tempo is established, so this is exactly the 0.9 s constant
    # until there is a beat to hold for.
    decay = float(np.exp(-max(ctx.dt, 0.0) / (0.9 + 0.5 * ctx.pulse)))
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
    codes = pack_octant_bits(dots) if octant else pack_braille(dots)
    cidx = ctx.ramp(cell_max(buf * dots))
    return codes, cidx


@mode("Sonar", group="particles",
      blurb="one sweep, not the whole spectrum — returns fade like CRT phosphor")
def sonar(ctx: Ctx):
    return _sonar(ctx, octant=False)


@mode("Sonar Fine", hidden=True, after="Sonar", group="particles",
      blurb="the same sweep drawn solid instead of stippled — needs a terminal that draws Unicode 16 octants")
def sonar_fine(ctx: Ctx):
    """Sonar on octant cells.

    Separate mode rather than a switch on the original, for the same reason
    Kaleidoscope Fine is: octants are Unicode 16 and an older terminal or
    font draws a grid of tofu, which is a thing to opt into rather than to
    discover when a mode you liked stops working. The two versions also keep
    separate phosphor buffers in scratch, so switching between them never
    advances the other's decay.
    """
    return _sonar(ctx, octant=True)


@mode("Orbit", group="particles", blurb="bodies on real elliptical orbits; loud bands swing out")
def orbit(ctx: Ctx):
    """Kepler, not a wheel of dots.

    The first version put each band at a radius equal to its level and gave it
    a constant angular speed. Two things went wrong with that, and they are the
    same thing twice: with a flat-ish spectrum every body sits at the *same*
    radius, so sixteen of them pile into one ring, and a constant angular rate
    means each one traces a perfect circle. What you got was a set of
    concentric circular trails -- ``Radial`` with motion, which is not a reason
    for a mode to exist.

    Here each band owns a fixed **semi-major axis**, spread across the disc, so
    the bodies keep their own lanes however the spectrum moves. What the level
    drives is **eccentricity**: quiet bands run near-circular, loud ones stretch
    into long ellipses that dive through the middle and swing far out. The
    picture reacts by changing the *shape* of the paths rather than by moving
    dots along fixed ones.

    Motion is Keplerian, which is what makes it read as orbiting. The ellipse
    in polar form about a focus at the centre is ``r = a(1-e^2) / (1 + e cos f)``,
    and conservation of angular momentum gives ``df/dt = L / r^2`` -- so a body
    whips through periapsis and crawls at apoapsis, all from one integration
    with no special-casing. Taking ``L`` proportional to ``sqrt(a(1-e^2))``
    falls out as Kepler's third law, period proportional to ``a^1.5``, so the
    outer bands genuinely orbit slower than the inner ones instead of the mode
    having to fake a speed gradient.

    The orbits also precess: each ellipse's long axis turns at its own rate, so
    the paths never close into a static figure and the trails weave. Precession
    speed follows overall energy.

    Brightness is speed. A body is hottest at periapsis where it is moving
    fastest, which is free -- ``df/dt`` is already computed.

    Kept from the previous version: the phosphor trail buffer, which is the one
    part that was right, and the single vectorised ``maximum.at`` that writes
    every body at once.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 10 or dc < 10:
        return empty(ctx.w, ctx.h)

    n = int(min(18, max(6, ctx.n_display + 4)))
    lv = ctx.display_bands(n).astype(np.float64)

    cx, cy = dc / 2.0, dr / 2.0
    # Braille dots are square, so a circle in dot space is a circle on screen.
    # The orbits are still drawn wider than tall on purpose: a terminal is a
    # wide letterbox and a height-limited circle leaves most of it empty.
    wide = max(1.0, (dc / max(dr, 1)) * 0.82)
    max_r = max(2.0, cy - 1.0)

    def spawn():
        rng = np.random.default_rng(41)
        return {
            # Fixed lanes. Squared spacing so the inner orbits are not all
            # crammed together, which is where the crowding used to happen.
            # 0.32..0.98 of the disc, not 0.16..0.96. Kepler makes the period
            # ratio the *1.5 power* of the axis ratio, so a 6x spread of lanes
            # is a 15x spread of orbital periods and the outer bands crawl.
            # A 3x spread keeps them all moving at watchable rates.
            "a": max_r * (0.32 + 0.66 * (np.arange(n) / max(n - 1, 1)) ** 0.85),
            "f": rng.uniform(0.0, 2 * math.pi, n),      # true anomaly
            "w": rng.uniform(0.0, 2 * math.pi, n),      # argument of periapsis
            "prec": rng.uniform(0.10, 0.42, n) * rng.choice((-1.0, 1.0), n),
            "e": np.zeros(n),
        }

    st = ctx.scratch("orbit", spawn)
    if len(st["f"]) != n:
        st = spawn()
        ctx.state[("orbit", ctx.w, ctx.h)] = st

    # Eccentricity eases rather than snapping: an orbit whose shape changed
    # discontinuously would teleport its body, since r depends on e.
    target_e = np.clip(0.06 + lv * 0.78, 0.0, 0.86)
    st["e"] += (target_e - st["e"]) * min(1.0, ctx.dt / 0.18)
    e = st["e"]
    a = st["a"]

    # r from the focal-polar form, then df/dt = L / r^2 with
    # L = k * sqrt(a(1 - e^2)) -- Kepler's third law, so outer bands are slower.
    # ``grav`` sets the timescale, and it is derived rather than guessed: with
    # L = grav * sqrt(a(1-e^2)) the period is 2*pi*a^1.5 / grav, so
    # grav = C * max_r^1.5 makes the period independent of terminal size, and
    # C = 0.63 puts the innermost lane at about 1.8 s and the outermost at
    # about 9 s. The first cut used a constant near 0.42*max_r, which works out
    # to a ~35 s inner orbit — slower than the 0.35 s trail decay by two orders
    # of magnitude, so nothing ever drew an arc at all.
    grav = 0.63 * max_r ** 1.5
    semi_latus = a * (1.0 - e * e)
    r = semi_latus / (1.0 + e * np.cos(st["f"]))
    rate = (np.sqrt(np.maximum(semi_latus, 1e-6)) * grav) / np.maximum(r * r, 1e-6)
    st["f"] = (st["f"] + rate * ctx.dt) % (2 * math.pi)
    st["w"] = (st["w"] + st["prec"] * (0.35 + ctx.energy * 1.6) * ctx.dt) % (2 * math.pi)

    # recompute r at the new anomaly so position and brightness agree
    r = semi_latus / (1.0 + e * np.cos(st["f"]))
    ang = st["f"] + st["w"]
    fx = cx + np.cos(ang) * r * wide
    fy = cy + np.sin(ang) * r

    buf = ctx.scratch("orbit_buf", lambda: np.zeros((dr, dc), dtype=np.float32))
    # Trail length, and it was the whole problem with this mode.
    #
    # At a 0.55 s decay against a 0.04 threshold a stroke stays lit for
    # 0.55 * ln(1/0.04) = 1.77 seconds. The inner bodies orbit in a couple of
    # seconds, so each one was drawing most of a full revolution at once and
    # sixteen of them overlapped into a solid scribble -- 41% of the frame
    # filled, almost all of it the heaviest glyph, with no individual path
    # readable anywhere in it. The Keplerian motion underneath was correct
    # and completely invisible.
    #
    # Short enough now (0.42 s to the threshold) that a body pulls a comet
    # tail rather than painting its whole orbit, which is what makes the
    # ellipse legible: you see where it has just been, not everywhere it has
    # ever been.
    buf *= float(np.exp(-max(ctx.dt, 0.0) / 0.16))

    # Speed is the brightness. Normalised against each body's own mean rate so
    # a slow outer orbit still lights up at its own periapsis rather than being
    # permanently dim next to the inner ones.
    mean_rate = (np.sqrt(np.maximum(semi_latus, 1e-6)) * grav) / np.maximum(a * a, 1e-6)
    hot = np.clip(rate / np.maximum(mean_rate, 1e-6), 0.35, 2.2)
    vals = np.clip(0.30 + 0.34 * hot + 0.30 * lv, 0.0, 1.0)

    # The path between frames, not just its endpoints. A body at periapsis
    # covers several dots per frame, so stamping only where it lands leaves a
    # dashed arc of disconnected blobs instead of a curve -- which is what the
    # ellipses looked like until this went in. Interpolating gives the
    # continuous line the trail buffer was always meant to be smearing.
    prev = st.get("prev")
    if prev is None or len(prev[0]) != n:
        prev = (fx.copy(), fy.copy())
    ox, oy = prev
    span = float(np.max(np.hypot(fx - ox, fy - oy))) if n else 0.0
    steps = int(min(28, max(1, math.ceil(span))))
    tt = np.linspace(0.0, 1.0, steps, dtype=np.float64)[None, :]
    lx = np.rint(ox[:, None] + (fx - ox)[:, None] * tt).astype(np.int32)
    ly = np.rint(oy[:, None] + (fy - oy)[:, None] * tt).astype(np.int32)
    ok = (ly >= 0) & (ly < dr) & (lx >= 0) & (lx < dc)
    np.maximum.at(
        buf,
        (ly[ok], lx[ok]),
        np.broadcast_to(vals[:, None], lx.shape)[ok].astype(np.float32),
    )
    st["prev"] = (fx.copy(), fy.copy())

    # The head gets the cross, so the body reads as an object on the line
    # rather than as the brightest pixel of it.
    px = np.clip(np.rint(fx).astype(np.int32), 0, dc - 1)
    py = np.clip(np.rint(fy).astype(np.int32), 0, dr - 1)
    dy_off, dx_off = _BUBBLE_RINGS[1]
    hy = py[:, None] + dy_off[None, :]
    hx = px[:, None] + dx_off[None, :]
    ok = (hy >= 0) & (hy < dr) & (hx >= 0) & (hx < dc)
    np.maximum.at(
        buf,
        (hy[ok], hx[ok]),
        np.broadcast_to(np.minimum(vals + 0.25, 1.0)[:, None], hy.shape)[ok].astype(np.float32),
    )

    dots = buf > 0.07
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(buf))
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
            "launch_acc": 0.0, "launched": 0,
            "rng": np.random.default_rng(41),
        }

    st = ctx.scratch("fireworks", spawn_state)
    rng = st["rng"]

    # One shell per detected onset. Previously a pair of bass envelopes
    # differenced against a hand-tuned 0.1 — an onset detector in miniature,
    # and a worse one than the analyser's, which looks at the whole spectrum
    # rather than the bottom fifth of it and so still fires on a snare.
    hit = bool(ctx.onsets)

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
    #
    # This rate is also what keeps the mode alive where the detector is weak.
    # Onsets punctuate the barrage; they no longer constitute it. Recall falls
    # to roughly 0.4 on dense fast drums, and a sky that only lights on
    # detected onsets would visibly thin out on exactly the music that should
    # fill it.
    st["launch_acc"] += (0.35 + ctx.energy * 7.0) * ctx.dt
    # Open with a shell instead of making the accumulator earn the first one:
    # at a moderate level that takes most of a second, and a fireworks mode
    # that begins on an empty sky reads as not working.
    if st["launched"] == 0 and st["launch_acc"] < 1.0:
        st["launch_acc"] = 1.0
    want = int(st["launch_acc"])
    if want:
        st["launch_acc"] -= want
        st["launched"] += want
    if hit:
        # A harder onset throws more shells, and several onsets inside one
        # frame each earn their own — at a low frame rate or on fast drums
        # ctx.onsets can be 2 or 3, and collapsing that to a single shell
        # would quietly drop beats the analyser did detect.
        want += ctx.onsets + int(min(2, ctx.onset_strength * 2.5))

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
            # Loud throws higher and faster. Height and speed were drawn from
            # a fixed range, so a shell launched during a quiet passage was
            # indistinguishable from one launched at full tilt — the music
            # chose when a rocket went up and, after the kind, nothing about
            # how it flew.
            lift = 0.55 + ctx.energy * 0.75
            st["rvy"][i] = dr * rng.uniform(1.1, 1.5) * lift
            # Burst height, as a distance down from the top of the screen —
            # so a smaller number is a higher shell.
            #
            # Recalibrated for the energy real music actually produces.
            # ``ctx.energy`` is the mean over every band, and the analyser's
            # autosens normalises the loudest band to about 1.0, so a mean of
            # 0.25-0.35 is a busy track, not a quiet one. The previous mapping
            # wanted 0.77 before it would send a shell near the top, which
            # nothing short of white noise reaches: measured across the range,
            # rockets burst at 56-65% of the screen height and the sky above
            # them was simply never used.
            top = dr * (0.42 - 0.30 * min(1.0, ctx.energy * 2.2))
            st["rtarget"][i] = rng.uniform(max(dr * 0.06, top * 0.7), max(top, dr * 0.10))
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
            # Isotropic, because a shell bursts as a sphere and only gravity
            # is allowed to make it anything else.
            #
            # The vertical component used to be halved, which reads as an
            # aspect correction and is not one: a terminal cell is twice as
            # tall as it is wide, and braille puts four dot rows and two dot
            # columns in it, so a dot is already square and needs no
            # correction at all. Halving it a second time made every burst an
            # ellipse — measured at 2.4 to 3.6 times wider than tall across
            # the energy range, which is what a firework looks like if you
            # sit on it.
            st["svy"][slots] = -np.sin(ang) * spd
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
    # The seed is constant, so the hash is too — it was being recomputed over
    # 320,000 cells every frame to produce the same field. Scratch.
    sand = ctx.scratch("dune_grain", lambda: noise((dr, dc), 0))
    grain = sand < np.clip(0.35 + depth * 0.05, 0.35, 0.97)
    dots = body & grain

    heat = np.clip((0.35 + 0.65 * (1.0 - depth / max(1, dr))) * dots, 0.0, 1.0)
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(heat))
    return codes, cidx


_MURM_CAP = 240


def _blob_offsets(r: float) -> tuple:
    """A soft round stamp, as (dy, dx, weight). Braille dots are ~square."""
    k = int(math.ceil(r))
    dy, dx = np.mgrid[-k:k + 1, -k:k + 1]
    d2 = (dy * dy + dx * dx).astype(np.float64)
    keep = d2 <= r * r
    w = np.exp(-d2 / (2.0 * (r * 0.55) ** 2))
    return dy[keep].astype(np.int32), dx[keep].astype(np.int32), w[keep]


#: One bird's deposit into the density field. Small on purpose: the mass has
#: to come from birds overlapping each other, not from each bird being big.
_MURM_BLOB = _blob_offsets(2.5)


def _murmuration_state(dr: int, dc: int) -> dict:
    rng = np.random.default_rng(211)
    # Enough birds that overlapping blobs actually make a mass. At the old
    # floor of 90 an 80x24 terminal got a scatter of separate specks, because
    # the flock is drawn as density and 90 agents cannot be dense.
    n = int(np.clip(dc * 0.55, 150, _MURM_CAP))
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
        "scatter_t": -99.0,
        # last frame's density bounding box, so only that region needs clearing
        "box": (0, dr, 0, dc),
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
    lofi modes. What the spectrum plays is the flock's *size and shape*: bass
    stiffens the centroid spring and shrinks the birds' comfort spacing, so
    the whole thing contracts into a dense ball; treble drives separation and
    speed and blows it back open into a churning sheet. On top of that, bass
    adds a wind force in a slowly rotating direction (``ctx.t * constant``,
    the safe half of the pattern ``Tunnel``'s docstring warns about), treble
    adds per-agent jitter (genuinely random every frame on purpose — this is
    stochastic forcing on the physics, not a texture that should hold still
    like ``Dune``'s grain), and a hard bass onset gives the whole flock an
    outward kick from its own centroid. Nothing scripts the flock back
    together afterward — cohesion is already pulling every agent toward its
    neighbours every frame, so it reforms because that's what the same three
    rules that hold it together always do.

    It is drawn as a *density field*, not as bodies. Each bird deposits a soft
    blob into one shared buffer and the buffer is dithered against fixed
    grain; nothing on screen corresponds to one agent. A real murmuration
    reads as mass — you see the shape of the whole thing and only pick out
    individuals at the thin edge — and giving each agent a bright head and a
    tapering trail instead made two hundred tadpoles.
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
    # The scatter has to stay an *event*, and that is a different requirement
    # from "was there a beat". The analyser answers the second one now, so
    # this only has to answer the first: the centroid spring needs about a
    # second to haul the flock back in, so scattering on every kick leaves it
    # permanently blown out against the walls, and a flock that is always
    # scattered is indistinguishable from one that never scatters.
    #
    # So: a real onset, hard enough to be worth reacting to, and not too soon
    # after the last one. What this replaces was a pair of envelopes over the
    # bass band differenced against a 0.22 threshold — a hand-rolled onset
    # detector that fired on the track getting louder as readily as on a drum,
    # and needed a 2.2 s refractory to be usable at all.
    hit = (
        ctx.onsets
        and ctx.onset_strength > 0.35
        and (ctx.t - st["scatter_t"]) > 1.4
    )

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
    # Comfort spacing. Bass shrinks it as well as stiffening the spring below:
    # a flock contracts because the birds tolerate being closer, not only
    # because something outside is squeezing them.
    r_sep = r_perc * (0.34 - 0.12 * bass)
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

    # Separation with a real distance falloff. This used to be a mean of pure
    # *unit* offsets, which is purely directional: it says which way to go but
    # not how badly, so inside a uniformly compressed cloud the unit vectors
    # cancel and there is no force left resisting the squeeze. The flock had
    # no spacing floor at all and cohesion crushed it to a blob a tenth of the
    # screen wide. Weighting each push by ``r_sep/d - 1`` makes it zero at the
    # comfort distance and steep below it, so the flock has a size set by how
    # many birds are in it.
    # Separation as a *sum* of distance-weighted pushes, which is what gives
    # the flock a size. It used to be a mean of pure unit offsets, and that is
    # doubly toothless: unit offsets carry no "how badly", and averaging them
    # divides by neighbour count, so the term gets *weaker* exactly as the
    # flock gets denser. Interior pushes cancel by symmetry either way — that
    # part is correct, it is how a fluid works — but with a sum the birds on
    # the boundary feel an outward force proportional to the density behind
    # them, which is the pressure that balances cohesion. Without it there is
    # no equilibrium radius at all and the flock crushes to a dot.
    sf = sep_m.astype(np.float64)
    d = np.sqrt(np.maximum(dist2, 1e-9))
    push = sf * np.clip(r_sep / d - 1.0, 0.0, 4.0) / d
    sep_ax = (push * raw_dx).sum(axis=1)
    sep_ay = (push * raw_dy).sum(axis=1)

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
    coh_gain = (0.25 + bass * 0.40) * scale
    # Separation sets *spacing inside* the flock; it is not the thing that
    # makes treble read. Driving it hard (it used to reach 2.8x scale) simply
    # overpowers cohesion and the flock stops being a flock — every bird ends
    # up an isolated agent obeying rules against an empty neighbourhood, which
    # is the scattered-dots picture this mode had. Treble goes into speed and
    # churn below instead, where an agitated flock actually shows it.
    sep_gain = (0.80 + treble * 0.60) * scale
    ax = sep_ax * sep_gain + (avg_vx - vx) * 2.0 + coh_x * inv_perc * coh_gain
    ay = sep_ay * sep_gain + (avg_vy - vy) * 2.0 + coh_y * inv_perc * coh_gain

    # Global cohesion toward the flock's own centroid, on top of the local
    # rule. Local cohesion only reaches one perception radius, so as soon as
    # the flock grows past that it fragments into unconnected sub-clusters and
    # the sub-clusters drift apart for good — there is no force left that can
    # find them again. This is the term that keeps it one readable mass, and
    # it is a spring, so bass tightening it makes the whole flock contract.
    # The spring constant is what the bass actually plays: slack, and the
    # flock spreads into a loose sheet that separation alone shapes; stiff,
    # and it contracts into a dense ball. That swing is the mode's whole
    # reaction, so the range has to be wide — a narrow one just moves a
    # constant-looking blob around the screen.
    gcx, gcy = float(x.mean()), float(y.mean())
    # Sizing note, because the numbers are not arbitrary: at the flock's edge
    # the outward separation force (max ``sep_gain``) balances the inward
    # spring (``pull`` x radius), so the flock settles at a radius of about
    # ``sep_gain / pull`` dots. At 80x24 (96 dot rows) that is a ~33-dot mass
    # when nothing is playing, ~14 dots under heavy bass, and ~57 dots when
    # treble drives separation and blows it open — a real change of shape at
    # every end, with a flock still on screen at all of them. A slacker spring
    # than this does not read as "spread out", it reads as dissolved.
    pull = 1.00 + 1.30 * bass
    ax += (gcx - x) * pull
    ay += (gcy - y) * pull

    # The wind veers a full turn every ~20s. At the original 0.05 rad/s it
    # took two minutes to come round, which is long enough that it acts as a
    # steady push: the flock rides it into a wall and sits there balanced
    # against the boundary force for most of a track.
    wind_ang = st["wind_dir"] + ctx.t * 0.32
    wind_mag = bass * scale * 0.55
    ax += math.cos(wind_ang) * wind_mag
    ay += math.sin(wind_ang) * wind_mag

    # a weak pull toward mid-screen so the flock stays where it can be seen
    ax += ((dc - 1) * 0.5 - x) * 0.35
    ay += ((dr - 1) * 0.5 - y) * 0.35

    jitter_mag = treble * scale * 0.30
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
        kick = scale * 0.85
        vx = vx + (ox / od) * kick
        vy = vy + (oy / od) * kick

    speed = np.hypot(vx, vy)
    max_speed = scale * (0.50 + bass * 0.35 + treble * 0.45)
    min_speed = scale * (0.12 + treble * 0.18)
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

    px = np.clip(np.rint(x).astype(np.int32), 0, dc - 1)
    py = np.clip(np.rint(y).astype(np.int32), 0, dr - 1)

    # Density, not bodies. Drawing each bird as a bright cross with a
    # tapering three-step trail behind it gives every agent a head and a tail,
    # and two hundred of those read as a field of tadpoles rather than as a
    # flock. A real murmuration reads as *mass*: you see the shape of the
    # whole thing and only pick out individuals at the thin edge. So every
    # bird deposits a soft round blob into one shared field and the field is
    # what gets drawn — overlap is what makes the middle solid.
    # Everything below only ever touches the flock's bounding box. The field
    # is exactly zero outside it, and at 400x100 a full-grid pass is 320k
    # cells — three or four of those cost more than the entire n-squared
    # flocking simulation does. The buffer is kept in scratch and only the
    # previous frame's box is cleared, so the cost tracks the flock's size
    # rather than the terminal's.
    field = ctx.scratch("murm_field", lambda: np.zeros((dr, dc), dtype=np.float32))
    py0, py1, px0, px1 = st["box"]
    field[py0:py1, px0:px1] = 0.0

    y0, y1 = max(0, int(py.min()) - 3), min(dr, int(py.max()) + 4)
    x0, x1 = max(0, int(px.min()) - 3), min(dc, int(px.max()) + 4)
    st["box"] = (y0, y1, x0, x1)

    by, bx, bw = _MURM_BLOB
    ys = py[:, None] + by[None, :]
    xs = px[:, None] + bx[None, :]
    ok = (ys >= 0) & (ys < dr) & (xs >= 0) & (xs < dc)
    np.add.at(field, (ys[ok], xs[ok]), np.broadcast_to(bw[None, :], ys.shape)[ok])

    # Saturating rather than clipped: one bird alone is faint, a pile of them
    # goes solid, and there is no flat ceiling where a dense core stops
    # getting denser and the interior structure disappears.
    box = field[y0:y1, x0:x1]
    vis = np.zeros((dr, dc), dtype=np.float32)
    vis[y0:y1, x0:x1] = 1.0 - np.exp(-box * np.float32(0.60 + 1.70 * ctx.energy))

    # Dithering against a *fixed* noise field, so the edge of the mass breaks
    # up into individual dots on its own. The threshold field is per-size and
    # never regenerated: the grain has to belong to the screen, not to the
    # frame, or the whole flock boils (invariant: animated randomness is
    # almost always wrong).
    thr = ctx.scratch(
        "murm_grain",
        lambda: (np.random.default_rng(77).random((dr, dc)) * 0.80 + 0.06).astype(np.float32),
    )
    dots = np.zeros((dr, dc), dtype=bool)
    dots[y0:y1, x0:x1] = box_vis = vis[y0:y1, x0:x1] > thr[y0:y1, x0:x1]
    np.multiply(vis[y0:y1, x0:x1], box_vis, out=vis[y0:y1, x0:x1])

    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(vis))
    return codes, cidx

