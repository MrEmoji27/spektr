"""Dither Storm — a one-bit field that moves, driven band by band.

A companion to Dither rather than a replacement. That mode is deliberately
still: its own docstring says a frozen spectrum is a frozen frame, because it
has no clock and its eight wave fields are baked once and only re-weighted.
That stillness is the point there and the limit here — this one is built to
react, so every term in it moves.
"""

from __future__ import annotations

import math

import numpy as np

from ..analysis import resample_bands
from ..render import cell_max, pack_braille
from . import Ctx, empty, mode

#: Directional waves in the mix, one per band group. Eight is the same count
#: Dither uses, and for the same reason: enough directions that the crosshatch
#: never resolves into stripes, few enough that the whole field is still one
#: small matrix multiply. See ``_field`` for why the count barely costs here.
LAYERS = 8

#: Extra transient directions in the layer bank, beyond the eight base ones.
#: They sit on the half-steps between the base angles and carry a finer grain,
#: and they are silent until an onset wakes one — round-robin — after which it
#: decays over a few tenths of a second. Waking one is a wider matrix multiply
#: for a few frames rather than a new pass over the grid, so a busy passage
#: reads as a *different* crosshatch, not just a faster one.
STORM = 6

#: Ring slots. A beat throws a ring and a burst of beats throws several; when
#: the slots run out the oldest ring is retired to make room, so a drum fill
#: reads as a drum fill instead of being throttled to one ring at a time.
RING_SLOTS = 8

#: Side of the blue-noise threshold tile. Tiled by absolute dot position, so
#: 64 is a compromise: large enough that the eye cannot find the repeat in a
#: moving field, small enough that building it is free.
TILE = 64

#: Radius bins for the onset rings, and the radius they span. x and y are
#: normalised to +/-0.5, so the far corner of the grid sits at sqrt(0.5) ~
#: 0.707 and a ring is fully gone just past that. 384 bins across it is finer
#: than a dot at any grid size this renders at, so the quantisation is not
#: visible in the ring's edge.
RAD_BINS = 384
RAD_MAX = 0.80

#: Levels the colour is folded onto before it reaches the ramp. This is a
#: render-cost control rather than an aesthetic one — see the note at the end
#: of ``dither_storm`` for what an unquantised field costs downstream.
COLOUR_STEPS = 10


@mode("Dither Storm", group="fields",
      blurb="Dither's one-bit crosshatch, but the field moves — each band drives its own wave, and beats throw rings through it")
def dither_storm(ctx: Ctx):
    """The spectrum as eight travelling waves, thresholded to one bit.

    Same family as Dither and a different animal. Dither bakes eight wave
    fields once and re-weights them per frame, so its texture restates the
    spectrum; what motion it has is one slow global drift shared by the whole
    field (``dither_drift``, about 0.05-0.25 of a cycle a second). Measured on
    a frozen spectrum that is 0.4% of cells changing per frame against 14%
    here, because in this mode every wave carries its own phase advancing at a
    rate set by the band that owns it — a busy band visibly races while a
    quiet one crawls. The spectrum picks the texture in both; only this one
    lets it pick the motion.

    (Dither's own docstring claims it has no clock at all and that a frozen
    spectrum is a frozen frame. That is not true of the code as it stands —
    the drift accumulator is right there — so do not take that sentence as the
    contrast this mode is drawn against. The contrast is one global drift
    versus a phase per band.)

    **Why the layer count is nearly free.** Summing eight travelling waves
    over the dot grid looks like eight passes over several hundred thousand
    dots a frame, which is what makes an animated field expensive. It is not,
    because a plane wave is separable once the angle-sum identity is applied::

        sin(kx·x + ky·y + p) = sin(kx·x + p)·cos(ky·y)
                             + cos(kx·x + p)·sin(ky·y)

    Every factor there is one-dimensional. Stacking the layers turns the whole
    sum into a single ``(dr, 2Q) @ (2Q, dc)`` matrix multiply, so the per-frame
    trigonometry is ``2·Q·dc`` values — about thirteen thousand at 400x100
    rather than 2.5 million — and BLAS does the rest. Adding layers widens a
    matrix multiply instead of adding a pass over the grid; even the storm
    layers below are just a wider matrix for the few frames they are alive.

    That is also why the phases can move at all. Dither had to bake its sine
    fields precisely because recomputing them was most of its cost; baked
    fields cannot have a moving phase. Factoring the wave instead of baking it
    buys the motion back.

    **Beats throw rings.** On an onset a ring is released at the centre and
    travels outward, added into the field before thresholding, so a hit
    arrives as a visible wavefront crossing the texture rather than a global
    brightening. Rings are held in scratch with a radius of their own and
    retire off the edge; the field's own motion is unaffected by them, so a
    beat reads as something passing *through* the surface. The falloff is a
    triangular window rather than a gaussian — over a dot grid this size the
    difference is invisible and ``exp`` is not. A ring is a hard wall, not a
    soft bump: a bright crest with a darker edge just ahead of it, so the hit
    reads as a shockwave crossing the grain. Up to eight rings can be in
    flight, and when the slots run out the oldest ring is retired rather than
    a new hit going unanswered.

    **Hits go inverted.** An onset also kicks the threshold itself. For about
    a tenth of a second the blue-noise tile is inverted — the field's troughs
    light instead of its crests, the texture itself flips — then the threshold
    stays pulled down for a few more tenths of a second as a bloom, so a beat
    is a flash that tears through the whole field before settling back to the
    texture. The same kick also speeds the phase of every wave, base and storm
    alike: hits drive the *motion* as well as the brightness, so the field
    visibly lurches on the beat.

    **Bass bends the field.** The low end does not just brighten the field, it
    deforms it. A low-frequency phase offset is added along each axis, its
    amplitude set by the bottom of the spectrum and its phase advancing on its
    own clock, so the waves wrinkle and writhe with the bass. The offset is
    one-dimensional — a per-column and per-row phase shift, scaled per layer
    by how far that layer's direction runs along the axis — so it is folded
    into the same factored sine/cosine passes and costs nothing structural.

    **Storm layers.** A transient wakes extra waves. Six finer-grained layers
    sit at the half-angles between the base directions, normally silent; each
    onset wakes one of them round-robin and it decays over a few tenths of a
    second, carrying its own fast phase. After a hit the crosshatch visibly
    re-orients into a denser, sharper weave, then settles — a busy passage is
    structurally different from a loud one, not just faster.

    **Blue noise, not Bayer.** Dither thresholds against an ordered Bayer
    matrix, which is what gives it its newspaper crosshatch. This one
    thresholds against a blue-noise tile: white noise, high-passed in the
    frequency domain, then rank-mapped back to a uniform distribution so it is
    still a fair threshold. The grain that produces is isotropic and has no
    preferred direction, so a moving field reads as an organic stipple rather
    than a moving screen door — Bayer's regular lattice beats against motion
    and produces crawling moiré, which is exactly the artefact to avoid once
    the field is no longer still.

    The tile is laid down by absolute dot position on both axes, like Dither's
    Bayer tiling and for the same reason: it keeps the surface one continuous
    skin instead of restarting at a row boundary or folding on a mirror line.

    Colour walks the ramp by each cell's strongest dot, so the crest of a
    passing ring sits highest in the ramp.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 8 or dc < 8:
        return empty(ctx.w, ctx.h)

    def build():
        # Normalised coordinates, so the texture keeps its structure at any
        # terminal size rather than getting finer as the grid grows. Kept in
        # the scratch tuple as well as folded into ax/ay, because the bass
        # warp phase offsets are evaluated over them every frame.
        x = (np.arange(dc, dtype=np.float32) - np.float32((dc - 1) / 2.0)) / np.float32(max(dc - 1, 1))
        y = (np.arange(dr, dtype=np.float32) - np.float32((dr - 1) / 2.0)) / np.float32(max(dr - 1, 1))

        # The layer bank: eight base directions plus the storm layers, all
        # factored the same way. The bank is a few hundred one-dimensional
        # values, so precomputing the whole thing costs nothing; per frame
        # only the live subset is touched.
        #
        # Cycles across the field, not per dot. Dither's docstring records
        # what happens when this is read as a fraction of the dot count: the
        # high bands land at three dots per cycle, which is speckle rather
        # than structure and reads as a static mode. Three to seventeen
        # cycles is broad swells through fine grain, all of it visible.
        ax = np.empty((LAYERS + STORM, dc), dtype=np.float32)
        ay = np.empty((LAYERS + STORM, dr), dtype=np.float32)
        wqx = np.empty(LAYERS + STORM, dtype=np.float32)
        wqy = np.empty(LAYERS + STORM, dtype=np.float32)
        for q in range(LAYERS + STORM):
            if q < LAYERS:
                th = math.pi * q / LAYERS
                k = 2.0 * math.pi * (1.4 + 5.2 * (q / max(LAYERS - 1, 1)))
            else:
                # Half-steps between the base angles, so the storm weave cuts
                # across the base crosshatch instead of reinforcing it, and a
                # finer grain than the base can reach.
                th = math.pi * (2 * (q - LAYERS) + 1) / (2 * LAYERS)
                k = 2.0 * math.pi * (6.5 + 2.5 * ((q - LAYERS) % 3))
            ax[q] = np.float32(math.cos(th) * k) * x
            ay[q] = np.float32(math.sin(th) * k) * y
            # How much a layer's direction runs along each axis; the bass
            # warp moves a layer along its own travel direction.
            wqx[q] = abs(math.cos(th))
            wqy[q] = abs(math.sin(th))

        # Distance from centre, quantised once into a lookup index.
        #
        # The rings only ever depend on a dot's radius, so evaluating them
        # per ring over the dot grid is wasted work: it was six passes over
        # several hundred thousand dots *per ring*, and with four in flight
        # that alone pushed the worst frame to 12 ms -- past the 11 ms
        # slow-mode threshold, where the widget starts reusing every other
        # frame. Binning the radius here turns the whole ring stack into one
        # small profile plus a single gather, whatever the ring count.
        rad = np.sqrt(x[None, :] ** 2 + y[:, None] ** 2).astype(np.float32)
        ridx = np.clip(rad * np.float32((RAD_BINS - 1) / RAD_MAX),
                       0, RAD_BINS - 1).astype(np.int32)

        return ax, ay, wqx, wqy, x, y, ridx, _blue_noise(TILE), _tile_index(dr, dc)

    ax, ay, wqx, wqy, xn, yn, ridx, tile, (ti, tj) = ctx.scratch("dither_storm", build)

    st = ctx.state.setdefault("dither_storm_st", {
        "phase": np.zeros(LAYERS, dtype=np.float32),
        "tp": np.zeros(STORM, dtype=np.float32),        # storm layer phases
        "tq": np.zeros(STORM, dtype=np.float32),        # storm layer amplitudes
        "storm_i": 0,                                   # round-robin wake index
        "r": np.zeros(RING_SLOTS, dtype=np.float32),    # ring radii; 0 means the slot is free
        "a": np.zeros(RING_SLOTS, dtype=np.float32),    # ring amplitudes
        "inv": np.float32(0.0),                         # threshold inversion, 0..1
        "bloom": np.float32(0.0),                       # threshold pull-down, 0..1.3
        "kick": np.float32(0.0),                        # phase-rate boost, 0..1.3
        "wp_x": np.float32(0.0),                        # bass warp phases
        "wp_y": np.float32(0.0),
    })

    # ── the band weights ──
    # Area-averaged down to one weight per layer, so widening the terminal
    # never makes a layer pop between values.
    amp = resample_bands(ctx.bands, LAYERS).astype(np.float32)
    total = float(amp.sum())
    # Normalised by the band total so the mix rotates with the music instead
    # of saturating: what matters to the texture is the *shape* of the
    # spectrum, not how loud it happens to be. Level comes back in below as
    # depth, where it belongs.
    amp = amp / np.float32(max(total, 1e-3))

    dt = np.float32(max(ctx.dt, 0.0))

    # ── bass warp ──
    # The bottom of the spectrum sets how far the field's phase is bent along
    # each axis; the warp phases advance on their own clocks, so even a steady
    # bass note keeps the field writhing instead of freezing into a bent
    # static pattern. Both axes share the same band, at different rates so the
    # two bends never lock into a standing figure.
    bass = ctx.range(0.0, 0.30)
    st["wp_x"] = (st["wp_x"] + np.float32(0.7 + 5.5 * bass) * dt) % np.float32(2.0 * math.pi)
    st["wp_y"] = (st["wp_y"] + np.float32(0.9 + 7.0 * bass) * dt) % np.float32(2.0 * math.pi)
    wx_off = np.float32(2.2 * bass) * np.sin(np.float32(2.0 * math.pi * 1.5) * xn + st["wp_x"])
    wy_off = np.float32(2.2 * bass) * np.sin(np.float32(2.0 * math.pi * 1.5) * yn + st["wp_y"])

    # ── onsets: flash, storm, rings ──
    # ctx.onsets is the count for THIS frame. Never difference ctx.onset_seq
    # privately: scratch survives a mode switch, so a mode that keeps its own
    # last-seen counter replays every beat that played while it was not
    # drawing. The contract is at spektr/modes/__init__.py:70-91.
    if ctx.onsets:
        s = np.float32(min(1.0, ctx.onset_strength))
        # Threshold inversion: for roughly a tenth of a second the tile is
        # lerped toward its own inverse, so the field's troughs light instead
        # of its crests — the texture itself flips. The bloom then keeps the
        # threshold pulled down for a few more tenths, and the kick speeds
        # every phase, so a hit is a flash and a lurch, not just a brightening.
        st["inv"] = max(st["inv"], np.float32(0.5 + 0.5 * s))
        st["bloom"] = min(np.float32(1.3), st["bloom"] + np.float32(0.45 + 0.55 * s))
        st["kick"] = min(np.float32(1.3), st["kick"] + np.float32(0.4 + 0.6 * s))

        # One storm layer per hit, round-robin. The amplitude caps a little
        # past one so back-to-back hits read as a pile-up rather than a clamp.
        for _ in range(ctx.onsets):
            i = st["storm_i"]
            st["tq"][i] = min(np.float32(1.2), st["tq"][i] + np.float32(0.55 + 0.45 * s))
            st["storm_i"] = (i + 1) % STORM

        # Rings fill the free slots, retiring the oldest live ring when a
        # burst outruns the slot count — a drum fill must read as a drum
        # fill, not be throttled to one ring at a time.
        need = ctx.onsets
        free = np.flatnonzero(st["r"] <= 0.0)
        if need > free.size:
            drop = np.argsort(st["r"])[-(need - free.size):]
            st["r"][drop] = 0.0
            free = np.flatnonzero(st["r"] <= 0.0)
        for i in free[:need]:
            st["r"][i] = np.float32(1e-3)
            st["a"][i] = np.float32(1.3 * (0.55 + 1.15 * s))

    st["inv"] = max(np.float32(0.0), st["inv"] - np.float32(8.5) * dt)
    st["bloom"] = max(np.float32(0.0), st["bloom"] - np.float32(2.3) * dt)
    st["kick"] = max(np.float32(0.0), st["kick"] - np.float32(3.5) * dt)

    # Storm layers decay; their phases keep advancing on their own fast clock,
    # boosted by the kick, so the wake left by a hit keeps moving as it fades.
    st["tq"] *= np.exp(-dt / np.float32(0.32))
    st["tp"] = (st["tp"] + (np.float32(16.0) + np.float32(30.0) * st["kick"]) * dt) % np.float32(2.0 * math.pi)

    # Each layer drifts at a rate set by its own band, kicked hard by any
    # onset. A band carrying nothing barely moves, so silence is a slow field
    # rather than a still one, and a band that suddenly fills visibly takes
    # off. Wrapped so the phase cannot grow until float32 loses resolution in
    # the fractional part -- a mode left running for hours would otherwise
    # start stepping its own animation coarsely.
    st["phase"] += (np.float32(0.9) + amp * np.float32(34.0) + np.float32(16.0) * st["kick"]) * dt
    st["phase"] %= np.float32(2.0 * math.pi)

    # ── rings ──
    live = st["r"] > 0.0
    if live.any():
        st["r"][live] += (np.float32(1.5) + np.float32(1.4) * ctx.energy) * dt
        # Retirement is measured, not guessed. x and y are normalised to
        # +/-0.5, so ``rad`` tops out at sqrt(0.5) ~ 0.707 in the corners:
        # a ring is fully off the grid just past that, and anything larger
        # holds a slot on a ring nobody can see. This first read 1.6, which
        # is more than twice the far corner -- the ring spent most of its
        # life outside the frame and beats read as a faint flicker near the
        # centre rather than a wave crossing the field.
        gone = st["r"] > np.float32(0.78)
        st["r"][gone] = 0.0
        st["a"][gone] = 0.0

    # ── the field, in one matrix multiply ──
    # The live set is the eight base layers plus whatever storm layers a
    # recent hit woke, so the matrix is a little wider for a few frames after
    # an onset and no wider at all the rest of the time.
    live_t = np.flatnonzero(st["tq"] > np.float32(0.02))
    sel = np.concatenate((np.arange(LAYERS, dtype=np.int32), LAYERS + live_t))
    ph = np.concatenate((st["phase"], st["tp"][live_t]))
    am = np.concatenate((amp, st["tq"][live_t]))
    # The bass warp rides on the phase: each layer bends along its own axis
    # by wq·offset — still one-dimensional, still factored, still one GEMM.
    # Both warp offsets are laid along the layer axis the same way: the per-
    # layer weight is the column and the offset is the row. wy_off is a 1-D
    # array over dot ROWS, so it broadcasts as [None, :] here exactly as
    # wx_off does over columns -- writing it as [:, None] makes numpy try to
    # line the layer count up against the row count and the mode dies with
    # "operands could not be broadcast together with shapes (8,1) (96,1)"
    # at every grid size.
    px = ax[sel] + ph[:, None] + wqx[sel, None] * wx_off[None, :]
    py = ay[sel] + wqy[sel, None] * wy_off[None, :]
    field = _field(px, py, am)

    hot = np.flatnonzero(st["r"] > 0.0)
    if hot.size:
        # The rings are summed over RAD_BINS radii -- a few hundred values --
        # and then read back with one gather. Triangular window rather than a
        # gaussian: over a grid this size the shapes are indistinguishable and
        # ``exp`` is not free.
        centres = np.linspace(0.0, RAD_MAX, RAD_BINS, dtype=np.float32)
        profile = np.zeros(RAD_BINS, dtype=np.float32)
        for i in hot:
            r = st["r"][i]
            crest = np.float32(1.0) - np.clip(np.abs(centres - r) * np.float32(5.0), 0.0, 1.0)
            inner = np.float32(1.0) - np.clip(np.abs(centres - (r - np.float32(0.06))) * np.float32(5.0), 0.0, 1.0)
            outer = np.float32(1.0) - np.clip(np.abs(centres - (r + np.float32(0.06))) * np.float32(5.0), 0.0, 1.0)
            # A hard crest with a faint interior glow and a darker edge just
            # ahead of it: the ring reads as a shockwave passing through the
            # grain rather than a soft brightening.
            profile += crest * st["a"][i]
            profile += inner * st["a"][i] * np.float32(0.5)
            profile -= outer * st["a"][i] * np.float32(0.35)
        field += profile[ridx]

    # ── to one bit ──
    # Depth is what loudness drives: a louder passage is a deeper, denser
    # field and a quiet one flattens toward dark. The swing is deliberately
    # wide — loud is near-solid — which is the point of a storm.
    lvl = np.float32(min(1.0, total / max(LAYERS * 0.55, 1e-3)))
    field *= np.float32(0.40) + np.float32(1.35) * lvl
    # The baseline is what silence looks like, and it is a dot density rather
    # than a cell count: at 0.30 roughly a third of dots survive the
    # threshold, and since a braille cell carries eight of them that lights
    # 94% of *cells* -- a silent track read as a solid sheet. 0.07 is
    # near-black but still an even speckle, which is the quiet end of the
    # swing; it never reads as a black screen or a solid sheet.
    field += np.float32(0.07) + np.float32(0.66) * lvl

    th = tile[ti[:, None], tj[None, :]]
    # The onset flash lives here: while inv > 0 the tile is lerped toward its
    # own inverse, so the field's troughs light instead of its crests, and the
    # bloom pulls the whole threshold down — a hit blooms the field toward
    # solid, then settles back to the texture.
    th = th * (np.float32(1.0) - np.float32(2.0) * st["inv"]) + st["inv"] - np.float32(0.5) * st["bloom"]
    lit = field > th

    codes = pack_braille(lit)

    # Quantised before ramping, and the reason is downstream of this mode.
    #
    # ``make_strips`` run-length encodes each row into spans of one colour, so
    # what it costs depends on how many times the ramp index CHANGES along a
    # row, not on the grid size. A continuous field lands on a different index
    # almost every cell, and measured in tests/bench.py at 400x100 that put
    # strips at 8.61 ms against plain Dither's 1.02 ms -- a total of 13.35 ms,
    # past the 11 ms slow-mode threshold, where the widget starts reusing every
    # other frame and the mode reads as stuttering. The build here was never
    # the problem; it is 3.64 ms against Dither's 3.23 ms.
    #
    # This is the cost recorded in Chladni Flow's docstring, reached from a
    # different direction, and the fix is the same one: fold the field onto a
    # few levels first so rows come out as runs. Ten steps is enough that the
    # texture still reads as shaded rather than posterised.
    val = np.clip(cell_max(np.where(lit, field, np.float32(0.0))), 0.0, 1.0)
    val = np.floor(val * np.float32(COLOUR_STEPS)) * np.float32(1.0 / COLOUR_STEPS)
    cidx = ctx.ramp(val)
    return codes, cidx


def _field(px: np.ndarray, py: np.ndarray, amp: np.ndarray) -> np.ndarray:
    """Sum the travelling waves as one ``(dr, 2Q) @ (2Q, dc)`` product.

    ``sin(kx·x + ky·y + p)`` expands to ``sin(kx·x+p)·cos(ky·y) +
    cos(kx·x+p)·sin(ky·y)``; stacking the layers along the shared axis makes
    the whole weighted sum a single GEMM. The amplitude rides on the x half so
    it is folded in by the same multiply rather than scaling the result after.

    ``px`` and ``py`` are the fully assembled per-layer phase terms — base
    phase, storm phase and the bass warp all arrive already folded in — so
    everything trigonometric here is one-dimensional — ``Q·dc`` and ``Q·dr``
    values — which is the entire reason the phases are allowed to move.
    """
    a = amp[:, None]
    bs = np.sin(px) * a
    bc = np.cos(px) * a
    left = np.concatenate((np.cos(py), np.sin(py)), axis=0).T   # (dr, 2Q)
    right = np.concatenate((bs, bc), axis=0)                    # (2Q, dc)
    return left @ right


def _blue_noise(n: int) -> np.ndarray:
    """A blue-noise threshold tile in [0, 1), built once per grid size.

    White noise, high-passed in the frequency domain, then rank-mapped back
    onto a uniform distribution. The rank step matters: high-passing alone
    leaves a roughly gaussian spread, and thresholding a field against a
    non-uniform mask biases the density it produces. Ranking restores a fair
    threshold while keeping the *arrangement* the filter produced, which is
    the part that makes it blue.

    Blue rather than white because the point is that no low-frequency clumps
    survive: white noise thresholds into visible blotches, blue noise into an
    even stipple that stays even while the field underneath it moves.
    """
    rng = np.random.default_rng(0x5EED)          # fixed, so a size looks the same every run
    w = rng.standard_normal((n, n))

    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    r = np.sqrt(fy * fy + fx * fx)
    # Suppress the low frequencies that produce clumping. r itself is a
    # gentle enough ramp to leave some mid energy, which keeps the grain from
    # looking like a regular lattice.
    w = np.real(np.fft.ifft2(np.fft.fft2(w) * r))

    flat = w.ravel()
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(flat.size, dtype=np.float32)
    ranks[order] = (np.arange(flat.size, dtype=np.float32) + 0.5) / flat.size
    return ranks.reshape(n, n)


def _tile_index(dr: int, dc: int) -> tuple[np.ndarray, np.ndarray]:
    """Absolute dot position modulo the tile, precomputed per size.

    Absolute rather than per-row or mirrored, so the threshold runs
    continuously across the whole grid — the same reason Dither tiles its
    Bayer matrix this way. A mirror line is precisely the seam the eye finds.
    """
    return (np.arange(dr, dtype=np.int32) % TILE,
            np.arange(dc, dtype=np.int32) % TILE)