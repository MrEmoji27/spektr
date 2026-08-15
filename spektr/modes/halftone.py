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
#:
#: Ten until the strip cost was measured against the alternatives on a live
#: frame sequence at 400x100: 10 steps is 4.93 ms and 7734 segments, 8 is
#: 3.44 ms and 5936, 6 is 2.80 ms and 4559. Colour here rides a one-bit
#: stipple that is already carrying the detail, so the ramp is doing less
#: work than the number suggests and 8 is not visibly flatter than 10 — 6 is,
#: which is where this stops. Smoothing the field before quantising was tried
#: instead and is worse on both counts: a 3-tap blur puts values *between* the
#: existing levels and took the same sequence to 8.78 ms and 15811 segments.
COLOUR_STEPS = 8

#: Target fraction of *dots* lit, at rest and at full level, plus how far a
#: flash lifts it. These are the mode's density budget — see the note by the
#: ``field.mean()`` correction for why density is aimed at rather than left to
#: emerge from the sum of everything upstream.
#:
#: Dots, not cells, and the difference is the whole reason these numbers look
#: low. A braille cell carries eight dots and lights if any one of them
#: survives, so a uniform 0.30 dot density already lights 94% of cells.
#:
#: Calibrated against plain Dither, which is the mode this one is a companion
#: to and which reads correctly: measured over a level sweep it runs 0.18
#: dots per cell when quiet, 1.33 at half level and 2.75 when loud. A first
#: attempt here at 0.22/0.30 gave 2.80 -> 3.87 — denser than Dither at its
#: loudest even in near-silence, and a range of one dot per cell where Dither
#: has two and a half. Wrong on both counts: nothing to see at the quiet end
#: and nowhere to go at the loud one.
#:
#: 0.06 at rest is a sparse stipple rather than the black screen the mode's
#: own docstring rules out, and the ceiling below keeps the loudest bar of
#: the loudest track from going solid.
_REST_DENSITY = 0.06
_LOUD_DENSITY = 0.30
_FLASH_DENSITY = 0.14


@mode("Dither Storm", group="fields",
      blurb="Dither's one-bit crosshatch, but the field moves — each band drives its own wave, and beats throw rings through it")
def dither_storm(ctx: Ctx):
    """Dither Storm with its density held. See :func:`_dither_storm`."""
    return _dither_storm(ctx, extreme=False)


@mode("Dither Storm Extreme", group="fields",
      blurb="Dither Storm with nothing holding it back — hits pile up and a dense passage blows the field to white")
def dither_storm_extreme(ctx: Ctx):
    """Dither Storm with every governor removed. See :func:`_dither_storm`.

    Same field, same rings, same blue noise; what differs is that the onset
    accumulators add instead of taking a maximum, the bloom pulls the
    threshold down directly, and nothing aims the dot density. Each of those
    is a knob pushing on how much of the frame lights, none of them knows
    what the others are doing, and the sum is a function of *beat rate*
    rather than of a beat.

    In the plain mode that is a bug and it is fixed. Here it is the point.
    Measured over a beat-density sweep at full level, dots lit:

        beats/min      plain      extreme
             none        42%          59%
              120        38%          75%
              160        38%          84%
              480        34%          99%

    A sparse groove looks close to the plain mode. As the material gets
    busier the field loads up, the crosshatch fills in, and a drum fill takes
    it to a solid white sheet before it falls back — the visualiser
    equivalent of letting something clip. That is a legitimate thing to want
    on the right track and a terrible default, which is why it is two modes
    and not a setting.
    """
    return _dither_storm(ctx, extreme=True)


def _dither_storm(ctx: Ctx, extreme: bool = False):
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

    **Continuous inputs under the discrete ones.** Everything above is
    onset-driven, and an onset only exists on the frames where the peak
    picker committed to one. Two continuous fields sit underneath so the mode
    keeps answering to the music in the gaps: ``flux`` — the raw
    detection function, i.e. how percussive the signal is right now — adds to
    every layer's phase rate, so a dense passage the detector is conservative
    about still races; and ``beat_phase`` swells the field's depth on the
    pulse and lets it settle through the bar. Both are gated the way the Ctx
    contract requires — ``beat_phase`` on ``tempo_bpm``, which is 0.0 until a
    tempo is established and takes the phase to 0.0 with it.

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

        # The threshold laid out over the whole grid, gathered ONCE per size.
        #
        # This used to be ``tile[ti[:, None], tj[None, :]]`` evaluated every
        # frame, which is a two-dimensional fancy index producing a
        # (dr, dc) array — 0.91 ms at 400x100, measured, and the single
        # largest item in this mode's build. Nothing in it depends on the
        # audio: the tile is fixed at construction and the index arrays are
        # pure functions of the grid size, so it produced a bit-identical
        # array sixty times a second. What the beat actually moves is the
        # affine transform applied to it below, and that is two fused passes
        # over a cached buffer.
        ti, tj = _tile_index(dr, dc)
        th0 = _blue_noise(TILE)[ti[:, None], tj[None, :]].astype(np.float32)

        # Output buffers, sized once. Both are written in full every frame,
        # so there is nothing to clear and nothing stale to read.
        return (ax, ay, wqx, wqy, x, y, ridx, th0,
                np.empty((dr, dc), dtype=np.float32),
                np.empty((dr, dc), dtype=np.float32))

    ax, ay, wqx, wqy, xn, yn, ridx, th0, th_buf, lit_buf = ctx.scratch(
        "dither_storm", build)

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
        "bloom_dc": np.float32(0.0),                    # bloom's running mean
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
        # of its crests — the texture itself flips. The bloom then lifts the
        # field for a few more tenths, and the kick speeds every phase, so a
        # hit is a flash and a lurch, not just a brightening.
        #
        # **Every one of these is a max, not a sum, and that is the fix for
        # the mode falling apart on dense material.** They were ``+=`` with a
        # cap, which makes each accumulator a function of the beat *rate*
        # rather than of a beat: at one hit a second bloom averaged 0.19, at
        # four a second 0.97, at eight 1.14, because hits arrived faster than
        # the decay could clear them. Since bloom pulled the threshold down,
        # dot density rose with beat rate and kept rising — measured 59% of
        # dots at rest, 84% at 160 BPM, 99% at a drum fill, i.e. a solid sheet
        # with no texture left in it at exactly the moment the music was most
        # interesting.
        #
        # A max cannot pile up. Two hits close together give the same peak as
        # one, which is what "a hit is a flash" has to mean if it is to mean
        # anything at any tempo; what a faster rate now buys is more flashes,
        # not a brighter floor. The remaining rate dependence — the decay
        # never reaching zero between hits — is taken out downstream by
        # subtracting bloom's own running mean.
        st["inv"] = max(st["inv"], np.float32(0.5 + 0.5 * s))
        if extreme:
            st["bloom"] = min(np.float32(1.3), st["bloom"] + np.float32(0.45 + 0.55 * s))
            st["kick"] = min(np.float32(1.3), st["kick"] + np.float32(0.4 + 0.6 * s))
        else:
            st["bloom"] = max(st["bloom"], np.float32(0.45 + 0.55 * s))
            st["kick"] = max(st["kick"], np.float32(0.4 + 0.6 * s))

        # One storm layer per hit, round-robin, each refreshed rather than
        # topped up — same reason. Piling meant a busy passage pinned all six
        # layers near their cap and the storm weave became a constant, so
        # "busy is structurally different from loud" stopped being true right
        # where it mattered.
        for _ in range(ctx.onsets):
            i = st["storm_i"]
            if extreme:
                st["tq"][i] = min(np.float32(1.2), st["tq"][i] + np.float32(0.55 + 0.45 * s))
            else:
                st["tq"][i] = max(st["tq"][i], np.float32(0.55 + 0.45 * s))
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
    # Slower in the extreme mode, which is half of why it loads up: a bloom
    # that has not cleared when the next hit lands is a floor the next hit
    # adds to.
    st["bloom"] = max(np.float32(0.0),
                      st["bloom"] - np.float32(2.3 if extreme else 3.4) * dt)
    st["kick"] = max(np.float32(0.0), st["kick"] - np.float32(3.5) * dt)

    # Bloom's own slow average, which is what gets subtracted from it below.
    #
    # Capping the peak stops a *single* accumulator running away; it does not
    # stop the floor creeping, because at eight hits a second the decay never
    # returns bloom to zero between them and the residue is a DC offset that
    # still scales with rate. Removing the running mean makes the flash a
    # mean-zero excursion: dense material gets more flashes about a baseline
    # that has not moved, sparse material gets fewer, and the resting texture
    # is the same in both. This is the analyser's own trick — sensitivity
    # there is driven by overshoot rather than by level, for the same reason.
    #
    # ~1.5 s, i.e. slow against a beat and fast against a section, so a hit
    # reads fully but a change of density settles out within a bar or two.
    st["bloom_dc"] += (st["bloom"] - st["bloom_dc"]) * np.float32(
        1.0 - math.exp(-float(dt) / 1.5))

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
    #
    # ``flux`` is in here alongside the kick, and the two are answering
    # different questions. The kick is discrete: it exists only on frames
    # where the peak picker committed to an onset, and it decays from there.
    # Flux is the raw detection function — how percussive the signal is *right
    # now* — and it is continuous and safe at any frame rate. On material the
    # detector is conservative about (brushed drums, a busy mix where nothing
    # clears the adaptive threshold) the kick never fires and the field was
    # left crawling at its band rate through a passage a listener hears as
    # dense. Flux keeps the motion honest between hits; the kick still
    # supplies the lurch on the ones that land.
    st["phase"] += (
        np.float32(0.9)
        + amp * np.float32(34.0)
        + np.float32(16.0) * st["kick"]
        + np.float32(11.0) * np.float32(min(1.0, ctx.flux))
    ) * dt
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
    # Beat-locked depth. The rings and the bloom are onset-driven, so between
    # hits the depth was a pure function of level and the field breathed only
    # when something was detected. ``beat_phase`` is continuous and available
    # in the gaps, so the surface swells on the pulse and settles through the
    # bar whether or not the peak picker found that particular beat.
    #
    # Gated on ``tempo_bpm``: it is 0.0 until a tempo is established and takes
    # ``beat_phase`` to 0.0 with it, which read ungated is a permanent
    # on-the-beat swell — including through silence, where it is most wrong.
    if ctx.tempo_bpm > 0.0:
        lvl = np.float32(min(1.0, float(lvl) * (1.0 + 0.13 * (1.0 - ctx.beat_phase) ** 2)))
    field *= np.float32(0.40) + np.float32(1.35) * lvl

    # ── dot density, aimed rather than hoped for ──
    #
    # The threshold tile is uniform on [0, 1) by construction — that is what
    # the rank step in ``_blue_noise`` is for — so the expected fraction of
    # dots that survive the comparison is ``E[clip(field, 0, 1)]``. Density is
    # therefore something this mode can *set* rather than something that falls
    # out of everything upstream.
    #
    # The clip is not a formality and getting it wrong is a real error: this
    # first used the raw mean, which is only the density if the field already
    # lies inside [0, 1). It does not — it is a sum of travelling waves with
    # ring crests added on top and a long tail either side — and clipping
    # folds the negative tail up to 0 while the raw mean still counts it as
    # negative. Aiming the raw mean at 0.10 measured 0.25 in practice, i.e.
    # two and a half times the density asked for, worst at the quiet end
    # where the target is small against the spread.
    #
    # One Newton step closes it. ``d`` is the density the field would give as
    # it stands and ``inside`` is exactly d(density)/d(shift) — the fraction
    # of dots whose comparison a small shift could still change, since dots
    # already clipped at either end are decided. The floor on it keeps the
    # step finite when almost everything is saturated.
    #
    # It used to fall out, and it fell out badly. Depth, baseline, ring
    # amplitudes, how many storm layers happened to be awake and the bloom
    # offset on the threshold all pushed on density independently, none of
    # them knew what the others were doing, and the sum ran to a solid sheet:
    # 59% of dots at rest against a docstring claiming "near-black but still
    # an even speckle", and 99% on a drum fill. A one-bit medium has exactly
    # one currency and nothing was accounting for it.
    #
    # Aiming it also makes every other knob safe to touch. Ring crests, storm
    # weave and bass warp now redistribute the lit dots — which is the part
    # that carries structure — instead of adding to them.
    target = _REST_DENSITY + _LOUD_DENSITY * lvl

    # The flash rides on top as a mean-zero excursion, so it brightens
    # relative to whatever the recent baseline has been rather than adding to
    # it. Clipped because a section change can move the running mean faster
    # than 1.5 s and there is no reading of "flash" that should darken the
    # field below its resting density.
    flash = float(np.clip(st["bloom"] - st["bloom_dc"], 0.0, 1.0))
    target = min(0.50, target + _FLASH_DENSITY * flash)

    if extreme:
        # None of the above. The extreme mode keeps the original fixed
        # baseline, which is the knob that made density a function of beat
        # rate: nothing measures what the field is actually doing, so depth,
        # rings, storm layers and the bloom offset below all add up
        # unsupervised and a busy passage runs to white. Left in deliberately
        # — see the mode's own docstring for the numbers it produces.
        field += np.float32(0.07) + np.float32(0.66) * lvl
    else:
        np.clip(field, 0.0, 1.0, out=lit_buf)
        d = float(lit_buf.mean())
        inside = float(np.count_nonzero((field > 0.0) & (field < 1.0))) / field.size
        field += np.float32((target - d) / max(inside, 0.20))

    # The onset inversion lives here: while inv > 0 the tile is lerped toward
    # its own inverse, so the field's troughs light instead of its crests and
    # the texture itself flips.
    #
    # Density-neutral by construction, which is why it is the one onset effect
    # that can stay on the threshold. ``1 - th0`` is uniform on [0, 1) exactly
    # as ``th0`` is, so inverting changes *which* dots light and never how
    # many — the flip reads as a flip rather than as a flash, and the flash is
    # handled above where it can be accounted for.
    #
    # The tile itself is gathered once per size (see ``build``); this is the
    # only part that moves. Written into a scratch buffer with fused ops
    # rather than three fresh (dr, dc) temporaries a frame.
    np.multiply(th0, np.float32(1.0) - np.float32(2.0) * st["inv"], out=th_buf)
    if extreme:
        # The bloom pulls the whole threshold down as well, on top of the
        # inversion — a hit blooms the field toward solid rather than merely
        # flipping it. This is the other half of the pile-up: bloom is already
        # accumulating, and here it moves the bar every dot is measured
        # against, so a run of hits walks the entire frame toward lit.
        th_buf += st["inv"] - np.float32(0.5) * st["bloom"]
    else:
        th_buf += st["inv"]
    lit = field > th_buf

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
    # ``field * lit`` rather than ``np.where(lit, field, 0.0)``. The two are
    # bit-identical here — the false branch is a zero and the true branch is
    # ``field`` unchanged, which is exactly multiplication by the bool — but
    # ``where`` against a python scalar builds a fresh (dr, dc) array through
    # a slow path: 1.03 ms at 400x100 against 0.09 ms for a multiply into a
    # buffer that already exists. It was the largest single line in this
    # mode's frame.
    np.multiply(field, lit, out=lit_buf)
    val = np.clip(cell_max(lit_buf), 0.0, 1.0)
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