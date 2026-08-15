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

    Same family as Dither and a different animal. Dither sums eight *fixed*
    wave fields and re-weights them, so the texture restates the spectrum
    without ever moving on its own. Here each wave also has a phase that
    advances, at a rate set by the level of the band that owns it, so a busy
    band visibly races and a quiet one drifts. The spectrum chooses the
    texture in both modes; only this one lets it choose the motion too.

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
    matrix multiply instead of adding a pass over the grid.

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
    difference is invisible and ``exp`` is not.

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
        # terminal size rather than getting finer as the grid grows.
        x = (np.arange(dc, dtype=np.float32) - np.float32((dc - 1) / 2.0)) / np.float32(max(dc - 1, 1))
        y = (np.arange(dr, dtype=np.float32) - np.float32((dr - 1) / 2.0)) / np.float32(max(dr - 1, 1))

        # Cycles across the field, not per dot. Dither's docstring records
        # what happens when this is read as a fraction of the dot count: the
        # high bands land at three dots per cycle, which is speckle rather
        # than structure and reads as a static mode. Three to seventeen
        # cycles is broad swells through fine grain, all of it visible.
        ax = np.empty((LAYERS, dc), dtype=np.float32)
        ay = np.empty((LAYERS, dr), dtype=np.float32)
        for q in range(LAYERS):
            th = math.pi * q / LAYERS
            k = 2.0 * math.pi * (1.4 + 5.2 * (q / max(LAYERS - 1, 1)))
            ax[q] = np.float32(math.cos(th) * k) * x
            ay[q] = np.float32(math.sin(th) * k) * y

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

        return ax, ay, ridx, _blue_noise(TILE), _tile_index(dr, dc)

    ax, ay, ridx, tile, (ti, tj) = ctx.scratch("dither_storm", build)

    st = ctx.state.setdefault("dither_storm_st", {
        "phase": np.zeros(LAYERS, dtype=np.float32),
        "r": np.zeros(4, dtype=np.float32),      # ring radii; 0 means the slot is free
        "a": np.zeros(4, dtype=np.float32),      # ring amplitudes
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
    # Each layer drifts at a rate set by its own band. A band carrying
    # nothing barely moves, so silence is a slow field rather than a still
    # one, and a band that suddenly fills visibly takes off.
    st["phase"] += (np.float32(0.6) + amp * np.float32(26.0)) * dt
    # Wrapped so the phase cannot grow until float32 loses resolution in the
    # fractional part -- a mode left running for hours would otherwise start
    # stepping its own animation coarsely.
    st["phase"] %= np.float32(2.0 * math.pi)

    # ── rings ──
    # ctx.onsets is the count for THIS frame. Never difference ctx.onset_seq
    # privately: scratch survives a mode switch, so a mode that keeps its own
    # last-seen counter replays every beat that played while it was not
    # drawing. The contract is at spektr/modes/__init__.py:70-91.
    if ctx.onsets:
        free = np.flatnonzero(st["r"] <= 0.0)[:ctx.onsets]
        for i in free:
            st["r"][i] = np.float32(1e-3)
            st["a"][i] = np.float32(0.45 + 0.55 * min(1.0, ctx.onset_strength))

    live = st["r"] > 0.0
    if live.any():
        st["r"][live] += (np.float32(1.15) + np.float32(0.9) * ctx.energy) * dt
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
    field = _field(ax, ay, st["phase"], amp)

    hot = np.flatnonzero(st["r"] > 0.0)
    if hot.size:
        # The rings are summed over RAD_BINS radii -- a few hundred values --
        # and then read back with one gather. Triangular window rather than a
        # gaussian: over a grid this size the shapes are indistinguishable and
        # ``exp`` is not free.
        centres = np.linspace(0.0, RAD_MAX, RAD_BINS, dtype=np.float32)
        profile = np.zeros(RAD_BINS, dtype=np.float32)
        for i in hot:
            d = np.abs(centres - st["r"][i]) * np.float32(7.0)
            profile += (np.float32(1.0) - np.clip(d, 0.0, 1.0)) * st["a"][i] * np.float32(1.3)
        field += profile[ridx]

    # ── to one bit ──
    # Depth is what loudness drives: a louder passage is a deeper, denser
    # field and a quiet one flattens toward even grey, which is the same
    # bargain Dither makes.
    lvl = np.float32(min(1.0, total / max(LAYERS * 0.55, 1e-3)))
    field *= np.float32(0.55) + np.float32(0.85) * lvl
    # The baseline is what silence looks like, and it is a dot density rather
    # than a cell count: at 0.30 roughly a third of dots survive the
    # threshold, and since a braille cell carries eight of them that lights
    # 94% of *cells* -- a silent track read as a solid sheet. 0.16 is an even
    # grey with the texture still legible in it, which is what a quiet
    # passage should be.
    field += np.float32(0.16) + np.float32(0.34) * lvl

    th = tile[ti[:, None], tj[None, :]]
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


def _field(ax: np.ndarray, ay: np.ndarray, phase: np.ndarray, amp: np.ndarray) -> np.ndarray:
    """Sum the travelling waves as one ``(dr, 2Q) @ (2Q, dc)`` product.

    ``sin(kx·x + ky·y + p)`` expands to ``sin(kx·x+p)·cos(ky·y) +
    cos(kx·x+p)·sin(ky·y)``; stacking the layers along the shared axis makes
    the whole weighted sum a single GEMM. The amplitude rides on the x half so
    it is folded in by the same multiply rather than scaling the result after.

    Everything trigonometric here is one-dimensional — ``Q·dc`` and ``Q·dr``
    values — which is the entire reason the phases are allowed to move.
    """
    px = ax + phase[:, None]
    a = amp[:, None]
    bs = np.sin(px) * a
    bc = np.cos(px) * a
    left = np.concatenate((np.cos(ay), np.sin(ay)), axis=0).T   # (dr, 2Q)
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
