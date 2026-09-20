"""Full-field modes: waterfall, plasma, and level meters."""

from __future__ import annotations

import math

import numpy as np

from ..analysis import resample_bands
from ..render import (
    cell_mean,
    pack_braille,
)
from . import Ctx, empty, mode


@mode("Dither", group="fields", blurb="the spectrum printed as a newspaper halftone, in one-bit crosshatch")
def dither(ctx: Ctx):
    """The spectrum as a continuous two-dimensional field, thresholded to one
    bit by an ordered dither — so the whole frame is one textured surface,
    not a row of bars.

    The field is the sum of eight directional plane waves, one per band, at
    angles spreading around the circle. Each wave's amplitude is that band's
    level, so the *shape* of the spectrum steers the texture: heavy bass
    swells the low-frequency waves into broad slow undulations while the
    high bands draw fine grain on top, and the sum is normalized by the band
    total so the mix rotates with the music instead of saturating. The
    wavelengths run from a twentieth of the width up to a quarter of it, so
    every braille cell sees a different slice of the wave mix and the
    cross-hatch pattern varies across the frame instead of repeating. The
    level drives two scalars — a baseline lift and the texture depth — so a
    louder signal reads as a denser, deeper field and a quiet one as a flat,
    even grey.

    The one piece of state is a slow drift phase integrated through ``ctx.dt``,
    which rides a single low-frequency swell across the field so the crosshatch
    does not look like a frozen print and a quiet passage still has something
    to do. It is lighting, not subject: the pattern itself is a pure function
    of the spectrum. This docstring claimed the opposite — no clock, no state,
    a frozen spectrum giving a frozen frame — for as long as the drift has
    existed.

    The threshold is an s x s Bayer matrix tiled by *absolute* dot position
    on both axes. The tile alignment is what makes this a texture rather
    than stripes: because the threshold varies with the column inside every
    row, a row whose field is uniform still breaks into the cross-hatch
    pattern of the matrix, and because the field varies along x as well as
    y, no row ever resolves to a single solid line. Wherever the field is
    near the threshold the matrix turns the gradient into structured dots
    and wedges — the ordered-dither effect — and the density of the pattern
    stands in for the field's value: a grey-scale rendering of a surface
    that is only ever lit or unlit.

    One guard on top of that: a row whose field rides high across its whole
    width would saturate every dot. Each row's maximum threshold is known,
    so the top threshold cell of any such row is forced dark — a uniformly
    solid row becomes structurally impossible without dimming anything
    else, and the baseline keeps every row lit somewhere.

    There is deliberately no left-right mirroring here. A mirror line
    through the centre is exactly the seam the eye locks onto, splitting
    one surface into two panels; a dithered field should read as a single
    continuous skin, and the absolute tiling keeps it that way across the
    whole frame.

    Colour walks the ramp by the lit density of each cell, so brighter
    patches of the texture sit higher in the ramp.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 8 or dc < 8:
        return empty(ctx.w, ctx.h)

    def grid():
        # Eight directional plane waves, one per band. Band q runs at angle
        # pi*q/8; its wavelength goes from a twentieth of the width (fine
        # grain, high bands) up to a quarter of the width (broad swells,
        # low bands), scaled to the terminal so the texture keeps its
        # structure at any size.
        x = (np.arange(dc, dtype=np.float32) - np.float32((dc - 1) / 2.0)) / np.float32(max(dc - 1, 1))
        y = (np.arange(dr, dtype=np.float32) - np.float32((dr - 1) / 2.0)) / np.float32(max(dr - 1, 1))
        twopi = np.float32(2 * math.pi)
        ph = []
        for q in range(8):
            thq = np.float32(math.pi * q / 8.0)
            # Cycles across the field, not a fraction of the dot count.
            #
            # This read ``dc * (0.06 + 0.28 * q/7)``, which at a 400x100
            # terminal is 48 cycles for the lowest band and 272 for the
            # highest -- three dots per cycle. Nothing at that frequency is
            # visible as structure; it is the speckle failure recorded in
            # Plasma's docstring, and it is why the mode read as static.
            # The comment above it described the intent correctly ("a
            # twentieth of the width up to a quarter") and the arithmetic
            # did something else entirely.
            #
            # Three to seventeen cycles is broad swells for the bass through
            # fine grain for the treble, all of it coarse enough to see, and
            # it is resolution-independent because x and y are normalised.
            kq = np.float32(1.4 + 5.2 * (q / 7.0))
            wx = np.cos(thq) * twopi * kq
            wy = np.sin(thq) * twopi * kq
            # Stored already evaluated. The wave's geometry depends only on
            # the grid, so sin() of it is a constant for the whole life of
            # this size — but it was being recomputed every frame, eight
            # times over the dot grid. At 400x100 that is 2.56 million
            # transcendentals a frame and it was most of the mode's cost.
            # Only the band weights change, so the per-frame work is a
            # weighted sum of fixed fields, with no sin() in it at all.
            ph.append(np.sin(wx * x[None, :] + wy * y[:, None]).astype(np.float32))
        # The ordered-dither threshold: an s x s Bayer matrix (the standard
        # recursion, normalized to [0, 1)) tiled by ABSOLUTE dot position on
        # both axes, so the cross-hatch runs continuously across the whole
        # grid instead of restarting at each row or folding at the centre.
        s = 8 if (dr >= 16 and dc >= 16) else 4
        b = np.zeros((1, 1), dtype=np.float32)
        size = 1
        while size < s:
            b = np.block([[4 * b, 4 * b + 2], [4 * b + 3, 4 * b + 1]])
            size *= 2
        b = b / np.float32(size * size)
        th = b[np.arange(dr, dtype=np.int32) % s][:, np.arange(dc, dtype=np.int32) % s]
        # One contiguous (8, dr, dc) block so the per-frame combine is a
        # single tensordot rather than eight separate multiply-adds each
        # walking the whole grid.
        driftx = (np.arange(dc, dtype=np.float32) / np.float32(max(dc - 1, 1)))[None, :] * np.float32(1.5)
        return {"sinph": np.ascontiguousarray(np.stack(ph)), "driftx": driftx,
                "th": th, "th_rowmax": th.max(axis=1).astype(np.float32),
                "th_argmax": np.argmax(th, axis=1).astype(np.int32)}

    g = ctx.scratch("dither_grid", grid)

    # The field is the spectrum's own silhouette, not a wave interference
    # pattern.
    #
    # This mode used to sum eight directional plane waves. That produced a
    # texture, and the texture was the problem: an interference field is what
    # Plasma, Chladni and Maelstrom already draw, so dithering one only
    # changed how it was shaded, not what it was. A halftone is a printing
    # technique, and a printing technique needs a subject.
    #
    # The subject here is the spectrum, drawn the way a newspaper would print
    # it: the band profile as a filled silhouette, given a vertical tone ramp
    # so it is dense along the floor and thins toward its own upper edge, and
    # then thresholded to pure black and white. What survives is a picture of
    # the spectrum made entirely of crosshatch, which nothing else in the set
    # does.
    prof = resample_bands(ctx.bands, dc).astype(np.float32)
    # A little horizontal smoothing so band boundaries do not read as steps;
    # the halftone exaggerates any hard vertical edge into a visible seam.
    prof = (prof + np.roll(prof, 1) + np.roll(prof, -1)) * np.float32(1.0 / 3.0)

    v = (np.arange(dr - 1, -1, -1, dtype=np.float32) / np.float32(max(dr - 1, 1)))[:, None]
    # Height above the floor, in the same 0..1 units as the profile. Soft
    # rather than a hard cut, so the top edge of the silhouette dissolves
    # into dots instead of terminating in a line -- that dissolve is the
    # halftone's signature and the reason the edge reads as tone.
    edge = np.clip((prof[None, :] * np.float32(1.05) - v) * np.float32(6.0), 0.0, 1.0)

    # Tone inside the silhouette: densest at the floor, lighter toward the
    # top, so the fill carries a gradient for the dither to resolve rather
    # than being one flat grey.
    shade = np.float32(0.30) + np.float32(0.62) * (np.float32(1.0) - v)

    # A slow drift keeps the crosshatch from looking like a frozen print, and
    # gives quiet passages something to do. Integrated through ctx.dt, and
    # the wave that carries it is a single low-frequency swell rather than a
    # field of them -- it is lighting, not subject.
    ph = ctx.scratch("dither_drift", lambda: {"v": 0.0})
    ph["v"] += (0.05 + ctx.energy * 0.20) * max(ctx.dt, 0.0)
    swell = np.sin((g["driftx"] + np.float32(ph["v"])) * np.float32(2 * math.pi)) * np.float32(0.10)

    field = (edge * (shade + swell)).astype(np.float32)
    lit = field > g["th"]
    # guard: a row whose field stays above its own row-max threshold would
    # saturate every dot. The top threshold cell of such a row is forced
    # dark, which makes a uniformly solid row structurally impossible
    # without dimming anything else.
    hot = field.max(axis=1) > g["th_rowmax"]
    if bool(hot.any()):
        cols = g["th_argmax"]
        lit[hot, cols[hot]] = False
    codes = pack_braille(lit)

    # Colour follows the underlying field, not the dithered result.
    #
    # Reading it off the lit dots looks equivalent and is not, for two
    # reasons. Visually, the dither pattern is already carrying the texture;
    # colouring by lit density paints per-cell noise on top of it and fights
    # the halftone the mode exists to draw. And structurally it is the worst
    # possible input to the strip builder: neighbouring cells almost never
    # agree, so run-length encoding finds no runs and emits a segment per
    # cell. Measured at 400x100 that was 24.61 ms in make_strips alone,
    # against 15.79 ms to build the frame -- the mode was over the 16.7 ms
    # budget almost entirely on colour it did not need.
    #
    # Sampling the smooth field instead gives neighbours the same ramp index
    # over stretches, which is what run-length encoding is for.
    # Quantised to a few buckets before ramping, the same lesson Chladni's
    # docstring records. The field is a sum of eight waves, so even sampled
    # smoothly it crosses ramp buckets almost every cell, and neighbours that
    # never agree are neighbours run-length encoding cannot merge. Eight
    # levels is invisible against a one-bit texture -- the dither is already
    # doing the shading -- and collapses the segment count enough to bring
    # make_strips from 15.4 ms back to something ordinary.
    col = cell_mean(np.clip(field, 0.0, 1.0))
    np.rint(col * 8.0, out=col)
    idx = ctx.ramp(np.clip(col * 0.125, 0.0, 1.0))
    return codes, idx
