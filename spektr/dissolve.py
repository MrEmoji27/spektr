"""Dissolve one mode's frame into another's.

Shuffle used to cut. This mixes the two frames instead, dot by dot, in the
order :mod:`spektr.bluenoise` lays down: at progress ``p`` a dot belongs to the
incoming frame when its rank is below ``p``. Blue noise is what keeps that
even — with random ranks the change arrives in clumps and reads as tearing.

Two frame shapes exist and both are handled. A braille frame is ``(codes,
cidx)``, where every cell is eight dots packed into one codepoint, so a cell
can hold dots from both frames at once and the dissolve happens inside it. A
half-block or octant frame is ``(codes, fg, bg)``, where the glyph already
spends its subcells on colour; there the whole cell switches at once, which at
the sizes these modes run is still fine-grained enough to read as a dissolve
rather than a wipe.

Nothing here is per-frame expensive: the mask is cached, and each call is a
handful of whole-array operations.
"""
from __future__ import annotations

import numpy as np

from . import bluenoise
from .render import BRAILLE_BASE, BRAILLE_BITS, pack_braille

#: How long a dissolve lasts. Long enough to read as a change of scene rather
#: than a glitch, short enough that a shuffle interval still shows the mode.
SECONDS = 0.6

#: Size of the mask tiled over the frame. 32 is enough that the pattern does
#: not repeat visibly at terminal sizes, and it builds in about half a second.
MASK = 32


def ease(p: float) -> float:
    """Smoothstep: start and finish gently, travel quickly in the middle."""
    p = 0.0 if p < 0.0 else 1.0 if p > 1.0 else float(p)
    return p * p * (3.0 - 2.0 * p)


def _dot_mask(rows: int, cols: int, progress: float) -> np.ndarray:
    """Per dot: True where the incoming frame has arrived."""
    return bluenoise.tile(bluenoise.mask(MASK), rows, cols) < progress


def _braille_dots(codes: np.ndarray) -> np.ndarray:
    """Unpack braille codepoints into a (rows*4, cols*2) boolean dot grid."""
    h, w = codes.shape
    bits = np.where(
        (codes >= BRAILLE_BASE) & (codes < BRAILLE_BASE + 256),
        codes - BRAILLE_BASE,
        0,
    ).astype(np.int32)
    dots = np.zeros((h * 4, w * 2), dtype=bool)
    for dy in range(4):
        for dx in range(2):
            dots[dy::4, dx::2] = (bits & int(BRAILLE_BITS[dy, dx])) != 0
    return dots


def blend(old: tuple, new: tuple, progress: float) -> tuple:
    """The frame part way from ``old`` to ``new``.

    ``progress`` is 0 at the start and 1 at the end, eased by the caller or
    by :func:`ease`. Either frame may be a 2-tuple (braille) or a 3-tuple
    (half-block or octant). When the two frames are not the same shape there
    is nothing meaningful to mix — a braille cell and an octant cell do not
    share a subcell grid — so the frame simply changes over at the halfway
    point, where the eye is least likely to catch it.
    """
    if progress <= 0.0:
        return old
    if progress >= 1.0:
        return new

    codes_old, codes_new = old[0], new[0]
    if codes_old.shape != codes_new.shape:
        return new  # a resize mid-dissolve: there is nothing to blend against

    rows, cols = codes_old.shape
    same_shape = len(old) == len(new)
    braille = same_shape and len(old) == 2

    if not same_shape:
        return new if progress >= 0.5 else old

    if braille:
        dots = _dot_mask(rows * 4, cols * 2, progress)
        mixed = np.where(dots, _braille_dots(codes_new), _braille_dots(codes_old))
        codes = pack_braille(mixed)
        # A cell's colour follows whichever frame owns more of its dots, so
        # the colour arrives with the shape rather than ahead of it.
        new_share = dots.reshape(rows, 4, cols, 2).sum(axis=(1, 3))
        take_new = new_share >= 4
        return codes, np.where(take_new, new[1], old[1])

    cells = _dot_mask(rows, cols, progress)
    return tuple(np.where(cells, n, o) for o, n in zip(old, new))
