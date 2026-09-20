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
from .render import BRAILLE_BASE, BRAILLE_BITS, SPACE, pack_braille

#: How long a morph lasts. Long enough to be seen as a change of scene —
#: half a second reads as a glitch when the two modes look nothing alike —
#: and short enough that a fifteen-second shuffle still shows the mode.
SECONDS = 0.9

#: How far the picture gathers in on itself at the half-way point, as a share
#: of its own height. The travel alone is invisible between two modes that
#: happen to fill the frame the same way; drawing in and blooming back out is
#: what makes a change of family read as one picture becoming another.
GATHER = 0.22

#: When the handover from old ink to new runs, as a share of the morph. The
#: first third is the old picture bending into the new shape, the last third
#: is the new one settling; the swap happens in between, where the two are
#: closest and it is least visible.
HANDOVER = (0.30, 0.75)

#: Size of the mask tiled over the frame. 32 is enough that the pattern does
#: not repeat visibly at terminal sizes, and it builds in about half a second.
MASK = 32


def _handover(progress: float) -> float:
    """Progress through the swap itself, held back until the ink has moved."""
    lo, hi = HANDOVER
    return ease((progress - lo) / (hi - lo))


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


def _profile(lit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where each column's ink sits: how much, its centre, and its spread.

    Three numbers a column at a time is enough to move one picture onto
    another: a bar that is tall on the left and short on the right becomes a
    wave by sliding and stretching each column, which is what the eye reads as
    the picture changing shape rather than being replaced.
    """
    rows = lit.shape[0]
    ys = np.arange(rows, dtype=np.float32)[:, None]
    count = lit.sum(axis=0).astype(np.float32)
    safe = np.maximum(count, 1.0)
    centre = (lit * ys).sum(axis=0) / safe
    var = (lit * (ys - centre) ** 2).sum(axis=0) / safe
    spread = np.sqrt(np.maximum(var, 0.0))
    # An empty column has no centre of its own; it keeps the grid's, so ink
    # arriving into it grows from the middle instead of leaping in from row 0.
    empty = count < 1.0
    centre = np.where(empty, (rows - 1) * 0.5, centre)
    # A single lit dot has no spread; one dot-row keeps the scale finite.
    spread = np.maximum(spread, 1.0)
    return count, centre.astype(np.float32), spread.astype(np.float32)


def _warp(arrays: tuple, centre: np.ndarray, spread: np.ndarray,
          centre_t: np.ndarray, spread_t: np.ndarray) -> tuple:
    """Resample each column so its ink sits at ``centre_t``/``spread_t``.

    Returns the moved arrays and the mask of cells the move actually reached.
    What to put outside that mask is the caller's business: blank for ink,
    but a colour or a glyph code has no zero that means "nothing", and filling
    those with 0 paints real ramp colour and invalid codepoints into the gap.
    """
    rows = arrays[0].shape[0]
    ys = np.arange(rows, dtype=np.float32)[:, None]
    src = centre + (ys - centre_t) * (spread / spread_t)
    idx = np.rint(src).astype(np.int32)
    inside = (idx >= 0) & (idx < rows)
    idx = np.clip(idx, 0, rows - 1)
    return tuple(np.take_along_axis(a, idx, axis=0) for a in arrays), inside


def _morph(old: tuple, new: tuple, lit_old: np.ndarray, lit_new: np.ndarray,
           progress: float) -> tuple[tuple, tuple]:
    """Both pictures, each moved a share of the way to the other's shape.

    A column with ink on only one side has nothing to move towards, so it
    keeps its own geometry and simply fades. Without that, ink heading into an
    empty column gets squeezed onto a single row on its way out, which looks
    like the picture collapsing rather than changing.
    """
    n_old, c_old, s_old = _profile(lit_old)
    n_new, c_new, s_new = _profile(lit_new)
    lonely = (n_old < 1.0) | (n_new < 1.0)
    c_new = np.where(lonely, c_old, c_new)
    s_new = np.where(lonely, s_old, s_new)
    c_t = c_old + (c_new - c_old) * progress
    s_t = s_old + (s_new - s_old) * progress
    # Gather in, then bloom back out: a half-cycle of sine, strongest at the
    # half-way point and nothing at either end, so the morph starts and ends
    # on the real picture.
    s_t = s_t * (1.0 - GATHER * float(np.sin(np.pi * progress)))
    return (_warp(old, c_old, s_old, c_t, s_t),
            _warp(new, c_new, s_new, c_t, s_t))


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
        dots_old = _braille_dots(codes_old)
        dots_new = _braille_dots(codes_new)
        ((moved_old,), keep_old), ((moved_new,), keep_new) = _morph(
            (dots_old,), (dots_new,), dots_old, dots_new, progress,
        )
        # Ink that moved off the frame simply is not there any more.
        warped_old = moved_old & keep_old
        warped_new = moved_new & keep_new
        # Shapes are aligned now, so the handover from one to the other lands
        # on ink that is already in the right place: it reads as the picture
        # becoming the new one rather than as dots being swapped underneath it.
        arrived = _dot_mask(rows * 4, cols * 2, _handover(progress))
        mixed = np.where(arrived, warped_new, warped_old)
        codes = pack_braille(mixed)
        # Colour travels with the shape: the ramp indices are moved by the
        # same per-column slide, judged on where each cell's ink is, and a
        # cell takes the incoming colour once most of its dots have arrived.
        cells_old = dots_old.reshape(rows, 4, cols, 2).any(axis=(1, 3))
        cells_new = dots_new.reshape(rows, 4, cols, 2).any(axis=(1, 3))
        ((moved_cidx_old,), cell_keep_old), ((moved_cidx_new,), cell_keep_new) = _morph(
            (old[1],), (new[1],), cells_old, cells_new, progress,
        )
        # A cell the move did not reach keeps its own colour rather than
        # taking ramp index 0, which is a colour like any other.
        cidx_old = np.where(cell_keep_old, moved_cidx_old, old[1])
        cidx_new = np.where(cell_keep_new, moved_cidx_new, new[1])
        new_share = arrived.reshape(rows, 4, cols, 2).sum(axis=(1, 3))
        return codes, np.where(new_share >= 4, cidx_new, cidx_old)

    # Half-block and octant frames have no free subcells to mix inside a cell,
    # so the morph happens cell by cell: the columns still slide and stretch
    # into place, and the handover is per cell rather than per dot.
    lit_old = codes_old != SPACE
    lit_new = codes_new != SPACE
    (moved_old, keep_old), (moved_new, keep_new) = _morph(
        old, new, lit_old, lit_new, progress)
    # Outside the move: a blank cell where the glyph was, and the cell's own
    # colour everywhere else.
    blanks = (SPACE,) + old[1:]
    warped_old = tuple(np.where(keep_old, m, f if np.isscalar(f) else f)
                       for m, f in zip(moved_old, blanks))
    blanks_new = (SPACE,) + new[1:]
    warped_new = tuple(np.where(keep_new, m, f if np.isscalar(f) else f)
                       for m, f in zip(moved_new, blanks_new))
    cells = _dot_mask(rows, cols, _handover(progress))
    return tuple(np.where(cells, n, o) for o, n in zip(warped_old, warped_new))
