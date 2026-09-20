"""One frame dissolves into another, dot by dot, without inventing anything."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spektr import dissolve  # noqa: E402
from spektr.render import BRAILLE_BASE  # noqa: E402

H, W = 8, 16


def braille(fill: int, colour: int) -> tuple:
    """A frame where every cell holds the same dots and the same ramp index."""
    return (
        np.full((H, W), BRAILLE_BASE + fill, dtype=np.int32),
        np.full((H, W), colour, dtype=np.int32),
    )


def octant(code: int, fg: int, bg: int) -> tuple:
    return (
        np.full((H, W), code, dtype=np.int32),
        np.full((H, W), fg, dtype=np.int32),
        np.full((H, W), bg, dtype=np.int32),
    )


def dots_from(frame) -> int:
    """How many dots are lit across a braille frame."""
    bits = frame[0] - BRAILLE_BASE
    return int(sum(bin(int(v)).count("1") for v in bits.ravel()))


def test_no_progress_is_the_old_frame():
    old, new = braille(0xFF, 3), braille(0x00, 9)
    assert dissolve.blend(old, new, 0.0) is old


def test_full_progress_is_the_new_frame():
    old, new = braille(0xFF, 3), braille(0x00, 9)
    assert dissolve.blend(old, new, 1.0) is new


@pytest.mark.parametrize("progress", [0.4, 0.5, 0.6])
def test_part_way_holds_dots_from_both(progress):
    """Mid-morph, during the handover window, both pictures are on screen.

    The share is not tied to progress: the swap runs between
    ``dissolve.HANDOVER``, so the picture is all old before it and all new
    after, and only mixed in between.
    """
    old, new = braille(0xFF, 3), braille(0x00, 9)  # all dots lit -> none lit
    lit = dots_from(dissolve.blend(old, new, progress))
    assert 0 < lit < H * W * 8


def test_the_dissolve_only_moves_forward():
    old, new = braille(0xFF, 3), braille(0x00, 9)
    lit = [dots_from(dissolve.blend(old, new, p)) for p in (0.2, 0.4, 0.6, 0.8)]
    assert lit == sorted(lit, reverse=True), lit


def test_it_never_invents_a_dot_neither_frame_has():
    old, new = braille(0x0F, 3), braille(0xF0, 9)
    mixed = dissolve.blend(old, new, 0.5)
    for cell in np.unique(mixed[0] - BRAILLE_BASE):
        assert int(cell) & ~0xFF == 0
        assert int(cell) | 0xFF == 0xFF


def test_colour_comes_from_whichever_frame_owns_the_cell():
    old, new = braille(0xFF, 3), braille(0xFF, 9)
    mixed = dissolve.blend(old, new, 0.5)
    assert set(np.unique(mixed[1])) <= {3, 9}


def test_octant_frames_change_over_cell_by_cell():
    old, new = octant(0x1CD00, 2, 4), octant(0x1CD10, 7, 8)
    mixed = dissolve.blend(old, new, 0.5)
    assert len(mixed) == 3
    # SPACE appears where the gather moved a cell off the frame; every other
    # cell holds one of the two glyphs, never anything invented.
    from spektr.render import SPACE

    assert set(np.unique(mixed[0])) <= {0x1CD00, 0x1CD10, SPACE}
    assert set(np.unique(mixed[1])) <= {2, 7}


def test_a_resize_part_way_through_just_shows_the_new_frame():
    old = braille(0xFF, 3)
    new = (np.full((H + 2, W), BRAILLE_BASE, dtype=np.int32),
           np.zeros((H + 2, W), dtype=np.int32))
    assert dissolve.blend(old, new, 0.3) is new


def test_frames_of_different_kinds_change_over_at_the_halfway_point():
    old, new = braille(0xFF, 3), octant(0x1CD10, 7, 8)
    assert dissolve.blend(old, new, 0.4) is old
    assert dissolve.blend(old, new, 0.6) is new


def test_easing_starts_and_ends_gently():
    assert dissolve.ease(0.0) == 0.0
    assert dissolve.ease(1.0) == 1.0
    assert dissolve.ease(0.5) == pytest.approx(0.5)
    assert dissolve.ease(0.1) < 0.1      # slow at the start
    assert dissolve.ease(0.9) > 0.9      # slow at the end
    assert dissolve.ease(-1.0) == 0.0    # clamped
    assert dissolve.ease(2.0) == 1.0


def test_ink_travels_between_the_two_shapes():
    """The point of the morph: ink moves, rather than one picture swapping
    for the other. Old ink sits in the top quarter, new ink in the bottom;
    part way through, it should be somewhere between the two."""
    rows, cols = H * 4, W * 2
    top = np.zeros((rows, cols), bool)
    top[: rows // 4] = True
    bottom = np.zeros((rows, cols), bool)
    bottom[-rows // 4:] = True
    from spektr.render import pack_braille

    old = (pack_braille(top), np.full((H, W), 5, dtype=np.int32))
    new = (pack_braille(bottom), np.full((H, W), 9, dtype=np.int32))

    ys = np.arange(rows)[:, None]

    def centre(frame) -> float:
        dots = dissolve._braille_dots(frame[0])
        return float((dots * ys).sum() / max(1, dots.sum()))

    walk = [centre(dissolve.blend(old, new, p)) for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert walk == sorted(walk), f"ink did not travel steadily: {walk}"
    assert walk[0] < rows * 0.2 and walk[-1] > rows * 0.8
    # and it is the same picture moving, not two pictures overlapping
    lit = [dissolve._braille_dots(dissolve.blend(old, new, p)[0]).sum()
           for p in (0.25, 0.5, 0.75)]
    assert max(lit) <= top.sum() * 1.2, f"ink appeared out of nowhere: {lit}"
