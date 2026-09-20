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

#: The mechanics of a morph with none of its manner: no sweep, no wavefront,
#: nothing landing past its target, no lean. Tests about travelling, arriving
#: and not inventing anything use this, so that how a morph moves cannot
#: quietly move the goalposts under them — the manner has tests of its own
#: below, and each of those turns one thing on at a time.
PLAIN = dissolve.Style(sweep=0.0, wavefront=0.0, overshoot=0.0, sway=0.0)


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


def test_a_resize_part_way_through_carries_on_at_the_new_size():
    """A resize used to cut to the incoming frame on the spot.

    Two frames of different sizes cannot be mixed, but the morph does not have
    to die there: the outgoing picture is resampled onto the size the new one
    is drawn at and the morph finishes at that size, so the change is still a
    morph on the frame the terminal changed shape.
    """
    old = braille(0xFF, 3)
    new = (np.full((H + 2, W * 2), BRAILLE_BASE + 0x0F, dtype=np.int32),
           np.full((H + 2, W * 2), 9, dtype=np.int32))
    mixed = dissolve.blend(old, new, 0.3)
    assert mixed is not new and mixed is not old
    assert mixed[0].shape == new[0].shape, "drawn at the size it finishes at"
    assert mixed[1].shape == new[1].shape


def test_frames_of_different_kinds_grow_out_of_the_old_shape():
    """A braille frame against a half-block or octant one cannot be mixed.

    A braille cell and an octant cell are not on the same subcell grid, so the
    two are never drawn in the same frame. What is drawn is the incoming
    picture, starting out in the outgoing picture's geometry and settling into
    its own, which is what makes the change a morph rather than a cut half way
    through.
    """
    from spektr.render import SPACE, pack_braille

    rows, cols = H * 4, W * 2
    # the old picture: an octant band across the top of the frame
    codes = np.full((H, W), SPACE, dtype=np.int32)
    codes[: H // 4] = 0x1CD10
    old = (codes, np.full((H, W), 4, dtype=np.int32),
           np.full((H, W), 0, dtype=np.int32))
    # the new one: a braille band through the middle, its own centre
    middle = np.zeros((rows, cols), bool)
    middle[rows // 3: 2 * rows // 3] = True
    new = (pack_braille(middle), np.full((H, W), 7, dtype=np.int32))

    ys = np.arange(rows)[:, None]

    def centre(frame) -> float:
        dots = dissolve._braille_dots(frame[0])
        return float((dots * ys).sum() / max(1, dots.sum()))

    # the incoming kind is what is drawn, from the first frame on
    assert all(len(dissolve.blend(old, new, p)) == 2 for p in (0.05, 0.4, 0.9))
    # and nothing of the other kind: a 2-tuple is braille or it is not one
    for p in (0.05, 0.4, 0.9):
        assert dissolve._braille(dissolve.blend(old, new, p)[0]), p
    # with the picture not being resampled across the frame, nothing is
    # invented either: every glyph is one the incoming frame drew
    for p in (0.05, 0.4, 0.9):
        mixed = dissolve.blend(old, new, p, style=PLAIN)
        assert set(np.unique(mixed[0])) <= set(np.unique(new[0])) | {SPACE}

    walk = [centre(dissolve.blend(old, new, p, style=PLAIN))
            for p in (0.05, 0.4, 0.7, 0.95)]
    own = centre(new)
    assert walk == sorted(walk), f"the new picture did not settle: {walk}"
    assert walk[0] < rows * 0.25, f"did not start in the old shape: {walk[0]}"
    assert abs(walk[-1] - own) < rows * 0.1, f"did not arrive: {walk[-1]} vs {own}"


def _band(rows: int, cols: int, first: int, last: int,
          across: slice | None = None) -> np.ndarray:
    """A dot grid with a band of ink across it, rows ``first``..``last``."""
    dots = np.zeros((rows, cols), bool)
    dots[first:last, across] = True
    return dots


def _x_marks(frame) -> tuple[float, float]:
    """Where a braille frame's ink sits across the frame: centre and spread."""
    dots = dissolve._braille_dots(frame[0])
    xs = np.arange(dots.shape[1], dtype=np.float64)[None, :]
    total = max(1, int(dots.sum()))
    centre = float((dots * xs).sum() / total)
    spread = float(np.sqrt(((dots * (xs - centre)) ** 2).sum() / total))
    return centre, spread


def test_ink_that_moved_sideways_travels_across():
    """The other half of the same idea, in x.

    A picture whose mass sits on the left and one whose mass sits on the right
    travel across to meet. Matching column for column — which is what the
    warp did before it had an x half — would leave each one's ink where it was
    and only slide it up and down.
    """
    from spektr.render import pack_braille

    rows, cols = H * 4, W * 2
    left = _band(rows, cols, 0, rows, slice(0, cols // 4))
    right = _band(rows, cols, 0, rows, slice(-cols // 4, None))
    old = (pack_braille(left), np.full((H, W), 5, dtype=np.int32))
    new = (pack_braille(right), np.full((H, W), 9, dtype=np.int32))

    start, width = _x_marks(old)
    walk = [_x_marks(dissolve.blend(old, new, p))[0]
            for p in (0.0, 0.2, 0.4, 0.6, 0.8)]
    assert walk == sorted(walk), f"ink did not travel across: {walk}"
    assert abs(walk[0] - start) < 1.0
    assert walk[2] > start + cols * 0.2, f"not across by the middle: {walk[2]}"
    assert walk[-1] > cols * 0.7, f"did not arrive: {walk[-1]}"
    # a slide, not a stretch: the ink keeps its own width on the way over
    mid = _x_marks(dissolve.blend(old, new, 0.4))[1]
    assert 0.7 * width < mid < 1.4 * width, f"the ink was smeared: {width} -> {mid}"


def test_a_beat_pushes_the_travel_along():
    """Onsets landing mid-morph shove the shapes on towards the picture they
    are becoming, and leave the handover to its own clock."""
    from spektr.render import pack_braille

    rows, cols = H * 4, W * 2
    left = _band(rows, cols, 0, rows, slice(0, cols // 4))
    right = _band(rows, cols, 0, rows, slice(-cols // 4, None))
    old = (pack_braille(left), np.full((H, W), 5, dtype=np.int32))
    new = (pack_braille(right), np.full((H, W), 9, dtype=np.int32))

    plain = _x_marks(dissolve.blend(old, new, 0.3))[0]
    pushed = _x_marks(dissolve.blend(old, new, 0.3, push=dissolve.PUSH_MAX))[0]
    assert pushed > plain, f"the beat did not move the shapes on: {plain} -> {pushed}"
    assert pushed < _x_marks(new)[0], "and not past the picture it is becoming"

    # the colours stay where the handover puts them, beat or no beat
    def arrived(frame) -> float:
        return float((frame[1] == 9).mean())

    assert arrived(dissolve.blend(old, new, 0.6, push=dissolve.PUSH_MAX)) == pytest.approx(
        arrived(dissolve.blend(old, new, 0.6))
    )


def test_ink_leaving_the_frame_fades_at_the_edge():
    """Ink with nothing to be at the other end of its travel is not cut off on
    the frame it crosses the boundary: it keeps drawing at the edge and thins
    out over the next few rows' worth of travel."""
    rank = np.tile(np.linspace(0.0, 0.8, 8), (4, 1))
    src = np.zeros((4, 8), np.float32)
    src[0] = -1.0  # a row that has just left the top of the frame
    keep = dissolve._reached(src, None, rank)
    assert keep[1:].all(), "ink that is on the frame was dropped"
    assert keep[0].any() and not keep[0].all(), "the edge cut the ink off"
    # and ink far enough out has gone entirely
    assert not dissolve._reached(src - 4.0, None, rank)[0].any()


def test_the_shapes_arrive_before_the_colours_do():
    """Travel and handover run on separate clocks.

    The travel finishes with the morph still going and the handover runs past
    it, so the last of the old ink swaps under a picture that has already
    stopped moving, which is the part the eye would otherwise catch.
    """
    assert dissolve._travel(dissolve.TRAVEL) == 1.0
    assert dissolve._travel(1.0) == 1.0  # and holds there
    assert dissolve._handover(dissolve.TRAVEL) < 1.0  # colours still going
    assert dissolve._handover(1.0) == 1.0
    travel = [dissolve._travel(p, over=0.0) for p in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert travel == sorted(travel), travel
    # A beat pushes the travel further along its curve. On the way up, which
    # is where a beat has somewhere to push it to: past the peak the travel is
    # settling onto its target and there is nothing left to shove along.
    assert dissolve._travel(0.25, dissolve.PUSH_MAX) > dissolve._travel(0.25)
    assert dissolve._travel(1.0, dissolve.PUSH_MAX) == 1.0


def test_the_travel_lands_past_its_target_and_settles_back():
    """Idea two: the shapes overshoot the mark and come to rest on it.

    Nothing with weight stops dead where it was aimed, and a travel that eases
    flat into place is most of what makes a morph read as a slide rather than
    as a move. The picture is at its target early and the rest of the clock is
    the settle, so the overshoot is the whole of the extra travel.
    """
    walk = [dissolve._travel(p) for p in np.linspace(0.0, dissolve.TRAVEL, 201)]
    peak = int(np.argmax(walk))
    assert walk[peak] == pytest.approx(1.0 + dissolve.OVERSHOOT, abs=1e-5), (
        f"the peak is not the overshoot asked for: {walk[peak]}"
    )
    # it rises to the target, overshoots once, and comes back onto it
    assert walk[:peak] == sorted(walk[:peak]), "not on its way to the target"
    assert walk[peak:] == sorted(walk[peak:], reverse=True), "more than one settle"
    assert walk[peak] > 1.0, "it never went past its target"
    assert walk[-1] == 1.0, "it did not settle on the target"

    # taken out, the travel is the plain ease again
    plain = [dissolve._travel(p, over=0.0) for p in (0.3, 0.5, 0.7, 0.8)]
    assert max(plain) == 1.0 and plain == sorted(plain)


def test_the_frames_travel_is_split_into_a_sweep():
    """Idea one, on its own switch: the change crosses the frame.

    With a sweep the columns set off one after another, so part way through
    the ones on the left are further along than the ones on the right. Without
    one every column is at the same place at the same moment, which is what
    the morph did before it had a sweep and reads as a change that happens
    rather than one that arrives.
    """
    from spektr.render import pack_braille

    rows, cols = H * 4, W * 2
    top = _band(rows, cols, 0, rows // 4)
    bottom = _band(rows, cols, -rows // 4, rows)
    old = (pack_braille(top), np.full((H, W), 5, dtype=np.int32))
    new = (pack_braille(bottom), np.full((H, W), 9, dtype=np.int32))

    def per_column(frame) -> np.ndarray:
        dots = dissolve._braille_dots(frame[0])
        ys = np.arange(dots.shape[0], dtype=np.float64)[:, None]
        lit = dots.sum(axis=0)
        return np.where(lit > 0, (dots * ys).sum(axis=0) / np.maximum(lit, 1), np.nan)

    ahead = dissolve.Style(sweep=0.25, wavefront=0.0, overshoot=0.0, sway=0.0)
    walked = per_column(dissolve.blend(old, new, 0.35, style=ahead))
    left, right = np.nanmean(walked[: cols // 4]), np.nanmean(walked[-cols // 4:])
    assert left > right + 1.0, f"the frame did not sweep: {left:.1f} then {right:.1f}"

    even = per_column(dissolve.blend(old, new, 0.35, style=PLAIN))
    left, right = np.nanmean(even[: cols // 4]), np.nanmean(even[-cols // 4:])
    assert abs(left - right) < 0.5, f"the sweep leaked into the plain morph: {left}"


def test_the_handover_crosses_the_frame_behind_the_sweep():
    """Idea four, on its own switch: the swap has a front.

    An even scatter of dots swapping everywhere at once reads as a flicker;
    a front crossing the frame is a change you can watch travel. The column
    that trails keeps the window the whole frame used to share, so the colours
    still finish where they finished before.
    """
    cols = 16
    delay = dissolve._sweep(cols)  # 0 leading, 1 trailing
    lead = dissolve._handover(0.6, 0.3, delay)
    assert lead[0] > lead[-1], "the swap does not lead on the left"
    assert lead[-1] == pytest.approx(dissolve._handover(0.6)), "the window moved"
    assert lead[0] > 0.0 and lead[-1] < 1.0, "the front is not inside the window"

    flat = dissolve._handover(0.6, 0.0, delay)
    assert np.ndim(flat) == 0, "a threshold per column for no wavefront"
    assert flat == pytest.approx(dissolve._handover(0.6))


def test_the_picture_leans_into_its_flight():
    """Idea three, on its own switch: the ink travels through an arc.

    The lean is a bow across the frame's width, nothing at either end of the
    morph, so the picture sets off straight and lands straight and bulges out
    of the way in between.
    """
    from spektr.render import pack_braille

    rows, cols = H * 4, W * 2
    left = _band(rows, cols, 0, rows, slice(0, cols // 4))
    right = _band(rows, cols, 0, rows, slice(-cols // 4, None))
    old = (pack_braille(left), np.full((H, W), 5, dtype=np.int32))
    new = (pack_braille(right), np.full((H, W), 9, dtype=np.int32))

    leaning = dissolve.Style(sweep=0.0, wavefront=0.0, overshoot=0.0, sway=0.1)
    straight = _x_marks(dissolve.blend(old, new, 0.35, style=PLAIN))[0]
    bowed = _x_marks(dissolve.blend(old, new, 0.35, style=leaning))[0]
    assert bowed > straight + 1.0, f"the picture did not lean: {straight} -> {bowed}"

    # nothing at either end: it starts and lands where it would have anyway
    for p, styles in ((0.0, (PLAIN, leaning)), (1.0, (PLAIN, leaning))):
        ends = [_x_marks(dissolve.blend(old, new, p, style=s))[0] for s in styles]
        assert ends[0] == pytest.approx(ends[1]), f"a lean at {p}: {ends}"


def test_the_music_reorders_the_sweep():
    """Idea five: the front runs through the loud parts of the picture.

    The bands are read against their own mean, so the pattern holds whatever
    the volume is, and a frame with nothing in it sweeps in order of position
    like any other.
    """
    cols = 40
    ordered = dissolve._sweep(cols, None)
    assert np.array_equal(ordered, np.arange(cols, dtype=np.float32) / (cols - 1))
    assert np.array_equal(dissolve._sweep(cols, np.zeros(4, np.float32)), ordered)

    # loud on the left, quiet on the right: the left sets off, the right waits
    loud = np.array([2.0, 0.5], dtype=np.float32)
    moved = dissolve._sweep(cols, loud)
    assert moved[10] < ordered[10], "the loud half did not go first"
    assert moved[30] > ordered[30], "the quiet half did not wait"
    assert moved[0] == 0.0 and moved.max() <= 1.0, "the sweep left its window"
    assert moved[19] <= moved[20], "the front doubled back on itself"


def test_the_incoming_picture_is_settled_by_the_time_the_swap_ends():
    """By the last dot of the handover the new picture is in its own geometry,
    not still arriving in the old one's."""
    from spektr.render import pack_braille

    rows, cols = H * 4, W * 2
    top = _band(rows, cols, 0, rows // 4)
    middle = _band(rows, cols, rows // 3, 2 * rows // 3)
    old = (pack_braille(top), np.full((H, W), 5, dtype=np.int32))
    new = (pack_braille(middle), np.full((H, W), 9, dtype=np.int32))

    ys = np.arange(rows)[:, None]

    def centre(frame) -> float:
        dots = dissolve._braille_dots(frame[0])
        return float((dots * ys).sum() / max(1, dots.sum()))

    settled = centre(dissolve.blend(old, new, dissolve.HANDOVER[1]))
    assert abs(settled - centre(new)) < 1.0, (
        f"{settled:.1f} is not {centre(new):.1f}: still arriving"
    )


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

    walk = [centre(dissolve.blend(old, new, p, style=PLAIN))
            for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert walk == sorted(walk), f"ink did not travel steadily: {walk}"
    assert walk[0] < rows * 0.2 and walk[-1] > rows * 0.8
    # and it is the same picture moving, not two pictures overlapping
    lit = [dissolve._braille_dots(dissolve.blend(old, new, p, style=PLAIN)[0]).sum()
           for p in (0.25, 0.5, 0.75)]
    assert max(lit) <= top.sum() * 1.2, f"ink appeared out of nowhere: {lit}"
