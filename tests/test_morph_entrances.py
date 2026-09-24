"""Each family arrives its own way, and a morph moves until it is over.

Every switch used to hand over in the same order: a blue-noise scatter behind
one left-to-right front. Between two bar modes and between a starfield and a
fluid it was the same gesture. Now the incoming picture enters the way its
family moves -- bars rise from the floor, particles burst out of the centre,
a trace is drawn across, a field ripples outward, a scene closes in from the
edges, rain falls -- and still dithered, so it is never a hard wipe.

The morph also used to go still a third of the way before it ended: the
widget eases progress and the travel eased it again inside an 80% window, so
the frames from about two thirds on were identical.
"""
from __future__ import annotations

import numpy as np
import pytest

import spektr.modes as M
from spektr import dissolve
from spektr.render import BRAILLE_BASE

ROWS, COLS = 24, 80
OLD_GLYPH, NEW_GLYPH = ord("█"), ord("▓")


def _shaped_old(rows: slice = slice(None), cols: slice = slice(None)):
    """A block-glyph picture occupying only part of the frame, so a picture
    grown into its shape is visibly not the incoming one yet."""
    codes = np.full((ROWS, COLS), ord(" "), np.int32)
    codes[rows, cols] = OLD_GLYPH
    return codes, np.full((ROWS, COLS), 5, np.int32)


def _braille_new(seed: int):
    dots = np.random.default_rng(seed).integers(1, 256, size=(ROWS, COLS))
    return (BRAILLE_BASE + dots).astype(np.int32), np.full((ROWS, COLS), 40, np.int32)


def _block_pair():
    old = (np.full((ROWS, COLS), OLD_GLYPH, np.int32), np.full((ROWS, COLS), 5, np.int32))
    new = (np.full((ROWS, COLS), NEW_GLYPH, np.int32), np.full((ROWS, COLS), 40, np.int32))
    return old, new


def _arrived(entrance: str, progress: float, push: float = 0.0) -> np.ndarray:
    old, new = _block_pair()
    style = dissolve.Style(entrance=entrance)
    out = dissolve.blend(old, new, progress, gather=0.0, push=push, style=style)
    return out[0] == NEW_GLYPH


def _thirds(mask: np.ndarray, axis: int) -> tuple[float, float]:
    n = mask.shape[axis]
    lo = np.take(mask, range(n // 3), axis=axis).mean()
    hi = np.take(mask, range(n - n // 3, n), axis=axis).mean()
    return float(lo), float(hi)


def _rings(mask: np.ndarray) -> tuple[float, float]:
    y, x = np.indices(mask.shape)
    d = np.hypot((y - (ROWS - 1) / 2) * 2.0, x - (COLS - 1) / 2)
    d /= d.max()
    return float(mask[d < 0.3].mean()), float(mask[d > 0.7].mean())


MID = 0.55


def test_bars_rise_from_the_floor():
    top, bottom = _thirds(_arrived("rise", MID), axis=0)
    assert bottom > top + 0.3


def test_rain_falls_from_the_top():
    top, bottom = _thirds(_arrived("fall", MID), axis=0)
    assert top > bottom + 0.3


def test_a_trace_is_drawn_across():
    left, right = _thirds(_arrived("draw", MID), axis=1)
    assert left > right + 0.3


def test_particles_burst_out_of_the_centre():
    centre, edge = _rings(_arrived("burst", MID))
    assert centre > edge + 0.3


def test_a_scene_closes_in_from_the_edges():
    centre, edge = _rings(_arrived("dive", MID))
    assert edge > centre + 0.3


def test_a_field_ripples_outward():
    centre, edge = _rings(_arrived("ripple", MID))
    assert centre > edge + 0.2


@pytest.mark.parametrize("entrance", dissolve.ENTRANCE_NAMES)
def test_an_entrance_is_dithered_not_wiped(entrance):
    """A hard edge between old and new reads as a wipe from a slideshow. Along
    the front, old and new cells are mixed."""
    mask = _arrived(entrance, MID)
    assert 0.1 < mask.mean() < 0.9
    mixed_rows = sum(1 for r in mask if 0.05 < r.mean() < 0.95)
    mixed_cols = sum(1 for c in mask.T if 0.05 < c.mean() < 0.95)
    assert mixed_rows + mixed_cols > (ROWS + COLS) // 4


@pytest.mark.parametrize("entrance", dissolve.ENTRANCE_NAMES)
def test_every_entrance_starts_old_and_ends_new(entrance):
    assert not _arrived(entrance, 0.02).any()
    assert _arrived(entrance, 0.999).all()


def test_a_beat_carries_the_front_on():
    assert _arrived("rise", MID, push=0.2).mean() > _arrived("rise", MID).mean()


def test_every_family_has_an_entrance():
    groups = {m.group for m in M.MODES if m.group != "off"}
    missing = groups - set(dissolve.ENTRANCES)
    assert not missing, f"no entrance for {sorted(missing)}"
    assert set(dissolve.ENTRANCES.values()) <= set(dissolve.ENTRANCE_NAMES)


def test_two_kinds_of_frame_get_the_entrance_too():
    """A bar mode into a braille mode cannot share one frame, so the incoming
    picture grows out of the old shape -- and snaps to its own form along
    the entrance, rather than everywhere at once."""
    old = _shaped_old(rows=slice(0, ROWS // 2))
    new = _braille_new(1)
    style = dissolve.Style(entrance="rise")
    out = dissolve.blend(old, new, MID, gather=0.0, style=style)
    settled = out[0] == new[0]
    top, bottom = _thirds(settled, axis=0)
    assert bottom > top + 0.2


def test_the_morph_is_still_moving_near_its_end():
    """Consecutive frames in the last third must differ: a morph that has
    finished before its clock has is a pause, not a morph."""
    old = _shaped_old(cols=slice(0, COLS // 3))
    new = _braille_new(2)
    style = dissolve.Style(entrance="burst")
    frames = [
        dissolve.blend(old, new, float(dissolve.ease(p)), style=style)[0]
        for p in (0.7, 0.8, 0.9)
    ]
    assert not np.array_equal(frames[0], frames[1])
    assert not np.array_equal(frames[1], frames[2])
