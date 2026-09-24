"""Locket Beat: one ring for every beat, and none for anything else."""
from __future__ import annotations

import numpy as np

import spektr.modes as M
from spektr.analysis import N_BANDS
from spektr.modes import Ctx
from spektr.palette import BUILTIN, Palette
from spektr.render import BRAILLE_BASE

PAL = Palette(BUILTIN["gruvbox"])
DT = 1 / 60
W, H = 100, 30


def frame(state, t, onsets=0, level=0.3, **kw):
    bands = np.full(N_BANDS, level)
    ctx = Ctx(w=W, h=H, bands=bands, peaks=bands, bands_l=bands, bands_r=bands,
              wave=np.zeros(512), stereo=np.zeros((512, 2)), frame=int(t / DT),
              t=t, dt=DT, energy=level, silent=False, palette=PAL, state=state,
              onsets=onsets, onset_strength=0.9 if onsets else 0.0, **kw)
    return M.get("Locket Beat").fn(ctx)


def ring_dots(codes, state) -> int:
    """Lit dots well outside the resting heart: rings, and only rings."""
    geo = next(v for k, v in state.items() if k[0] == "locket_geo")
    dots = np.zeros((H * 4, W * 2), dtype=bool)
    bits = codes - BRAILLE_BASE
    # unpack the standard braille layout
    layout = ((0, 0, 0x01), (1, 0, 0x02), (2, 0, 0x04), (0, 1, 0x08),
              (1, 1, 0x10), (2, 1, 0x20), (3, 0, 0x40), (3, 1, 0x80))
    for r, c, bit in layout:
        dots[r::4, c::2] = (bits & bit) != 0
    return int((dots & (geo["scale"] > 0.45)).sum())


def test_a_beat_throws_a_ring():
    st: dict = {}
    frame(st, 0.0)
    codes, _ = frame(st, DT, onsets=1)
    for i in range(6):
        codes, _ = frame(st, DT * (2 + i))
    assert ring_dots(codes, st) > 20


def test_nothing_is_thrown_between_beats():
    st: dict = {}
    for i in range(120):
        codes, _ = frame(st, i * DT)
        assert ring_dots(codes, st) == 0, f"a ring with no beat, frame {i}"


def test_the_ring_is_gone_within_the_beat():
    st: dict = {}
    frame(st, 0.0, onsets=1, tempo_bpm=120.0)
    t = 0.0
    for _ in range(int(0.52 / DT)):
        t += DT
        codes, _ = frame(st, t, tempo_bpm=120.0)
    assert ring_dots(codes, st) == 0


def test_two_onsets_in_one_frame_are_one_ring():
    one, two = {}, {}
    frame(one, 0.0, onsets=1)
    frame(two, 0.0, onsets=2)
    a = frame(one, 0.1)[0]
    b = frame(two, 0.1)[0]
    assert np.array_equal(a, b)


def test_the_downbeat_ring_is_the_big_one():
    counts = []
    for beat in (0, 2):
        st: dict = {}
        frame(st, 0.0, onsets=1, tempo_bpm=120.0, beat_in_bar=beat, bar_confidence=0.9)
        codes, _ = frame(st, 0.15, tempo_bpm=120.0, beat_in_bar=beat, bar_confidence=0.9)
        counts.append(ring_dots(codes, st))
    assert counts[0] > counts[1] * 1.2, counts


def test_every_beat_shows(monkeypatch):
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import reactivity as R
    from onset_eval import SCENARIOS

    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    samples, rate, _ = SCENARIOS["four_on_floor"]()
    got = R.measure("Locket Beat", samples, rate, now)
    assert got["lost"] == 0
    # Measured at 3.3 when it was added; Locket, with its cascade, is 1.05.
    assert got["beat_ratio"] > 3.3 * 0.8, got
