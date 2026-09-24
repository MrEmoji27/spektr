"""Locket Beat: a ring shot for every hit, shaped by it, and none for anything else."""
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

#: Beat response on four_on_floor, measured when the rings became shots.
RESPONSE = 2.55


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


def test_a_ring_clears_in_its_flight_time():
    from spektr.modes.field_hearts import _SHOT_LIFE_S

    st: dict = {}
    frame(st, 0.0, onsets=1)
    t = 0.0
    for _ in range(int(_SHOT_LIFE_S * 1.25 / DT)):
        t += DT
        codes, _ = frame(st, t)
    assert ring_dots(codes, st) == 0


def test_two_onsets_in_one_frame_are_one_ring():
    one, two = {}, {}
    frame(one, 0.0, onsets=1)
    frame(two, 0.0, onsets=2)
    a = frame(one, 0.1)[0]
    b = frame(two, 0.1)[0]
    assert np.array_equal(a, b)


def _ring_after(drums, secs=0.12):
    st: dict = {}
    frame(st, 0.0, onsets=1, drums=drums)
    codes, _ = frame(st, secs, drums=drums)
    return ring_dots(codes, st)


def test_a_kick_s_ring_is_heavier_than_a_hat_s():
    kick = _ring_after({"kick": 1.0, "snare": 0.0, "hat": 0.0})
    hat = _ring_after({"kick": 0.0, "snare": 0.0, "hat": 1.0}, secs=0.06)
    assert kick > hat * 1.3, (kick, hat)


def test_several_hits_are_several_rings_in_flight():
    one: dict = {}
    many: dict = {}
    for i in range(30):
        t = i * DT
        frame(one, t, onsets=int(i == 0))
        a, _ = frame(one, t)
        frame(many, t, onsets=int(i in (0, 10, 20)))
        b, _ = frame(many, t)
    assert ring_dots(b, many) > ring_dots(a, one) * 1.5


def test_with_a_tempo_the_rings_come_on_the_beat_clock():
    """No hit needed: a ring leaves the heart on every beat of the tempo."""
    st: dict = {}
    born = 0
    phase = 0.0
    for i in range(int(2.0 / DT)):                    # two seconds at 120 BPM
        phase = (i * DT * 2.0) % 1.0
        frame(st, i * DT, tempo_bpm=120.0, beat_phase=phase)
        rings = next(v for k, v in st.items() if k[0] == "locket_beat")
        born = int((rings["born"] > -1).sum())
    assert born == 3, f"{born} rings in two seconds of 120 BPM"


def test_the_rings_are_a_stream_not_a_volley():
    """Half way through a beat, two rings are in flight: one is always leaving
    the heart as the one before it is part way out, so there is no gap."""
    st: dict = {}
    for i in range(int(3.25 / DT)):
        frame(st, i * DT, tempo_bpm=120.0, beat_phase=(i * DT * 2.0) % 1.0)
    rings = next(v for k, v in st.items() if k[0] == "locket_beat")
    t = int(3.25 / DT) * DT
    flying = [(t - b) for b, life in zip(rings["born"], rings["life"]) if 0 <= t - b < life]
    assert len(flying) == 2, flying


def test_the_rings_land_on_the_beats_of_real_audio(monkeypatch):
    """Through the real analyser: once the tempo has locked, rings are born
    within a few tens of milliseconds of the true beats, and almost every
    beat gets one. A ring that waited for the detector was 30 ms late before
    the widget drew it, and was then born inside the heart."""
    import dataclasses
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import reactivity as R
    from onset_eval import SCENARIOS

    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    samples, rate, truth = SCENARIOS["kick_snare"]()
    truth = np.array(truth)
    births: set[float] = set()
    mode = M.get("Locket Beat")
    original = mode.fn

    def spy(ctx):
        out = original(ctx)
        rings = next((v for k, v in ctx.state.items() if k[0] == "locket_beat"), None)
        if rings is not None:
            births.update(float(b) for b in rings["born"] if b > -1)
        return out

    spied = dataclasses.replace(mode, fn=spy)
    at = M.MODES.index(mode)
    M.MODES[at] = spied
    M._BY_NAME["Locket Beat"] = spied
    try:
        R.measure("Locket Beat", samples, rate, now)
    finally:
        M.MODES[at] = mode
        M._BY_NAME["Locket Beat"] = mode

    lags = []
    for b in sorted(births):
        d = b - truth
        d = d[np.abs(d) < 0.2]
        if d.size:
            lags.append(d[np.argmin(np.abs(d))])
    assert len(births) >= truth.size - 3, f"{len(births)} rings for {truth.size} beats"
    locked = np.abs(np.array(lags[len(lags) // 3:]))
    assert locked.max() < 0.035, f"rings up to {locked.max() * 1000:.0f} ms off the beat"
