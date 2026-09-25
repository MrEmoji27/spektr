"""Fireworks: with a tempo, the beat's rockets burst on the beat, not after it."""
from __future__ import annotations

import dataclasses

import numpy as np

import spektr.modes as M
from spektr.analysis import N_BANDS
from spektr.modes import Ctx
from spektr.palette import BUILTIN, Palette

PAL = Palette(BUILTIN["gruvbox"])
DT = 1 / 60


def ctx(state, t, bpm=120.0, level=0.3, **kw) -> Ctx:
    bands = np.full(N_BANDS, level)
    beat = t * bpm / 60.0 if bpm else 0.0
    base = dict(
        w=120, h=30, bands=bands, peaks=bands, bands_l=bands, bands_r=bands,
        wave=np.zeros(512), stereo=np.zeros((512, 2)), frame=int(t / DT), t=t,
        dt=DT, energy=float(level), silent=level < 0.02, palette=PAL, state=state,
        tempo_bpm=bpm, beat_phase=beat % 1.0,
    )
    base.update(kw)
    return Ctx(**base)


def _state(state):
    return next(v for k, v in state.items() if k[0] == "fireworks")


def _timed_bursts(seconds, **kw) -> list[float]:
    """When the beat's rockets burst, driving the mode frame by frame."""
    fn = M.get("Fireworks").fn
    state: dict = {}
    out = []
    for i in range(int(seconds / DT)):
        t = i * DT
        st = next((v for k, v in state.items() if k[0] == "fireworks"), None)
        before = (st["ry"] >= 0) & st["rtimed"] if st is not None else None
        fn(ctx(state, t, **kw))
        st = _state(state)
        if before is not None and (before & (st["ry"] < 0)).any():
            out.append(t)
    return out


def test_with_a_tempo_the_salvo_bursts_on_the_beat_clock():
    bursts = _timed_bursts(4.0)
    beats = np.arange(1, 8) * 0.5                  # 120 BPM
    assert len(bursts) >= len(beats) - 1
    for b in bursts:
        off = np.min(np.abs(beats - b))
        assert off <= DT + 1e-9, f"a salvo burst {off * 1000:.0f} ms off the beat"


def test_a_salvo_climbs_before_it_bursts():
    """A beat rocket is seen going up: it is lit a flight before its beat."""
    fn = M.get("Fireworks").fn
    state: dict = {}
    first_seen = None
    for i in range(int(1.2 / DT)):
        fn(ctx(state, i * DT))
        st = _state(state)
        if first_seen is None and ((st["ry"] >= 0) & st["rtimed"]).any():
            first_seen = i * DT
    assert first_seen is not None and first_seen < 0.5 - 0.2


def test_silence_sends_no_salvo():
    assert not _timed_bursts(2.0, level=0.0)


def test_without_a_tempo_a_hit_sends_rockets_up():
    fn = M.get("Fireworks").fn
    state: dict = {}
    fn(ctx(state, 0.0, bpm=0.0))
    for i in range(1, 30):                          # let the opening shell go
        fn(ctx(state, i * DT, bpm=0.0))
    before = int((_state(state)["ry"] >= 0).sum())
    fn(ctx(state, 30 * DT, bpm=0.0, onsets=1, onset_strength=0.9))
    after = int((_state(state)["ry"] >= 0).sum())
    assert after >= before + 2


def test_the_salvos_land_on_the_beats_of_real_audio(monkeypatch):
    """Through the real analyser: once the tempo has locked, every salvo
    bursts within a few tens of milliseconds of a true beat."""
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
    bursts: list[float] = []
    mode = M.get("Fireworks")
    original = mode.fn

    def spy(c):
        st = next((v for k, v in c.state.items() if k[0] == "fireworks"), None)
        before = (st["ry"] >= 0) & st["rtimed"] if st is not None else None
        out = original(c)
        st = next((v for k, v in c.state.items() if k[0] == "fireworks"), None)
        if before is not None and (before & (st["ry"] < 0)).any():
            bursts.append(c.t)
        return out

    spied = dataclasses.replace(mode, fn=spy)
    at = M.MODES.index(mode)
    M.MODES[at] = spied
    M._BY_NAME["Fireworks"] = spied
    try:
        R.measure("Fireworks", samples, rate, now)
    finally:
        M.MODES[at] = mode
        M._BY_NAME["Fireworks"] = mode

    assert len(bursts) >= truth.size // 2, f"{len(bursts)} salvos for {truth.size} beats"
    offs = [np.min(np.abs(truth - b)) for b in bursts]
    assert max(offs) < 0.035, f"a salvo burst {max(offs) * 1000:.0f} ms off the beat"
