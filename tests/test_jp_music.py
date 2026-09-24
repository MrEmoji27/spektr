"""JP Sequencer shows the bar it was told, and claims no downbeat it was not."""
from __future__ import annotations

import numpy as np
import pytest

import spektr.modes as M
from spektr.analysis import N_BANDS
from spektr.modes import Ctx
from spektr.modes.jp import _LED, _PEAK, _TRAIL
from spektr.palette import BUILTIN, Palette
from spektr.render import SPACE

PAL = Palette(BUILTIN["gruvbox"])
DT = 1 / 60
W, H = 96, 30
NEW = ("JP Sequencer",)
NO_DRUMS = {"kick": 0.0, "snare": 0.0, "hat": 0.0}


def ctx(state, t=0.0, level=0.3, **kw) -> Ctx:
    bands = np.full(N_BANDS, level)
    base = dict(
        w=W, h=H, bands=bands, peaks=bands, bands_l=bands, bands_r=bands,
        wave=np.zeros(512), stereo=np.zeros((512, 2)), frame=int(t / DT), t=t,
        dt=DT, energy=float(level), silent=level < 0.02, palette=PAL,
        state=state,
    )
    base.update(kw)
    return Ctx(**base)


def draw(name, state, **kw):
    return M.get(name).fn(ctx(state, **kw))


@pytest.mark.parametrize("name", NEW)
def test_they_are_in_the_family(name):
    m = M.get(name)
    assert m is not None and m.group == "jp" and not m.hidden


@pytest.mark.parametrize("name", NEW)
def test_silence_lights_nothing(name):
    codes, _ = draw(name, {}, level=0.0, chroma=np.zeros(12, np.float32))
    assert not (codes == _LED).any()


# ── JP Sequencer ─────────────────────────────────────────────────────────────

KICK = {"kick": 0.9, "snare": 0.0, "hat": 0.0}
SNARE = {"kick": 0.0, "snare": 0.9, "hat": 0.0}
BAR = dict(tempo_bpm=120.0, bar_confidence=0.9)


def _grid(state) -> tuple[np.ndarray, np.ndarray]:
    st = next(v for k, v in state.items() if k[0] == "jp_seq")
    return st["cur"], st["prev"]


def test_a_hit_is_written_on_its_drum_s_row_at_its_step():
    st: dict = {}
    draw("JP Sequencer", st, onsets=1, onset_strength=0.9, drums=KICK, bar_phase=0.0, **BAR)
    draw("JP Sequencer", st, t=0.5, onsets=1, onset_strength=0.9, drums=SNARE,
         bar_phase=0.25, **BAR)
    cur, _ = _grid(st)
    kick, snare = 2, 1                       # rows: hat, snare, kick
    assert cur[kick, 0] > 0 and cur[snare, 4] > 0
    assert cur.astype(bool).sum() == 2


def test_nothing_is_written_without_a_drum():
    st: dict = {}
    draw("JP Sequencer", st, onsets=1, onset_strength=0.9, drums=NO_DRUMS, bar_phase=0.1, **BAR)
    cur, _ = _grid(st)
    assert not cur.any()


def test_the_bar_just_played_becomes_the_last_bar():
    st: dict = {}
    draw("JP Sequencer", st, onsets=1, onset_strength=0.9, drums=KICK, bar_phase=0.0, **BAR)
    draw("JP Sequencer", st, t=1.9, bar_phase=0.95, **BAR)
    draw("JP Sequencer", st, t=2.05, bar_phase=0.02, **BAR)       # over the one
    cur, prev = _grid(st)
    assert not cur.any() and prev[2, 0] > 0
    codes, _ = draw("JP Sequencer", st, t=2.1, bar_phase=0.05, **BAR)
    assert (codes == _TRAIL).any(), "last bar's hit is not drawn"


def _lamps(codes):
    return [int(c) for c in codes[0] if c != SPACE]


def test_the_playhead_follows_the_bar_and_marks_one_when_it_is_known():
    codes, _ = draw("JP Sequencer", {}, bar_phase=0.55, **BAR)
    lamps = _lamps(codes)
    assert lamps.index(_LED) == 8 and lamps[0] == _PEAK


def test_no_one_is_claimed_for_a_bar_that_is_not_known():
    codes, _ = draw("JP Sequencer", {}, tempo_bpm=120.0, bar_confidence=0.1,
                    beat_phase=0.3)
    assert _PEAK not in _lamps(codes)


def test_without_a_beat_the_playhead_runs_dim():
    codes, _ = draw("JP Sequencer", {}, tempo_bpm=0.0)
    lamps = _lamps(codes)
    assert _LED not in lamps and _TRAIL in lamps


# ── through the real widget, on the drum corpus ──────────────────────────────

#: Beat response on ``four_on_floor``, measured when these were added, as in
#: ``tests/test_reactivity.py``. Not pinned there: an LED panel is sparse on
#: purpose and sits under that file's churn floor, as ``JP Bars`` does.
RESPONSE = {"JP Sequencer": 2.46}


@pytest.mark.parametrize("name", NEW)
def test_the_beat_shows(name, monkeypatch):
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import reactivity as R
    from onset_eval import SCENARIOS

    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    samples, rate, _ = SCENARIOS["four_on_floor"]()
    got = R.measure(name, samples, rate, now)
    assert got["lost"] == 0
    assert got["beat_ratio"] >= RESPONSE[name] * 0.8, got
