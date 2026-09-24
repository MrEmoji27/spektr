"""JP Sequencer shows the bar it was told, and claims no downbeat it was not."""
from __future__ import annotations

import numpy as np
import pytest

import spektr.modes as M
from spektr.analysis import N_BANDS
from spektr.modes import Ctx
from spektr.modes.jp import _LED, _OFF, _PEAK, _TRAIL
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

def _lamps(codes):
    row = codes[0]
    return [int(c) for c in row if c != SPACE]


def test_the_live_page_follows_the_bar():
    st: dict = {}
    codes, _ = draw("JP Sequencer", st, beat_in_bar=2, bar_confidence=0.9,
                    tempo_bpm=120.0)
    assert _lamps(codes) == [_PEAK, _OFF, _LED, _OFF]


def test_no_downbeat_is_claimed_for_a_bar_that_is_not_known():
    st: dict = {}
    codes, _ = draw("JP Sequencer", st, beat_in_bar=2, bar_confidence=0.1,
                    tempo_bpm=0.0, onsets=1)
    lamps = _lamps(codes)
    assert _PEAK not in lamps and lamps.count(_LED) == 1


def test_without_a_bar_the_pages_still_step_on_the_beat():
    st: dict = {}
    seen = []
    phase = 0.0
    for i in range(int(2.2 / DT)):
        phase = (phase + DT * 2.0) % 1.0          # 120 BPM
        codes, _ = draw("JP Sequencer", st, t=i * DT, tempo_bpm=120.0,
                        beat_phase=phase)
        seen.append(_lamps(codes).index(_LED))
    assert len(set(seen)) == 4


def test_held_pages_are_drawn_a_weight_lighter():
    st: dict = {}
    draw("JP Sequencer", st, level=0.8, beat_in_bar=0, bar_confidence=0.9, tempo_bpm=120.0)
    codes, _ = draw("JP Sequencer", st, level=0.1, beat_in_bar=1,
                    bar_confidence=0.9, tempo_bpm=120.0)
    page_w = (W - 3) // 4
    held, live = codes[2:, :page_w], codes[2:, page_w + 1: 2 * page_w + 1]
    assert (held == _TRAIL).any() and not (held == _LED).any()
    assert (live == _LED).any()


# ── through the real widget, on the drum corpus ──────────────────────────────

#: Beat response on ``four_on_floor``, measured when these were added, as in
#: ``tests/test_reactivity.py``. Not pinned there: an LED panel is sparse on
#: purpose and sits under that file's churn floor, as ``JP Bars`` does.
RESPONSE = {"JP Sequencer": 4.46}


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
