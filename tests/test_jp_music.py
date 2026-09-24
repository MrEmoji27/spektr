"""The JP panels that hear the music say what they heard, and only that.

Each of these stands on one stream of the 0.6.0 analysis, and each is held
here to two things: it shows the thing it claims to (which drum, which beat,
which key), and it draws nothing it was not told -- no ring without a drum,
no downbeat without a known bar, no tonic without a settled key.
"""
from __future__ import annotations

import numpy as np
import pytest

import spektr.modes as M
from spektr.analysis import N_BANDS
from spektr.modes import Ctx
from spektr.modes.jp import _LED, _OFF, _PEAK, _TRAIL, bulb_row_index
from spektr.palette import BUILTIN, Palette
from spektr.render import SPACE

PAL = Palette(BUILTIN["gruvbox"])
DT = 1 / 60
W, H = 96, 30
NEW = ("JP Kit", "JP Sequencer", "JP Key")
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


# ── JP Kit ───────────────────────────────────────────────────────────────────

def _rings_after(drums, secs=0.15):
    """The ring bulbs a single hit leaves, with the plain panel taken away."""
    st: dict = {}
    t = 0.0
    draw("JP Kit", st, t=t, level=0.1, onsets=1, onset_strength=0.9, drums=drums)
    for _ in range(int(secs / DT)):
        t += DT
        out = draw("JP Kit", st, t=t, level=0.1, drums=drums)
    plain = draw("JP Kit", {}, t=t, level=0.1, drums=NO_DRUMS)
    return (out[0] == _LED) & (plain[0] != _LED)


def _corners(mask):
    rows, cols = mask.shape
    low_left = mask[rows // 2:, : cols // 3].mean()
    high_right = mask[: rows // 2, 2 * cols // 3:].mean()
    return low_left, high_right


def test_a_kick_rolls_out_of_the_bass_corner():
    ll, hr = _corners(_rings_after({"kick": 1.0, "snare": 0.0, "hat": 0.0}))
    assert ll > hr and ll > 0.02


def test_a_hat_flickers_out_of_the_treble_corner():
    ll, hr = _corners(_rings_after({"kick": 0.0, "snare": 0.0, "hat": 1.0}, secs=0.08))
    assert hr > ll and hr > 0.02


def test_no_drum_throws_no_ring():
    assert not _rings_after(NO_DRUMS).any()


def test_a_trace_of_a_drum_is_not_that_drum():
    assert not _rings_after({"kick": 0.2, "snare": 0.1, "hat": 0.1}).any()


def test_the_rings_never_cover_the_reading():
    st: dict = {}
    draw("JP Kit", st, level=0.5, onsets=1, onset_strength=1.0,
         drums={"kick": 1.0, "snare": 1.0, "hat": 1.0})
    ringed = draw("JP Kit", st, t=0.1, level=0.5, drums=NO_DRUMS)[0]
    plain = draw("JP Kit", {}, t=0.1, level=0.5, drums=NO_DRUMS)[0]
    assert ((plain == _LED) <= (ringed == _LED)).all()


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


# ── JP Key ───────────────────────────────────────────────────────────────────

FIFTHS = tuple((7 * k) % 12 for k in range(12))


def _columns(codes):
    """For each of the 12 note columns, its cells, in fifths order."""
    from spektr.modes import band_columns

    col, active = band_columns(W, 12)
    return [codes[:, (col == k) & active] for k in range(12)]


def test_a_key_powers_its_seven_notes_and_no_others():
    codes, _ = draw("JP Key", {}, level=0.3, chroma=np.full(12, 0.5, np.float32),
                    key="C major", key_confidence=0.9)
    powered = [bool((c == _OFF).any() or (c == _LED).any() and (c == _OFF).any())
               for c in _columns(codes)]
    names = [("C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F")[k]
             for k in range(12)]
    lattice = {names[k] for k, c in enumerate(_columns(codes)) if (c == _OFF).any()}
    assert lattice == {"F", "C", "G", "D", "A", "E", "B"}, (lattice, powered)


def test_the_tonic_has_its_lamp():
    codes, _ = draw("JP Key", {}, level=0.05, chroma=np.full(12, 0.05, np.float32),
                    key="A minor", key_confidence=0.9)
    top = bulb_row_index(H)[-1]
    cols = _columns(codes)
    a = FIFTHS.index(9)
    assert (cols[a][top] == _PEAK).any()
    assert not any((c[top] == _PEAK).any() for k, c in enumerate(cols) if k != a)


def test_an_uncertain_key_powers_everything_and_marks_nothing():
    codes, _ = draw("JP Key", {}, level=0.3, chroma=np.full(12, 0.02, np.float32),
                    key=None, key_uncertain=True, key_confidence=0.3)
    assert all((c == _OFF).any() for c in _columns(codes))
    assert not (codes == _PEAK).any()


def test_a_note_lights_its_own_column():
    chroma = np.zeros(12, np.float32)
    chroma[2] = 1.0                                   # D
    codes, _ = draw("JP Key", {}, level=0.5, chroma=chroma)
    lit = [bool((c == _LED).any()) for c in _columns(codes)]
    assert lit == [k == FIFTHS.index(2) for k in range(12)]


# ── through the real widget, on the drum corpus ──────────────────────────────

#: Beat response on ``four_on_floor``, measured when these were added, as in
#: ``tests/test_reactivity.py``. Not pinned there: an LED panel is sparse on
#: purpose and sits under that file's churn floor, as ``JP Bars`` does.
RESPONSE = {"JP Kit": 3.72, "JP Sequencer": 4.46, "JP Key": 5.22}


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
