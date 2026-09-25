"""The JP music machines: pad grid, chord display, drum panel and tracker."""
from __future__ import annotations

import numpy as np
import pytest

import spektr.modes as M
from spektr.analysis import N_BANDS
from spektr.modes import Ctx
from spektr.modes.jp import _LED, _TRAIL
from spektr.palette import BUILTIN, Palette
from spektr.render import SPACE

PAL = Palette(BUILTIN["gruvbox"])
DT = 1 / 60
W, H = 96, 30
NEW = ("JP Sequencer", "JP Chords", "JP Panel", "JP Tracker")
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


# ── JP Sequencer: the pad grid ───────────────────────────────────────────────

KICK = {"kick": 0.9, "snare": 0.0, "hat": 0.0}
SNARE = {"kick": 0.0, "snare": 0.9, "hat": 0.0}
HAT = {"kick": 0.0, "snare": 0.0, "hat": 0.9}
BAR = dict(tempo_bpm=120.0, bar_confidence=0.9)
HIT = dict(onsets=1, onset_strength=0.9)


def _chroma(*pitches):
    c = np.zeros(12, np.float32)
    c[list(pitches)] = 1.0
    return c


def _state(state, key):
    return next(v for k, v in state.items() if k[0] == key)


@pytest.mark.parametrize("drums, kind", [(KICK, "ring"), (SNARE, "cross"), (NO_DRUMS, "wipe")])
def test_each_drum_plays_its_own_show(drums, kind):
    st: dict = {}
    draw("JP Sequencer", st, drums=drums, onsets=1, onset_strength=0.2, **BAR)
    assert [s[0] for s in _state(st, "jp_pads")["shows"]] == [kind]


def test_a_hard_hit_sends_its_show_twice():
    st: dict = {}
    draw("JP Sequencer", st, drums=KICK, **HIT, **BAR)
    shows = _state(st, "jp_pads")["shows"]
    assert [s[0] for s in shows] == ["ring", "ring"] and shows[1][3] > shows[0][3]


def test_snares_take_turns_between_a_cross_and_an_x():
    st: dict = {}
    for i in range(2):
        draw("JP Sequencer", st, t=i * 0.02, drums=SNARE, onsets=1, onset_strength=0.2, **BAR)
    assert [s[0] for s in _state(st, "jp_pads")["shows"]] == ["cross", "x"]


def test_the_bar_opens_with_a_square():
    st: dict = {}
    draw("JP Sequencer", st, bar_phase=0.97, **BAR)
    draw("JP Sequencer", st, t=0.02, bar_phase=0.01, **BAR)
    assert [s[0] for s in _state(st, "jp_pads")["shows"]] == ["square"]


def test_a_kick_rings_out_from_the_centre():
    st: dict = {}
    draw("JP Sequencer", st, drums=KICK, **HIT, **BAR)
    draw("JP Sequencer", st, t=0.1, **BAR)
    glow = _state(st, "jp_pads")["glow"]
    assert glow[3:5, 3:5].max() > 0.2, "the ring has not left the centre"
    assert glow[0, 0] == 0.0 and glow[7, 7] == 0.0, "the ring is at the corners already"


def test_a_hat_blinks_a_few_pads_at_once():
    st: dict = {}
    draw("JP Sequencer", st, drums=HAT, **HIT, **BAR)
    lit = int((_state(st, "jp_pads")["glow"] > 0.5).sum())
    assert 2 <= lit <= 4 and not _state(st, "jp_pads")["shows"]


def test_the_grid_goes_dark_between_hits():
    st: dict = {}
    draw("JP Sequencer", st, drums=SNARE, **HIT, **BAR)
    for i in range(1, 90):
        draw("JP Sequencer", st, t=i / 60, **BAR)
    assert _state(st, "jp_pads")["glow"].max() < 0.02


def _lamps(codes):
    top = next(r for r in codes if (r != SPACE).any())
    return [int(c) for c in top if c != SPACE]


def test_the_top_lamps_run_across_the_bar():
    lamps = _lamps(draw("JP Sequencer", {}, bar_phase=0.55, **BAR)[0])
    assert len(lamps) == 8 and lamps.index(_LED) == 4


def test_without_a_beat_the_lamps_run_dim():
    lamps = _lamps(draw("JP Sequencer", {}, tempo_bpm=0.0)[0])
    assert _LED not in lamps and _TRAIL in lamps


# ── JP Chords ────────────────────────────────────────────────────────────────

def _listen(st, chroma, since=0.0, seconds=0.6):
    for i in range(int(seconds * 60)):
        draw("JP Chords", st, t=since + i / 60, chroma=chroma)
    return _state(st, "jp_chords")


@pytest.mark.parametrize("pitches, chord", [
    ((0, 4, 7), (0, "")), ((9, 0, 4), (9, "m")), ((9, 0, 4, 7), (9, "m7")),
    ((7, 11, 2, 5), (7, "7")), ((0, 7), (0, "5")),
])
def test_it_names_the_chord_it_hears(pitches, chord):
    assert _listen({}, _chroma(*pitches))["chord"] == chord


def test_no_chord_in_silence():
    st: dict = {}
    for i in range(30):
        draw("JP Chords", st, t=i / 60, level=0.0, chroma=_chroma(0, 4, 7))
    assert _state(st, "jp_chords")["chord"] is None


def test_a_passing_note_is_not_a_chord_change():
    st: dict = {}
    _listen(st, _chroma(0, 4, 7))
    got = _listen(st, _chroma(7, 11, 2), since=0.6, seconds=0.15)
    assert got["chord"] == (0, "")
    got = _listen(st, _chroma(7, 11, 2), since=0.75, seconds=0.8)
    assert got["chord"] == (7, "") and got["hist"] == [(0, "")]


# ── JP Panel and JP Tracker ──────────────────────────────────────────────────

def test_panel_lights_the_pad_the_drum_landed_on():
    st: dict = {}
    draw("JP Panel", st, onsets=1, onset_strength=0.9, drums=SNARE, bar_phase=0.25, **BAR)
    cur = _state(st, "jp_panel")["cur"]
    assert cur[1, 4] > 0 and cur.astype(bool).sum() == 1


def test_tracker_scrolls_a_hit_to_the_left():
    st: dict = {}
    draw("JP Tracker", st, onsets=1, onset_strength=0.9, drums=KICK, **BAR)
    at = int(np.flatnonzero(_state(st, "jp_tracker")["drums"][2])[-1])
    draw("JP Tracker", st, t=0.5, dt=0.5, **BAR)
    now = int(np.flatnonzero(_state(st, "jp_tracker")["drums"][2])[-1])
    assert now == at - 8, "half a second at 120 BPM is a beat: eight columns"


# ── through the real widget, on the drum corpus ──────────────────────────────

#: Beat response on ``four_on_floor``, measured when these were added, as in
#: ``tests/test_reactivity.py``. Not pinned there: an LED panel is sparse on
#: purpose and sits under that file's churn floor, as ``JP Bars`` does.
#: The Sequencer's light shows and the Tracker's scroll keep moving after a
#: hit by design, so they read near 1.0 here while in time; this only guards
#: them from falling.
RESPONSE = {"JP Sequencer": 1.6, "JP Chords": 5.4, "JP Panel": 3.9, "JP Tracker": 0.98}


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
