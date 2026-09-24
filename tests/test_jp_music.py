"""The JP displays that hear the music say what they heard, and only that.

Each of these stands on one stream of the 0.6.0 analysis, and each is held
here to two things: it shows the thing it claims to (which drum, where the
bar is), and it draws nothing it was not told -- no blast without a kick, no
downbeat without a known bar, no beat marks without a beat.
"""
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
NEW = ("JP Demo", "JP Clock", "JP Ribbons", "JP Sequencer")
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
RESPONSE = {"JP Demo": 4.63, "JP Clock": 1.70, "JP Ribbons": 1.90, "JP Sequencer": 4.46}


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


# ── JP Demo ──────────────────────────────────────────────────────────────────

from spektr.render import BRAILLE_BASE  # noqa: E402


def lit_dots(codes) -> int:
    bits = (codes - BRAILLE_BASE).clip(0, 255).astype(np.uint8)
    return int(np.unpackbits(bits).sum())


def _demo_after(drums, secs=0.1, onset=True):
    st: dict = {}
    draw("JP Demo", st, t=0.0, level=0.1, onsets=int(onset), onset_strength=0.9, drums=drums)
    return draw("JP Demo", st, t=secs, level=0.1, drums=drums)[0]


def test_a_kick_blasts_the_burst():
    blast = _demo_after({"kick": 1.0, "snare": 0.0, "hat": 0.0})
    calm = _demo_after(NO_DRUMS)
    # The ring is a band two segments deep through every ray: on this frame
    # about a sixth more dots than the burst alone.
    assert lit_dots(blast) > lit_dots(calm) * 1.1


def test_a_snare_turns_the_burst():
    turned = _demo_after({"kick": 0.0, "snare": 1.0, "hat": 0.0}, secs=0.3)
    calm = _demo_after(NO_DRUMS, secs=0.3)
    assert not np.array_equal(turned, calm)


def test_no_drum_no_gesture():
    assert np.array_equal(_demo_after(NO_DRUMS), _demo_after(NO_DRUMS, onset=False))


# ── JP Clock ─────────────────────────────────────────────────────────────────

def test_each_bar_steps_inward():
    st: dict = {}
    for i in range(120):                       # one bar at 120 BPM
        draw("JP Clock", st, t=i * DT, level=0.6, tempo_bpm=120.0,
             bar_phase=(i * DT / 2.0) % 1.0, bar_confidence=0.9)
    rec = next(v for k, v in st.items() if k[0] == "jp_clock")["rec"]
    written = rec[0].copy()
    for i in range(120, 125):                  # over the downbeat
        draw("JP Clock", st, t=i * DT, level=0.0, tempo_bpm=120.0,
             bar_phase=(i * DT / 2.0) % 1.0, bar_confidence=0.9)
    rec = next(v for k, v in st.items() if k[0] == "jp_clock")["rec"]
    assert written.max() > 0.3
    assert np.allclose(rec[1], written)


def _marks_lit(**kw) -> int:
    """Cells of the beat marks drawn in a lit colour rather than the glass's.
    A lit mark and an unlit one are the same dots; only the colour says."""
    from spektr.modes.jp import recede_index

    st: dict = {}
    codes, cidx = draw("JP Clock", st, level=0.0, **kw)
    geo = next(v for k, v in st.items() if k[0] == "jp_clock_geo")
    # Cells holding mark dots and nothing from the face inside them: the tip
    # of the hand can share a cell with the mark at twelve.
    cells = geo["tick"].reshape(H, 4, W, 2).any(axis=(1, 3))
    inner = (geo["r"] < 0.87).reshape(H, 4, W, 2).any(axis=(1, 3))
    cells &= ~inner
    return int((cells & (cidx != recede_index(PAL))).sum())


def test_the_beat_mark_lights_and_only_with_a_beat():
    assert _marks_lit(tempo_bpm=120.0, bar_phase=0.01, bar_confidence=0.9) > 0
    assert _marks_lit(tempo_bpm=0.0) == 0


# ── JP Ribbons ───────────────────────────────────────────────────────────────

def test_a_kick_throws_the_low_ribbon():
    st_hit: dict = {}
    st_calm: dict = {}
    draw("JP Ribbons", st_hit, level=0.2, onsets=1, onset_strength=1.0,
         drums={"kick": 1.0, "snare": 0.0, "hat": 0.0}, tempo_bpm=120.0)
    draw("JP Ribbons", st_calm, level=0.2, tempo_bpm=120.0)
    hit = draw("JP Ribbons", st_hit, t=DT, level=0.2, tempo_bpm=120.0)[0]
    calm = draw("JP Ribbons", st_calm, t=DT, level=0.2, tempo_bpm=120.0)[0]
    low = slice(H // 2, H)
    assert lit_dots(hit[low]) > lit_dots(calm[low])


def test_the_ribbons_come_round_with_the_bar():
    a = draw("JP Ribbons", {}, level=0.3, tempo_bpm=120.0, bar_phase=0.3, bar_confidence=0.9)[0]
    b = draw("JP Ribbons", {}, t=7.0, level=0.3, tempo_bpm=120.0, bar_phase=0.3, bar_confidence=0.9)[0]
    assert np.array_equal(a, b)
