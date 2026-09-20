"""The runtime cache follows selection without changing the visible roster."""
from __future__ import annotations

from spektr.config import Settings
from spektr.widget import AudioVisualizer


def _viz() -> AudioVisualizer:
    return AudioVisualizer(settings=Settings())


def test_the_first_five_slots_begin_at_the_selected_mode():
    viz = _viz()
    assert viz._mode_window() == ["Bars", "Bricks", "Columns", "Ladder", "Mirror"]
    assert list(viz._mode_state) == ["Bars", "Bricks", "Columns", "Ladder", "Mirror"]


def test_the_window_advances_in_the_existing_mode_order():
    viz = _viz()
    original = viz.mode_names
    viz.set_mode("Wave")
    assert viz._mode_window() == ["Wave", "Scope", "ECG", "Strings", "Helix"]
    assert viz.mode_names == original


def test_the_five_slots_wrap_at_the_end_of_the_roster():
    viz = _viz()
    viz.set_mode("None")
    assert viz._mode_window() == ["None", "Bars", "Bricks", "Columns", "Ladder"]
    assert len(viz._mode_state) == 5


def test_state_is_reused_inside_the_window_and_evicted_outside_it():
    viz = _viz()
    bricks = viz._mode_state["Bricks"]
    bricks["kept"] = True

    viz.set_mode("Bricks")
    assert viz._mode_state["Bricks"] is bricks

    viz.set_mode("Wave")
    assert "Bricks" not in viz._mode_state
    assert list(viz._mode_state) == ["Wave", "Scope", "ECG", "Strings", "Helix"]
    assert len(viz._mode_state) == 5
