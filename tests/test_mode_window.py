"""The runtime cache follows what is likely to be drawn next.

Five modes keep their working memory. Which five used to be the selected mode
and the four after it in menu order, which is what the cycle keys reach -- and
nothing else does: shuffle jumps anywhere, the picker jumps anywhere, and the
mode you just left was the first thing evicted, so going back to it paid its
whole setup again. Now the five are, in order: the mode on screen, the one it
is morphing out of, the one shuffle has already picked for next, the one used
before this, and then the menu order as before.
"""
from __future__ import annotations

from spektr.config import Settings
from spektr.widget import AudioVisualizer


def _viz() -> AudioVisualizer:
    return AudioVisualizer(settings=Settings())


def test_a_fresh_start_holds_the_selected_mode_and_the_next_four():
    viz = _viz()
    assert viz._mode_window() == ["Bars", "Bricks", "Columns", "Ladder", "Mirror"]
    assert list(viz._mode_state) == ["Bars", "Bricks", "Columns", "Ladder", "Mirror"]


def test_the_mode_you_left_is_kept():
    viz = _viz()
    viz.set_mode("Wave")
    assert viz._mode_window() == ["Wave", "Bars", "Scope", "ECG", "Strings"]


def test_the_roster_order_is_untouched():
    viz = _viz()
    original = viz.mode_names
    viz.set_mode("Wave")
    viz.set_mode("Pulse")
    assert viz.mode_names == original


def test_what_shuffle_picked_next_is_held_before_it_is_shown():
    viz = _viz()
    viz.set_mode("Wave")
    viz.expect("Valentine")
    window = viz._mode_window()
    assert window[:3] == ["Wave", "Valentine", "Bars"]
    assert "Valentine" in viz._mode_state


def test_the_morph_source_outranks_everything_but_the_mode_on_screen():
    viz = _viz()
    viz.set_mode("Wave")
    viz.expect("Valentine")
    viz.set_mode("Pulse", dissolve=True)
    assert viz._mode_window()[:2] == ["Pulse", "Wave"]


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
    viz.set_mode("Pulse")
    assert "Bricks" not in viz._mode_state
    assert len(viz._mode_state) == 5
