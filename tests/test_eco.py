"""Eco mode: what it does, when it switches itself on, and what it leaves alone."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spektr import config  # noqa: E402
from spektr.ui.app import Spektr  # noqa: E402
from spektr.ui.widget import AudioVisualizer  # noqa: E402


def _viz(**settings) -> AudioVisualizer:
    app = Spektr(settings=config.Settings(**settings))
    return AudioVisualizer(settings=app.settings)


def test_auto_is_the_default_and_starts_off():
    viz = _viz()
    assert viz.settings.eco == "auto"
    assert viz.eco_active() is False


@pytest.mark.parametrize("choice, active", [("on", True), ("off", False)])
def test_an_explicit_choice_is_obeyed(choice, active):
    assert _viz(eco=choice).eco_active() is active


def test_auto_turns_on_when_frames_cost_most_of_the_budget():
    viz = _viz()
    viz._build_ms = 1000.0 / 60 * 0.95          # 95% of a 60 fps budget
    for _ in range(viz.ECO_SETTLE_FRAMES):
        viz._maybe_eco()
    assert viz.eco_active() is True


def test_auto_stays_off_on_a_machine_that_keeps_up():
    viz = _viz()
    viz._build_ms = 2.0
    for _ in range(viz.ECO_SETTLE_FRAMES * 2):
        viz._maybe_eco()
    assert viz.eco_active() is False


def test_auto_turns_itself_back_off_when_the_cost_drops():
    viz = _viz()
    viz._build_ms = 1000.0 / 60 * 0.95
    for _ in range(viz.ECO_SETTLE_FRAMES):
        viz._maybe_eco()
    assert viz.eco_active() is True
    viz._build_ms = 1.0
    for _ in range(viz.ECO_SETTLE_FRAMES):
        viz._maybe_eco()
    assert viz.eco_active() is False


@pytest.mark.parametrize("choice", ["on", "off"])
def test_an_explicit_choice_is_never_overridden_by_measurement(choice):
    viz = _viz(eco=choice)
    viz._build_ms = 1000.0 / 60 * 0.95
    for _ in range(viz.ECO_SETTLE_FRAMES * 2):
        viz._maybe_eco()
    assert viz.eco_active() is (choice == "on")


def test_eco_holds_the_band_count_down():
    viz = _viz(eco="on", fps=60, bands=32)
    viz._apply_eco()
    assert viz.settings.bands <= config.ECO_BANDS


def test_leaving_eco_restores_the_band_count_the_user_chose():
    viz = _viz(eco="on", fps=60, bands=32)
    viz._apply_eco()
    viz.settings.eco = "off"
    viz._apply_eco()
    assert viz.settings.bands == 32


def test_shuffle_only_skips_modes_measured_over_the_budget():
    viz = _viz(eco="on")
    viz._target_fps = 60
    viz._mode_ms["Terra"] = 40.0        # far over a 16.7 ms budget
    viz._mode_ms["Bars"] = 1.0
    assert viz.affordable("Bars") is True
    assert viz.affordable("Terra") is False
    # a mode nobody has drawn yet is not excluded on a guess
    assert viz.affordable("Flame") is True
