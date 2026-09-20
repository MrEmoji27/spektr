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


def test_eco_is_off_unless_asked_for():
    viz = _viz()
    assert viz.settings.eco == "off"
    assert viz.eco_active() is False


@pytest.mark.parametrize("choice, active", [("on", True), ("off", False)])
def test_an_explicit_choice_is_obeyed(choice, active):
    assert _viz(eco=choice).eco_active() is active


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


# ── morph between modes ──────────────────────────────────────────────────────


def test_modes_in_one_family_are_recognised():
    viz = _viz()
    assert viz._same_family("Bars", "Bricks") is True        # both spectrum
    assert viz._same_family("Bars", "Terra") is False
    assert viz._same_family("Bars", "Not A Mode") is False


def test_a_family_switch_keeps_the_outgoing_mode_drawing():
    """Its bars must go on bouncing while they become the new shape."""
    viz = _viz()
    viz._target_fps = viz._fps = 60
    viz.mode_name = "Bricks"
    viz._mode_ms.update({"Bars": 1.0, "Bricks": 1.0})
    viz._frozen_old = ("frozen",)
    drawn = []
    viz._render_mode = lambda name, *a: drawn.append(name) or ("live",)
    assert viz._outgoing("Bars", None, 80, 24, 0) == ("live",)
    assert drawn == ["Bars"]


def test_an_expensive_switch_across_families_holds_the_last_frame():
    import numpy as np

    viz = _viz()
    viz._target_fps = viz._fps = 60
    viz.mode_name = "Terra"
    viz._mode_ms.update({"Terra": 30.0, "Chladni Extreme (o)": 20.0})
    frozen = (np.zeros((24, 80), dtype=np.int32), np.zeros((24, 80), dtype=np.int32))
    viz._frozen_old = frozen
    viz._render_mode = lambda *a: pytest.fail("should not re-render an expensive pair")
    assert viz._outgoing("Chladni Extreme (o)", None, 80, 24, 0) is frozen
