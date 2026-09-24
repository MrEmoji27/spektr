"""The motion profile setting, held to the things that would break it.

``motion`` is a display-feel switch wired through four layers that have to
agree: ``config`` owns the persisted name, ``motion`` owns the parameter
table those names map to, the widget applies them live, and the settings
panel offers them as stops. A name that exists on one side and not another
means either a row that saves a value the springs have never heard of or a
profile nobody can pick — so the first test here is that the two tables are
the same table.

The rest is the physics: ``spread`` may only ever raise a bar (it is a
maximum against a leak, not an average), a flat field must be a fixed point,
and the glide profile really must be slower than snappy — a profile toggle
that doesn't change the response is a row that lies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spektr import config  # noqa: E402
from spektr.app import Spektr  # noqa: E402
from spektr.config import Settings  # noqa: E402
from spektr.motion import (  # noqa: E402
    GLIDE_SPREAD_DECAY,
    PROFILES,
    Spring,
    spread,
)
from spektr.widget import AudioVisualizer  # noqa: E402

DT = 1.0 / 60.0


def _viz(**settings) -> tuple[AudioVisualizer, Settings]:
    """A widget built the way the real app builds one, plus its settings."""
    app = Spektr(settings=Settings(**settings))
    return AudioVisualizer(settings=app.settings), app.settings


# ── the two tables are one ───────────────────────────────────────────────────

def test_config_choices_and_spring_profiles_are_the_same_set():
    """A choice the panel offers must be a profile the widget can tune to."""
    assert set(config.MOTION_CHOICES) == set(PROFILES)
    assert config.MOTION_DEFAULT in PROFILES


@pytest.mark.parametrize("name", config.MOTION_CHOICES)
def test_every_choice_survives_clamp(name):
    """Picking a stop, closing the panel, and finding it did not stick."""
    assert Settings(motion=name).clamp().motion == name


def test_an_unknown_profile_falls_back_rather_than_breaking():
    """A config written by another build carries no guarantees."""
    assert Settings(motion="junk").clamp().motion == config.MOTION_DEFAULT


# ── spread: cava's monstercat analogue ───────────────────────────────────────

def test_a_hot_band_leaks_to_its_neighbours_and_nowhere_lower():
    """Distance-d neighbours get decay^d; the peak itself is untouched."""
    out = spread(np.array([0.0, 0.0, 1.0, 0.0, 0.0]))
    assert out[2] == 1.0
    assert out[1] == pytest.approx(GLIDE_SPREAD_DECAY)
    assert out[3] == pytest.approx(GLIDE_SPREAD_DECAY)
    assert out[0] == pytest.approx(GLIDE_SPREAD_DECAY * GLIDE_SPREAD_DECAY)
    assert out[4] == pytest.approx(GLIDE_SPREAD_DECAY * GLIDE_SPREAD_DECAY)


def test_spread_can_only_raise_a_bar():
    """It is a maximum against a leak, not a blur — nothing may lose height."""
    rng = np.random.default_rng(7)
    values = rng.uniform(0.0, 1.0, 32)
    assert (spread(values) >= values - 1e-12).all()


def test_spread_leaves_a_flat_field_exactly_as_it_was():
    """A plateau is a fixed point: max(v, v·decay) is v."""
    values = np.full(16, 0.42)
    assert spread(values) == pytest.approx(values)


def test_spread_preserves_the_maximum():
    values = np.zeros(8)
    values[3] = 0.9
    assert spread(values).max() == pytest.approx(0.9)


# ── the profiles move differently ────────────────────────────────────────────

def _height_after(profile: str, hold: float, tail: float) -> float:
    spring = Spring(4, **PROFILES[profile])
    high = np.full(4, 1.0)
    zero = np.zeros(4)
    for _ in range(int(hold / DT)):
        spring.step(high, DT)
    for _ in range(int(tail / DT)):
        spring.step(zero, DT)
    return float(spring.x.mean())


def test_glide_rises_more_slowly_than_snappy():
    """The whole point of the profile: transients swell instead of snapping."""
    spring = Spring(4, **PROFILES["glide"])
    snappy = Spring(4, **PROFILES["snappy"])
    target = np.full(4, 1.0)
    for _ in range(int(0.15 / DT)):
        spring.step(target, DT)
        snappy.step(target, DT)
    assert spring.x.mean() < snappy.x.mean()


def test_glide_falls_more_slowly_than_snappy():
    """And bars sink lazily rather than collapsing — cava's gravity look."""
    assert _height_after("glide", 0.6, 0.3) > _height_after("snappy", 0.6, 0.3)


# ── live application through the widget ──────────────────────────────────────

def test_set_motion_retunes_the_live_springs():
    """Switching profiles changes the constants in place, not just the label."""
    viz, settings = _viz()
    before = viz._spring._wa
    settled = viz.set_motion("glide")
    assert settled == "glide"
    assert settings.motion == "glide"
    assert viz._motion == "glide"
    assert viz._spring._wa == pytest.approx(5.0 / PROFILES["glide"]["attack"])
    # Stereo follows the main spring — one personality at a time.
    assert viz._stereo_l._wa == pytest.approx(viz._spring._wa)
    assert before != viz._spring._wa


def test_set_motion_rejects_unknown_names_without_touching_anything():
    viz, settings = _viz()
    wa = viz._spring._wa
    assert viz.set_motion("nope") == viz._motion
    assert settings.motion != "nope"
    assert viz._spring._wa == wa


def test_a_saved_profile_is_built_in_at_construction():
    viz, settings = _viz(motion="glide")
    assert viz._motion == "glide"
    assert viz._spring._wa == pytest.approx(5.0 / PROFILES["glide"]["attack"])


# ── the panel binding ────────────────────────────────────────────────────────

def test_the_panel_offers_motion_and_reads_it_back():
    """The row exists, lists every profile, and shows the current one."""
    app = Spektr(settings=Settings())
    viz = AudioVisualizer(settings=app.settings)
    rows, values = app._settings_rows(viz, app.settings)
    by_key = {r.key: r for r in rows}
    assert "motion" in by_key
    row = by_key["motion"]
    assert len(row.choices) == len(config.MOTION_CHOICES)
    assert values["motion"] == Settings().motion


def test_the_panel_row_applies_through_the_widget():
    """Stepping the row lands in settings AND in the moving parts."""
    app = Spektr(settings=Settings())
    viz = AudioVisualizer(settings=app.settings)
    rows, values = app._settings_rows(viz, app.settings)
    row = next(r for r in rows if r.key == "motion")
    row.apply(row.choices[-1])
    assert app.settings.motion == row.choices[-1]
    assert viz._spring._wa == pytest.approx(
        5.0 / PROFILES[row.choices[-1]]["attack"]
    )



# ── accents punch through glide ──────────────────────────────────────────────

def _glide_widget(monkeypatch):
    """An offline visualiser on glide, driven by a clock the test owns."""
    import time

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import reactivity as R

    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    viz = R._widget(now)
    viz.analyser = R._Analyser()
    viz.set_motion("glide")
    return viz, now


def _rise(monkeypatch, accented: bool, frames: int = 4) -> float:
    """How high a hit gets in ``frames`` ticks at 60 fps, from a quiet bed."""
    from spektr.analysis import Frame

    viz, now = _glide_widget(monkeypatch)
    quiet = np.full(32, 0.05)
    loud = np.full(32, 0.9)
    for _ in range(40):
        now[0] += 1 / 60
        viz.analyser.frame = Frame(bands=quiet, bands_l=quiet, bands_r=quiet)
        viz._tick()
    hit = Frame(bands=loud, bands_l=loud, bands_r=loud,
                accent_seq=1 if accented else 0)
    for _ in range(frames):
        now[0] += 1 / 60
        viz.analyser.frame = hit
        viz._tick()
    return float(viz._spring.x.mean())


def test_glide_lets_the_beat_through(monkeypatch):
    """Glide rounds off transients, which is the point of it -- and what made
    the kick and snare look mushy while the hats still set off every beat
    effect. An accented hit rises at snappy's speed; the same level without
    an accent keeps glide's."""
    beat = _rise(monkeypatch, accented=True)
    other = _rise(monkeypatch, accented=False)
    assert beat >= 0.6, f"an accented hit only reached {beat:.2f}"
    assert other < beat * 0.6, (beat, other)


def test_the_punch_lasts_one_hit_not_the_passage(monkeypatch):
    """Once the hit has landed, glide is glide again: a sustained rise long
    after the accent must not be rushed."""
    from spektr.analysis import Frame
    from spektr.motion import GLIDE_PUNCH_S

    viz, now = _glide_widget(monkeypatch)
    quiet = np.full(32, 0.05)
    frame = Frame(bands=quiet, bands_l=quiet, bands_r=quiet, accent_seq=1)
    for _ in range(int((GLIDE_PUNCH_S + 0.3) * 60)):
        now[0] += 1 / 60
        viz.analyser.frame = frame
        viz._tick()
    assert now[0] >= viz._punch_until
