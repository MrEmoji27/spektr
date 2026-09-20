"""Shuffle transitions and song-change timing stay separate from manual input."""
from __future__ import annotations

import asyncio

import numpy as np

from spektr import config, dissolve
from spektr.analysis import Frame
from spektr.app import Spektr
from spektr.platform.nowplaying import Track


def test_shuffle_requests_a_dissolve(monkeypatch, tmp_path):
    async def run() -> None:
        app = Spektr(
            settings=config.Settings(shuffle_scope="modes", fps=15),
            config_dir=tmp_path,
        )
        app.notify = lambda *a, **k: None  # type: ignore[method-assign]
        async with app.run_test(size=(80, 24)):
            calls = []
            original = app.viz.set_mode

            def record(name, **kwargs):
                calls.append((name, kwargs))
                return original(name, **kwargs)

            monkeypatch.setattr(app.viz, "set_mode", record)
            monkeypatch.setattr("spektr.ui.app.random.choice", lambda choices: choices[0])
            app._shuffle_tick()
            assert calls and calls[0][1] == {"dissolve": True}

    asyncio.run(run())


def test_manual_mode_switches_morph_too(tmp_path):
    """Only shuffle used to morph. Every switch a person makes does now.

    Including the ones that come from a key, which is why the span is the
    short one: a morph the length of a shuffle under an `m` press reads as a
    key that did nothing.
    """
    async def run() -> None:
        app = Spektr(settings=config.Settings(fps=15), config_dir=tmp_path)
        async with app.run_test(size=(80, 24)):
            viz = app.viz
            viz.set_mode("Wave")  # a stored or restored mode is not a switch
            assert viz._dissolve_from is None
            viz.cycle_mode()
            assert viz._dissolve_from == "Wave"
            assert viz._morph_span() == dissolve.FAMILY_SECONDS

            viz.cycle_mode(-1)
            assert viz._dissolve_from is not None

            # a mode set from a flag, a config file or a plugin swap still cuts
            viz.set_mode("Bars")
            assert viz._dissolve_from is None

    asyncio.run(run())


def test_a_removed_mode_does_not_morph(tmp_path):
    """Quarantine moves you somewhere safe rather than animating a broken
    picture as it leaves: there is nothing worth showing for the length of a
    morph, and the mode it came from is the one that just failed."""
    async def run() -> None:
        app = Spektr(settings=config.Settings(fps=15), config_dir=tmp_path)
        async with app.run_test(size=(80, 24)):
            viz = app.viz
            viz.set_mode("Wave")
            viz._quarantine_mode("Wave", "boom")
            assert viz.mode_name != "Wave"
            assert viz._dissolve_from is None

    asyncio.run(run())


def test_the_picker_previews_instantly_and_morphs_on_the_commit(tmp_path):
    """Browsing the list changes modes as fast as you can press a key, so it
    cuts. The choice morphs — out of the mode the picker was opened on, since
    the one it commits has been on screen the whole time."""
    async def run() -> None:
        app = Spektr(settings=config.Settings(fps=15), config_dir=tmp_path)
        async with app.run_test(size=(80, 24)):
            viz = app.viz
            viz.set_mode("Bars")
            viz._build()  # Bars has drawn the frame being held
            viz.preview_mode("Wave")
            assert viz.mode_name == "Wave"
            assert viz._dissolve_from is None, "browsing must not morph"
            viz._build()  # and now Wave has

            viz.commit_mode(dissolve=True)
            assert viz.mode_name == "Wave"
            assert viz._dissolve_from == "Bars"
            assert viz._morph_span() == dissolve.FAMILY_SECONDS
            # The held frame is Wave's, so it cannot stand in for Bars: the
            # outgoing picture is redrawn rather than frozen from the wrong one.
            assert viz._frozen_old is None

            # and the morph frame itself builds: two modes drawn, blended, and
            # handed to the strip builder
            assert viz._build()

    asyncio.run(run())


def test_cancelling_a_preview_morphs_back(tmp_path):
    """Escape is the end of the interaction, not another preview, so it is a
    mode change like any other and gets the morph."""
    async def run() -> None:
        app = Spektr(settings=config.Settings(fps=15), config_dir=tmp_path)
        async with app.run_test(size=(80, 24)):
            viz = app.viz
            viz.set_mode("Bars")
            viz.preview_mode("Wave")
            viz.cancel_mode_preview()
            assert viz.mode_name == "Bars"
            assert viz._dissolve_from == "Wave"
            assert viz._preview_mode is None

    asyncio.run(run())


def test_both_modes_render_during_a_dissolve(monkeypatch, tmp_path):
    async def run() -> None:
        app = Spektr(settings=config.Settings(fps=15), config_dir=tmp_path)
        async with app.run_test(size=(80, 24)):
            viz = app.viz
            calls: list[str] = []

            def frame(name, _audio, _w, _h, _onsets):
                calls.append(name)
                codes = np.zeros((viz.size.height, viz.size.width), dtype=np.int32)
                colours = np.zeros_like(codes)
                return codes, colours

            monkeypatch.setattr(viz, "_render_mode", frame)
            viz.set_mode("Wave", dissolve=True)
            calls.clear()
            viz._build()
            assert calls == ["Wave", "Bars"]

    asyncio.run(run())


def test_a_beat_mid_morph_pushes_the_travel(tmp_path):
    """Onsets that land while a morph is running move its travel on, and are
    capped: a drum fill must not finish the morph on its own."""
    async def run() -> None:
        app = Spektr(settings=config.Settings(fps=15), config_dir=tmp_path)
        async with app.run_test(size=(80, 24)):
            viz = app.viz
            viz.set_mode("Wave", dissolve=True, quick=True)
            assert viz._dissolve_from == "Bars"
            assert viz._dissolve_push == 0.0

            viz._last_onset_seq = 0
            viz._frame_data = Frame(onset_seq=1)  # one beat before the next frame
            viz._build()
            assert viz._dissolve_push == dissolve.PUSH

            viz._frame_data = Frame(onset_seq=1)  # a frame with no beat on it
            viz._build()
            assert viz._dissolve_push == dissolve.PUSH

            viz._frame_data = Frame(onset_seq=3)  # two at once, counted as two
            viz._build()
            assert viz._dissolve_push == dissolve.PUSH * 3

            viz._frame_data = Frame(onset_seq=30)  # and the cap holds
            viz._build()
            assert viz._dissolve_push == dissolve.PUSH_MAX

    asyncio.run(run())


def test_track_timing_uses_changes_after_the_first_observation(monkeypatch, tmp_path):
    async def run() -> None:
        app = Spektr(
            settings=config.Settings(shuffle=True, shuffle_timing="track"),
            config_dir=tmp_path,
        )
        shuffled = []
        monkeypatch.setattr(app, "_shuffle_tick", lambda: shuffled.append(True))
        tracks = iter(
            [
                Track("First", "Artist"),
                Track("First", "Artist"),
                None,
                Track("Second", "Artist"),
            ]
        )

        async def current():
            return next(tracks)

        monkeypatch.setattr("spektr.ui.app.nowplaying.current", current)
        for _ in range(4):
            await app._poll_now_playing()
        assert shuffled == [True]

    asyncio.run(run())


def test_song_change_timing_is_off_by_default_and_clamped():
    assert config.Settings().shuffle_timing == "timer"
    assert config.Settings(shuffle_timing="track").clamp().shuffle_timing == "track"
    assert config.Settings(shuffle_timing="unknown").clamp().shuffle_timing == "timer"
