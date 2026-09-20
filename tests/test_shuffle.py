"""Shuffle transitions and song-change timing stay separate from manual input."""
from __future__ import annotations

import asyncio

import numpy as np

from spektr import config
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


def test_manual_mode_switches_are_instant(tmp_path):
    async def run() -> None:
        app = Spektr(settings=config.Settings(fps=15), config_dir=tmp_path)
        async with app.run_test(size=(80, 24)):
            app.viz.set_mode("Wave")
            assert app.viz._dissolve_from is None
            app.viz.cycle_mode()
            assert app.viz._dissolve_from is None

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
