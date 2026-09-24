"""A mode's first frame is paid for before it is on screen.

Import is cheap. What is not is the first frame: a mode builds its geometry,
look-up tables and grids into scratch the first time it draws at a size, and
for the heaviest that is 20 to 35 ms at 200x50 against a 16.7 ms frame --
landing on exactly the frame a morph starts, which is where a stutter shows
most. The warmer runs that first frame on a background thread for the modes
the window says are likely next, so the switch finds its scratch already
built.
"""
from __future__ import annotations

import threading
import time

import numpy as np

import spektr.modes as M
from spektr.config import Settings
from spektr.palette import BUILTIN, Palette
from spektr.ui.warm import ModeWarmer, warm_ctx
from spektr.widget import AudioVisualizer

W, H = 200, 50
PAL = Palette(BUILTIN["gruvbox"])


def _first_frame_ms(name: str, state: dict) -> float:
    ctx = warm_ctx(W, H, PAL, state)
    t0 = time.perf_counter()
    M.get(name).fn(ctx)
    return (time.perf_counter() - t0) * 1000.0


def test_a_warmed_mode_draws_its_first_frame_at_its_steady_cost():
    cold = _first_frame_ms("Valentine", {})
    warmer = ModeWarmer(settle=0.0)
    state: dict = {}
    warmer.request("Valentine", state, W, H, PAL)
    assert warmer.wait("Valentine", timeout=10.0)
    warm = _first_frame_ms("Valentine", state)
    assert warm < cold * 0.5, (cold, warm)
    warmer.stop()


def test_the_warmer_works_off_the_calling_thread():
    seen = []

    def spy(ctx):
        seen.append(threading.current_thread())
        return M.empty(ctx.w, ctx.h)

    M.mode("Warm Spy", group="test")(spy)
    warmer = ModeWarmer(settle=0.0)
    try:
        warmer.request("Warm Spy", {}, W, H, PAL)
        assert warmer.wait("Warm Spy", timeout=10.0)
    finally:
        m = M.get("Warm Spy")
        M.MODES.remove(m)
        M._BY_NAME.pop("Warm Spy", None)
        warmer.stop()
    assert seen and all(t is not threading.main_thread() for t in seen)


def test_a_mode_that_fails_to_warm_is_left_for_the_render_path():
    """The render path owns failures: it quarantines, it reports. A warm-up
    that raised must not take the app down or mark the mode as ready."""
    warmer = ModeWarmer(settle=0.0)
    state: dict = {}

    def boom(ctx):
        raise RuntimeError("nope")

    M.mode("Warm Boom", group="test")(boom)
    try:
        warmer.request("Warm Boom", state, W, H, PAL)
        assert warmer.wait("Warm Boom", timeout=10.0)
    finally:
        m = M.get("Warm Boom")
        M.MODES.remove(m)
        M._BY_NAME.pop("Warm Boom", None)
        warmer.stop()


def test_the_warm_up_does_not_touch_what_the_mode_will_draw():
    """A warmed mode's first real frame must be the frame it would have drawn
    cold: the warm-up builds scratch, it does not advance the animation."""
    cold_state: dict = {}
    warm_state: dict = {}
    warmer = ModeWarmer(settle=0.0)
    warmer.request("Bars", warm_state, W, H, PAL)
    assert warmer.wait("Bars", timeout=10.0)
    warmer.stop()
    a = M.get("Bars").fn(warm_ctx(W, H, PAL, cold_state, level=0.5))
    b = M.get("Bars").fn(warm_ctx(W, H, PAL, warm_state, level=0.5))
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_the_widget_warms_what_its_window_holds():
    viz = AudioVisualizer(settings=Settings())
    viz.set_mode("Wave")
    viz.expect("Valentine")
    viz._warm_window(W, H)
    assert viz._warmer.wait("Valentine", timeout=10.0)
    assert viz._mode_state["Valentine"], "Valentine's scratch was not built"
    viz._warmer.stop()


def test_rendering_a_mode_mid_warm_waits_rather_than_racing():
    """Two threads inside one mode's scratch at once is corruption. The render
    path waits for a warm-up of the mode it is about to draw."""
    viz = AudioVisualizer(settings=Settings())
    gate = threading.Event()
    started = threading.Event()
    original = M.get("Bars").fn

    import dataclasses

    def slow(ctx):
        started.set()
        gate.wait(5.0)
        return original(ctx)

    m = M.get("Bars")
    spy = dataclasses.replace(m, fn=slow)
    idx = M.MODES.index(m)
    M.MODES[idx] = spy
    M._BY_NAME["Bars"] = spy
    try:
        viz._warmer.request("Bars", viz._mode_state["Bars"], W, H, viz.palette)
        assert started.wait(5.0)
        threading.Timer(0.2, gate.set).start()
        t0 = time.perf_counter()
        viz._warmer.wait("Bars", timeout=10.0)
        assert time.perf_counter() - t0 >= 0.1
        assert not viz._warmer.busy("Bars")
    finally:
        M.MODES[idx] = m
        M._BY_NAME["Bars"] = m
        viz._warmer.stop()
