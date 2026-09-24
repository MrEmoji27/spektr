"""Do the beats reach the picture, and does the picture move on them?

``golden.py`` pins what a mode *draws* for a given set of bands. ``onset_score``
pins what the detector *finds* in a piece of audio. Neither of them covers the
stretch in between: the widget, the springs, the onset counter, the morph and
the two painters. A change there can leave every fingerprint and every F-score
exactly where it was while the app on screen stops moving to the music, which
is the one thing spektr is for.

This drives the real widget over the real analyser on the corpus from
``onset_eval``, one tick at a time on the signal's own clock, and measures two
things per mode:

``lost``
    beats the analyser published that no mode ever saw. Must be zero. The
    counter is differenced in exactly one place and consumed by whichever
    painter runs, so a second caller appearing on that path would eat beats
    without any other test noticing.

``beat_ratio``
    how much more the picture changes on the frames around a beat than on the
    frames between them. This is "reactive" written down. A mode whose ratio
    falls to 1.0 is still animating -- it has simply stopped animating to the
    music.

What this cannot measure
------------------------
The corpus is mono: ``onset_eval._stereo`` duplicates one channel into both,
so every frame's left and right are identical. A mode whose subject is the
*difference* between the channels therefore has nothing to draw here --
``Gonio`` is a stereo phase scope and sees a straight diagonal line whatever
the music does, so the 0.22 it measures is a property of this corpus and not
of the mode. Read a low number for a stereo mode as "not measured".

A low number is not by itself a fault either. Ask what the mode promises
before believing it: ``Bubbles`` never reads ``ctx.onsets`` and sits at 1.0
honestly. That cuts both ways -- ``Bars`` reads no rhythm field at all and
still measures 2.45, because it is redrawn from the level every frame -- so
"reads no onset field" excuses nothing on its own.

The clock is injected, like everywhere else offline. The widget reads the wall
clock for ``dt``, for ``ctx.t`` and for the morph, and a harness that renders
ten seconds of audio in one second of CPU would hand all three a ``dt`` of
nothing. Physics integrated in seconds then produces a picture that does not
move, and every number here would be a measurement of the harness.
"""
from __future__ import annotations

import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import spektr.modes as M  # noqa: E402
from spektr.analysis import HOP, Analyser, Frame  # noqa: E402
from spektr.capture import RingBuffer  # noqa: E402

#: Big enough that a mode has room to draw something, small enough that a few
#: hundred frames of it are quick.
WIDTH, HEIGHT = 120, 30

#: The rate the widget is driven at. Fixed rather than probed: this measures
#: the picture, and letting the pacer move would measure the machine.
FPS = 60


class _Driver:
    is_inline = False

    def write(self, text):
        pass

    def flush(self):
        pass


class _App:
    """Just enough app for the direct painter to decide it may write."""

    _overlay = None
    _notifications = ()

    def __init__(self):
        self._driver = _Driver()


class _Analyser:
    """Hands back whichever frame the harness has made current."""

    def __init__(self):
        self.frame = Frame()
        self.sensitivity = 1.0

    def set_bands(self, n):
        pass

    def start(self):
        pass

    def stop(self):
        pass


def published_frames(signal: np.ndarray, samplerate: int) -> list[tuple[float, Frame]]:
    """Every frame the analyser publishes, stamped on the signal's clock."""
    ring = RingBuffer(1 << 16)
    now = [0.0]
    an = Analyser(ring, lambda: samplerate, clock=lambda: now[0])
    an._ensure_plan(samplerate)
    out = []
    for start in range(0, signal.shape[0] - HOP + 1, HOP):
        ring.push(signal[start:start + HOP])
        now[0] = (start + HOP) / samplerate
        an._analyse_once()
        out.append((now[0], an._frame))
    return out


def _widget(clock):
    """A visualiser with a size, an app and a clock, and no terminal."""
    try:
        from spektr.ui.widget import AudioVisualizer
    except ImportError:  # pre-0.5.5 layout
        from spektr.widget import AudioVisualizer

    from textual.geometry import Region, Size

    class Offline(AudioVisualizer):
        def __init__(self):
            super().__init__()
            self._app = _App()

        @property
        def size(self):
            return Size(WIDTH, HEIGHT)

        @property
        def app(self):
            return self._app

        def refresh(self, *a, **kw):
            # Textual's compositor is not running; the direct painter is what
            # this harness is here to exercise.
            return self

        def _direct_region(self):
            return Region(0, 0, WIDTH, HEIGHT)

    del clock
    return Offline()


def measure(mode_name: str, signal, samplerate, clock, *, ticks=None) -> dict:
    """Drive ``mode_name`` over ``signal`` and report what the picture did.

    ``clock`` is the one-element list the caller has already patched
    ``time.monotonic`` to read, so the widget and this loop agree on when
    "now" is.
    """
    frames = published_frames(signal, samplerate)
    clock[0] = 0.0
    viz = _widget(clock)
    viz.analyser = _Analyser()
    viz._fps = FPS
    viz.mode_name = mode_name
    if hasattr(viz, "_refresh_mode_window"):
        viz._refresh_mode_window()

    mode = M.get(mode_name)
    assert mode is not None, f"no mode called {mode_name!r}"

    drawn: list[tuple[int, int, np.ndarray]] = []
    tick = [0]
    original = mode.fn

    def spy(ctx):
        out = original(ctx)
        drawn.append((tick[0], int(getattr(ctx, "onsets", 0)), _cells(out)))
        return out

    # Mode is frozen, so both registries are swapped rather than mutated: the
    # list the pickers read, and the dict every lookup goes through.
    spied = dataclasses.replace(mode, fn=spy)
    index = M.MODES.index(mode)
    by_name = getattr(M, "_BY_NAME", None)
    M.MODES[index] = spied
    if by_name is not None:
        by_name[mode_name] = spied

    published = 0
    try:
        dt = 1.0 / FPS
        t, i = 0.0, 0
        while i < len(frames) and (ticks is None or tick[0] < ticks):
            while i < len(frames) and frames[i][0] <= t:
                i += 1
            if i == 0:
                t += dt
                continue
            frame = frames[i - 1][1]
            viz.analyser.frame = frame
            # The beats the widget hands modes; before 0.6.0 every onset was one.
            published = max(published, getattr(frame, "accent_seq", frame.onset_seq))
            clock[0] = t
            tick[0] += 1
            viz._tick()
            if not hasattr(viz, "_picture"):
                # Before the direct painter, the strips were the only frame.
                viz._strips = None
                viz._build()
            t += dt
    finally:
        M.MODES[index] = mode
        if by_name is not None:
            by_name[mode_name] = mode

    return _summarise(drawn, published)


def _cells(out) -> np.ndarray:
    """Every layer a mode returned, flattened into one row of cells.

    All of them, not just the glyphs. A bar mode animates by changing which
    character sits in a cell, but a field mode fills the whole screen with one
    block and animates entirely in the colour index -- so a measure that reads
    ``codes`` alone calls Plasma, Swell and the Chladni modes perfectly still
    while they are visibly moving.
    """
    return np.concatenate([
        np.asarray(layer).ravel().astype(np.int64)
        for layer in out if layer is not None
    ])


def _summarise(drawn, published: int) -> dict:
    """The first draw of each tick is the one that reaches the screen."""
    seen_ticks: set[int] = set()
    onsets, pictures = [], []
    for tk, n, arr in drawn:
        if tk in seen_ticks:
            continue
        seen_ticks.add(tk)
        onsets.append(n)
        pictures.append(arr)

    out = {
        "published": published,
        "on_screen": int(sum(onsets)),
        "lost": published - int(sum(onsets)),
        "frames": len(pictures),
    }
    if len(pictures) < 4:
        return out

    churn = np.array([
        float(np.count_nonzero(pictures[i] != pictures[i - 1])) / pictures[i].size
        for i in range(1, len(pictures))
    ])
    hit = np.array(onsets[1:]) > 0
    # The frame a beat lands on and the three after it. A mode answers a beat
    # over a few frames; asking only about the frame itself would measure the
    # phase of the sampling rather than the response.
    near = hit.copy()
    for shift in (1, 2, 3):
        near[shift:] |= hit[:-shift]

    out["churn"] = round(float(churn.mean()), 5)
    if near.any() and (~near).any():
        out["beat_churn"] = round(float(churn[near].mean()), 5)
        out["idle_churn"] = round(float(churn[~near].mean()), 5)
        out["beat_ratio"] = round(
            out["beat_churn"] / max(out["idle_churn"], 1e-9), 3
        )
    return out
