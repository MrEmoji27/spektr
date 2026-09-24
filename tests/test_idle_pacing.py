"""In silence, with nothing moving, spektr stops drawing sixty times a second.

People leave it open. A still picture redrawn at full rate is a mode, a
picture and a diff every frame, to write nothing. After a second of silence
with nothing on screen changing, the widget drops to ``IDLE_FPS``; the first
sound, or anything changing on screen, puts the rate back where it was.
"""
from __future__ import annotations

import time

import numpy as np

from spektr.analysis import Frame
from spektr.config import Settings
from spektr.ui import widget as W
from spektr.widget import AudioVisualizer


class _Stub:
    def __init__(self):
        self.frame = Frame()


def _viz(monkeypatch, fps=60):
    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    viz = AudioVisualizer(settings=Settings(fps=fps))
    viz.analyser = _Stub()
    viz._fps = viz._target_fps = fps
    rates = []

    def retime(f, *, requested=False):
        if requested:
            viz._idle_from = None
            viz._still_since = None
        viz._fps = f
        rates.append(f)

    monkeypatch.setattr(viz, "_retime", retime)
    monkeypatch.setattr(viz, "_painting_directly", lambda: True)
    wrote = [False]

    def paint():
        viz._painted_nothing = not wrote[0]

    monkeypatch.setattr(viz, "_paint", paint)
    return viz, now, rates, wrote


def _run(viz, now, secs, frame, fps=60):
    for _ in range(int(secs * fps)):
        now[0] += 1.0 / fps
        viz.analyser.frame = frame
        viz._tick()


SILENT = Frame(silent=True)
LOUD = Frame(bands=np.full(32, 0.5), bands_l=np.full(32, 0.5),
             bands_r=np.full(32, 0.5), silent=False)


def test_silence_with_a_still_picture_idles(monkeypatch):
    viz, now, rates, _ = _viz(monkeypatch)
    _run(viz, now, 0.5, SILENT)
    assert viz._fps == 60, "idled before the silence had lasted"
    _run(viz, now, 1.0, SILENT)
    assert viz._fps == W.IDLE_FPS


def test_the_first_sound_puts_the_rate_back(monkeypatch):
    viz, now, rates, _ = _viz(monkeypatch)
    _run(viz, now, 2.0, SILENT)
    assert viz._fps == W.IDLE_FPS
    _run(viz, now, 1 / 60, LOUD)
    assert viz._fps == 60


def test_a_picture_that_moves_in_silence_is_not_idled(monkeypatch):
    """Modes that animate on their own in silence keep their full rate: the
    painter writes something every frame, so the widget is not still."""
    viz, now, rates, wrote = _viz(monkeypatch)
    wrote[0] = True
    _run(viz, now, 3.0, SILENT)
    assert viz._fps == 60 and not rates


def test_music_never_idles(monkeypatch):
    viz, now, rates, _ = _viz(monkeypatch)
    _run(viz, now, 3.0, LOUD)
    assert viz._fps == 60


def test_a_rate_asked_for_while_idle_is_kept(monkeypatch):
    viz, now, rates, _ = _viz(monkeypatch)
    _run(viz, now, 2.0, SILENT)
    viz._retime(30, requested=True)
    _run(viz, now, 1 / 60, LOUD)
    assert viz._fps == 30
