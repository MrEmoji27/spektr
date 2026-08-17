"""The Android bridge, checked against the engine it actually wraps.

This exists because the bridge rotted silently. It was written on 2026-08-15
against ``Analyser.snapshot()`` and a ``Frame`` carrying ``peaks``, ``energy``
and ``bars``; none of those are real, and by the time the port was picked up
again the engine had moved 101 commits. Every mode raised ``AttributeError``
on the first render and nothing said so, because no test imported the file.

The bridge is Python that ships inside an Android app, so it cannot be checked
by running the desktop app. It has to be checked here or not at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "android" / "app" / "src" / "main" / "python"))

spektr_android = pytest.importorskip("spektr_android")

RATE = 48000


def _pcm(seconds: float = 4096 / RATE) -> bytes:
    """Interleaved 16-bit stereo, the shape AudioRecord hands over."""
    t = np.arange(int(RATE * seconds), dtype=np.float32) / RATE
    sig = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 3000 * t)
    return (np.stack([sig, sig], axis=1).ravel() * 32767).astype("<i2").tobytes()


@pytest.fixture
def engine():
    return spektr_android.Engine(samplerate=RATE)


def test_the_engine_builds_and_lists_what_the_desktop_has(engine):
    import spektr.modes as M
    from spektr.palette import BUILTIN

    assert len(engine.mode_names()) == len(M.MODES)
    assert len(engine.theme_names()) == len(BUILTIN)


def test_every_mode_renders_through_the_wire_format(engine):
    """The failure this file was written for: all 64 raised on the first call."""
    pcm = _pcm()
    bad = []
    for name in engine.mode_names():
        try:
            for _ in range(6):
                engine.push(pcm)
                engine.render(name, 80, 24)
        except Exception as exc:                     # noqa: BLE001 — reporting
            bad.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not bad, "modes failed through the bridge:\n" + "\n".join(bad[:10])


def test_the_buffer_is_exactly_the_size_the_header_describes(engine):
    pcm = _pcm()
    for _ in range(6):
        engine.push(pcm)
    for w, h in ((80, 24), (37, 11), (200, 50)):
        buf = engine.render("Bars", w, h)
        magic, ver, planes, bw, bh = spektr_android._HEADER.unpack_from(buf, 0)
        assert magic == spektr_android._MAGIC
        assert ver == spektr_android.WIRE_VERSION
        assert (bw, bh) == (w, h)
        # codes are int32, each colour plane one byte a cell
        want = spektr_android._HEADER.size + w * h * 4 + w * h * (planes - 1)
        assert len(buf) == want, f"{w}x{h}: {len(buf)} bytes, header describes {want}"


def test_some_modes_carry_three_planes_and_the_count_says_so():
    """The trap the design document called out.

    Modes drawing with the half-block trick return a background ramp index as
    well. Kotlin that assumed two planes would not crash — it would render
    every one of them half-wrong, which is the worst kind of failure to ship.
    """
    engine = spektr_android.Engine(samplerate=RATE)
    pcm = _pcm()
    counts = {2: 0, 3: 0}
    for name in engine.mode_names():
        for _ in range(4):
            engine.push(pcm)
            buf = engine.render(name, 40, 12)
        planes = spektr_android._HEADER.unpack_from(buf, 0)[2]
        assert planes in (2, 3), f"{name}: {planes} planes"
        counts[planes] += 1
    assert counts[3] > 0, "no mode reported a background plane — is bidx dropped?"
    assert counts[2] > 0


def test_colour_indices_stay_inside_the_ramp(engine):
    """They are packed as one byte; a value past the ramp would wrap silently."""
    from spektr.palette import RAMP_STEPS

    pcm = _pcm()
    w, h = 60, 20
    for name in engine.mode_names():
        for _ in range(4):
            engine.push(pcm)
            buf = engine.render(name, w, h)
        planes = spektr_android._HEADER.unpack_from(buf, 0)[2]
        off = spektr_android._HEADER.size + w * h * 4
        for p in range(planes - 1):
            plane = np.frombuffer(buf, np.uint8, w * h, off + p * w * h)
            assert plane.max() < RAMP_STEPS, f"{name}: ramp index {plane.max()}"


def test_switching_mode_does_not_hand_over_the_last_one_s_scratch(engine):
    """Same rule the desktop widget follows, and for the same reason."""
    pcm = _pcm()
    engine.push(pcm)
    engine.render("Tunnel", 60, 20)
    engine.render("Chladni", 60, 20)          # different shapes in scratch
    engine.render("Tunnel", 60, 20)           # must not read Chladni's arrays


def test_a_resize_mid_session_is_just_two_different_numbers(engine):
    pcm = _pcm()
    engine.push(pcm)
    for w, h in ((80, 24), (200, 60), (40, 10), (80, 24)):
        buf = engine.render("Bars", w, h)
        assert spektr_android._HEADER.unpack_from(buf, 0)[3:] == (w, h)


def test_silence_renders_rather_than_raising(engine):
    """No audio pushed at all — the app opens before the user consents."""
    engine.render("Bars", 80, 24)


def test_an_unknown_mode_is_a_keyerror_not_a_wrong_picture(engine):
    with pytest.raises(KeyError):
        engine.render("No Such Mode", 80, 24)


def test_the_engine_smooths_like_the_desktop_widget():
    """The motion layer is between the analyser and the modes, and easy to miss.

    ``Analyser`` publishes raw bands and no peaks at all; every bit of the
    easing that makes spektr look like spektr lives in the widget. A port that
    forwards the Frame straight to the modes renders the same shapes, jittering
    — so this asserts the springs are actually in the path, by checking that
    one loud block does not arrive at full height in one frame.
    """
    engine = spektr_android.Engine(samplerate=RATE)
    engine.push(_pcm())
    engine.render("Bars", 80, 24)
    first = float(engine._spring.x.max())
    for _ in range(30):
        engine.push(_pcm())
        engine.render("Bars", 80, 24)
    settled = float(engine._spring.x.max())
    assert first < settled, "bands arrived instantly — the spring is not in the path"
    assert engine._peaks.value is not None
