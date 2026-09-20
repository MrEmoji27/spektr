"""Chroma: twelve pitch classes folded out of a spectrum."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spektr.audio import chroma  # noqa: E402

SR = 48000
N = 4096


def spectrum(*freqs: float) -> np.ndarray:
    t = np.arange(N) / SR
    x = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    return np.abs(np.fft.rfft(x * np.hanning(N)))


def heard(*freqs: float) -> tuple[str, float]:
    return chroma.strongest(chroma.fold(spectrum(*freqs), SR))


@pytest.mark.parametrize("name, hz", [
    ("C", 261.63), ("D", 293.66), ("E", 329.63), ("G", 196.00),
    ("A", 440.00), ("A#", 466.16),
])
def test_a_single_note_is_named(name, hz):
    got, margin = heard(hz)
    assert got == name, f"{hz} Hz heard as {got}, not {name}"
    assert margin > 0.5, f"{name} stood only {margin:.2f} above the rest"


def test_every_octave_of_a_note_is_the_same_class():
    """The whole point of folding: C is C wherever it is played."""
    for freqs in ((130.81,), (261.63,), (523.25,), (130.81, 261.63, 523.25)):
        assert heard(*freqs)[0] == "C", f"{freqs} did not read as C"


def test_a_chord_puts_its_own_notes_on_top():
    c = chroma.fold(spectrum(261.63, 329.63, 392.00), SR)   # C major
    top = {n for n, _ in sorted(zip(chroma.NOTES, c), key=lambda p: -p[1])[:3]}
    assert top == {"C", "E", "G"}, f"C major read as {top}"


def test_silence_is_twelve_zeros_not_twelve_equals():
    """Equal values would read as every note at once, which is not silence."""
    out = chroma.fold(np.zeros(N // 2 + 1), SR)
    assert out.shape == (12,)
    assert not out.any()
    assert chroma.strongest(out) == ("", 0.0)


def test_noise_has_no_note_worth_acting_on():
    noise = np.abs(np.fft.rfft(np.random.default_rng(1).normal(size=N)))
    _, margin = chroma.strongest(chroma.fold(noise, SR))
    tonal = heard(440.0)[1]
    assert margin < 0.3, f"noise claimed a note by {margin:.2f}"
    assert tonal > margin * 2, "noise and a note should not look alike"


def test_the_result_says_nothing_about_loudness():
    quiet = chroma.fold(spectrum(440.0) * 0.001, SR)
    loud = chroma.fold(spectrum(440.0) * 1000.0, SR)
    assert np.allclose(quiet, loud, atol=1e-5)


def test_the_weights_are_built_once_per_rate():
    chroma.weights.cache_clear()
    bins = N // 2 + 1
    first = chroma.weights(bins, SR)
    assert chroma.weights(bins, SR) is first
    assert chroma.weights(bins, 44100) is not first


def test_every_class_carries_the_same_weight():
    """Without normalising, the high classes cover far more bins than the low
    ones and the picture describes the fold rather than the music."""
    w = chroma.weights(N // 2 + 1, SR)
    totals = w.sum(axis=1)
    assert np.allclose(totals, 1.0, atol=1e-5), f"uneven classes: {totals}"


def test_nothing_outside_the_readable_range_contributes():
    w = chroma.weights(N // 2 + 1, SR)
    freqs = np.fft.rfftfreq(N, d=1.0 / SR)
    outside = (freqs < chroma.LOW_HZ) | (freqs > chroma.HIGH_HZ)
    assert not w[:, outside].any()
