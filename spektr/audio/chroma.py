"""The twelve pitch classes, folded out of a spectrum spektr already has.

A band tells you where the energy is in hertz. Chroma tells you what note it
is: every C in every octave lands in the same bin, so a chord reads as a shape
rather than as a spread. It is what lets a mode colour by harmony instead of by
height, and it is the first half of naming a key.

The work is a matrix multiply against weights that depend only on the sample
rate and the window size, so they are built once and reused. Nothing here
decides what key a piece is in: that needs a long memory and a confidence, and
:mod:`spektr.audio.key` is where it will live.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

#: Note names, starting at C, which is where pitch-class 0 sits by convention.
NOTES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

#: Concert pitch. A4 = 440 Hz puts every other note where the rest of the
#: world expects it; spektr never tunes to anything else, and a recording that
#: is a few cents off simply spreads over two bins rather than failing.
A4_HZ = 440.0

#: The range chroma is read over. Below this, one FFT bin is wider than a
#: semitone and the fold is a guess; above it, what is left is mostly cymbals
#: and air, which have no pitch to contribute and would flatten the picture.
LOW_HZ = 65.0      # about C2
HIGH_HZ = 2100.0   # about C7


def midi_of(hz: float) -> float:
    """The MIDI note number of a frequency, fractional."""
    return 69.0 + 12.0 * np.log2(max(hz, 1e-9) / A4_HZ)


@lru_cache(maxsize=8)
def weights(bins: int, rate: float) -> np.ndarray:
    """A ``(12, bins)`` matrix folding an FFT magnitude spectrum into classes.

    Each bin is spread across the two nearest pitch classes in proportion to
    how close it sits to them, so a note between two bins does not jump from
    one class to the other as it drifts. Bins outside :data:`LOW_HZ` to
    :data:`HIGH_HZ` contribute nothing.
    """
    freqs = np.fft.rfftfreq((bins - 1) * 2, d=1.0 / rate)
    out = np.zeros((12, bins), dtype=np.float32)
    usable = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    if not usable.any():
        return out

    midi = 69.0 + 12.0 * np.log2(np.where(usable, freqs, A4_HZ) / A4_HZ)
    lower = np.floor(midi)
    frac = (midi - lower).astype(np.float32)
    lo = (lower.astype(np.int64) % 12)
    hi = ((lower.astype(np.int64) + 1) % 12)

    idx = np.flatnonzero(usable)
    np.add.at(out, (lo[idx], idx), 1.0 - frac[idx])
    np.add.at(out, (hi[idx], idx), frac[idx])

    # Each class ends up covering a different number of bins — the high ones
    # far more than the low — so without this a picture of the chroma would
    # say more about the fold than about the music.
    per_class = out.sum(axis=1, keepdims=True)
    np.divide(out, np.maximum(per_class, 1e-9), out=out)
    return out


def fold(spectrum: np.ndarray, rate: float) -> np.ndarray:
    """Twelve numbers from one magnitude spectrum, each 0..1.

    Normalised by the loudest class, so it describes the shape of the harmony
    rather than how loud the passage is. Silence gives twelve zeros rather
    than twelve equal values, which would read as every note at once.
    """
    spec = np.asarray(spectrum, dtype=np.float32)
    w = weights(spec.size, float(rate))
    out = w @ spec
    peak = float(out.max())
    if peak <= 1e-9:
        return np.zeros(12, dtype=np.float32)
    return (out / peak).astype(np.float32)


def strongest(chroma: np.ndarray) -> tuple[str, float]:
    """The loudest pitch class and how far it stands above the rest.

    The margin is what stops a mode acting on noise: a chord gives one or
    three classes well clear of the others, while a cymbal or a drum fills all
    twelve about equally and leaves almost nothing between the top and the
    average.
    """
    c = np.asarray(chroma, dtype=np.float32)
    if c.size != 12 or not c.any():
        return "", 0.0
    top = int(np.argmax(c))
    rest = float((c.sum() - c[top]) / 11.0)
    return NOTES[top], float(c[top] - rest)
