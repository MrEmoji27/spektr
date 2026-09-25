"""A blue-noise rank mask, for dissolving one picture into another.

A dissolve needs an order to reveal dots in. White noise gives each dot a
random rank, and the eye reads that as clumping: holes open in one place while
another area is still solid. Blue noise spreads the ranks so that any two dots
close together get ranks far apart, and the picture then dissolves evenly —
the same reason ordered dithering uses it rather than random thresholds.

The mask is built once by the void-and-cluster algorithm (Ulichney, 1993) and
tiled over the dot grid. At progress ``p`` a dot shows the incoming picture
when ``mask < p``. Generation is not cheap and is not meant to run per frame:
:func:`mask` caches per size, and 64x64 takes well under a second.

    m = mask(64)              # values in [0, 1), each appearing once
    show_new = m[rows, cols] < progress
"""
from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

#: Width of the gaussian used to measure how clustered a dot is. 1.5 samples
#: is Ulichney's value and it is what makes the result blue rather than merely
#: scattered: wider blurs the distinction between neighbours, narrower stops
#: the energy reaching the next dot at all.
SIGMA = 1.5


def _energy(binary: np.ndarray, sigma: float = SIGMA) -> np.ndarray:
    """How crowded each cell is, wrapping at the edges.

    A gaussian is separable, so this is two 1-D convolutions rather than one
    2-D one, and wrapping makes the mask tileable: the dot pattern stays even
    across the seam when the mask repeats over a large grid.
    """
    n = binary.shape[0]
    half = n // 2
    offsets = np.arange(-half, n - half)
    kernel = np.exp(-(offsets ** 2) / (2.0 * sigma * sigma))
    kernel = np.roll(kernel, -half)  # centre at index 0, so the roll wraps
    spectrum = np.fft.rfft2(binary)
    blur = np.outer(kernel, kernel)
    return np.fft.irfft2(spectrum * np.fft.rfft2(blur), s=binary.shape)


def _tightest_cluster(energy: np.ndarray, binary: np.ndarray) -> tuple[int, int]:
    masked = np.where(binary == 1, energy, -np.inf)
    return np.unravel_index(int(np.argmax(masked)), energy.shape)


def _largest_void(energy: np.ndarray, binary: np.ndarray) -> tuple[int, int]:
    masked = np.where(binary == 0, energy, np.inf)
    return np.unravel_index(int(np.argmin(masked)), energy.shape)


def _initial_pattern(n: int, seed: int) -> np.ndarray:
    """A scattered starting set, made even by swapping clusters into voids."""
    rng = np.random.default_rng(seed)
    binary = np.zeros((n, n), dtype=np.int8)
    flat = rng.choice(n * n, size=max(1, (n * n) // 10), replace=False)
    binary.flat[flat] = 1
    while True:
        energy = _energy(binary)
        cy, cx = _tightest_cluster(energy, binary)
        binary[cy, cx] = 0
        energy = _energy(binary)
        vy, vx = _largest_void(energy, binary)
        if (vy, vx) == (cy, cx):
            binary[cy, cx] = 1
            return binary
        binary[vy, vx] = 1


#: One build at a time. ``lru_cache`` does not stop two threads that miss the
#: cache together from both building the mask, and it costs a few hundred
#: milliseconds of solid work: the warmer and the render path asking at once
#: would each pay it, and every caller after the first can simply wait.
_BUILDING = threading.Lock()


def mask(n: int = 64, seed: int = 0x5EED) -> np.ndarray:
    """An ``n`` x ``n`` rank mask of floats in ``[0, 1)``, each value once.

    Deterministic for a given size and seed, so a dissolve looks the same on
    every machine and every run — the same rule the rest of spektr follows for
    anything that should look settled.
    """
    with _BUILDING:
        return _mask(n, seed)


@lru_cache(maxsize=8)
def _mask(n: int, seed: int) -> np.ndarray:
    if n < 2 or n & (n - 1):
        raise ValueError("mask size must be a power of two, 2 or larger")

    binary = _initial_pattern(n, seed)
    rank = np.zeros((n, n), dtype=np.int32)
    total = n * n

    # Phase 1: take the initial dots away, tightest cluster first. Those are
    # the last dots a dissolve should reveal, so they rank downward from the
    # size of the initial pattern.
    working = binary.copy()
    for r in range(int(working.sum()) - 1, -1, -1):
        cy, cx = _tightest_cluster(_energy(working), working)
        working[cy, cx] = 0
        rank[cy, cx] = r

    # Phase 2: fill the rest in, largest void first, so every added dot lands
    # as far as possible from the ones already there.
    working = binary.copy()
    for r in range(int(binary.sum()), total):
        vy, vx = _largest_void(_energy(working), working)
        working[vy, vx] = 1
        rank[vy, vx] = r

    out = rank.astype(np.float64) / float(total)
    out.setflags(write=False)  # cached: a caller must not edit it in place
    return out


def tile(m: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """The mask repeated to cover a ``rows`` x ``cols`` dot grid."""
    n = m.shape[0]
    return m[np.arange(rows) % n][:, np.arange(cols) % n]
