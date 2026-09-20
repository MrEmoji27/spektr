"""Which drum it sounded like, from where the spectrum moved.

The detector says *that* something hit. This says what it sounded like: a
kick, a snare, a hat, or none of them. It reads the same rectified flux the
peak picker already has, so it adds no analysis of its own and cannot change
what the detector finds.

These are likelihoods, never verdicts. A kick and a floor tom differ by
almost nothing a spectrum can see, a rim shot lands between snare and hat, and
plenty of music has no drums at all. A mode reading these should lean on them,
not branch on them, and :func:`classify` returning three small numbers is the
honest answer to a bar of piano.

The method is deliberately the simple one: how the energy that *arrived* is
spread in frequency, and how broad it is. A kick is a narrow burst near the
bottom; a hat is a wide burst near the top; a snare is a body low down with
noise above it, which is why it is the hardest of the three and why its
likelihood is the one to trust least.
"""
from __future__ import annotations

import numpy as np

#: Where each drum lives, in hertz. Wider than a drum-tuning chart, because
#: these are bands of a spectrum rather than the fundamental of an instrument:
#: a kick's thump is at 50 Hz but its click is far above it.
KICK_HZ = (20.0, 160.0)
SNARE_HZ = (160.0, 1200.0)
HAT_HZ = (3500.0, 16000.0)

#: How broad a burst has to be, as a share of the bands in its range, before
#: it reads as noise rather than as a note. A hat lights nearly everything
#: above it; a bass note lights two bands and a couple of harmonics.
BROAD = 0.45

#: Below this share of the frame's total rise, a drum is not claimed at all.
#: Without it, a quiet piece of music with no percussion in it still produces
#: a winner every time something moves.
FLOOR = 0.18


def _share(flux: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    """How much of the rise landed in a range, and how much of it it filled."""
    inside = (freqs >= lo) & (freqs < hi)
    if not inside.any():
        return 0.0, 0.0
    total = float(flux.sum())
    if total <= 1e-9:
        return 0.0, 0.0
    part = flux[inside]
    share = float(part.sum()) / total
    lit = float(np.count_nonzero(part > part.max() * 0.25)) / part.size
    return share, lit


def classify(flux: np.ndarray, freqs: np.ndarray) -> dict[str, float]:
    """Kick, snare and hat likelihoods for one rectified flux frame.

    ``flux`` is the per-band rise, already rectified, and ``freqs`` the centre
    frequency of each band. All three come back 0..1 and do not sum to
    anything in particular: two drums can land together, and often do.
    """
    f = np.asarray(flux, dtype=np.float32)
    hz = np.asarray(freqs, dtype=np.float32)
    if f.size != hz.size or f.size == 0 or float(f.sum()) <= 1e-9:
        return {"kick": 0.0, "snare": 0.0, "hat": 0.0}

    low, low_lit = _share(f, hz, *KICK_HZ)
    mid, mid_lit = _share(f, hz, *SNARE_HZ)
    high, high_lit = _share(f, hz, *HAT_HZ)

    # A kick is the low end carrying the frame, and narrow with it: a bass
    # note that happens to be loud fills the same range but keeps its shape.
    kick = low * (1.0 - 0.5 * max(0.0, low_lit - BROAD))

    # A hat is the top end carrying the frame *and* filling it. A cymbal and
    # a high melody line divide on the second half of that, not the first.
    hat = high * min(1.0, high_lit / BROAD)

    # A snare is a body in the middle with noise above it. Neither half alone
    # is enough: the mid on its own is most of a mix, and the noise on its own
    # is a hat.
    snare = mid * min(1.0, mid_lit / BROAD) * (0.4 + 0.6 * min(1.0, high / 0.25))

    out = {"kick": kick, "snare": snare, "hat": hat}
    return {k: round(float(v), 4) if v >= FLOOR else 0.0 for k, v in out.items()}


def band_freqs(n: int, low: float = 50.0, high: float = 10000.0) -> np.ndarray:
    """Centre frequency of each of ``n`` log-spaced bands.

    The same distribution the analyser lays its bands out on, so a caller with
    a band vector and nothing else can still ask this module a question.
    """
    edges = np.geomspace(low, high, n + 1)
    return np.sqrt(edges[:-1] * edges[1:]).astype(np.float32)
