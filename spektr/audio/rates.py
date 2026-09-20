"""One place that knows how fast the analyser runs.

The onset detector's constants — how long its history is, how far apart two
hops are, how fast its peak memory decays — were all tuned at 48 kHz, where
one 256-frame hop arrives every 5.33 ms and the analyser runs at 187.5 hops a
second. Those numbers were written into the code as the literal ``187.5``,
which is only true at 48 kHz. A 96 kHz device delivers twice as many hops a
second, so every one of those constants meant half as much time as intended:
the detector's memory, its peak decay and its peak-picking window all shrank,
and detection accuracy dropped with them (F 0.948 at 48 kHz against 0.888 at
96 kHz, with breakbeat recall falling from 0.79 to 0.48).

Deriving the rate from the device instead keeps every constant meaning the
same number of seconds whatever the hardware runs at.
"""
from __future__ import annotations

#: Frames per analysis hop. The analyser always consumes the ring in these,
#: no matter how the capture backend pushes.
HOP = 256

#: The sample rate the detector's constants were tuned at. Everything derived
#: below equals its historical value at this rate.
DESIGN_RATE = 48000

#: Fallback when a device does not report a usable rate.
DEFAULT_RATE = DESIGN_RATE


def analyses_per_sec(samplerate: float | None = None) -> float:
    """Hops a second at ``samplerate`` — 187.5 at 48 kHz."""
    rate = float(samplerate or DEFAULT_RATE)
    if rate <= 0:
        rate = DEFAULT_RATE
    return rate / HOP


def hop_seconds(samplerate: float | None = None) -> float:
    """Seconds between hops — 5.33 ms at 48 kHz."""
    return 1.0 / analyses_per_sec(samplerate)
