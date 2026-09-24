"""Which hits are the beat, and which are only in it.

The onset detector finds every hit it can, and it has to: tempo, the drum
streams and the bar tracker are all built from the whole pattern, hi-hats and
ghost notes included. Its ``strength`` is measured inside whichever sub-band
found the hit, so a hat that is the loudest thing the treble has done lately
reads as a hard hit however quiet it is in the mix. Measured on a backbeat
with hats 28 dB under it: the hats fired 31 times out of 32 at two thirds of a
kick's strength, and more than half of every beat a mode was given was
background.

A listener does not hear it that way. The kick and the snare are the beat
because they are the biggest things that happen, and this measures exactly
that: how far the signal's energy jumped at the hit, against the biggest jumps
of the last few seconds.

**Energy, not the bands.** The first version summed the rise in the band
levels, which is what the bars draw. That scored the kick at an eighth of the
snare: the display's treble tilt makes anything broadband look enormous, and a
55 Hz kick lives in two bands. The jump in RMS over a short window has no
opinion about where in the spectrum the energy is, which is what "loud" means.

**Relative, and forgetting.** Against a decaying memory of recent accents
rather than a fixed level, so the whole thing is blind to volume, and so a
breakdown after a drop gets its own beat back once the drop has been
forgotten instead of being judged against it for the rest of the song.

**No added latency.** The detector confirms a peak :attr:`OnsetDetector.PEAK_SPAN
<spektr.audio.analysis.OnsetDetector.PEAK_SPAN>` hops after it happened, and
the energy after the hit that is already in hand by then is enough to judge
it. Waiting longer changed no answer on the test material and would put every
beat on screen late.
"""
from __future__ import annotations

import math
from collections import deque

#: The energy before a hit is its quietest level over this long beforehand, so
#: a hit landing on the previous one's tail is measured from where that tail
#: had fallen to, not from silence.
PRE_S = 0.06

#: How long the memory of a big accent takes to fade, as a time constant.
#: Long enough that a hat between two kicks is judged against those kicks,
#: short enough that a quiet section is judged against itself within a few
#: bars of it starting.
MEMORY_S = 3.0

#: How much energy history is kept. The window before a hit plus the
#: detector's confirmation delay, with room to spare.
KEEP_S = 0.4

#: A hit is a beat for the modes if its jump is at least this share of the
#: biggest recent one. Half is -6 dB: a kick and a snare a little softer than
#: it both clear it, a hat or a pluck that sits under the drums does not.
ACCENT_MIN = 0.5


class AccentMeter:
    """How much each hit stands out from the passage around it.

    :meth:`feed` takes one short-window RMS reading per hop, and :meth:`judge`
    is asked about a hit once the detector has confirmed it.
    """

    __slots__ = ("_hist", "_peak", "_peak_t")

    def __init__(self) -> None:
        self._hist: deque[tuple[float, float]] = deque()
        self._peak = 0.0
        self._peak_t = 0.0

    def reset(self) -> None:
        self._hist.clear()
        self._peak = 0.0
        self._peak_t = 0.0

    def feed(self, t: float, rms: float) -> None:
        self._hist.append((t, float(rms)))
        while self._hist and t - self._hist[0][0] > KEEP_S:
            self._hist.popleft()

    def judge(self, t: float) -> float:
        """The accent of a hit at ``t``, 0..1: its jump over the recent best."""
        before = [v for u, v in self._hist if t - PRE_S <= u < t]
        after = [v for u, v in self._hist if u >= t]
        if not before or not after:
            return 0.0
        rise = max(0.0, max(after) - min(before))
        if rise <= 0.0:
            return 0.0
        faded = self._peak * math.exp(-max(0.0, t - self._peak_t) / MEMORY_S)
        self._peak = max(faded, rise)
        self._peak_t = t
        return rise / self._peak
