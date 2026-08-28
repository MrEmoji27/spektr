"""Maelstrom's dissipation is a rate, so it has to be per second.

The sim is careful about ``dt`` nearly everywhere — advection is
semi-Lagrangian, vorticity and diffusion take it as a parameter, and the
forcing terms were deliberately converted to per-second with a ``dt * 60``
factor and a comment explaining why. The three dissipation multiplies at the
end of the step were left behind:

    vy *= 0.995;  vx *= 0.995;  dye *= 0.997

A bare per-frame multiply is a rate in disguise. At 24 fps the dye kept
``0.997**24`` = 93% of itself per second; at 240 fps it kept ``0.997**240``
= 49%. The smoke lingered nearly twice as long on a slow display, and the
same passage lit 82% of the screen at 24 fps against 40% at 240.

What remains after the fix is *not* the same bug. Semi-Lagrangian advection
is unconditionally stable at any ``dt`` but not equally accurate: each step
interpolates, so a 240 fps run takes ten times as many interpolations as a
24 fps one and smears the dye correspondingly more. That is inherent to the
scheme and needs a higher-order one (BFECC, MacCormack) to remove — measured
here, not assumed: quadrupling the Jacobi iterations moved the spread by
0.003, and disabling diffusion outright left most of it. So the bound below
is deliberately loose. It is a ratchet against the rate bug coming back, not
a claim that the mode is frame-rate perfect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spektr.modes as M  # noqa: E402
from spektr.analysis import N_BANDS  # noqa: E402
from spektr.modes import Ctx  # noqa: E402
from spektr.palette import BUILTIN, Palette  # noqa: E402
from spektr.render import BRAILLE_BASE, SPACE  # noqa: E402

PAL = Palette(BUILTIN["gruvbox"])
MAELSTROM = M.get("Maelstrom").fn


def _lit(fps, secs=25.0, w=80, h=24):
    """Mean fraction of cells lit in the steady state, at a given frame rate."""
    state: dict = {}
    dt = 1.0 / fps
    last_beat = -1.0
    lit: list[float] = []
    for f in range(int(secs * fps)):
        t = f * dt
        bands = np.full(N_BANDS, 0.85, dtype=np.float32)
        wave = (np.sin(np.linspace(0.0, 40.0, 512)) * 0.85).astype(np.float32)
        onset = 0
        if t - last_beat >= 1.0 / 3.0:
            onset, last_beat = 1, t
        ctx = Ctx(
            w=w, h=h, bands=bands, peaks=bands, bands_l=bands, bands_r=bands,
            wave=wave, stereo=np.stack([wave, wave], 1), frame=f, t=t, dt=dt,
            energy=0.85, silent=False, palette=PAL, state=state,
            onsets=onset, onset_strength=0.8 if onset else 0.0,
        )
        codes, _ = MAELSTROM(ctx)
        # The first seconds are the fluid filling; judge the steady state.
        if t > 3.0:
            lit.append(float(((codes != BRAILLE_BASE) & (codes != SPACE)).mean()))
    return float(np.mean(lit))


def test_the_smoke_does_not_linger_longer_on_a_slower_display():
    """The same passage looks broadly the same at 24 fps and at 240.

    The bound is a ratchet, not a target: with the per-frame dissipation the
    spread measured 0.425 (0.82 lit against 0.40), and the residual from
    advection accuracy alone is about 0.18. Anything back above 0.25 means a
    rate has gone per-frame again.
    """
    lit = {fps: round(_lit(fps), 3) for fps in (24, 60, 240)}
    spread = max(lit.values()) - min(lit.values())
    assert spread < 0.25, (
        f"the frame rate is changing the fluid again: {lit} (spread {spread:.3f}); "
        f"check that every decay in the step is scaled per second, not per frame")
