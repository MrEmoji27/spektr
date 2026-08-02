"""Backwards-compatible shim.

The visualiser used to be one 900-line module. It is now split across
``capture`` (device probing and the ring buffer), ``analysis`` (the overlapped
FFT), ``motion`` (easing), ``palette`` (themes), ``render`` (grid primitives and
Strips), ``modes/`` (the render modes) and ``widget`` (the Textual widget).

Anything importing ``spektr.visualizer.AudioVisualizer`` still works.
"""

from __future__ import annotations

from .analysis import N_BANDS
from .modes import names as _mode_names
from .widget import AudioVisualizer

#: kept so old code doing ``from spektr.visualizer import VIS_MODES`` still runs
VIS_MODES = _mode_names()

__all__ = ["AudioVisualizer", "VIS_MODES", "N_BANDS"]
