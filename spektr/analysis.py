"""Compatibility alias for :mod:`spektr.audio.analysis`."""

import sys

from .audio import analysis as _analysis

sys.modules[__name__] = _analysis
