"""Compatibility alias for :mod:`spektr.audio.capture`."""

import sys

from .audio import capture as _capture

sys.modules[__name__] = _capture
