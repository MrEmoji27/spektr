"""Compatibility alias for :mod:`spektr.platform.display`."""

import sys

from .platform import display as _display

sys.modules[__name__] = _display
