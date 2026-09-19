"""Compatibility alias for :mod:`spektr.platform.nowplaying`."""

import sys

from .platform import nowplaying as _nowplaying

sys.modules[__name__] = _nowplaying
