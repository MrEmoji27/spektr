"""Compatibility alias for :mod:`spektr.render.palette`."""

import sys

from .render import palette as _palette

sys.modules[__name__] = _palette
