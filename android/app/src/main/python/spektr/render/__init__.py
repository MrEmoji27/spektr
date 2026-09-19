"""Rendering primitives and colour-theme support.

This package remains the compatibility surface for ``spektr.render`` while
the implementations live in focused submodules.
"""

from . import palette as _palette
from . import render as _render
from .palette import *  # noqa: F403
from .render import *  # noqa: F403


def __getattr__(name):
    """Forward private and mutable attributes to their implementation module."""
    for module in (_render, _palette):
        try:
            return getattr(module, name)
        except AttributeError:
            continue
    raise AttributeError(f"module 'spektr.render' has no attribute {name!r}")
