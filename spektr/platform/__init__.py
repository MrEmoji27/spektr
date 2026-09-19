"""Operating-system integration implementations."""

from . import display as _display
from . import nowplaying as _nowplaying


def __getattr__(name):
    """Forward public, private, and mutable attributes to their owner."""
    for module in (_display, _nowplaying):
        try:
            return getattr(module, name)
        except AttributeError:
            continue
    raise AttributeError(f"module 'spektr.platform' has no attribute {name!r}")
