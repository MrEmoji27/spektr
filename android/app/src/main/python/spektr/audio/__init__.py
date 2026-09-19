"""Audio capture and spectrum analysis implementations."""

from . import analysis as _analysis
from . import capture as _capture
from .analysis import *  # noqa: F403
from .capture import *  # noqa: F403


def __getattr__(name):
    """Forward private and mutable attributes to the owning module."""
    for module in (_analysis, _capture):
        try:
            return getattr(module, name)
        except AttributeError:
            continue
    raise AttributeError(f"module 'spektr.audio' has no attribute {name!r}")
