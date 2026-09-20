"""Compatibility surface for the UI picker and settings modules."""

from .ui import pickers as _pickers
from .ui import settings as _settings

_MODULES = (_pickers, _settings)


def __getattr__(name):
    for module in _MODULES:
        try:
            return getattr(module, name)
        except AttributeError:
            continue
    raise AttributeError(f"module 'spektr.pickers' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()).union(*(vars(module) for module in _MODULES)))
