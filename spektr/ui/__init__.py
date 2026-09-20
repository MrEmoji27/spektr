"""Textual application shell, screens, and widgets."""

from . import app as _app
from . import pickers as _pickers
from . import settings as _settings
from . import widget as _widget

_MODULES = (_app, _widget, _pickers, _settings)


def __getattr__(name):
    for module in _MODULES:
        try:
            return getattr(module, name)
        except AttributeError:
            continue
    raise AttributeError(f"module 'spektr.ui' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()).union(*(vars(module) for module in _MODULES)))
