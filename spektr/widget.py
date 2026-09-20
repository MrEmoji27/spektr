"""Compatibility surface for :mod:`spektr.ui.widget`."""

from .ui import widget as _widget


def __getattr__(name):
    try:
        return getattr(_widget, name)
    except AttributeError:
        raise AttributeError(f"module 'spektr.widget' has no attribute {name!r}") from None


def __dir__():
    return sorted(set(globals()).union(vars(_widget)))
