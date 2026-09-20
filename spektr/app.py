"""Compatibility surface for the UI shell and command-line entry point.

Installed launchers still resolve ``spektr.app:main``. Attribute access is
forwarded rather than copied so monkeypatching and embedders observe the live
implementation after the package split.
"""

from . import cli as _cli
from .ui import app as _app

_MODULES = (_app, _cli)


def __getattr__(name):
    for module in _MODULES:
        try:
            return getattr(module, name)
        except AttributeError:
            continue
    raise AttributeError(f"module 'spektr.app' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()).union(*(vars(module) for module in _MODULES)))


if __name__ == "__main__":
    _cli.main()
