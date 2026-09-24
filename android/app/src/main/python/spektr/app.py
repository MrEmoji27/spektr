"""Compatibility surface for the UI shell and command-line entry point.

Installed launchers still resolve ``spektr.app:main``. Attribute access is
forwarded rather than copied so monkeypatching and embedders observe the live
implementation after the package split.

Names the command line defines are answered from it without importing the UI
shell: the launcher resolving ``main`` used to import the whole UI toolkit
before ``spektr --version`` could print a line. The two modules share no name
of their own, so looking in the command line first changes no answer.
"""

from . import cli as _cli


def _shell():
    from .ui import app as _app

    return _app


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    if name in vars(_cli):
        return getattr(_cli, name)
    try:
        return getattr(_shell(), name)
    except AttributeError:
        raise AttributeError(f"module 'spektr.app' has no attribute {name!r}") from None


def __dir__():
    return sorted(set(globals()).union(vars(_shell()), vars(_cli)))


if __name__ == "__main__":
    _cli.main()
