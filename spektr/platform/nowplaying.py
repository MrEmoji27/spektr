"""Now-playing metadata, from the OS media session rather than the audio.

spektr taps raw samples off a loopback device — it has no idea what's making
the sound, only that something is. Track title/artist has to come from
somewhere else entirely: the operating system's own media session, which
whatever player is running (Spotify, a browser tab, VLC...) already reports
to for lock-screen widgets and hardware media keys. Windows exposes this as
System Media Transport Controls; Linux media players mostly speak MPRIS over
D-Bus. Both are optional in every sense — an unsupported platform, a missing
backend package, or nothing playing all end at the same ``None``, not an
error, because a header line is not worth taking the visualiser down over.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Track:
    title: str
    #: May be empty — not every source reports one (a browser tab playing a
    #: podcast, for instance), and a title alone is still worth showing.
    artist: str

    def __str__(self) -> str:
        return f"{self.artist} — {self.title}" if self.artist else self.title


def _reader(name: str):
    """The per-OS reader, through this module so a patched one is honoured.

    Tests replace ``nowplaying._windows``; resolving the name here rather than
    importing the OS module directly means such a patch is still what runs.
    """
    import sys as _sys
    return getattr(_sys.modules[__name__], name)


def _os(name: str):
    """The per-OS reader, imported only on the OS it belongs to."""
    from importlib import import_module
    return getattr(import_module(f".{name}", __package__), f"_{'nowplaying' if name == 'macos' else name}")


async def current() -> Track | None:
    """The track the OS says is playing right now, or ``None``.

    Every reason there might not be an answer — unsupported platform, the
    backend package isn't installed, no media session is active, the active
    session declined to report anything, or the call simply took too long —
    collapses to the same ``None`` here, so the one caller in app.py never
    needs to know which.
    """
    import asyncio

    try:
        if sys.platform == "win32":
            return await asyncio.wait_for(_reader("_windows")(), timeout=3.0)
        if sys.platform.startswith("linux"):
            return await asyncio.wait_for(_reader("_linux")(), timeout=3.0)
    except Exception:
        pass
    return None




def __getattr__(name: str):
    """``_windows`` and ``_linux`` still resolve, from their own modules.

    They moved to platform/windows.py and platform/linux.py; tests and any
    caller that reached for them here keep working.
    """
    if name in ("_windows", "_linux"):
        return _os(name[1:])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
