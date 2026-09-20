"""Display refresh-rate detection.

There is no portable way to ask "how fast is the screen?" from a terminal
program. The terminal gives us a character grid and nothing else — no window
handle we own, no display server connection, no refresh rate. So this is three
separate platform probes behind one function, each of which is allowed to fail.

Failure is normal here, not exceptional: a headless server, an SSH session, a
Wayland compositor with no discovery path, a macOS built-in panel that reports
0.0. Every probe therefore returns ``None`` rather than raising, and the caller
picks the fallback. This follows the same convention as ``load_user_themes()``
and ``plugins.discover()`` — a bad environment degrades to a default instead of
taking the app down.

Call this **once**, at startup, and cache it. It shells out on Linux and it is
not something to touch per frame.

Multi-monitor: every probe reports the *highest* rate among active displays. A
terminal can be on any monitor and we cannot tell which, so the rate is used as
an upper bound on how fast it is ever worth drawing — capping to the slowest
display would throttle someone whose terminal is on the fast one.
"""

from __future__ import annotations

import sys

#: Used when every probe fails. 60 Hz is the safe assumption — it is the most
#: common panel and erring low costs smoothness, where erring high costs CPU
#: for frames the display cannot show.
FALLBACK_HZ = 60


def _probe(name: str):
    """The per-OS probe, imported only on the OS it belongs to."""
    from importlib import import_module
    return getattr(import_module(f".{name}", __package__), f"_{name}_hz")


def refresh_hz() -> float | None:
    """Highest active display refresh rate, or ``None`` if it can't be found."""
    if sys.platform == "win32":
        return _probe("windows")()
    if sys.platform == "darwin":
        return _probe("macos")()
    if sys.platform.startswith("linux"):
        return _probe("linux")()
    return None


def refresh_hz_or(default: int = FALLBACK_HZ) -> int:
    """``refresh_hz()`` rounded to an int, or ``default`` when unknown."""
    hz = refresh_hz()
    if hz is None or hz <= 1.0:
        return int(default)
    return int(round(hz))


#: Resolved once, then reused. The Linux path shells out and none of the
#: probes are cheap enough to call per frame — and the answer is not allowed
#: to change mid-session anyway, because ``_target_fps`` is a fixed number the
#: adaptive pacer compares against.
_UNLIMITED: tuple[int, int | None] | None = None


def unlimited_fps(cap: int) -> tuple[int, int | None]:
    """Resolve "unlimited" to a real frame rate. Returns ``(fps, detected)``.

    ``detected`` is the probed refresh rate, or ``None`` when every probe
    failed — the caller needs to tell those apart, because "we measured 60"
    and "we gave up and assumed 60" look identical in the number alone and
    only one of them is a bug worth reporting.

    ``cap`` is an upper bound from the analysis rate: past roughly twice the
    rate at which new spectra exist, extra frames carry no new audio, only
    interpolation. In practice it does not bind on any current panel — at
    HOP 256 the cap is 375 — but a display faster than the analyser is a real
    configuration and the ceiling says what happens in it.

    Never raises and never returns 0: a failed probe yields ``FALLBACK_HZ``.
    """
    global _UNLIMITED
    if _UNLIMITED is None:
        try:
            hz = refresh_hz()
        except Exception:
            hz = None
        detected = int(round(hz)) if hz and hz > 1.0 else None
        _UNLIMITED = (max(15, min(detected or FALLBACK_HZ, int(cap))), detected)
    return _UNLIMITED


#: Where each name went when the per-OS probes moved out of this module.
_MOVED = {
    "_windows_hz": "windows",
    "_macos_hz": "macos",
    "_linux_hz": "linux",
    "_parse_xrandr": "linux",
    "_parse_wlr": "linux",
}


def __getattr__(name: str):
    """The probes and their parsers still resolve from here."""
    where = _MOVED.get(name)
    if where is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    return getattr(import_module(f".{where}", __package__), name)
