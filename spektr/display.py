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

import subprocess
import sys

#: Used when every probe fails. 60 Hz is the safe assumption — it is the most
#: common panel and erring low costs smoothness, where erring high costs CPU
#: for frames the display cannot show.
FALLBACK_HZ = 60


def _windows_hz() -> float | None:
    """``EnumDisplaySettingsW`` over every attached display.

    ``dmDisplayFrequency`` is an integer field, so 59.94 Hz panels report 59 or
    60 — close enough for a frame cap. It also reports 0 or 1 to mean "whatever
    the hardware default is" on some drivers, which is why those are rejected
    below rather than taken literally.
    """
    try:
        import ctypes
        from ctypes import wintypes

        class POINTL(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        class _DMDisplay(ctypes.Structure):
            _fields_ = [
                ("dmPosition", POINTL),
                ("dmDisplayOrientation", wintypes.DWORD),
                ("dmDisplayFixedOutput", wintypes.DWORD),
            ]

        class _DMUnion(ctypes.Union):
            _fields_ = [("display", _DMDisplay), ("_pad", ctypes.c_byte * 16)]

        class DEVMODEW(ctypes.Structure):
            _anonymous_ = ("u",)
            _fields_ = [
                ("dmDeviceName", wintypes.WCHAR * 32),
                ("dmSpecVersion", wintypes.WORD),
                ("dmDriverVersion", wintypes.WORD),
                ("dmSize", wintypes.WORD),
                ("dmDriverExtra", wintypes.WORD),
                ("dmFields", wintypes.DWORD),
                ("u", _DMUnion),
                ("dmColor", ctypes.c_short),
                ("dmDuplex", ctypes.c_short),
                ("dmYResolution", ctypes.c_short),
                ("dmTTOption", ctypes.c_short),
                ("dmCollate", ctypes.c_short),
                ("dmFormName", wintypes.WCHAR * 32),
                ("dmLogPixels", wintypes.WORD),
                ("dmBitsPerPel", wintypes.DWORD),
                ("dmPelsWidth", wintypes.DWORD),
                ("dmPelsHeight", wintypes.DWORD),
                ("dmDisplayFlags", wintypes.DWORD),
                ("dmDisplayFrequency", wintypes.DWORD),
                ("dmICMMethod", wintypes.DWORD),
                ("dmICMIntent", wintypes.DWORD),
                ("dmMediaType", wintypes.DWORD),
                ("dmDitherType", wintypes.DWORD),
                ("dmReserved1", wintypes.DWORD),
                ("dmReserved2", wintypes.DWORD),
                ("dmPanningWidth", wintypes.DWORD),
                ("dmPanningHeight", wintypes.DWORD),
            ]

        class DISPLAY_DEVICEW(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("DeviceName", wintypes.WCHAR * 32),
                ("DeviceString", wintypes.WCHAR * 128),
                ("StateFlags", wintypes.DWORD),
                ("DeviceID", wintypes.WCHAR * 128),
                ("DeviceKey", wintypes.WCHAR * 128),
            ]

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.EnumDisplaySettingsW.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(DEVMODEW)
        ]
        user32.EnumDisplaySettingsW.restype = wintypes.BOOL
        user32.EnumDisplayDevicesW.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(DISPLAY_DEVICEW), wintypes.DWORD
        ]
        user32.EnumDisplayDevicesW.restype = wintypes.BOOL

        current = ctypes.c_ulong(-1).value      # ENUM_CURRENT_SETTINGS
        attached = 0x00000001                   # DISPLAY_DEVICE_ACTIVE

        best = 0.0
        i = 0
        while True:
            dd = DISPLAY_DEVICEW()
            dd.cb = ctypes.sizeof(DISPLAY_DEVICEW)
            if not user32.EnumDisplayDevicesW(None, i, ctypes.byref(dd), 0):
                break
            if dd.StateFlags & attached:
                dm = DEVMODEW()
                dm.dmSize = ctypes.sizeof(DEVMODEW)
                if user32.EnumDisplaySettingsW(dd.DeviceName, current, ctypes.byref(dm)):
                    best = max(best, float(dm.dmDisplayFrequency))
            i += 1

        if best <= 1.0:   # 0 and 1 both mean "driver default", not a real rate
            dm = DEVMODEW()
            dm.dmSize = ctypes.sizeof(DEVMODEW)
            if user32.EnumDisplaySettingsW(None, current, ctypes.byref(dm)):
                best = float(dm.dmDisplayFrequency)
        return best if best > 1.0 else None
    except Exception:
        return None


def _parse_xrandr(text: str) -> float:
    """Highest rate among modes marked current with ``*``. 0.0 if none."""
    import re

    best = 0.0
    for line in text.splitlines():
        if "*" not in line:
            continue
        s = line.strip()
        # mode rows start with a resolution; the header and connector rows
        # can also contain '*' in some locales, so this is not optional
        if not re.match(r"^\d+x\d+", s):
            continue
        for m in re.finditer(r"(\d+\.\d+)(\*)?", s):
            if m.group(2):
                best = max(best, float(m.group(1)))
    return best


def _linux_hz() -> float | None:
    """``xrandr`` first, then ``wlr-randr`` for wlroots-based Wayland.

    Wayland has no standard way to query this — there is no protocol for it
    and each compositor differs — so a Wayland session outside wlroots will
    correctly fall through to the default rather than guessing.
    """
    for argv, parse in (
        (["xrandr", "--query"], _parse_xrandr),
        (["wlr-randr"], _parse_wlr),
    ):
        try:
            out = subprocess.run(
                argv, capture_output=True, text=True, timeout=1.5, check=False
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode != 0:
            continue
        hz = parse(out.stdout)
        if hz > 1.0:
            return hz
    return None


def _parse_wlr(text: str) -> float:
    """``wlr-randr`` prints ``1920x1080 px, 144.000000 Hz (current)``."""
    import re

    best = 0.0
    for line in text.splitlines():
        if "current" not in line:
            continue
        m = re.search(r"([\d.]+)\s*Hz", line)
        if m:
            try:
                best = max(best, float(m.group(1)))
            except ValueError:
                pass
    return best


def _macos_hz() -> float | None:
    """CoreGraphics ``CGDisplayModeGetRefreshRate``.

    Returns 0.0 for a great many built-in Apple panels, which is why a zero
    result is treated as unknown rather than as a rate. On ProMotion displays
    this reports the top of the variable range, which is the correct bound for
    a frame cap.
    """
    try:
        import ctypes
        import ctypes.util

        path = ctypes.util.find_library("CoreGraphics")
        if not path:
            return None
        cg = ctypes.cdll.LoadLibrary(path)

        cg.CGMainDisplayID.restype = ctypes.c_uint32
        cg.CGDisplayCopyDisplayMode.argtypes = [ctypes.c_uint32]
        cg.CGDisplayCopyDisplayMode.restype = ctypes.c_void_p
        cg.CGDisplayModeGetRefreshRate.argtypes = [ctypes.c_void_p]
        cg.CGDisplayModeGetRefreshRate.restype = ctypes.c_double
        cg.CGDisplayModeRelease.argtypes = [ctypes.c_void_p]

        mode = cg.CGDisplayCopyDisplayMode(cg.CGMainDisplayID())
        if not mode:
            return None
        try:
            hz = float(cg.CGDisplayModeGetRefreshRate(mode))
        finally:
            cg.CGDisplayModeRelease(mode)
        return hz if hz > 1.0 else None
    except Exception:
        return None


def refresh_hz() -> float | None:
    """Highest active display refresh rate, or ``None`` if it can't be found."""
    if sys.platform == "win32":
        return _windows_hz()
    if sys.platform == "darwin":
        return _macos_hz()
    if sys.platform.startswith("linux"):
        return _linux_hz()
    return None


def refresh_hz_or(default: int = FALLBACK_HZ) -> int:
    """``refresh_hz()`` rounded to an int, or ``default`` when unknown."""
    hz = refresh_hz()
    if hz is None or hz <= 1.0:
        return int(default)
    return int(round(hz))
