"""Windows: the display refresh rate, and the system media session."""
from __future__ import annotations

from .nowplaying import Track


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


async def _windows() -> Track | None:
    """System Media Transport Controls — what the lock screen's media
    overlay and the keyboard's play/pause key already talk to."""
    try:
        from winrt.windows.media.control import (
            GlobalSystemMediaTransportControlsSessionManager as SessionManager,
        )
    except ImportError:
        return None

    manager = await SessionManager.request_async()
    session = manager.get_current_session()
    if session is None:
        return None

    info = await session.try_get_media_properties_async()
    title = (info.title or "").strip()
    if not title:
        return None
    return Track(title=title, artist=(info.artist or "").strip())
