"""macOS: the display refresh rate.

Now-playing has no implementation here yet, and system-audio capture still
needs a loopback driver; both are 0.6.5 work. This module exists so that
work has an obvious home instead of being threaded back through the
shared modules.
"""
from __future__ import annotations


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


async def _nowplaying():
    """macOS exposes no media session spektr can read yet."""
    return None
