"""Linux: the display refresh rate, and MPRIS players over D-Bus."""
from __future__ import annotations

import re
import subprocess

from .nowplaying import Track


def _parse_xrandr(text: str) -> float:
    """Highest rate among modes marked current with ``*``. 0.0 if none."""

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


async def _linux() -> Track | None:
    """MPRIS over the session D-Bus. Several players can be registered at
    once (a browser tab, a music app); the first one actually *playing* wins
    — a paused player from yesterday shouldn't outrank what's live now."""
    try:
        from dbus_next import BusType
        from dbus_next.aio import MessageBus
    except ImportError:
        return None

    bus = await MessageBus(bus_type=BusType.SESSION).connect()
    try:
        dbus_intro = await bus.introspect("org.freedesktop.DBus", "/org/freedesktop/DBus")
        dbus_obj = bus.get_proxy_object(
            "org.freedesktop.DBus", "/org/freedesktop/DBus", dbus_intro
        )
        names = await dbus_obj.get_interface("org.freedesktop.DBus").call_list_names()
        players = [n for n in names if n.startswith("org.mpris.MediaPlayer2.")]
        if not players:
            return None

        fallback: Track | None = None
        for name in players:
            track, playing = await _mpris_player(bus, name)
            if track is None:
                continue
            if playing:
                return track
            if fallback is None:
                fallback = track
        return fallback
    finally:
        bus.disconnect()


async def _mpris_player(bus, name: str) -> tuple[Track | None, bool]:
    """One MPRIS player's current track and whether it's actually playing."""
    intro = await bus.introspect(name, "/org/mpris/MediaPlayer2")
    obj = bus.get_proxy_object(name, "/org/mpris/MediaPlayer2", intro)
    props = obj.get_interface("org.freedesktop.DBus.Properties")

    status = await props.call_get("org.mpris.MediaPlayer2.Player", "PlaybackStatus")
    playing = status.value == "Playing"

    meta = await props.call_get("org.mpris.MediaPlayer2.Player", "Metadata")
    fields = meta.value
    title_v = fields.get("xesam:title")
    artist_v = fields.get("xesam:artist")
    title = (title_v.value or "").strip() if title_v else ""
    if not title:
        return None, playing
    artist = ", ".join(artist_v.value) if artist_v and artist_v.value else ""
    return Track(title=title, artist=artist), playing
