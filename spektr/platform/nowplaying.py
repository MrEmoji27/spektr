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
            return await asyncio.wait_for(_windows(), timeout=3.0)
        if sys.platform.startswith("linux"):
            return await asyncio.wait_for(_linux(), timeout=3.0)
    except Exception:
        pass
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
