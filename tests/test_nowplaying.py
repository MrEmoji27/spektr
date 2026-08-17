"""Now-playing metadata test — no real media session or D-Bus required.

Neither backend can be exercised against the real thing from a single dev
machine: SMTC only exists on Windows, MPRIS only exists on Linux, and neither
is guaranteed to have anything actually playing. Both are faked at the module
level instead, the same way test_audio.py stands in for soundcard/sounddevice
to test device enumeration off real hardware.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import spektr.nowplaying as nowplaying  # noqa: E402


class _Variant:
    """dbus_next wraps every D-Bus value in one of these; only .value is used."""

    def __init__(self, value):
        self.value = value


def _install(modules: dict[str, types.ModuleType]):
    """Swap fake modules into sys.modules, returning what to restore."""
    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    return saved


def _restore(saved: dict) -> None:
    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


# ── Windows: SMTC ─────────────────────────────────────────────────────────────
#
# Faked at "winrt.windows.media.control" — the split winrt-Windows.Media.Control
# / winrt-Windows.Foundation packages, not the monolithic "winsdk". winsdk's
# latest release has no cp313 wheel, which was a real bug here: the dependency
# silently never installed on this Python version and the feature silently
# never worked. Confirmed live against an actual Windows media session with
# winrt installed before switching — see nowplaying.py's dependency comment
# in pyproject.toml for the full story.

def _fake_winrt(title: str | None, artist: str | None, has_session: bool = True):
    class FakeProps:
        def __init__(self, title, artist):
            self.title = title
            self.artist = artist

    class FakeSession:
        def __init__(self, title, artist):
            self._title, self._artist = title, artist

        async def try_get_media_properties_async(self):
            return FakeProps(self._title, self._artist)

    class FakeManager:
        _session = None

        def get_current_session(self):
            return FakeManager._session

        @classmethod
        async def request_async(cls):
            return cls()

    FakeManager._session = FakeSession(title, artist) if has_session else None

    control = types.ModuleType("winrt.windows.media.control")
    control.GlobalSystemMediaTransportControlsSessionManager = FakeManager

    return {
        "winrt": types.ModuleType("winrt"),
        "winrt.windows": types.ModuleType("winrt.windows"),
        "winrt.windows.media": types.ModuleType("winrt.windows.media"),
        "winrt.windows.media.control": control,
    }


def check_windows_no_package() -> list[str]:
    """The overwhelmingly common case off Windows: the import just fails.

    Clearing sys.modules alone isn't enough here — winrt is a real installed
    dependency on the machine these tests happen to run on (needed to
    exercise the other winrt-backed checks below against a real session
    during development), so a cleared cache just re-imports it from disk.
    Blocking __import__ for the name is what test_audio.py's
    no_soundcard does for the same reason, one file over.
    """
    import builtins

    real_import = builtins.__import__

    def no_winrt(name, *args, **kwargs):
        if name == "winrt" or name.startswith("winrt."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    saved = _install({})
    for name in list(sys.modules):
        if name == "winrt" or name.startswith("winrt."):
            sys.modules.pop(name, None)
    builtins.__import__ = no_winrt
    try:
        track = asyncio.run(nowplaying._windows())
    finally:
        builtins.__import__ = real_import
        _restore(saved)
    return [] if track is None else [f"expected None without winrt installed, got {track}"]


def check_windows_reports_track() -> list[str]:
    saved = _install(_fake_winrt("Song Title", "The Artist"))
    try:
        track = asyncio.run(nowplaying._windows())
    finally:
        _restore(saved)
    bad = []
    if track is None:
        return ["expected a track, got None"]
    if track.title != "Song Title" or track.artist != "The Artist":
        bad.append(f"wrong fields: {track}")
    if str(track) != "The Artist — Song Title":
        bad.append(f"str() formatting wrong: {track!s}")
    return bad


def check_windows_no_active_session() -> list[str]:
    """Nothing playing at all — a real, common state, not an error."""
    saved = _install(_fake_winrt(None, None, has_session=False))
    try:
        track = asyncio.run(nowplaying._windows())
    finally:
        _restore(saved)
    return [] if track is None else [f"expected None with no session, got {track}"]


def check_windows_blank_title_is_nothing() -> list[str]:
    """A session can exist with no real metadata (e.g. a silent background
    app) — a blank title isn't a track, it's noise, and must not render as
    one in the header."""
    saved = _install(_fake_winrt("   ", "Someone"))
    try:
        track = asyncio.run(nowplaying._windows())
    finally:
        _restore(saved)
    return [] if track is None else [f"expected None for a blank title, got {track}"]


# ── Linux: MPRIS over D-Bus ───────────────────────────────────────────────────

def _fake_dbus_next(players: list[tuple[str, str | None, str | None, str]]):
    """``players``: (bus name, title, artist, PlaybackStatus) tuples."""

    class FakeDBusListing:
        async def call_list_names(self):
            return ["org.freedesktop.DBus"] + [p[0] for p in players]

    class FakePlayerProps:
        def __init__(self, title, artist, status):
            self._title, self._artist, self._status = title, artist, status

        async def call_get(self, iface, prop):
            if prop == "PlaybackStatus":
                return _Variant(self._status)
            if prop == "Metadata":
                meta = {}
                if self._title:
                    meta["xesam:title"] = _Variant(self._title)
                if self._artist:
                    meta["xesam:artist"] = _Variant([self._artist])
                return _Variant(meta)
            raise KeyError(prop)

    class FakeProxyObject:
        def __init__(self, name):
            self._name = name

        def get_interface(self, iface):
            if iface == "org.freedesktop.DBus":
                return FakeDBusListing()
            if iface == "org.freedesktop.DBus.Properties":
                for name, title, artist, status in players:
                    if name == self._name:
                        return FakePlayerProps(title, artist, status)
            raise RuntimeError(f"unexpected interface {iface!r} for {self._name!r}")

    class FakeBus:
        def __init__(self, bus_type=None):
            pass

        async def connect(self):
            return self

        async def introspect(self, name, path):
            return object()  # opaque — only fed back into get_proxy_object

        def get_proxy_object(self, name, path, intro):
            return FakeProxyObject(name)

        def disconnect(self):
            pass

    dbus_next_mod = types.ModuleType("dbus_next")
    dbus_next_mod.BusType = types.SimpleNamespace(SESSION="session")
    aio_mod = types.ModuleType("dbus_next.aio")
    aio_mod.MessageBus = FakeBus

    return {"dbus_next": dbus_next_mod, "dbus_next.aio": aio_mod}


class _BlockImport:
    """A meta-path finder that makes one package genuinely unimportable.

    Dropping a module from ``sys.modules`` does not make it missing — the next
    ``import`` simply loads it again from site-packages. This does.
    """

    def __init__(self, package: str) -> None:
        self.package = package

    def find_spec(self, name, path=None, target=None):
        if name == self.package or name.startswith(self.package + "."):
            raise ImportError(f"{name} is blocked for this test")
        return None


def check_linux_no_package() -> list[str]:
    """``_linux`` returns None when dbus_next is not installed.

    This used to simulate "not installed" by popping the module out of
    ``sys.modules``, which does nothing at all: the import inside ``_linux``
    reloads it from site-packages. So the check only meant something on a
    machine where dbus_next happened to be absent — and on one where it is
    present it did the opposite of its name, importing the real package,
    reaching for a session bus, and raising ``InvalidAddressError``.

    Not hypothetical. ``dbus-next`` is a declared dependency on Linux (see
    pyproject), so it is installed on every Linux CI runner, and this only
    stayed green while the runner image happened to export ``DISPLAY``. When
    that changed the job went red — and the fallback it was supposed to be
    guarding had never actually been exercised.
    """
    blocker = _BlockImport("dbus_next")
    stale = [n for n in list(sys.modules) if n == "dbus_next" or n.startswith("dbus_next.")]
    saved = {n: sys.modules.pop(n) for n in stale}
    sys.meta_path.insert(0, blocker)
    try:
        track = asyncio.run(nowplaying._linux())
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)
    return [] if track is None else [f"expected None without dbus_next installed, got {track}"]


def check_linux_no_players() -> list[str]:
    saved = _install(_fake_dbus_next([]))
    try:
        track = asyncio.run(nowplaying._linux())
    finally:
        _restore(saved)
    return [] if track is None else [f"expected None with no MPRIS players, got {track}"]


def check_linux_prefers_the_playing_one() -> list[str]:
    """A paused browser tab from this morning and Spotify actually playing
    right now must not be a coin flip — the one making sound wins."""
    players = [
        ("org.mpris.MediaPlayer2.firefox", "Old Podcast", "Someone", "Paused"),
        ("org.mpris.MediaPlayer2.spotify", "Current Song", "The Band", "Playing"),
    ]
    saved = _install(_fake_dbus_next(players))
    try:
        track = asyncio.run(nowplaying._linux())
    finally:
        _restore(saved)
    if track is None:
        return ["expected a track, got None"]
    return (
        []
        if track.title == "Current Song" and track.artist == "The Band"
        else [f"picked the wrong player: {track}"]
    )


def check_linux_falls_back_to_paused() -> list[str]:
    """Nothing is playing, but something is paused with real metadata — that
    is still more useful in the header than nothing at all."""
    players = [("org.mpris.MediaPlayer2.vlc", "Paused Track", "Artist", "Paused")]
    saved = _install(_fake_dbus_next(players))
    try:
        track = asyncio.run(nowplaying._linux())
    finally:
        _restore(saved)
    if track is None:
        return ["expected the paused track as a fallback, got None"]
    return [] if track.title == "Paused Track" else [f"wrong fallback track: {track}"]


def check_linux_ignores_players_with_no_metadata() -> list[str]:
    """A registered player with an empty title (just launched, nothing
    loaded yet) must be skipped rather than reported as a blank track."""
    players = [
        ("org.mpris.MediaPlayer2.empty", None, None, "Stopped"),
        ("org.mpris.MediaPlayer2.real", "Real Track", "Real Artist", "Playing"),
    ]
    saved = _install(_fake_dbus_next(players))
    try:
        track = asyncio.run(nowplaying._linux())
    finally:
        _restore(saved)
    if track is None:
        return ["expected the real track, got None"]
    return [] if track.title == "Real Track" else [f"wrong track: {track}"]


# ── shared ─────────────────────────────────────────────────────────────────────

def check_track_str_without_artist() -> list[str]:
    t = nowplaying.Track(title="Just A Title", artist="")
    return [] if str(t) == "Just A Title" else [f"str() with no artist: {t!s}"]


def check_current_dispatches_by_platform() -> list[str]:
    """An unsupported platform (macOS, BSD, ...) must fall straight through
    to None rather than attempting either backend."""
    real_platform = sys.platform
    sys.platform = "darwin"
    try:
        track = asyncio.run(nowplaying.current())
    finally:
        sys.platform = real_platform
    return [] if track is None else [f"expected None on an unsupported platform, got {track}"]


def check_current_times_out_rather_than_hangs() -> list[str]:
    """A stuck backend call must not stall the header forever — this is
    exactly why current() wraps each platform call in asyncio.wait_for."""
    import asyncio as _asyncio

    async def _hangs():
        await _asyncio.sleep(30)
        return None

    real_windows, real_linux = nowplaying._windows, nowplaying._linux
    real_platform = sys.platform
    nowplaying._windows = _hangs
    nowplaying._linux = _hangs
    sys.platform = "win32"
    try:
        # current()'s own timeout is 3s; this just proves it actually fires
        # rather than inheriting the coroutine's full 30s sleep.
        track = asyncio.run(_asyncio.wait_for(nowplaying.current(), timeout=5.0))
    finally:
        nowplaying._windows, nowplaying._linux = real_windows, real_linux
        sys.platform = real_platform
    return [] if track is None else [f"expected the timeout to produce None, got {track}"]


TESTS = [
    ("windows: no winrt installed", check_windows_no_package),
    ("windows: reports the active track", check_windows_reports_track),
    ("windows: no active session", check_windows_no_active_session),
    ("windows: blank title is not a track", check_windows_blank_title_is_nothing),
    ("linux: no dbus_next installed", check_linux_no_package),
    ("linux: no MPRIS players registered", check_linux_no_players),
    ("linux: prefers the playing player over a paused one", check_linux_prefers_the_playing_one),
    ("linux: falls back to a paused player's track", check_linux_falls_back_to_paused),
    ("linux: skips a player with no metadata", check_linux_ignores_players_with_no_metadata),
    ("Track.__str__ with no artist", check_track_str_without_artist),
    ("current() skips unsupported platforms", check_current_dispatches_by_platform),
    ("current() times out rather than hangs", check_current_times_out_rather_than_hangs),
]


# ── pytest entry points ───────────────────────────────────────────────────────

def test_windows_no_package() -> None:
    bad = check_windows_no_package()
    assert not bad, "\n".join(bad)


def test_windows_reports_track() -> None:
    bad = check_windows_reports_track()
    assert not bad, "\n".join(bad)


def test_windows_no_active_session() -> None:
    bad = check_windows_no_active_session()
    assert not bad, "\n".join(bad)


def test_windows_blank_title_is_nothing() -> None:
    bad = check_windows_blank_title_is_nothing()
    assert not bad, "\n".join(bad)


def test_linux_no_package() -> None:
    bad = check_linux_no_package()
    assert not bad, "\n".join(bad)


def test_linux_no_players() -> None:
    bad = check_linux_no_players()
    assert not bad, "\n".join(bad)


def test_linux_prefers_the_playing_one() -> None:
    bad = check_linux_prefers_the_playing_one()
    assert not bad, "\n".join(bad)


def test_linux_falls_back_to_paused() -> None:
    bad = check_linux_falls_back_to_paused()
    assert not bad, "\n".join(bad)


def test_linux_ignores_players_with_no_metadata() -> None:
    bad = check_linux_ignores_players_with_no_metadata()
    assert not bad, "\n".join(bad)


def test_track_str_without_artist() -> None:
    bad = check_track_str_without_artist()
    assert not bad, "\n".join(bad)


def test_current_dispatches_by_platform() -> None:
    bad = check_current_dispatches_by_platform()
    assert not bad, "\n".join(bad)


def test_current_times_out_rather_than_hangs() -> None:
    bad = check_current_times_out_rather_than_hangs()
    assert not bad, "\n".join(bad)


if __name__ == "__main__":
    failures = 0
    for name, fn in TESTS:
        bad = fn()
        mark = "ok  " if not bad else "FAIL"
        print(f"  [{mark}] {name}")
        for b in bad:
            print(f"         {b}")
        failures += len(bad)
    print("\nall good" if not failures else f"\n{failures} problems")
    raise SystemExit(1 if failures else 0)
