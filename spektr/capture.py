"""System audio capture into a ring buffer.

Two changes from the original design, both of which matter downstream:

1. **Stereo is preserved.** The old callback did ``indata.mean(axis=1)`` and
   threw away the channel difference before anything could use it. Half the
   information in a stereo mix was being discarded at the door; the goniometer
   and stereo meters need it.

2. **A ring buffer instead of a single slot.** The old callback overwrote
   ``_fft_buf`` unconditionally, so any block that arrived between two renders
   was lost, and any render that arrived between two blocks re-analysed stale
   samples. With a ring the analyser reads on its own clock and can overlap
   windows, which is where the smoothness comes from.

The device probing logic is kept close to the original — it was tuned against
real WASAPI quirks and the fallback order earns its keep.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Callable, NamedTuple, Optional

import numpy as np

BLOCK = 256          # callback granularity; small, so the ring stays current
RING_SECONDS = 1.0

#: HRESULTs worth translating, because "Error 0x800401f0" tells you nothing.
_HRESULTS = {
    0x800401F0: "CO_E_NOTINITIALIZED — COM was not initialised on this thread",
    0x80070005: "E_ACCESSDENIED — microphone/app permission is blocked in Windows privacy settings",
    0x88890001: "AUDCLNT_E_NOT_INITIALIZED",
    0x88890004: "AUDCLNT_E_DEVICE_INVALIDATED — the device was removed or reconfigured",
    0x8889000A: "AUDCLNT_E_DEVICE_IN_USE — another app holds it in exclusive mode",
    0x88890008: "AUDCLNT_E_UNSUPPORTED_FORMAT — the endpoint rejected this sample rate/channel count",
    0x8889000E: "AUDCLNT_E_SERVICE_NOT_RUNNING — the Windows Audio service is stopped",
}


def load_soundcard():
    """Import ``soundcard``, with the shutdown crash patched out.

    soundcard's ``_COMLibrary`` sets ``self.com_loaded`` inside a ``try`` in
    ``__init__`` and reads it in ``__del__``. When the attribute never gets set
    — a failed construction, or an instance dict already torn down at
    interpreter exit — the destructor raises, and Python prints

        Exception ignored in: <function _COMLibrary.__del__ ...>
        AttributeError: '_COMLibrary' object has no attribute 'com_loaded'

    as spektr closes. Harmless, but it is the last thing a user sees, and in
    the frozen exe it looks exactly like a crash on exit.

    A class-level default fixes it at the source rather than hiding the
    message, and the fallback value is the correct one: if the flag is missing,
    this object never initialised COM, so the destructor must not uninitialise
    it. Guarded so a future soundcard that has already fixed this is untouched.
    """
    import soundcard as sc

    try:
        from soundcard import mediafoundation as _mf  # Windows only

        if not hasattr(_mf._COMLibrary, "com_loaded"):
            _mf._COMLibrary.com_loaded = False
    except Exception:  # noqa: BLE001 — not Windows, or the internals moved
        pass
    return sc


def install_shutdown_filter() -> None:
    """Swallow destructor noise from the audio libraries at interpreter exit.

    ``__del__`` failures are routed to :func:`sys.unraisablehook`, which prints
    them and carries on. During shutdown, module globals are already being set
    to None, so any destructor that touches one can raise through no fault of
    its own — soundcard has several. Nothing in this class of error can be
    acted on, and the process is going away regardless.

    Deliberately narrow: only unraisables raised from inside the audio
    libraries' own destructors are dropped, and everything else is handed to
    whichever hook was installed before. A blanket hook would hide real bugs.
    """
    previous = sys.unraisablehook

    def hook(unraisable):
        func = getattr(unraisable.object, "__qualname__", "") or ""
        module = getattr(unraisable.object, "__module__", "") or ""
        if func.endswith("__del__") and module.startswith(("soundcard", "sounddevice")):
            return
        previous(unraisable)

    sys.unraisablehook = hook


def _pkg_version(module, name: str) -> str:
    """Version of an installed package. soundcard has no ``__version__``."""
    v = getattr(module, "__version__", None)
    if v:
        return str(v)
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001
        return "unknown"


def explain_hresult(exc: BaseException) -> str:
    """Append a human reading to a bare COM error code, if we know it."""
    text = str(exc)
    for code, meaning in _HRESULTS.items():
        if f"{code:08x}" in text.lower() or f"{code:#x}" in text.lower():
            return f"{text}  [{meaning}]"
    return text


def _com_init() -> bool:
    """Give the calling thread a COM apartment. Returns True if we made one.

    COM apartments are per-thread, and ``soundcard`` initialises COM once at
    import time — on whatever thread imported it. A recorder opened from any
    other thread fails with CO_E_NOTINITIALIZED (0x800401F0), which is exactly
    what happens when the capture pump runs in its own thread. So it has to be
    done here.
    """
    if os.name != "nt":
        return False
    try:
        import ctypes

        COINIT_MULTITHREADED = 0x0      # matches what soundcard itself uses
        hr = ctypes.windll.ole32.CoInitializeEx(None, COINIT_MULTITHREADED) & 0xFFFFFFFF
        if hr in (0x0, 0x1):            # S_OK, S_FALSE (already initialised)
            return True
        # RPC_E_CHANGED_MODE: this thread is already in a different apartment,
        # which is fine — we just must not uninitialise it on the way out.
        return False
    except Exception:  # noqa: BLE001
        return False


def _com_uninit() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.ole32.CoUninitialize()
    except Exception:  # noqa: BLE001
        pass


def default_tap(notes: "list[str] | None" = None) -> "Optional[Source]":
    """Loopback of whatever the OS currently calls the default output.

    This is cava's model, and it is the whole selection strategy: ask the
    system which device is playing and record that one. cava's PulseAudio
    input takes the default sink's monitor; its Windows capture takes the
    default render endpoint. Neither auditions devices looking for signal.

    The previous approach here enumerated every loopback endpoint and sorted
    the one whose ``name`` equalled ``default_speaker().name`` to the front.
    That comparison is fragile — WASAPI reports endpoint names that do not
    always match the render device's name character for character, and when it
    misses, the "default" ends up being whichever endpoint enumerated first,
    which on a laptop with HDMI attached is routinely a device that never
    plays anything. ``get_microphone`` does soundcard's own id resolution
    instead, which is the documented way to ask this question.
    """
    try:
        sc = load_soundcard()
        speaker = sc.default_speaker()
        mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        return Source(
            mic,
            48000,
            max(1, min(2, int(getattr(mic, "channels", 2) or 2))),
            None,
            f"loopback: {speaker.name}",
            "loopback",
            "sc",
        )
    except Exception as exc:  # noqa: BLE001
        if notes is not None:
            notes.append(f"default output loopback unavailable: {explain_hresult(exc)}")
        return None


class Source(NamedTuple):
    """One candidate capture device.

    ``kind`` matters more than it looks. A loopback tap that is silent is
    reporting the truth — the speakers are not playing anything — whereas a
    microphone always has *something* on it. Treating "carries signal" as the
    only selection criterion picks the microphone every time spektr is launched
    while nothing is playing, and then cheerfully visualises the room.
    """

    device: object
    samplerate: int
    channels: int
    extra: object
    label: str
    kind: str            # "loopback" | "monitor" | "mic"
    backend: str = "sd"  # "sd" = sounddevice/PortAudio, "sc" = soundcard/WASAPI

    @property
    def is_tap(self) -> bool:
        """True if this listens to output rather than to a room."""
        return self.kind in ("loopback", "monitor")


class _LoopbackStream:
    """A soundcard loopback recorder, pumped into the ring by its own thread.

    Why a second audio library at all: **PortAudio cannot do WASAPI loopback**,
    and neither can sounddevice on top of it. ``WasapiSettings`` exposes only
    ``exclusive``, ``auto_convert`` and ``explicit_sample_format`` — there has
    never been a ``loopback`` argument, and PortAudio has no such flag to wrap.
    An earlier version of this file assumed otherwise and reported "sounddevice
    too old", which sent people chasing an upgrade that could not help.

    ``soundcard`` talks to WASAPI directly and sets AUDCLNT_STREAMFLAGS_LOOPBACK
    itself, so it captures whatever an output endpoint is playing. It records
    by blocking rather than by callback, which suits us fine — the ring buffer
    already decouples capture from analysis.
    """

    def __init__(self, mic, samplerate: int, channels: int, blocksize: int, ring: "RingBuffer"):
        self._mic = mic
        self._sr = int(samplerate)
        self._ch = max(1, int(channels))
        self._bs = int(blocksize)
        self._ring = ring
        self._running = True
        self.error: BaseException | None = None
        #: frames handed to the ring, and whether the recorder ever opened
        self.pushed = 0
        self.opened = False
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        # surface an open failure to the caller rather than in a dead thread
        self._ready.wait(timeout=5.0)
        if self.error is not None:
            raise self.error

    def _pump(self) -> None:
        # must happen on *this* thread, before touching the recorder
        mine = _com_init()
        try:
            with self._mic.recorder(
                samplerate=self._sr, channels=self._ch, blocksize=self._bs
            ) as rec:
                self.opened = True
                self._ready.set()
                while self._running:
                    data = rec.record(numframes=self._bs)
                    if data is not None and len(data):
                        # Counted so a stream that opens cleanly but never
                        # delivers can be told apart from one that failed to
                        # open. On WASAPI loopback those look identical from
                        # the outside and have completely different causes.
                        self.pushed += len(data)
                        self._ring.push(np.asarray(data, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001
            self.error = exc
        finally:
            self._ready.set()
            if mine:
                _com_uninit()

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.5)


class RingBuffer:
    """Fixed-size circular buffer of interleaved stereo frames."""

    __slots__ = ("_buf", "_cap", "_lock", "_w", "_written")

    def __init__(self, capacity: int):
        self._cap = int(capacity)
        self._buf = np.zeros((self._cap, 2), dtype=np.float32)
        self._w = 0
        self._written = 0
        self._lock = threading.Lock()

    @property
    def written(self) -> int:
        return self._written

    def clear(self) -> None:
        with self._lock:
            self._buf[:] = 0.0
            self._w = 0
            self._written = 0

    def push(self, block: np.ndarray) -> None:
        """block: (n,) mono or (n, ch). Mono is duplicated across L/R."""
        if block.ndim == 1:
            block = block[:, None]
        if block.shape[1] == 1:
            block = np.repeat(block, 2, axis=1)
        elif block.shape[1] > 2:
            block = block[:, :2]

        n = block.shape[0]
        if n == 0:
            return
        if n >= self._cap:
            block = block[-self._cap :]
            n = self._cap

        with self._lock:
            end = self._w + n
            if end <= self._cap:
                self._buf[self._w : end] = block
            else:
                split = self._cap - self._w
                self._buf[self._w :] = block[:split]
                self._buf[: end - self._cap] = block[split:]
            self._w = end % self._cap
            self._written += n

    def latest(self, n: int) -> Optional[np.ndarray]:
        """The most recent n frames as ``(n, 2)``, or None if not filled yet."""
        n = min(int(n), self._cap)
        with self._lock:
            if self._written < n:
                return None
            start = (self._w - n) % self._cap
            if start + n <= self._cap:
                return self._buf[start : start + n].copy()
            split = self._cap - start
            out = np.empty((n, 2), dtype=np.float32)
            out[:split] = self._buf[start:]
            out[split:] = self._buf[: n - split]
            return out


class Capture:
    """Opens the best available loopback source and feeds a RingBuffer."""

    def __init__(self, device=None, on_status: Callable[[str], None] | None = None,
                 allow_mic: bool = False):
        self.ring = RingBuffer(int(48000 * RING_SECONDS))
        self.samplerate = 48000
        self.status = "starting capture…"
        self.label = ""
        self._forced = device
        self._allow_mic = allow_mic
        self._on_status = on_status
        #: True when we ended up on a microphone rather than an output tap
        self.on_mic = False
        self._stream = None
        self._running = False
        #: how many sources to step past — bumped by `d`
        self._skip = 0
        #: sources that would not open, with the reason. Shown in the status
        #: when we end up somewhere other than a loopback tap.
        self._skipped: list[str] = []
        #: candidates resolved by start(); None means "not enumerated yet"
        self._cands: "list[Source] | None" = None
        #: why enumeration produced what it did — surfaced by --monitor
        self.notes: list[str] = []
        self._thread: threading.Thread | None = None

    # ── lifecycle ──
    def start(self) -> None:
        # Enumerate here, on the caller's thread, rather than inside the
        # capture thread.
        #
        # This is not a style preference. Device discovery goes through COM,
        # and on at least one real machine ``soundcard`` returns *no loopback
        # endpoints at all* when it is first imported and queried from a
        # freshly created MTA thread, while the identical call from the thread
        # that started the process returns three. ``--diagnose`` enumerated
        # from the main thread and saw everything; the app enumerated from its
        # capture thread, silently got an empty list, and fell through to
        # Stereo Mix — which is muted by default on Realtek hardware, so the
        # display sat flat with no error anywhere.
        #
        # Opening still happens on the capture/pump thread, exactly as before.
        # Only the question "what devices exist" moves, and that is the call
        # we know works, because it is the one --diagnose has been making all
        # along.
        try:
            self._cands = self.candidates()
        except Exception as exc:  # noqa: BLE001
            self._cands = None
            self.notes.append(f"enumeration on the calling thread failed: {exc}")

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._close()

    def next_source(self) -> None:
        """Drop the current source and reopen on the next candidate."""
        self._skip += 1
        self._reopen()

    def reset_source(self) -> None:
        """Forget manual picks and go back to the OS default output.

        `d` is sticky for the rest of the session, which is right — a manual
        choice should not be silently undone — but there has to be a way back,
        otherwise cycling past the device you wanted means restarting spektr.
        """
        self._skip = 0
        self._reopen()

    def _reopen(self) -> None:
        self._running = False
        self._close()
        time.sleep(0.05)
        self.start()

    def _close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _say(self, text: str) -> None:
        self.status = text
        if self._on_status:
            try:
                self._on_status(text)
            except Exception:
                pass

    # ── device discovery ──
    def candidates(self) -> "list[Source]":
        """Ordered capture candidates, output taps first."""
        import sounddevice as sd

        out: list[Source] = []
        seen: set = set()

        if self._forced is not None:
            info = sd.query_devices(self._forced)
            loop = None
            if info["max_input_channels"] == 0:
                loop = sd.WasapiSettings(loopback=True)
            chans = int(info["max_output_channels"] or info["max_input_channels"] or 2)
            out.append(
                Source(
                    self._forced,
                    int(info.get("default_samplerate") or 48000),
                    max(1, min(2, chans)),
                    loop,
                    f"forced: {info['name']}",
                    "loopback" if loop is not None else "monitor",
                )
            )
            return out

        # 1. WASAPI output endpoints, default first — loopback hears the speakers.
        #
        # Everything here is reported into self.notes rather than swallowed. An
        # earlier version wrapped this whole block in one try/except, so a single
        # unsupported endpoint — or a sounddevice too old for loopback — silently
        # discarded *every* loopback candidate and left spektr quietly falling
        # back to Stereo Mix. The failure was invisible because the explanation
        # went to a status line that the next message immediately overwrote.
        self.notes = []

        # ── 1a. soundcard: the only path that actually does WASAPI loopback ──
        try:
            sc = load_soundcard()

            # The default output's own loopback, resolved the way soundcard
            # documents it. See default_tap() for why this is not the same as
            # picking the endpoint whose name matches the default speaker's.
            default_src = default_tap(self.notes)
            if default_src is not None:
                out.append(default_src)

            loops = [m for m in sc.all_microphones(include_loopback=True)
                     if getattr(m, "isloopback", False)]
            if not loops:
                self.notes.append("soundcard found no loopback endpoints")

            default_name = default_src.label.split(": ", 1)[-1] if default_src else None
            for m in loops:
                if default_name and m.name == default_name:
                    continue          # already first in the list
                out.append(
                    Source(
                        m, 48000, max(1, min(2, int(getattr(m, "channels", 2) or 2))),
                        None, f"loopback: {m.name}", "loopback", "sc",
                    )
                )
        except ImportError:
            self.notes.append(
                "soundcard is not installed — it is what provides WASAPI loopback. "
                "Run: pip install soundcard"
            )
        except Exception as exc:  # noqa: BLE001
            self.notes.append(f"soundcard loopback unavailable: {explain_hresult(exc)}")

        # ── 1b. sounddevice/PortAudio loopback, if it ever gains support ──
        loopback_settings = None
        try:
            loopback_settings = sd.WasapiSettings(loopback=True)
        except TypeError:
            # Expected on every sounddevice released so far. Not a version
            # problem — PortAudio has no loopback flag to expose at all.
            pass
        except Exception:  # noqa: BLE001
            pass

        if loopback_settings is not None:
            wasapi = [
                (i, api) for i, api in enumerate(sd.query_hostapis())
                if "wasapi" in api["name"].lower()
            ]
            if not wasapi:
                names = ", ".join(a["name"] for a in sd.query_hostapis())
                self.notes.append(f"no WASAPI host API found. Available: {names}")

            for _, api in wasapi:
                default = api.get("default_output_device", -1)
                devs = list(api.get("devices", []))
                if not devs:
                    self.notes.append("WASAPI host API reports no devices")
                if default is not None and default >= 0:
                    devs = [default] + [d for d in devs if d != default]

                for dev in devs:
                    # per device: one awkward endpoint must not cost us the rest
                    try:
                        info = sd.query_devices(dev)
                        if info["max_output_channels"] < 1 or dev in seen:
                            continue
                        seen.add(dev)
                        out.append(
                            Source(
                                dev,
                                int(info.get("default_samplerate") or 48000),
                                max(1, min(2, int(info["max_output_channels"]))),
                                loopback_settings,
                                f"loopback: {info['name']}",
                                "loopback",
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        self.notes.append(f"device {dev} skipped: {exc!r}")
                break

        if not any(s.kind == "loopback" for s in out):
            self.notes.append(
                "No WASAPI loopback endpoint was offered — this is spektr's main "
                "capture path on Windows, so something above needs fixing."
            )

        # 2. monitor-style inputs (Stereo Mix, BlackHole, PulseAudio monitor)
        try:
            for i, d in enumerate(sd.query_devices()):
                name = d["name"].lower()
                if d["max_input_channels"] > 0 and any(
                    k in name
                    for k in (
                        "stereo mix", "loopback", "what u hear", "monitor",
                        "blackhole", "soundflower", "virtual line", "pipewire",
                    )
                ):
                    out.append(
                        Source(
                            i,
                            int(d.get("default_samplerate") or 48000),
                            max(1, min(2, int(d["max_input_channels"]))),
                            None,
                            f"monitor: {d['name']}",
                            "monitor",
                        )
                    )
        except Exception:
            pass

        # 3. the microphone. Last resort only — see the note on Source.kind.
        #    Never selected automatically while any output tap is available.
        out.append(Source(None, 48000, 1, None, "microphone (NOT system audio)", "mic"))
        return out

    def _open_source(self, src: "Source"):
        """Open whichever backend this source belongs to."""
        if src.backend == "sc":
            stream = _LoopbackStream(src.device, src.samplerate, src.channels, 512, self.ring)
            self.samplerate = src.samplerate
            return stream
        return self._open(src.device, src.samplerate, src.channels, src.extra)

    def _open(self, dev, sr, ch, extra):
        import sounddevice as sd

        def callback(indata, frames, t, status):
            self.ring.push(np.asarray(indata, dtype=np.float32))

        # a loopback endpoint only yields audio at the rate windows is actually
        # running it at, which is often not the rate it reports
        rates = [sr] + [r for r in (48000, 44100, 96000) if r != sr]
        chans = [ch] + [c for c in (2, 1) if c != ch]
        last = None
        for rate in rates:
            for c in chans:
                try:
                    stream = sd.InputStream(
                        device=dev,
                        channels=c,
                        samplerate=rate,
                        blocksize=BLOCK,
                        latency="low",
                        callback=callback,
                        extra_settings=extra,
                    )
                    stream.start()
                    self.samplerate = rate
                    return stream
                except Exception as exc:
                    last = exc
        raise last

    def _run(self) -> None:
        # The capture thread enumerates devices as well as opening them, and
        # both go through COM. Give this thread an apartment for its lifetime.
        mine = _com_init()
        try:
            self._run_inner()
        finally:
            if mine:
                _com_uninit()

    def _run_inner(self) -> None:
        last_err = None
        self._skipped = []

        try:
            # Enumerated in start(), on the thread that asked us to run — see
            # the note there. Only re-enumerate if that could not be done.
            cands = self._cands if self._cands is not None else self.candidates()
        except (ImportError, OSError) as exc:
            # a missing library used to kill this thread silently, leaving the
            # status stuck on "starting capture…" forever
            self._say(backend_help(exc).splitlines()[0])
            return
        except Exception as exc:  # noqa: BLE001
            self._say(f"could not enumerate audio devices: {exc}")
            return

        # Take the first source that opens, and stay on it.
        #
        # This used to audition: open each tap, measure 2.5 seconds of signal,
        # and rotate onward if it was quiet, re-sweeping every few seconds
        # forever. The reasoning was that a silent tap might be the wrong tap.
        # In practice it is the wrong strategy, for the same reason cava does
        # not do it: *silence on the default output is the correct answer* when
        # nothing is playing, and auditioning cannot tell that apart from a bad
        # device. What it did instead was wander onto whatever endpoint
        # happened to enumerate first — an idle HDMI output, typically — and
        # then keep reopening devices, so pressing play produced nothing.
        #
        # The system already knows which device is playing. Ask it, take that,
        # hold it. `d` cycles manually when you want something else.
        taps = [c for c in cands if c.is_tap]
        mics = [c for c in cands if not c.is_tap]
        manual = self._skip > 0

        order = list(taps)
        if manual or self._allow_mic:
            order += mics              # an explicit request may reach the mic
        if order:
            skip = self._skip % len(order)
            order = order[skip:] + order[:skip]

        for src in order:
            if not self._running:
                return
            try:
                self._stream = self._open_source(src)
            except Exception as exc:
                # Not swallowed. Falling past the device the OS says is playing
                # is the single most confusing thing this class can do, and the
                # reason has to travel with the result — it used to be reported
                # only if *every* source failed, so landing on Stereo Mix said
                # nothing at all about the three loopbacks that came first.
                last_err = f"{src.label} -> {explain_hresult(exc)}"
                self._skipped.append(last_err)
                continue
            self._settle(src)
            return

        if not self._running:
            return

        # No output tap could be opened at all — the only case where a
        # microphone is a reasonable automatic choice, and it needs saying.
        for src in mics:
            if not self._running:
                return
            try:
                self._stream = self._open_source(src)
                self._settle(src)
                return
            except Exception as exc:
                last_err = f"{src.label} -> {exc}"

        self._say(f"no audio source. last error: {last_err}")

    def _settle(self, src: Source) -> None:
        """Hold the chosen source open until asked to stop.

        The status line still distinguishes "audio is arriving" from "nothing
        is playing", because that is the first question anyone asks when the
        display is flat. It just never acts on it — reporting silence and
        switching device in response to silence are very different things, and
        only the first one is useful.
        """
        self.label = src.label
        self.on_mic = src.kind == "mic"

        # Anything we had to skip to get here is part of the answer to "why is
        # it listening to *that*?", so it goes in the status rather than into a
        # note nobody reads. So is the fact that *you* chose it: `d` is sticky
        # for the rest of the session, and a manual pick looks identical to a
        # bad automatic one unless it says so.
        why = ""
        if self._skip > 0:
            why = f"  |  manual pick {self._skip + 1} — press D for the default output"
        elif self._skipped and src.kind != "loopback":
            why = "  |  skipped " + "; ".join(self._skipped[:2])

        if self.on_mic:
            self._say(f"microphone — {src.label}. No output tap opened.{why}")
        else:
            self._say(f"listening — {src.label}{why}")

        quiet_since = time.time()
        reported_quiet = False
        while self._running:
            buf = self.ring.latest(2048)
            if buf is not None and float(np.abs(buf).max()) > 3e-5:
                quiet_since = time.time()
                if reported_quiet and not self.on_mic:
                    self._say(f"listening — {src.label}")
                    reported_quiet = False
            elif not reported_quiet and time.time() - quiet_since > 3.0 and not self.on_mic:
                self._say(f"{src.label} — nothing playing (press d for another source)")
                reported_quiet = True
            time.sleep(0.2)


#: What to tell someone whose audio backend isn't usable. Both the TUI and
#: ``--devices`` route through this, so the advice stays consistent.
def backend_help(exc: BaseException) -> str:
    text = str(exc).lower()
    if isinstance(exc, ImportError):
        return "sounddevice is not installed — pip install sounddevice"
    if "portaudio" in text:
        return (
            "PortAudio not found — spektr needs it to reach your sound card.\n"
            "  Debian/Ubuntu:  sudo apt install libportaudio2\n"
            "  Fedora:         sudo dnf install portaudio\n"
            "  Arch:           sudo pacman -S portaudio\n"
            "  macOS:          brew install portaudio\n"
            "  Windows:        it ships with sounddevice; try pip install -U sounddevice"
        )
    return f"could not reach the audio backend: {exc}"


def diagnose(seconds: float = 2.0) -> str:
    """Open every candidate in turn and report what it actually delivers.

    This answers the only question that matters when the display isn't moving:
    is audio reaching spektr at all, from which device, and at what level. It
    prints raw numbers rather than a verdict, because a level that looks wrong
    against the gate is the whole diagnosis.
    """
    from .analysis import Analyser

    cap = Capture()
    try:
        cands = cap.candidates()
    except (ImportError, OSError) as exc:
        return backend_help(exc)

    probe = Analyser(cap.ring, lambda: cap.samplerate)
    gate = probe.gate

    import sounddevice as sd

    lines = ["environment"]
    lines.append(f"  sounddevice      {getattr(sd, '__version__', 'unknown')}  (playback devices, monitors, mic)")
    try:
        sc = load_soundcard()

        loops = [m for m in sc.all_microphones(include_loopback=True)
                 if getattr(m, "isloopback", False)]
        lines.append(
            f"  soundcard        {_pkg_version(sc, 'SoundCard')}  "
            f"— {len(loops)} WASAPI loopback endpoint(s)"
        )
        # This is now the entire selection rule, so it is the first thing to
        # check when the display is flat: is the OS pointing at the device you
        # are actually listening to?
        try:
            lines.append(f"  default output   {sc.default_speaker().name}")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"  default output   could not be read — {explain_hresult(exc)}")
        picked = default_tap()
        lines.append(
            f"  spektr will use  {picked.label}" if picked
            else "  spektr will use  (default output loopback unavailable — see notes below)"
        )
    except ImportError:
        lines.append("  soundcard        NOT INSTALLED — this is what provides loopback")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"  soundcard        present but unusable — {explain_hresult(exc)}")
    try:
        apis = sd.query_hostapis()
        lines.append(
            "  host APIs        "
            + ", ".join(f"{a['name']} ({len(a.get('devices', []))} devices)" for a in apis)
        )
    except Exception as exc:  # noqa: BLE001
        lines.append(f"  host APIs        could not query: {exc!r}")

    counts: dict = {}
    for c in cands:
        counts[c.kind] = counts.get(c.kind, 0) + 1
    lines.append(
        "  candidates       "
        + (", ".join(f"{n}x {k}" for k, n in counts.items()) or "none")
    )

    for note in getattr(cap, "notes", []):
        lines.append(f"  ! {note}")

    lines += [
        "",
        f"Play something now — probing each source for {seconds:.0f}s.",
        f"Noise gate is {gate:.2e}; anything below that is treated as silence.",
        "",
        f"  {'source':<44} {'rms':>10} {'peak':>8} {'x gate':>8}  verdict",
    ]

    best = None
    for src in cands:
        try:
            cap.ring.clear()
            stream = cap._open_source(src)
        except Exception as exc:
            lines.append(
                f"  {src.label[:44]:<44} {'—':>10} {'—':>8} {'—':>8}  "
                f"cannot open: {explain_hresult(exc)}"
            )
            continue

        end = time.time() + seconds
        peak = 0.0
        acc, n = 0.0, 0
        while time.time() < end:
            buf = cap.ring.latest(2048)
            if buf is not None:
                mono = buf.mean(axis=1)
                peak = max(peak, float(np.abs(mono).max()))
                acc += float(np.sqrt(np.mean(mono * mono)))
                n += 1
            time.sleep(0.05)
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

        rms = acc / max(n, 1)
        ratio = rms / gate if gate else 0.0
        if ratio < 1:
            verdict = "silent"
        elif ratio < 4:
            verdict = "barely above the gate"
        else:
            verdict = "AUDIO" if src.is_tap else "audio (microphone!)"
        if src.is_tap and ratio >= 4 and best is None:
            best = src

        lines.append(
            f"  {src.label[:44]:<44} {rms:10.2e} {peak:8.3f} {ratio:8.1f}  {verdict}"
        )

    lines.append("")
    if best is not None:
        lines.append(f"Looks good — spektr should select: {best.label}")
        return "\n".join(lines)

    has_loopback = any(c.kind == "loopback" for c in cands)
    if not has_loopback:
        lines += [
            "No loopback endpoint was offered at all — that is the problem.",
            "",
            "Loopback comes from the `soundcard` package, not sounddevice:",
            "PortAudio has no WASAPI loopback flag, so sounddevice cannot do it at",
            "any version. Without soundcard you are left with Stereo Mix, which",
            "Windows mutes by default.",
            "",
            "  pip install soundcard",
            "",
            "then re-run spektr --diagnose and check the environment block above.",
        ]
    else:
        lines += [
            "Loopback endpoints exist but none carried audio. Either nothing was",
            "playing during the probe, or Windows is outputting somewhere else.",
            "",
            "  - play something loud, then re-run this",
            "  - check the Windows volume mixer is sending to that endpoint",
            "  - spektr --devices, then spektr --device <n> to force one",
        ]

    stereo_mix = [c for c in cands if "stereo mix" in c.label.lower()]
    if stereo_mix:
        lines += [
            "",
            "Note: Stereo Mix showed up but read as near-silent. It is disabled or",
            "muted by default on most Realtek installs. Sound settings → Recording →",
            "Stereo Mix → Properties → Levels, and raise it — but fixing loopback",
            "above is the better answer, since Stereo Mix is lower quality.",
        ]
    return "\n".join(lines)


def monitor(seconds: float = 12.0) -> str:
    """Run the app's own capture path headlessly and report what arrives.

    ``--diagnose`` opens each device from the main thread, one at a time, and
    measures it. That answers "can this device deliver audio", which is not
    the same question as "is the running application receiving any". When the
    two disagree — a device that measures fine but a flat display — everything
    interesting is in the gap between them, and nothing in the TUI shows it.

    So this builds the same :class:`Capture`, on its own thread, exactly as the
    widget does, attaches the same analyser, and prints the pipeline stage by
    stage once a second:

        frames   samples the capture backend handed to the ring
        rms      level in the ring, which is what the analyser reads
        gate     whether the noise gate is open
        bars     peak band height, i.e. what would be drawn

    Where the zeros start is the answer.
    """
    from .analysis import Analyser

    out = []

    def say(text):
        out.append(text)
        print(text, flush=True)

    cap = Capture()
    an = Analyser(cap.ring, lambda: cap.samplerate)
    cap.start()
    an.start()

    # What the app is actually choosing between. When this disagrees with
    # --diagnose, the disagreement *is* the bug — enumeration is not returning
    # the same devices in both places.
    cands = cap._cands or []
    kinds = {}
    for c in cands:
        kinds[c.kind] = kinds.get(c.kind, 0) + 1
    say("  candidates: " + (", ".join(f"{n}x {k}" for k, n in kinds.items()) or "none"))
    for c in cands[:6]:
        say(f"    {c.label}")
    if cap.notes:
        say("  notes:")
        for note in cap.notes:
            say(f"    {note}")
    say("")
    say("Play something now. Watching the same pipeline the visualiser uses.")
    say("")
    say(f"  {'t':>5}  {'frames':>9}  {'rms':>9}  {'gate':>6}  {'sens':>9}  {'bars':>5}  source")
    try:
        end = time.time() + seconds
        while time.time() < end:
            time.sleep(1.0)
            frames = cap.ring.written
            buf = cap.ring.latest(4096)
            rms = float(np.sqrt(np.mean(buf * buf))) if buf is not None else 0.0
            frame = an.frame
            gate = "shut" if frame.silent else "open"
            peak = float(frame.bands.max()) if frame.bands.size else 0.0
            t = seconds - (end - time.time())
            say(f"  {t:5.1f}  {frames:>9}  {rms:9.2e}  {gate:>6}  {an._sens:9.1f}  "
                f"{peak:5.2f}  {cap.label or cap.status}")
    finally:
        an.stop()
        cap.stop()

    say("")
    stream = cap._stream
    pushed = getattr(stream, "pushed", None)
    if pushed is not None:
        say(f"  the capture backend pushed {pushed} frames "
            f"(recorder opened: {getattr(stream, 'opened', '?')})")
    if cap._skipped:
        say("  sources skipped on the way here:")
        for note in cap._skipped:
            say(f"    {note}")

    say("")
    if cap.ring.written == 0:
        say("  Nothing reached the ring: the capture backend is not delivering.")
    elif all("0.00e+00" in line for line in out[-int(seconds) - 4:] if "  " in line):
        say("  The device is open but every block is digital silence.")
    else:
        say("  Audio reached the ring — if the display is still flat the problem")
        say("  is downstream of capture, not in device selection.")
    return "\n".join(out)


def describe_devices() -> str:
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        return backend_help(exc)

    lines = [f"sounddevice {sd.__version__}", "", "host apis:"]
    for i, api in enumerate(sd.query_hostapis()):
        lines.append(f"  [{i}] {api['name']}  default_out={api.get('default_output_device')}")
    lines += ["", "devices:"]
    for i, d in enumerate(sd.query_devices()):
        lines.append(
            f"  [{i:>2}] in={d['max_input_channels']:>2} "
            f"out={d['max_output_channels']:>2} {d['name']}"
        )
    try:
        sc = load_soundcard()

        loops = [m for m in sc.all_microphones(include_loopback=True)
                 if getattr(m, "isloopback", False)]
        lines += ["", f"soundcard {_pkg_version(sc, 'SoundCard')}"
                      f" — {len(loops)} WASAPI loopback endpoint(s)"]
        for m in loops:
            lines.append(f"  {m.name}")
    except ImportError:
        lines += ["", "soundcard NOT installed — no WASAPI loopback. pip install soundcard"]
    except Exception as exc:
        lines += ["", f"soundcard present but unusable: {exc!r}"]
    return "\n".join(lines)
