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
                self._ready.set()
                while self._running:
                    data = rec.record(numframes=self._bs)
                    if data is not None and len(data):
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

    __slots__ = ("_buf", "_cap", "_w", "_written", "_lock")

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
        self._skip = 0
        self._rescan = 0
        self._thread: threading.Thread | None = None
        self._probe_seconds = 2.5
        #: base watch window when holding a silent tap; grows with each sweep
        self._watch_seconds = 6.0

    # ── lifecycle ──
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._close()

    def next_source(self) -> None:
        """Drop the current source and rescan starting from the next one."""
        self._skip += 1
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
            import soundcard as sc

            loops = [m for m in sc.all_microphones(include_loopback=True)
                     if getattr(m, "isloopback", False)]
            if not loops:
                self.notes.append("soundcard found no loopback endpoints")

            # put the current default output first — it's what you're hearing
            try:
                default_name = sc.default_speaker().name
            except Exception:  # noqa: BLE001
                default_name = None
            if default_name:
                loops.sort(key=lambda m: m.name != default_name)

            for m in loops:
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

    def _heard_signal(self, seconds: float) -> bool:
        self.ring.clear()
        end = time.time() + seconds
        while time.time() < end and self._running:
            buf = self.ring.latest(2048)
            if buf is not None and float(np.abs(buf).max()) > 3e-5:
                return True
            time.sleep(0.05)
        return False

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
        first_ok = None
        last_err = None

        try:
            cands = self.candidates()
        except (ImportError, OSError) as exc:
            # a missing library used to kill this thread silently, leaving the
            # status stuck on "starting capture…" forever
            self._say(backend_help(exc).splitlines()[0])
            return
        except Exception as exc:  # noqa: BLE001
            self._say(f"could not enumerate audio devices: {exc}")
            return

        # Output taps are auditioned; the microphone is not. Probing for "which
        # source carries signal" and taking the first hit sounds reasonable and
        # is wrong: launch spektr with nothing playing and every loopback tap
        # reports silence — correctly — while the microphone picks up the room,
        # so the room wins. spektr would then happily visualise your keyboard.
        #
        # Silence on a tap is a real answer. The mic is only reached if no tap
        # could be opened at all, or if you ask for it with `d` or --mic.
        taps = [c for c in cands if c.is_tap]
        mics = [c for c in cands if not c.is_tap]
        manual = self._skip > 0

        # Outer loop: keep hunting across taps for as long as we're running.
        # Launching spektr before pressing play, or output going to a
        # non-default endpoint, both leave us holding a tap that is real but
        # silent. Rather than sit there forever, hold it, watch for audio, and
        # rotate on if none arrives. Only ever taps — never a microphone.
        while self._running:
            order = list(taps)
            if self._rescan and taps:
                r = self._rescan % len(taps)
                order = order[r:] + order[:r]
            if manual or self._allow_mic:
                order += mics          # an explicit request may reach the mic
            if order:
                skip = self._skip % len(order)
                order = order[skip:] + order[:skip]

            first_ok = None
            for src in order:
                if not self._running:
                    return
                try:
                    stream = self._open_source(src)
                except Exception as exc:
                    last_err = f"{src.label} -> {exc}"
                    continue

                self._stream = stream
                self._say(f"probing — {src.label}")

                # a manually chosen source is honoured whether or not it has signal
                if manual or self._heard_signal(self._probe_seconds):
                    self._settle(src, heard=True)
                    return

                if first_ok is None:
                    first_ok = src
                self._close()

            if first_ok is None:
                break                  # nothing opened at all — fall through

            try:
                self._stream = self._open_source(first_ok)
            except Exception as exc:
                last_err = f"{first_ok.label} -> {exc}"
                break

            self.label = first_ok.label
            self.on_mic = False
            self._say(
                f"silent — {first_ok.label} (waiting for audio; press d to change source)"
            )
            # Back off as rounds go by. Someone who leaves spektr open on a
            # silent machine shouldn't have it reopening audio devices every
            # six seconds forever; a minute between sweeps is plenty to notice
            # playback starting, since audio on the *held* tap is seen instantly.
            watch = min(self._watch_seconds * (1 + self._rescan // max(1, len(taps))), 60.0)
            if self._wait_for_audio(first_ok, watch):
                return                 # audio arrived, or we were stopped

            if len(taps) < 2:
                self._settle(first_ok, heard=False)   # nowhere else to look
                return

            self._close()
            self._rescan += 1          # rotate and go round again

        if not self._running:
            return

        # No output tap could be opened at all — the only case where a
        # microphone is a reasonable automatic choice, and it needs saying.
        for src in mics:
            if not self._running:
                return
            try:
                self._stream = self._open_source(src)
                self._settle(src, heard=True)
                return
            except Exception as exc:
                last_err = f"{src.label} -> {exc}"

        self._say(f"no audio source. last error: {last_err}")

    def _wait_for_audio(self, src: Source, seconds: float | None = None) -> bool:
        """Hold a silent tap open and watch. True if audio turned up."""
        end = time.time() + (self._watch_seconds if seconds is None else seconds)
        while time.time() < end and self._running:
            buf = self.ring.latest(2048)
            if buf is not None and float(np.abs(buf).max()) > 3e-5:
                self._settle(src, heard=True)
                return True
            time.sleep(0.1)
        return not self._running

    def _settle(self, src: Source, heard: bool) -> None:
        """Hold a chosen source open until asked to stop."""
        self.label = src.label
        self.on_mic = src.kind == "mic"

        if self.on_mic:
            self._say(f"⚠ microphone — {src.label}. No output tap found; press d to cycle.")
        elif heard:
            self._say(f"listening — {src.label}")
        else:
            self._say(f"silent — {src.label} (nothing is playing; press d to change source)")

        while self._running:
            time.sleep(0.1)


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
        import soundcard as sc

        loops = [m for m in sc.all_microphones(include_loopback=True)
                 if getattr(m, "isloopback", False)]
        lines.append(
            f"  soundcard        {_pkg_version(sc, 'SoundCard')}  "
            f"— {len(loops)} WASAPI loopback endpoint(s)"
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
        import soundcard as sc

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
