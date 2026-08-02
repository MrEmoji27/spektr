"""Analysis pipeline test — no audio device required.

Feeds synthetic audio straight into the ring buffer, which is exactly what the
sound card callback does, and checks three things:

1. tones land in the right bands
2. the analyser really does run faster than the frame rate (this is the whole
   point of moving it off the render timer)
3. the spring settles to the same place regardless of frame rate, which the
   old per-frame damping multiply did not
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The default Windows console is cp1252, which cannot encode the ≈ and — this
# file prints. Without this the suite dies on a UnicodeEncodeError on the
# project's own primary platform.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from spektr.analysis import HOP, N_BANDS, Analyser, resample_bands  # noqa: E402
from spektr.capture import Capture, RingBuffer, Source              # noqa: E402
from spektr.motion import Peaks, Spring                             # noqa: E402

SR = 48000


def tone(freq: float, n: int, amp: float = 0.3, phase: float = 0.0) -> np.ndarray:
    t = (np.arange(n) + phase) / SR
    s = (np.sin(2 * math.pi * freq * t) * amp).astype(np.float32)
    return np.stack((s, s), axis=1)


def band_of(freq: float) -> int:
    edges = np.logspace(math.log10(20.0), math.log10(20000.0), N_BANDS + 1)
    return int(np.clip(np.searchsorted(edges, freq) - 1, 0, N_BANDS - 1))


def test_ring_roundtrip() -> list[str]:
    bad = []
    ring = RingBuffer(1024)
    for i in range(10):
        ring.push(np.full((300, 2), i, dtype=np.float32))
    got = ring.latest(600)
    if got is None or got.shape != (600, 2):
        return ["ring: wrong shape"]
    # the most recent 300 frames must all be the last value pushed
    if not np.allclose(got[-300:], 9.0):
        bad.append("ring: latest() did not return the newest samples")
    if ring.written != 3000:
        bad.append(f"ring: written={ring.written}, want 3000")

    mono = RingBuffer(64)
    mono.push(np.ones(32, dtype=np.float32))
    got = mono.latest(32)
    if got is None or got.shape[1] != 2:
        bad.append("ring: mono input was not duplicated to stereo")
    return bad


def test_bands() -> list[str]:
    bad = []
    ring = RingBuffer(SR)
    an = Analyser(ring, lambda: SR)

    for freq in (120.0, 1000.0, 8000.0):
        ring.clear()
        ring.push(tone(freq, 4096))
        an._analyse_once()
        f = an.frame
        if f.silent:
            bad.append(f"{freq:.0f} Hz: gated as silent")
            continue
        peak = int(np.argmax(f.bands))
        want = band_of(freq)
        if abs(peak - want) > 1:
            bad.append(f"{freq:.0f} Hz: peak band {peak}, expected ~{want}")
    return bad


def test_gate() -> list[str]:
    ring = RingBuffer(SR)
    an = Analyser(ring, lambda: SR)
    ring.push(tone(1000.0, 4096, amp=1e-7))
    an._analyse_once()
    if not an.frame.silent:
        return ["gate: near-silence was not gated"]

    ring.clear()
    ring.push(tone(1000.0, 4096, amp=0.2))
    an._analyse_once()
    if an.frame.silent:
        return ["gate: real audio was gated"]
    return []


def test_analysis_rate() -> list[str]:
    """The reason this rewrite exists: analysis must outpace the frame rate."""
    ring = RingBuffer(SR)
    an = Analyser(ring, lambda: SR)
    an.start()

    seen = []
    deadline = time.monotonic() + 1.0
    phase = 0
    try:
        while time.monotonic() < deadline:
            ring.push(tone(440.0, HOP, phase=phase))
            phase += HOP
            time.sleep(HOP / SR)          # feed at real time
            seen.append(an.frame.seq)
    finally:
        an.stop()

    rate = max(seen) - min(seen)
    if rate < 60:
        return [f"analysis ran at ~{rate} Hz over 1 s — must exceed 60 fps"]
    print(f"    analysis rate ≈ {rate} Hz (frame rate is 60)")
    return []


def test_framerate_independence() -> list[str]:
    """Same input, different frame rates — the spring must land in the same
    place. The old code's per-frame damping multiply failed this badly."""
    target = np.full(N_BANDS, 0.75)
    results = {}
    for fps in (15, 30, 60, 120):
        s = Spring(N_BANDS)
        dt = 1.0 / fps
        for _ in range(int(fps * 0.6)):     # 600 ms of settling
            s.step(target, dt)
        results[fps] = float(s.x.mean())

    spread = max(results.values()) - min(results.values())
    print("    settle after 600 ms: " + "  ".join(f"{k}fps={v:.4f}" for k, v in results.items()))
    if spread > 0.01:
        return [f"spring is frame-rate dependent: spread {spread:.4f} across 15-120 fps"]
    return []


def test_peak_hold_seconds() -> list[str]:
    """Peak hold is in seconds, so it must decay identically at any fps."""
    results = {}
    for fps in (15, 60):
        p = Peaks(4, hold=0.2, fall=0.5)
        dt = 1.0 / fps
        p.step(np.full(4, 0.9), dt)
        for _ in range(int(fps * 1.0)):
            p.step(np.zeros(4), dt)
        results[fps] = float(p.value.mean())
    spread = abs(results[15] - results[60])
    print("    peak after 1 s: " + "  ".join(f"{k}fps={v:.4f}" for k, v in results.items()))
    if spread > 0.02:
        return [f"peak hold is frame-rate dependent: spread {spread:.4f}"]
    return []


def test_resample() -> list[str]:
    bad = []
    src = np.linspace(0.0, 1.0, N_BANDS)
    for n in (8, 10, 16, 32, 48):
        out = resample_bands(src, n)
        if len(out) != n:
            bad.append(f"resample to {n}: got {len(out)}")
        elif out.min() < -1e-9 or out.max() > 1.0 + 1e-9:
            bad.append(f"resample to {n}: out of range {out.min()}..{out.max()}")
    return bad


def test_never_picks_mic() -> list[str]:
    """The regression that shipped: with nothing playing, every loopback tap
    correctly reports silence while the microphone picks up the room — so
    "first source with signal wins" selected the microphone and visualised the
    room. Silence on an output tap is a valid answer and must be preferred."""
    taps = [
        Source("spk", SR, 2, None, "loopback: Speakers", "loopback"),
        Source("hp", SR, 2, None, "loopback: Headphones", "loopback"),
        Source(3, SR, 2, None, "monitor: Stereo Mix", "monitor"),
    ]
    mic = Source(None, SR, 1, None, "microphone (NOT system audio)", "mic")
    bad = []

    def rig(sources):
        cap = Capture()
        cap.candidates = lambda: sources
        cap.chosen = None
        cap.kind = ""
        cap._close = lambda: None
        cap._settle = lambda src, heard: setattr(cap, "chosen", src)

        def fake_open(dev, sr, ch, extra):
            cap.kind = next((s.kind for s in sources if s.device == dev), "?")
            return object()

        cap._open = fake_open
        # only the microphone carries signal, exactly as when nothing is playing
        cap._heard_signal = lambda secs: cap.kind == "mic"
        # one sweep only: the real loop hunts forever, which is correct but
        # would never return here
        def one_pass(src, seconds=0.0):
            cap._running = False
            return False

        cap._wait_for_audio = one_pass
        cap._running = True
        cap._run()
        return cap

    cap = rig(taps + [mic])
    if cap.chosen is not None and cap.chosen.kind == "mic":
        bad.append("selected the microphone while output taps were available")
    if cap.on_mic:
        bad.append("on_mic was set despite taps being available")

    # with no output tap at all, the mic *is* the right answer
    cap2 = rig([mic])
    if cap2.chosen is None or cap2.chosen.kind != "mic":
        bad.append("with no output tap available, the mic should be used")

    return bad


def test_hunts_for_a_live_tap() -> list[str]:
    """Launching before you press play, or output going to a non-default
    endpoint, must both resolve themselves. spektr holds a silent tap, watches
    for audio, and rotates onto the next — without ever reaching for the mic."""
    import threading

    from spektr.capture import Source

    taps = [
        Source("A", SR, 2, None, "loopback: Speakers", "loopback"),
        Source("B", SR, 2, None, "loopback: HDMI", "loopback"),
        Source("C", SR, 2, None, "monitor: Stereo Mix", "monitor"),
    ]
    mic = Source(None, SR, 1, None, "microphone (NOT system audio)", "mic")

    cap = Capture()
    cap.candidates = lambda: taps + [mic]
    cap._probe_seconds = 0.3
    cap._watch_seconds = 0.8

    state = {"live": None, "cur": None}
    opens: list = []

    class FakeStream:
        def stop(self): pass
        def close(self): pass

    def fake_open(dev, sr, ch, extra):
        opens.append(dev)
        state["cur"] = dev
        return FakeStream()

    cap._open = fake_open

    def pump():
        while cap._running:
            amp = 0.3 if state["cur"] == state["live"] else 0.0
            cap.ring.push(np.full((256, 2), amp, dtype=np.float32))
            time.sleep(0.005)

    cap._running = True
    threading.Thread(target=pump, daemon=True).start()
    cap.start()

    bad = []
    try:
        time.sleep(2.5)                       # nothing playing yet
        if cap.on_mic:
            bad.append("fell back to the microphone while taps were silent")
        state["live"] = "B"                   # audio starts on a non-default tap
        time.sleep(3.5)
        if cap.label != "loopback: HDMI":
            bad.append(f"did not find the live tap; settled on {cap.label!r}")
        if cap.on_mic:
            bad.append("ended up on the microphone")
        if len(opens) > 40:
            bad.append(f"re-opened devices {len(opens)} times — probe loop is spinning")
    finally:
        cap.stop()
        time.sleep(0.2)

    print(f"    settled on {cap.label!r} after {len(opens)} device opens")
    return bad


def _fake_sounddevice(version, loopback_ok, devices, hostapis, bad_device=None):
    """A stand-in for sounddevice, so device enumeration is testable off-Windows."""
    import types

    fake = types.ModuleType("sounddevice")
    fake.__version__ = version

    class WasapiSettings:
        def __init__(self, **kw):
            if "loopback" in kw and not loopback_ok:
                raise TypeError("unexpected keyword argument 'loopback'")
            self.kw = kw

    fake.WasapiSettings = WasapiSettings

    def query_devices(i=None):
        if i is None:
            return devices
        if i == bad_device:
            raise OSError("Invalid device")
        return devices[i]

    fake.query_devices = query_devices
    fake.query_hostapis = lambda: hostapis
    return fake


def test_enumeration_failures_are_explained() -> list[str]:
    """Reported in the field: only Stereo Mix and a mic were offered, no loopback
    at all, and nothing said why. Two separate causes, both silent before —
    soundcard missing (it is what provides loopback), and one bad endpoint
    aborting the whole scan because a single try/except wrapped the entire loop.
    """
    import builtins
    import sys

    devices = [
        {"name": "Speakers (Realtek)", "max_input_channels": 0, "max_output_channels": 2,
         "default_samplerate": SR},
        {"name": "BROKEN HDMI", "max_input_channels": 0, "max_output_channels": 2,
         "default_samplerate": SR},
        {"name": "Headphones", "max_input_channels": 0, "max_output_channels": 2,
         "default_samplerate": SR},
        {"name": "Stereo Mix (Realtek)", "max_input_channels": 2, "max_output_channels": 0,
         "default_samplerate": SR},
    ]
    apis = [{"name": "Windows WASAPI", "default_output_device": 0, "devices": [0, 1, 2, 3]}]

    bad = []
    real_sd = sys.modules.get("sounddevice")
    real_sc = sys.modules.get("soundcard")
    real_import = builtins.__import__

    def no_soundcard(name, *args, **kwargs):
        if name == "soundcard":
            raise ImportError("No module named 'soundcard'")
        return real_import(name, *args, **kwargs)

    try:
        # (a) soundcard absent — no loopback, and the note must name the fix
        sys.modules.pop("soundcard", None)
        builtins.__import__ = no_soundcard
        sys.modules["sounddevice"] = _fake_sounddevice("0.5.5", False, devices, apis)
        cap = Capture()
        got = cap.candidates()
        if any(s.kind == "loopback" for s in got):
            bad.append("offered loopback with soundcard unavailable")
        if not any("pip install soundcard" in n for n in cap.notes):
            bad.append(f"note does not name the fix: {cap.notes}")
        builtins.__import__ = real_import

        # (b) one endpoint raises mid-scan — the others must survive.
        #     Exercised on the sounddevice path, which is where the old
        #     single try/except lost everything.
        sys.modules["soundcard"] = _fake_soundcard("none", [])
        sys.modules["sounddevice"] = _fake_sounddevice(
            "0.5.5", True, devices, apis, bad_device=1
        )
        cap = Capture()
        got = cap.candidates()
        loop = [s for s in got if s.kind == "loopback" and s.backend == "sd"]
        if len(loop) != 2:
            bad.append(f"one bad endpoint cost us the rest: {len(loop)} of 2 survived")
        if not any("device 1 skipped" in n for n in cap.notes):
            bad.append(f"the skipped device was not reported: {cap.notes}")
    finally:
        builtins.__import__ = real_import
        for name, mod in (("sounddevice", real_sd), ("soundcard", real_sc)):
            if mod is not None:
                sys.modules[name] = mod
            else:
                sys.modules.pop(name, None)

    return bad


def _fake_soundcard(live_name, endpoints):
    """Stand-in for soundcard, so the loopback path is testable off Windows."""
    import types

    class FakeRecorder:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def record(self, numframes=512):
            time.sleep(numframes / SR)
            amp = 0.25 if self.name == live_name else 0.0
            t = np.arange(numframes) / float(SR)
            sig = (np.sin(2 * math.pi * 440 * t) * amp).astype(np.float32)
            return np.stack((sig, sig), axis=1)

    class FakeMic:
        def __init__(self, name):
            self.name = name
            self.isloopback = True
            self.channels = 2

        def recorder(self, samplerate, channels=None, blocksize=None):
            return FakeRecorder(self.name)

    sc = types.ModuleType("soundcard")
    sc.__version__ = "0.4.6"
    sc.all_microphones = lambda include_loopback=False: (
        [FakeMic(n) for n in endpoints] if include_loopback else []
    )
    sc.default_speaker = lambda: types.SimpleNamespace(name=live_name)
    return sc


def test_soundcard_loopback_backend() -> list[str]:
    """The real fix for "barely any reaction" on Windows.

    PortAudio has no WASAPI loopback flag, so sounddevice cannot capture system
    audio at any version — it was falling back to Stereo Mix, which Windows
    mutes by default. soundcard talks to WASAPI directly. This checks that
    loopback endpoints are enumerated with the current default output first, and
    that audio actually reaches the ring through that backend.
    """
    import sys

    live = "Speakers (Realtek(R) Audio)"
    others = ["AI Noise-cancelling Output (ASUS Utility)", live]

    real_sc = sys.modules.get("soundcard")
    real_sd = sys.modules.get("sounddevice")
    devices = [
        {"name": live, "max_input_channels": 0, "max_output_channels": 2,
         "default_samplerate": SR},
        {"name": "Stereo Mix (Realtek)", "max_input_channels": 2,
         "max_output_channels": 0, "default_samplerate": SR},
    ]
    apis = [{"name": "Windows WASAPI", "default_output_device": 0, "devices": [0, 1]}]

    bad = []
    try:
        sys.modules["soundcard"] = _fake_soundcard(live, others)
        sys.modules["sounddevice"] = _fake_sounddevice("0.5.5", False, devices, apis)

        cap = Capture()
        got = cap.candidates()
        loops = [s for s in got if s.kind == "loopback"]
        if len(loops) != 2:
            bad.append(f"expected 2 loopback endpoints, got {len(loops)}")
        elif loops[0].label != f"loopback: {live}":
            bad.append(f"default output not listed first: {loops[0].label!r}")
        if any(s.backend != "sc" for s in loops):
            bad.append("loopback sources were not attributed to the soundcard backend")

        # and audio must actually flow through it
        cap2 = Capture()
        cap2._probe_seconds = 1.0
        cap2._watch_seconds = 0.8
        cap2.start()
        time.sleep(2.5)
        try:
            if cap2.on_mic:
                bad.append("ended up on the microphone despite live loopback")
            if not cap2.status.startswith("listening"):
                bad.append(f"did not lock on: {cap2.status!r}")
            buf = cap2.ring.latest(2048)
            rms = float(np.sqrt(np.mean(buf * buf))) if buf is not None else 0.0
            if rms < 0.05:
                bad.append(f"no audio reached the ring through soundcard (rms {rms:.4f})")
            else:
                print(f"    captured rms={rms:.3f} via {cap2.label!r}")
        finally:
            cap2.stop()
            time.sleep(0.2)
    finally:
        for name, mod in (("soundcard", real_sc), ("sounddevice", real_sd)):
            if mod is not None:
                sys.modules[name] = mod
            else:
                sys.modules.pop(name, None)

    return bad


def test_com_is_initialised_per_thread() -> list[str]:
    """Reported in the field as `cannot open: Error 0x800401f0`.

    COM apartments are per-thread. soundcard initialises COM once at import, on
    whatever thread imported it, so every loopback endpoint failed to open from
    the capture thread with CO_E_NOTINITIALIZED. Both the thread that enumerates
    devices and the thread that runs the recorder need their own apartment.
    """
    import sys
    import threading

    import spektr.capture as C

    live = "Speakers"
    real_sc = sys.modules.get("soundcard")
    real_sd = sys.modules.get("sounddevice")
    real_init, real_uninit = C._com_init, C._com_uninit

    calls: list = []
    bad = []
    try:
        C._com_init = lambda: (calls.append(("init", threading.current_thread().name)) or True)
        C._com_uninit = lambda: calls.append(("uninit", threading.current_thread().name))

        sys.modules["soundcard"] = _fake_soundcard(live, [live])
        sys.modules["sounddevice"] = _fake_sounddevice(
            "0.5.5", False, [],
            [{"name": "Windows WASAPI", "default_output_device": 0, "devices": []}],
        )

        cap = C.Capture()
        cap._probe_seconds = 0.8
        cap._watch_seconds = 0.5
        cap.start()
        time.sleep(2.0)
        try:
            buf = cap.ring.latest(2048)
            rms = float(np.sqrt(np.mean(buf * buf))) if buf is not None else 0.0
            if rms < 0.05:
                bad.append(f"no audio through the loopback backend (rms {rms:.4f})")
        finally:
            cap.stop()
            time.sleep(0.4)

        threads = {t for _, t in calls}
        if len(threads) < 2:
            bad.append(f"COM initialised on only {threads} — needs the capture and pump threads")
        inits = sum(1 for k, _ in calls if k == "init")
        uninits = sum(1 for k, _ in calls if k == "uninit")
        if inits != uninits:
            bad.append(f"COM init/uninit unbalanced: {inits} vs {uninits}")
        else:
            print(f"    COM balanced across {len(threads)} threads ({inits} apartments)")
    finally:
        C._com_init, C._com_uninit = real_init, real_uninit
        for name, mod in (("soundcard", real_sc), ("sounddevice", real_sd)):
            if mod is not None:
                sys.modules[name] = mod
            else:
                sys.modules.pop(name, None)

    # and the code that tripped us must read as English
    if "CO_E_NOTINITIALIZED" not in C.explain_hresult(RuntimeError("Error 0x800401f0")):
        bad.append("0x800401f0 is not translated")
    if "DEVICE_IN_USE" not in C.explain_hresult(RuntimeError("Error 0x8889000a")):
        bad.append("0x8889000a is not translated")
    return bad


def test_knee_suppresses_floor() -> list[str]:
    """A signal barely above the gate must not be drawn at full height."""
    SR = 48000
    ring = RingBuffer(SR)
    an = Analyser(ring, lambda: SR)
    gate = an.gate

    out = {}
    for name, rms_target in (("floor", gate * 1.05), ("music", gate * 400)):
        ring.clear()
        t = np.arange(4096) / SR
        # white-ish content so every band sees something
        sig = np.sin(2 * math.pi * 440 * t) + 0.5 * np.sin(2 * math.pi * 3000 * t)
        sig = sig / np.sqrt(np.mean(sig ** 2)) * rms_target
        ring.push(np.stack((sig, sig), axis=1).astype(np.float32))
        an._analyse_once()
        f = an.frame
        out[name] = (float(f.bands.max()), f.confidence, f.silent)

    bad = []
    floor_peak, floor_conf, floor_silent = out["floor"]
    music_peak, music_conf, _ = out["music"]
    print(f"    just above gate: peak={floor_peak:.3f} strength={floor_conf:.2f}")
    print(f"    real music:      peak={music_peak:.3f} strength={music_conf:.2f}")

    if not floor_silent and floor_peak > 0.15:
        bad.append(f"noise floor drawn at {floor_peak:.2f} — should be near zero")
    if music_conf < 0.99:
        bad.append(f"real music attenuated by the knee (strength {music_conf:.2f})")
    if music_peak < 0.3:
        bad.append(f"real music only reached {music_peak:.2f}")
    return bad


TESTS = [
    ("ring buffer", test_ring_roundtrip),
    ("never auto-selects the microphone", test_never_picks_mic),
    ("hunts across taps for live audio", test_hunts_for_a_live_tap),
    ("enumeration failures are explained", test_enumeration_failures_are_explained),
    ("soundcard loopback backend", test_soundcard_loopback_backend),
    ("COM initialised per thread", test_com_is_initialised_per_thread),
    ("soft knee suppresses the noise floor", test_knee_suppresses_floor),
    ("band mapping", test_bands),
    ("noise gate", test_gate),
    ("band resampling", test_resample),
    ("analysis rate", test_analysis_rate),
    ("frame-rate independence", test_framerate_independence),
    ("peak hold in seconds", test_peak_hold_seconds),
]


if __name__ == "__main__":
    failures = 0
    for name, fn in TESTS:
        print(f"  {name}…")
        bad = fn()
        for b in bad:
            print(f"    FAIL {b}")
        failures += len(bad)
    print("\nall good" if not failures else f"\n{failures} failures")
    raise SystemExit(1 if failures else 0)
