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

from spektr.analysis import HOP, N_BANDS, Analyser, BandPlan, resample_bands  # noqa: E402
from spektr.capture import Capture, RingBuffer, Source              # noqa: E402
from spektr.motion import Peaks, Spring                             # noqa: E402

SR = 48000

#: Long enough to fill the bass window at any supported rate.
FEED = 16384


def tone(freq: float, n: int, amp: float = 0.3, phase: float = 0.0) -> np.ndarray:
    t = (np.arange(n) + phase) / SR
    s = (np.sin(2 * math.pi * freq * t) * amp).astype(np.float32)
    return np.stack((s, s), axis=1)


def band_of(freq: float, plan: BandPlan | None = None) -> int:
    """Which bar a frequency belongs to, asked of the plan rather than guessed.

    An earlier version of this file recomputed the band edges from its own copy
    of the formula, which meant the test agreed with itself rather than with
    the analyser.
    """
    plan = plan or BandPlan(SR)
    return int(np.clip(np.searchsorted(plan.cutoff, freq) - 1, 0, N_BANDS - 1))


def settle(an: Analyser, ring: RingBuffer, signal: np.ndarray, rounds: int = 250) -> np.ndarray:
    """Feed a signal hop by hop until autosens has converged, return the bands.

    cava's sensitivity loop needs a few hundred analyses to find its level;
    reading one frame straight after the first push measures the transient, not
    the steady state.
    """
    ring.clear()
    ring.push(signal[:FEED])
    pos = FEED
    for _ in range(rounds):
        if pos + HOP > len(signal):
            pos = 0
        ring.push(signal[pos : pos + HOP])
        pos += HOP
        an._analyse_once()
    return an.frame.bands


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


def test_band_plan() -> list[str]:
    """Every bar must read its own bins.

    This is the bug the cava port exists to fix. The old layout spread 32 bands
    over 20 Hz-20 kHz and read them from one 2048-point FFT, so at 48 kHz the
    bottom bands all landed on the same two bins — five were exact duplicates
    of an earlier band and nine were under two bins wide. The bass moved as a
    slab because it was, quite literally, one number drawn four times.
    """
    bad = []
    for rate in (44100, 48000, 96000, 22050):
        p = BandPlan(rate)
        cut = p.bass_bar
        widths = p.upper - p.lower + 1

        if np.any(widths < 1):
            bad.append(f"{rate}: {int((widths < 1).sum())} bars read no bins at all")
        if not np.all(np.diff(p.lower[:cut]) > 0):
            bad.append(f"{rate}: bass bars do not advance through the spectrum")
        if not np.all(np.diff(p.lower[cut:]) > 0):
            bad.append(f"{rate}: mid bars do not advance through the spectrum")
        # disjoint within each window — the two windows have their own indices,
        # so a bass bar and a mid bar sharing a bin *number* is not a collision
        if not np.all(p.lower[1:cut] > p.upper[: cut - 1]):
            bad.append(f"{rate}: bass bars overlap")
        if not np.all(p.lower[cut + 1 :] > p.upper[cut:-1]):
            bad.append(f"{rate}: mid bars overlap")
        if not np.all(np.diff(p.cutoff) > 0):
            bad.append(f"{rate}: cut-off frequencies are not increasing")
        if p.bass_size <= p.mid_size:
            bad.append(f"{rate}: the bass window must be longer than the mid window")

    p = BandPlan(SR)
    print(f"    {SR} Hz: bass window {p.bass_size} ({SR / p.bass_size:.1f} Hz/bin), "
          f"mid {p.mid_size} ({SR / p.mid_size:.1f} Hz/bin), {p.bass_bar} bass bars")
    print(f"    narrowest bar {int((p.upper - p.lower + 1).min())} bins, "
          f"widest {int((p.upper - p.lower + 1).max())}, "
          f"range {p.cutoff[0]:.0f}-{p.cutoff[-1]:.0f} Hz")
    return bad


def test_bands() -> list[str]:
    bad = []
    ring = RingBuffer(SR)
    an = Analyser(ring, lambda: SR)
    plan = BandPlan(SR)

    for freq in (60.0, 120.0, 300.0, 1000.0, 3000.0, 8000.0):
        bands = settle(an, ring, tone(freq, SR))
        if an.frame.silent:
            bad.append(f"{freq:.0f} Hz: gated as silent")
            continue
        peak = int(np.argmax(bands))
        want = band_of(freq, plan)
        if abs(peak - want) > 1:
            bad.append(f"{freq:.0f} Hz: peak band {peak}, expected ~{want}")
    return bad


def test_treble_reaches_full_height() -> list[str]:
    """The eq tilt earns its place here.

    Spectra fall off with frequency, so without cava's ``f^0.85`` boost the top
    bars sit near the floor no matter what is playing, and a third of the
    display is decoration. A tone at 6 kHz should drive its bar as hard as a
    tone at 200 Hz drives that one.
    """
    bad = []
    ring = RingBuffer(SR)
    heights = {}
    for freq in (200.0, 6000.0):
        an = Analyser(ring, lambda: SR)      # a fresh sens per source
        bands = settle(an, ring, tone(freq, SR), rounds=400)
        heights[freq] = float(bands.max())
    for freq, h in heights.items():
        if h < 0.5:
            bad.append(f"{freq:.0f} Hz only reached {h:.2f} of full height")
    print("    peak height: " + "  ".join(f"{k:.0f}Hz={v:.2f}" for k, v in heights.items()))
    return bad


def test_autosens_converges() -> list[str]:
    """A quiet source and a loud one must both end up using the display.

    The old auto-gain normalised on frame RMS, which meant a kick raised the
    RMS, lowered the gain and shrank the whole display *on the beat* — exactly
    backwards. cava scales on overshoot instead: down while anything clips, up
    while nothing does.
    """
    bad = []
    results = {}
    for amp in (0.004, 0.05, 0.5):
        ring = RingBuffer(SR)
        an = Analyser(ring, lambda: SR)
        bands = settle(an, ring, tone(440.0, SR, amp=amp), rounds=600)
        results[amp] = float(bands.max())
    for amp, peak in results.items():
        if peak < 0.4:
            bad.append(f"amplitude {amp}: settled at {peak:.2f} — never reached the display")
        if peak > 1.0001:
            bad.append(f"amplitude {amp}: {peak:.2f} exceeds full height")
    print("    settled peak: " + "  ".join(f"amp{k}={v:.2f}" for k, v in results.items()))
    return bad


def test_band_count_is_settable() -> list[str]:
    """One control, two mechanisms — and the boundary has to hold.

    Asking for fewer bars than the analyser resolves is a drawing question:
    the modes resample and nothing about the analysis changes. Asking for more
    is an analysis question, and answering it by interpolating would be a lie —
    the extra bars have to be real bin ranges. So below 32 the plan must stay
    put, and above it the plan must actually grow.
    """
    bad = []
    ring = RingBuffer(SR)

    for want, expect in ((8, N_BANDS), (32, N_BANDS), (48, 48), (64, 64)):
        an = Analyser(ring, lambda: SR)
        an.set_bands(want)
        settle(an, ring, tone(440.0, SR), rounds=120)
        got = len(an.frame.bands)
        if got != expect:
            bad.append(f"asked for {want} bands, analyser resolved {got}, expected {expect}")
        plan = an.plan
        if len(plan.lower) != expect:
            bad.append(f"{want}: plan has {len(plan.lower)} bars, expected {expect}")
        # the guarantees from test_band_plan must survive a rebuild
        if np.any(plan.upper - plan.lower + 1 < 1):
            bad.append(f"{want}: some bars read no bins after the rebuild")
        cut = plan.bass_bar
        if not np.all(np.diff(plan.lower[cut:]) > 0):
            bad.append(f"{want}: mid bars stopped advancing after the rebuild")

    # a silent frame must be the same width as a live one, or the widget
    # resizes its springs every time the music pauses
    an = Analyser(ring, lambda: SR)
    an.set_bands(64)
    settle(an, ring, tone(440.0, SR), rounds=60)
    live = len(an.frame.bands)
    ring.clear()
    ring.push(np.zeros((FEED, 2), dtype=np.float32))
    time.sleep(0.35)          # past the gate's hold window
    an._analyse_once()
    if not an.frame.silent:
        bad.append("silence was not gated")
    elif len(an.frame.bands) != live:
        bad.append(f"silent frame is {len(an.frame.bands)} wide, live is {live}")

    print("    8/32 keep the plan at 32; 48 and 64 rebuild it")
    return bad


def test_shutdown_filter_is_narrow() -> list[str]:
    """The exit-noise filter must drop soundcard's destructor noise only.

    soundcard's ``_COMLibrary.__del__`` raises during interpreter shutdown and
    prints a traceback after the UI is gone, which looks like a crash on exit —
    especially in the frozen exe. Suppressing it is right; suppressing anything
    else would hide real bugs, so that boundary is worth a test.
    """
    import spektr.capture as C

    original = sys.unraisablehook
    seen = []
    try:
        sys.unraisablehook = lambda u: seen.append(getattr(u.object, "__qualname__", "?"))
        C.install_shutdown_filter()

        class _COMLibrary:                      # stands in for soundcard's
            def __del__(self):
                pass

        _COMLibrary.__del__.__module__ = "soundcard.mediafoundation"

        class Ours:
            def __del__(self):
                pass

        Ours.__del__.__module__ = "spektr.capture"

        def fire(fn, exc):
            u = type("U", (), {})()
            u.object, u.exc_type, u.exc_value = fn, type(exc), exc
            u.exc_traceback, u.err_msg = None, None
            sys.unraisablehook(u)

        fire(_COMLibrary.__del__, AttributeError("no attribute 'com_loaded'"))
        fire(Ours.__del__, AttributeError("one of ours broke"))
        fire(len, ValueError("not a destructor at all"))
    finally:
        sys.unraisablehook = original

    # these are nested classes, so qualnames carry a <locals> prefix
    bad = []
    if any("_COMLibrary" in name for name in seen):
        bad.append("soundcard's destructor noise was not suppressed")
    for want in ("Ours.__del__", "len"):
        if not any(want in name for name in seen):
            bad.append(f"the filter swallowed {want} — it is too broad")
    print(f"    dropped soundcard's, passed through {len(seen)} others")
    return bad


def test_gate() -> list[str]:
    ring = RingBuffer(SR)
    an = Analyser(ring, lambda: SR)
    ring.push(tone(1000.0, FEED, amp=1e-7))
    an._analyse_once()
    if not an.frame.silent:
        return ["gate: near-silence was not gated"]

    ring.clear()
    ring.push(tone(1000.0, FEED, amp=0.2))
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
        cap._settle = lambda src: setattr(cap, "chosen", src)

        def fake_open(dev, sr, ch, extra):
            cap.kind = next((s.kind for s in sources if s.device == dev), "?")
            return object()

        cap._open = fake_open
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


def test_takes_the_default_output_and_stays() -> list[str]:
    """cava's rule: ask the system which device is playing, take it, hold it.

    The previous behaviour auditioned every tap for 2.5 seconds of signal and
    rotated onward when one was quiet, re-sweeping forever. That cannot tell
    "wrong device" apart from "nothing is playing yet", so starting spektr
    before pressing play sent it wandering onto whichever endpoint enumerated
    first — usually an idle HDMI output — and it would keep reopening devices
    from there. Silence on the default output is a correct answer, and the
    only sane response to it is to say so.
    """
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

    opens: list = []

    class FakeStream:
        def stop(self): pass
        def close(self): pass

    def fake_open(dev, sr, ch, extra):
        opens.append(dev)
        return FakeStream()

    cap._open = fake_open
    cap.start()

    bad = []
    try:
        # nothing playing at all: it must still be on the first tap, and must
        # say so rather than going looking
        time.sleep(1.0)
        if cap.label != "loopback: Speakers":
            bad.append(f"did not take the default output; on {cap.label!r}")
        if cap.on_mic:
            bad.append("fell back to the microphone while a tap was open")
        if len(opens) != 1:
            bad.append(f"opened {len(opens)} devices — it is still hunting")

        # a status that reports the silence is useful; switching device is not
        time.sleep(2.5)
        if "nothing playing" not in cap.status:
            bad.append(f"silence was not reported: {cap.status!r}")

        # audio arrives on the device we are already holding
        def pump():
            for _ in range(120):
                cap.ring.push(np.full((256, 2), 0.3, dtype=np.float32))
                time.sleep(0.005)

        threading.Thread(target=pump, daemon=True).start()
        time.sleep(1.0)
        if "listening" not in cap.status:
            bad.append(f"audio arrived but the status still says {cap.status!r}")
        if len(opens) != 1:
            bad.append(f"reopened the device {len(opens)} times")
    finally:
        cap.stop()
        time.sleep(0.2)

    print(f"    held {cap.label!r} throughout, {len(opens)} device open(s)")
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

    def get_microphone(id, include_loopback=False):
        """soundcard resolves an id by fuzzy match, not by equality.

        That difference is the entire bug this replaced: WASAPI does not
        promise that a render device's name and its loopback endpoint's name
        are character-for-character identical, so comparing them directly
        misses and the "default" silently becomes whatever enumerated first.
        """
        want = str(id).lower()
        for name in endpoints:
            if want in name.lower() or name.lower() in want:
                return FakeMic(name)
        raise RuntimeError(f"no microphone matching {id!r}")

    sc.get_microphone = get_microphone
    return sc


def test_enumerates_off_the_capture_thread() -> list[str]:
    """Device discovery must happen where it is known to work.

    On a real machine, ``soundcard`` returned three WASAPI loopback endpoints
    when queried from the process's own thread and *none* when first imported
    and queried from a freshly spawned capture thread. Nothing raised — the
    list simply came back empty, spektr fell through to Stereo Mix (muted by
    default on Realtek hardware) and the display sat flat with no error
    anywhere. ``--diagnose`` enumerated from the main thread and reported
    everything as healthy, which is what made it so hard to see.

    So enumeration happens in start(), on the caller's thread. This checks it
    stays there: a backend that only works on the starting thread must still
    produce a full candidate list.
    """
    import threading

    from spektr.capture import Capture, Source

    home = threading.get_ident()
    calls = []

    loopback = Source("A", SR, 2, None, "loopback: Headphones", "loopback")
    stereo_mix = Source("B", SR, 2, None, "monitor: Stereo Mix", "monitor")

    def thread_sensitive_candidates():
        here = threading.get_ident()
        calls.append(here)
        # the failure mode, reproduced: loopbacks vanish off the home thread
        return [loopback, stereo_mix] if here == home else [stereo_mix]

    cap = Capture()
    cap.candidates = thread_sensitive_candidates

    class FakeStream:
        def stop(self): pass
        def close(self): pass

    cap._open = lambda dev, sr, ch, extra: FakeStream()
    cap.start()
    time.sleep(0.6)

    bad = []
    try:
        if cap.label != "loopback: Headphones":
            bad.append(f"enumerated on the wrong thread — settled on {cap.label!r}")
        if not calls:
            bad.append("candidates() was never called")
        elif calls[0] != home:
            bad.append("candidates() was first called off the starting thread")
    finally:
        cap.stop()
        time.sleep(0.2)

    print(f"    enumerated on the starting thread, chose {cap.label!r}")
    return bad


def test_soundcard_loopback_backend() -> list[str]:
    """The real fix for "barely any reaction" on Windows.

    PortAudio has no WASAPI loopback flag, so sounddevice cannot capture system
    audio at any version — it was falling back to Stereo Mix, which Windows
    mutes by default. soundcard talks to WASAPI directly. This checks that
    loopback endpoints are enumerated with the current default output first, and
    that audio actually reaches the ring through that backend.

    The endpoint order below is the point: a virtual "AI noise-cancelling"
    device enumerates ahead of the real speakers and plays nothing. Anything
    that picks by enumeration order lands on it and shows a flat display
    forever, which is exactly what happened on a real machine.
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

        # The endpoint name need not match the render device's name exactly.
        # Comparing them for equality is what used to fail; resolving the id
        # through soundcard is what fixes it.
        skewed = _fake_soundcard(live, ["AI Noise-cancelling Output (ASUS Utility)",
                                        live + " [Loopback]"])
        sys.modules["soundcard"] = skewed
        cap_skew = Capture()
        first = [s for s in cap_skew.candidates() if s.kind == "loopback"]
        if not first or live not in first[0].label:
            got = first[0].label if first else "nothing"
            bad.append(f"name mismatch defeated the default lookup: picked {got!r}")
        sys.modules["soundcard"] = _fake_soundcard(live, others)

        # and audio must actually flow through it
        cap2 = Capture()
        cap2.start()
        time.sleep(2.0)
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
        t = np.arange(SR) / SR
        # white-ish content so every band sees something
        sig = np.sin(2 * math.pi * 440 * t) + 0.5 * np.sin(2 * math.pi * 3000 * t)
        sig = sig / np.sqrt(np.mean(sig ** 2)) * rms_target
        # Both sources are run to convergence: autosens will happily normalise
        # the noise floor to full height, and the point of the knee is that the
        # floor stays small anyway.
        an = Analyser(ring, lambda: SR)
        settle(an, ring, np.stack((sig, sig), axis=1).astype(np.float32), rounds=400)
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
    ("takes the default output and stays", test_takes_the_default_output_and_stays),
    ("enumeration failures are explained", test_enumeration_failures_are_explained),
    ("enumerates off the capture thread", test_enumerates_off_the_capture_thread),
    ("soundcard loopback backend", test_soundcard_loopback_backend),
    ("COM initialised per thread", test_com_is_initialised_per_thread),
    ("soft knee suppresses the noise floor", test_knee_suppresses_floor),
    ("band plan (cava distribution)", test_band_plan),
    ("band mapping", test_bands),
    ("treble reaches full height", test_treble_reaches_full_height),
    ("autosens converges", test_autosens_converges),
    ("band count is settable", test_band_count_is_settable),
    ("shutdown filter stays narrow", test_shutdown_filter_is_narrow),
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
