"""Onset-evaluation corpus generator.

Not a test: a corpus. The files this writes are the ground truth the onset
detector is measured against, committed to the repo so that measurement is
reproducible without a synthesis step or an internet connection. They are
deliberately not wired into CI and not mentioned in the docs.

Tracks
------
``sim_115``   the default: continuous 4/4 at 115 BPM, a 16th-note stream of
    single sines at 4410 Hz (amplitude 0.25, legato, with a short fade at
    each note boundary so the stream has no clicks for a detector to trip
    on), plus one loud, separated kick per bar — a 60 Hz sine at 0.9 with a
    3 ms attack and an exponential decay. Steady notes under one isolated
    loud transient per bar exercise the dynamic-range and envelope paths: a
    correct detector fires exactly once per bar, at the kick.

``drive_130``  four-on-the-floor 4/4 at 130 BPM, a 16th-note stream of single
    sines at 9755 Hz, and five additional percussive hits per bar on the
    16th grid — the second (beat two), the three-and (the & of beat three),
    and three driving off-beat sixteenths (1&, 2&, 4&). Between the kicks
    and the extras the whole 16th grid is populated, so the score exercises
    a detector against a dense, driving beat rather than one hit per bar.

Every track is normalised to a peak of 0.95 — a kick landing on a note
boundary or a kick-plus-extra collision otherwise sums past 1.0 and clips
in s16le, smearing the very transients the corpus exists to time.

Format
------
Each track is ``<name>.wav`` — 44.1 kHz, stereo, s16le, identical channels;
the analyser's ring buffer expects two channels — plus ``<name>.sc``, a
plain-text score: the ground-truth onset times in seconds, one per line,
``#`` comments allowed. Every onset sits exactly on the track's 16th grid.

Run ``python make_corpus.py`` to regenerate. The output is deterministic.

Measuring
---------
A harness feeding a track through the analyser should skip the first
~0.2 s of output: the first kick sits at t=0, while the analyser's windows
are still warming up, so its transient is not visible to any detector that
does not give the ring buffer a moment to fill.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 44100
BARS = 5

#: Kick tone, Hz. Low, in the bass window.
KICK_HZ = 60.0
#: The extra percussive hits (drive track), Hz. Mid, clear of both the kicks
#: and the note stream.
HIT_HZ = 1500.0

TRACKS = {
    "sim_115": {
        "bpm": 115,
        "note_hz": 4410.0,
        "note_amp": 0.25,
        "kick_amp": 0.9,
        "kick_beats": (1,),                       # one loud, separated kick per bar
        "extra_beats": (),
    },
    "drive_130": {
        "bpm": 130,
        "note_hz": 9755.0,
        "note_amp": 0.2,
        "kick_amp": 0.85,
        "kick_beats": (1, 2, 3, 4),               # four-on-the-floor
        "extra_beats": (1.5, 2.0, 2.5, 3.5, 4.5),  # 1&, second, 2&, three-and, 4&
    },
}


def _note_stream(hz: float, amp: float, note_len: int, n_total: int,
                 fade_s: float = 0.002) -> np.ndarray:
    """A legato 16th-note stream: one sine per note, phase-continuous, with a
    short fade at each boundary so note changes are not themselves onsets.
    Sized to ``n_total`` exactly; the last note is truncated if the track
    does not end on a note boundary."""
    i = np.arange(n_total)
    pos = i % note_len
    fade = max(1, round(fade_s * SR))
    win = np.sin(np.linspace(0.0, np.pi / 2, fade)) ** 2
    gain = np.ones(n_total)
    inside = pos < fade
    gain[inside] = win[pos[inside]]
    tail = pos >= note_len - fade
    gain[tail] = win[::-1][pos[tail] - (note_len - fade)]
    return amp * gain * np.sin(2 * np.pi * hz * i / SR)


def _struck(hz: float, amp: float, dur_s: float, attack_s: float,
            decay_s: float) -> np.ndarray:
    """A percussive sine: fast attack, exponential decay, no end click."""
    n = round(dur_s * SR)
    t = np.arange(n) / SR
    env = np.exp(-np.maximum(t - attack_s, 0.0) / decay_s)
    a = max(1, round(attack_s * SR))
    env[:a] = np.linspace(0.0, 1.0, a)
    tail = max(1, round(0.005 * SR))          # kill any residual click
    env[-tail:] *= np.linspace(1.0, 0.0, tail)
    return amp * env * np.sin(2 * np.pi * hz * t)


def _build(name: str, spec: dict) -> tuple[np.ndarray, list[float]]:
    bar_s = 4 * 60.0 / spec["bpm"]
    total_s = bar_s * BARS
    n_total = round(total_s * SR)
    out = np.zeros(n_total)

    note_len = round(bar_s / 16 * SR)
    out += _note_stream(spec["note_hz"], spec["note_amp"], note_len, n_total)

    onsets: list[float] = []
    quarter = bar_s / 4.0
    kick = _struck(KICK_HZ, spec["kick_amp"], 0.4, 0.003, 0.08)
    hit = _struck(HIT_HZ, 0.45, 0.12, 0.002, 0.05)
    for b in range(BARS):
        t0 = b * bar_s
        for beat in spec["kick_beats"]:
            pos = t0 + (beat - 1) * quarter
            i0 = round(pos * SR)
            out[i0:i0 + len(kick)] += kick
            onsets.append(pos)
        for beat in spec["extra_beats"]:
            pos = t0 + (beat - 1) * quarter
            i0 = round(pos * SR)
            out[i0:i0 + len(hit)] += hit
            onsets.append(pos)
    # distinct times only — the drive track's "second" sits on the beat-two
    # kick — sorted, and quantised to the 16th grid
    onsets = sorted({round(x, 6) for x in onsets})
    # normalise so the loudest moment (kick plus coincident note or extra)
    # never exceeds 0.95 — the track must not clip in s16le
    peak = float(np.abs(out).max())
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out, onsets


def _write_wav(path: Path, mono: np.ndarray) -> None:
    data = (mono * 32767.0).astype(np.int16)
    stereo = np.stack((data, data), axis=1).ravel()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(stereo.tobytes())


def _write_sc(path: Path, name: str, spec: dict, onsets: list[float]) -> None:
    bar_s = 4 * 60.0 / spec["bpm"]
    note_hz = spec["note_hz"]
    lines = [
        "# ground-truth onset times for %s, seconds, one per line" % name,
        "# 4/4 at %g BPM, 16th-note sines at %g Hz, %d bars, %d onset(s) per bar"
        % (spec["bpm"], note_hz, BARS, len(onsets) // BARS),
        "# every time below sits exactly on the 16th grid (%.6f s)" % (bar_s / 16),
    ]
    lines += ["%.6f" % t for t in onsets]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(here: Path) -> dict[str, dict]:
    for name, spec in TRACKS.items():
        mono, onsets = _build(name, spec)
        _write_wav(here / f"{name}.wav", mono)
        _write_sc(here / f"{name}.sc", name, spec, onsets)
    return TRACKS


def verify(here: Path) -> list[str]:
    """Re-read what was written and check it against the intended grid."""
    problems = []
    for name, spec in TRACKS.items():
        wav = here / f"{name}.wav"
        with wave.open(str(wav), "rb") as w:
            nframes, ch, rate = w.getnframes(), w.getnchannels(), w.getframerate()
        bar_s = 4 * 60.0 / spec["bpm"]
        total_s = bar_s * BARS
        if rate != SR or ch != 2:
            problems.append(f"{name}: wav is {ch}ch {rate}Hz, want 2ch {SR}Hz")
        if nframes != round(total_s * SR):
            problems.append(f"{name}: {nframes} frames, want {round(total_s * SR)}")
        with wave.open(str(wav), "rb") as w:
            raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if np.abs(raw).max() > 32767 * 0.96:
            problems.append(f"{name}: clips in s16le (peak {np.abs(raw).max()})")

        sc = here / f"{name}.sc"
        times = [float(ln) for ln in sc.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        grid = bar_s / 16
        # distinct grid positions per bar — the drive track's "second" sits on
        # the beat-two kick, so kicks and extras union rather than add
        want_per_bar = len(set(spec["kick_beats"]) | set(spec["extra_beats"]))
        if len(times) != BARS * want_per_bar:
            problems.append(
                f"{name}: {len(times)} onsets, want {BARS * want_per_bar} "
                f"({want_per_bar} per bar)"
            )
        if len(times) != len(set(times)):
            problems.append(f"{name}: duplicate onset times")
        for t in times:
            if not 0.0 <= t <= total_s:
                problems.append(f"{name}: onset {t:.6f} outside the track")
            if abs(t - round(t / grid) * grid) > 1e-6:
                problems.append(f"{name}: onset {t:.6f} is off the 16th grid")
        print(f"    {name}: {len(times)} onsets, {nframes} frames, "
              f"{rate} Hz {ch}ch - first {times[0]:.3f}s, last {times[-1]:.3f}s")
    return problems


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    generate(here)
    bad = verify(here)
    if bad:
        print("\n".join("FAIL " + b for b in bad))
        raise SystemExit(1)
    print("corpus regenerated and verified")
