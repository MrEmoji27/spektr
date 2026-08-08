"""Onset-detector scorer and its synthetic evaluation corpus.

The detector contract
---------------------
``detect(samples, sr) -> list[float]``: ``samples`` is an (N, 2) float32
stereo array at ``sr`` Hz; the return value is a list of onset times in
seconds. That is the whole interface — a detector can be graded on it without
the analyser, the widget, or a display.

Scoring
-------
MIREX-style onset matching. A detection matches a ground-truth onset when it
falls within +/- ``tolerance`` (default 50 ms, the MIREX window) of it. Each
truth matches at most one detection and each detection at most one truth:
a detector that fires twice on one drum hit earns one true positive and one
false positive, not two true positives. Per scenario ``evaluate`` reports
``tp``, ``fp``, ``fn``, precision, recall, F-measure and the mean absolute
timing error over the matched pairs, and a pooled ``total`` entry that
aggregates the counts across every scenario.

The one ambiguous case is the empty one. When the truth is empty *and* the
detector fires nothing (a detector that is silent on the silence, noise and
note-stream scenarios), precision and recall are vacuous — 0/0 — and the
scenario scores F = 1.0, because there is nothing to miss and nothing false
fired. Any detection on an empty truth is a false positive and costs the
whole scenario: precision 0, recall vacuous, F 0. That is what makes the
false-positive scenarios discriminative at all.

Corpus
------
Eleven scenarios, all synthesised in numpy at the moment they are needed —
no WAV files, so the corpus costs nothing in git and runs anywhere CI or SSH
can run Python. Each generator returns ``(samples, sr, truth)``: stereo
float32 samples, a sample rate, and the ground-truth onset times in seconds.

``click``           impulses every quarter at 120 BPM, 4 bars (16 onsets).
``four_on_floor``   synth kick on every beat at 128 BPM, 4 bars (16).
``kick_snare``      kick on 1 and 3, snare on 2 and 4 at 100 BPM (16).
``breakbeat``       syncopated kicks and snares plus 16th-note hats at
                    174 BPM (48) — the density test.
``pad_under_kick``  a sustained harmonic pad at constant level under the
                    four-on-the-floor kick (16). The pad is a continuous
                    tone; only the kicks are onsets.
``swing``           kick_snare whose offbeats (the snares) land 60 ms late,
                    and the truth says so (16).
``tempo_ramp``      100 -> 140 BPM over 20 s, kick on every beat (~41).
``quiet``           four_on_floor at -40 dB (16) — a detector whose gate is
                    a fixed level will fire nothing here.
``silence``         digital silence, truth empty.
``noise``           white noise, truth empty.
``note_stream``     a legato 16th-note sine stream at 120 BPM: phase
                    continuous, a 2 ms fade at every boundary, the pitch
                    moving through a pentatonic melody. These are notes
                    that must NOT count as onsets — sustained energy at a
                    constant level, so only a spectral-flux detector can
                    even *see* them, and seeing them is a false positive.

pad_under_kick, silence, noise and note_stream are the ones that measure
false positives: no onset-free energy for a detector to trip on except its
own mistakes. With no scorer and no false-positive scenarios, nothing in the
codebase could tell a detector that fires on silence from one that does not.

Running
-------
``python tests/onset_eval.py`` runs the scorer's self-checks, verifies that
every scenario agrees with its own ground truth (an identity detector must
score F = 1.0 and zero timing error on all eleven), and prints the
evaluation table for a small reference detector. It exits 1 if any check
fails. ``evaluate`` is importable for the real detector:

    import sys; sys.path.insert(0, "tests")
    from onset_eval import evaluate
    from my_detector import detect
    print(evaluate(detect))
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

SR = 44100
TOLERANCE = 0.05


# ── synthesis helpers ────────────────────────────────────────────────────────

def _stereo(mono: np.ndarray) -> np.ndarray:
    """Duplicate a mono track into an (N, 2) float32 stereo array."""
    return np.stack((mono, mono), axis=1).astype(np.float32)


def _norm(mono: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Scale down (never up) so the loudest moment cannot exceed ``peak``."""
    m = float(np.abs(mono).max())
    return mono * (peak / m) if m > peak else mono


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


def _noise_burst(amp: float, dur_s: float, decay_s: float,
                 rng: np.random.Generator) -> np.ndarray:
    """A percussive noise hit: instant attack, exponential decay."""
    n = round(dur_s * SR)
    t = np.arange(n) / SR
    env = np.exp(-t / decay_s)
    tail = max(1, round(0.005 * SR))
    env[-tail:] *= np.linspace(1.0, 0.0, tail)
    return amp * env * rng.standard_normal(n)


def _fade_edges(n: int, edge_s: float) -> np.ndarray:
    """A 0..1 window that is flat in the body and ramps at both ends."""
    edge = round(edge_s * SR)
    if edge <= 0 or 2 * edge >= n:
        return np.ones(n)
    ramp = np.sin(np.linspace(0.0, np.pi / 2, edge)) ** 2
    win = np.ones(n)
    win[:edge] = ramp
    win[-edge:] = ramp[::-1]
    return win


def _quant(times: list[float]) -> list[float]:
    """Distinct onset times, sorted, quantised to microseconds."""
    return sorted({round(float(t), 6) for t in times})


# ── scenarios ────────────────────────────────────────────────────────────────

def scenario_click() -> tuple[np.ndarray, int, list[float]]:
    """Impulses at 120 BPM: a 5 ms burst with an instant attack, one per
    quarter, 4 bars. The cleanest possible ground truth — no sustained
    energy at all between onsets."""
    bpm, bars = 120.0, 4
    beat = 60.0 / bpm
    total = bars * 4 * beat
    n = round(total * SR)
    out = np.zeros(n)
    burst = _struck(2000.0, 0.8, 0.005, 0.001, 0.0015)
    onsets: list[float] = []
    t = 0.0
    while t < total:
        i0 = round(t * SR)
        if i0 + len(burst) <= n:
            out[i0:i0 + len(burst)] += burst
            onsets.append(t)
        t += beat
    return _stereo(_norm(out)), SR, _quant(onsets)


def _four_on_floor(bpm: float = 128.0, bars: int = 4, gain: float = 1.0,
                   kick_hz: float = 55.0) -> tuple[np.ndarray, list[float]]:
    """The shared kick bed: a synth kick on every beat, ``bars`` bars."""
    bar_s = 4 * 60.0 / bpm
    total = bar_s * bars
    n = round(total * SR)
    out = np.zeros(n)
    kick = _struck(kick_hz, gain, 0.4, 0.003, 0.08)
    quarter = bar_s / 4.0
    onsets: list[float] = []
    for b in range(bars):
        for k in range(4):
            pos = b * bar_s + k * quarter
            i0 = round(pos * SR)
            if i0 + len(kick) <= n:
                out[i0:i0 + len(kick)] += kick
                onsets.append(pos)
    return out, onsets


def scenario_four_on_floor() -> tuple[np.ndarray, int, list[float]]:
    """The kick alone: 55 Hz, four on the floor at 128 BPM, 4 bars."""
    mono, onsets = _four_on_floor()
    return _stereo(_norm(mono)), SR, _quant(onsets)


def scenario_kick_snare() -> tuple[np.ndarray, int, list[float]]:
    """Kick on 1 and 3, snare on 2 and 4 at 100 BPM, 4 bars — an alternating
    two-timbre pattern a level-based detector must not confuse."""
    bpm, bars = 100.0, 4
    bar_s = 4 * 60.0 / bpm
    total = bar_s * bars
    n = round(total * SR)
    out = np.zeros(n)
    rng = np.random.default_rng(7)
    kick = _struck(55.0, 0.9, 0.4, 0.003, 0.08)
    snare = _noise_burst(0.7, 0.15, 0.04, rng)
    quarter = bar_s / 4.0
    onsets: list[float] = []
    for b in range(bars):
        for beat in (1, 3):
            pos = b * bar_s + (beat - 1) * quarter
            i0 = round(pos * SR)
            out[i0:i0 + len(kick)] += kick
            onsets.append(pos)
        for beat in (2, 4):
            pos = b * bar_s + (beat - 1) * quarter
            i0 = round(pos * SR)
            out[i0:i0 + len(snare)] += snare
            onsets.append(pos)
    return _stereo(_norm(out)), SR, _quant(onsets)


def scenario_breakbeat() -> tuple[np.ndarray, int, list[float]]:
    """Syncopated drums at 174 BPM: kick on the 1 and the three-and, snare
    on 2 and 4, hats on every odd 16th — 12 hits per bar, 48 in all, with
    hits 86 ms apart. The density test: a detector with a lazy refractory
    window or a coarse hop will merge adjacent hits."""
    bpm, bars = 174.0, 4
    bar_s = 4 * 60.0 / bpm
    sixteenth = bar_s / 16.0
    total = bar_s * bars
    n = round(total * SR)
    out = np.zeros(n)
    rng = np.random.default_rng(11)
    kick = _struck(55.0, 0.9, 0.35, 0.002, 0.06)
    snare = _noise_burst(0.7, 0.12, 0.03, rng)
    hat = _noise_burst(0.35, 0.05, 0.012, rng)
    onsets: list[float] = []
    for b in range(bars):
        for step in (0, 10):                       # the 1 and the three-and
            pos = b * bar_s + step * sixteenth
            i0 = round(pos * SR)
            if i0 + len(kick) <= n:
                out[i0:i0 + len(kick)] += kick
                onsets.append(pos)
        for step in (4, 12):                       # 2 and 4
            pos = b * bar_s + step * sixteenth
            i0 = round(pos * SR)
            if i0 + len(snare) <= n:
                out[i0:i0 + len(snare)] += snare
                onsets.append(pos)
        for step in range(1, 16, 2):               # offbeat 16th hats
            pos = b * bar_s + step * sixteenth
            i0 = round(pos * SR)
            if i0 + len(hat) <= n:
                out[i0:i0 + len(hat)] += hat
                onsets.append(pos)
    return _stereo(_norm(out)), SR, _quant(onsets)


def scenario_pad_under_kick() -> tuple[np.ndarray, int, list[float]]:
    """The four_on_floor kick over a sustained harmonic pad (110, 165, 220
    and 275 Hz) held at constant level for the whole track. Only the ends
    ramp, over half a second, so the pad itself is not an onset. Truth is
    the 16 kicks — a detector that hears the pad at all is hearing the
    spectral change under the kick, and any onset it reports elsewhere on
    the pad is a false positive."""
    bpm, bars = 128.0, 4
    bar_s = 4 * 60.0 / bpm
    total = bar_s * bars
    n = round(total * SR)
    t = np.arange(n) / SR
    pad = np.zeros(n)
    for hz, amp in ((110.0, 0.12), (165.0, 0.10), (220.0, 0.08), (275.0, 0.06)):
        pad += amp * np.sin(2 * np.pi * hz * t)
    out = pad * _fade_edges(n, 0.5)
    kick = _struck(55.0, 0.9, 0.4, 0.003, 0.08)
    quarter = bar_s / 4.0
    onsets: list[float] = []
    for b in range(bars):
        for k in range(4):
            pos = b * bar_s + k * quarter
            i0 = round(pos * SR)
            if i0 + len(kick) <= n:
                out[i0:i0 + len(kick)] += kick
                onsets.append(pos)
    return _stereo(_norm(out)), SR, _quant(onsets)


def scenario_swing() -> tuple[np.ndarray, int, list[float]]:
    """kick_snare at 100 BPM whose offbeats — the snares on 2 and 4 — are
    delayed 60 ms, and the ground truth records the delay. A detector that
    quantises to a grid will flag the snares as missed and fire on the grid
    positions where they are not, so the swing costs it in both directions."""
    bpm, bars = 100.0, 4
    bar_s = 4 * 60.0 / bpm
    total = bar_s * bars
    n = round(total * SR)
    out = np.zeros(n)
    rng = np.random.default_rng(7)
    kick = _struck(55.0, 0.9, 0.4, 0.003, 0.08)
    snare = _noise_burst(0.7, 0.15, 0.04, rng)
    quarter = bar_s / 4.0
    onsets: list[float] = []
    for b in range(bars):
        for beat in (1, 3):
            pos = b * bar_s + (beat - 1) * quarter
            i0 = round(pos * SR)
            if i0 + len(kick) <= n:
                out[i0:i0 + len(kick)] += kick
                onsets.append(pos)
        for beat in (2, 4):
            pos = b * bar_s + (beat - 1) * quarter + 0.06
            i0 = round(pos * SR)
            if i0 + len(snare) <= n:
                out[i0:i0 + len(snare)] += snare
                onsets.append(pos)
    return _stereo(_norm(out)), SR, _quant(onsets)


def scenario_tempo_ramp() -> tuple[np.ndarray, int, list[float]]:
    """Kick on every beat while the tempo rises linearly from 100 to 140 BPM
    over 20 s; beat times come from integrating the instantaneous tempo, and
    the truth is exactly that integration. ~41 onsets at a spacing that
    shrinks from 0.6 to 0.43 s — no steady grid to lock onto."""
    sr, dur = SR, 20.0
    bpm0, bpm1 = 100.0, 140.0
    n = round(dur * sr)
    out = np.zeros(n)
    kick = _struck(55.0, 0.9, 0.4, 0.003, 0.08)
    onsets: list[float] = []
    t = 0.0
    while t < dur:
        i0 = round(t * sr)
        if i0 + len(kick) <= n:
            out[i0:i0 + len(kick)] += kick
            onsets.append(t)
        t += 60.0 / (bpm0 + (bpm1 - bpm0) * (t / dur))
    return _stereo(_norm(out)), sr, _quant(onsets)


def scenario_quiet() -> tuple[np.ndarray, int, list[float]]:
    """four_on_floor at -40 dB. The truth is exactly as loud as the audio
    is quiet: a detector with a fixed level gate fires nothing and scores
    zero — this scenario exists to punish absolute thresholds."""
    mono, onsets = _four_on_floor(gain=0.01)
    return _stereo(mono), SR, _quant(onsets)


def scenario_silence() -> tuple[np.ndarray, int, list[float]]:
    """Digital silence. Any detection here is a false positive, which is the
    whole point: there is no energy, so only a detector's own noise can fire."""
    return np.zeros((round(4.0 * SR), 2), dtype=np.float32), SR, []


def scenario_noise() -> tuple[np.ndarray, int, list[float]]:
    """White noise, one independent channel per side, normalised to 0.8.
    Loud, featureless, and entirely onset-free: a detector with a trigger
    that twitches at random spectral motion will score here exactly what it
    deserves."""
    n = round(5.0 * SR)
    rng = np.random.default_rng(23)
    x = rng.standard_normal((n, 2)).astype(np.float32)
    x *= 0.8 / max(1e-9, float(np.abs(x).max()))
    return x, SR, []


def scenario_note_stream() -> tuple[np.ndarray, int, list[float]]:
    """A legato 16th-note sine stream at 120 BPM: 64 notes over 4 bars, the
    pitch cycling through a pentatonic melody, phase-continuous at every
    splice with a 2 ms fade at each boundary. Sustained energy at constant
    level — the note changes are audible but are not onsets, and the truth
    says so: empty. A spectral-flux detector sees the pitch move and fires
    at every boundary; that is the exact false positive this scenario
    exists to count."""
    bpm, bars = 120.0, 4
    step = 60.0 / bpm / 4.0
    total = bars * 4 * step
    n = round(total * SR)
    out = np.zeros(n)
    melody = (440.0, 523.25, 587.33, 659.25, 783.99)
    note_len = round(step * SR)
    fade = max(1, round(0.002 * SR))
    ramp = np.sin(np.linspace(0.0, np.pi / 2, fade)) ** 2
    phase = 0.0
    i, k = 0, 0
    while i < n:
        ln = min(note_len, n - i)
        hz = melody[k % len(melody)]
        seg = np.sin(2 * np.pi * hz * np.arange(ln) / SR + phase)
        phase = (phase + 2 * np.pi * hz * ln / SR) % (2 * np.pi)
        win = np.ones(ln)
        if fade < ln:
            win[:fade] = ramp
            win[-fade:] = ramp[::-1]
        out[i:i + ln] += 0.35 * seg * win
        i += ln
        k += 1
    return _stereo(out), SR, []


SCENARIOS: dict[str, Callable[[], tuple[np.ndarray, int, list[float]]]] = {
    "click": scenario_click,
    "four_on_floor": scenario_four_on_floor,
    "kick_snare": scenario_kick_snare,
    "breakbeat": scenario_breakbeat,
    "pad_under_kick": scenario_pad_under_kick,
    "swing": scenario_swing,
    "tempo_ramp": scenario_tempo_ramp,
    "quiet": scenario_quiet,
    "silence": scenario_silence,
    "noise": scenario_noise,
    "note_stream": scenario_note_stream,
}


# ── the scorer ───────────────────────────────────────────────────────────────

def _match(truth: np.ndarray, det: np.ndarray,
           tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one MIREX matching, greedily.

    ``truth`` and ``det`` are sorted arrays of times. Walking the detections
    in ascending order and giving each the leftmost truth it can reach keeps
    the maximum number of matches — the classic interval-scheduling greedy,
    where every truth a detection could reach is an interval and claiming the
    leftmost one leaves the most room for the detections still to come.
    Returns the matched truth times and the matched detection times.
    """
    mt: list[float] = []
    md: list[float] = []
    i = 0                                        # first unclaimed truth
    for d in det:
        j = max(i, int(np.searchsorted(truth, d - tolerance, side="left")))
        if j < len(truth) and truth[j] <= d + tolerance:
            mt.append(float(truth[j]))
            md.append(float(d))
            i = j + 1
    return np.asarray(mt), np.asarray(md)


def _metrics(truth, det, tolerance: float) -> dict:
    """One scenario's numbers: tp/fp/fn counts and derived scores."""
    truth_a = np.sort(np.asarray(truth, dtype=np.float64))
    det_a = np.sort(np.asarray(det, dtype=np.float64))
    mt, md = _match(truth_a, det_a, tolerance)
    tp = len(mt)
    fp = len(det_a) - tp
    fn = len(truth_a) - tp
    if tp == 0 and fp == 0 and fn == 0:
        # nothing fired and nothing to fire on: the only perfect empty case
        precision = recall = f = 1.0
    else:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mae = float(np.abs(md - mt).mean()) if len(mt) else None
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f": f, "mae": mae,
    }


def evaluate(detect, scenarios: Sequence[str] | None = None,
             tolerance: float = TOLERANCE) -> dict[str, dict]:
    """Grade ``detect(samples, sr) -> list[float]`` against the corpus.

    ``scenarios`` is a sequence of scenario names (all of them by default).
    The result maps each name to its metrics — ``tp``, ``fp``, ``fn``,
    ``precision``, ``recall``, ``f``, and ``mae`` (the mean absolute timing
    error over the matched pairs; ``None`` when nothing matched) — plus a
    ``total`` entry pooling the counts across every scenario.
    """
    if scenarios is None:
        scenarios = tuple(SCENARIOS)
    unknown = [s for s in scenarios if s not in SCENARIOS]
    if unknown:
        raise ValueError(f"unknown scenario(s): {unknown}")

    out: dict[str, dict] = {}
    tp = fp = fn = 0
    mae_sum = mae_n = 0.0
    for name in scenarios:
        samples, sr, truth = SCENARIOS[name]()
        det = detect(samples, sr)
        m = _metrics(truth, det, tolerance)
        out[name] = m
        tp += m["tp"]
        fp += m["fp"]
        fn += m["fn"]
        if m["mae"] is not None:
            mae_sum += m["mae"] * m["tp"]
            mae_n += m["tp"]

    if tp == 0 and fp == 0 and fn == 0:
        precision = recall = f = 1.0
    else:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out["total"] = {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f": f,
        "mae": mae_sum / mae_n if mae_n else None,
    }
    return out


# ── the standalone reference detector ────────────────────────────────────────

def _reference_detect(samples: np.ndarray, sr: int) -> list[float]:
    """A deliberately small mean-spectral-flux detector.

    The placeholder the standalone table runs — not part of the scoring
    contract, and naive on purpose: a fixed 1024/512 frame, the mean
    difference of successive FFT magnitudes, an adaptive threshold, and a
    40 ms refractory window. It fires on kicks and snares, stays silent on
    silence, trips on every note_stream boundary (the pitch moves, so the
    spectrum moves, even though nothing starts), and fires spuriously on
    noise — exactly the false-positive behaviour the corpus measures.
    """
    mono = samples.mean(axis=1).astype(np.float64)
    win, hop = 1024, 512
    starts = np.arange(0, len(mono) - win + 1, hop)
    if starts.size < 3:
        return []
    idx = starts[:, None] + np.arange(win)
    mag = np.abs(np.fft.rfft(mono[idx] * np.hanning(win), axis=1))
    flux = np.abs(np.diff(mag, axis=0)).mean(axis=1)
    window = max(1, round(0.5 * sr / hop))
    out: list[float] = []
    last = -1.0
    for i in range(len(flux)):
        f = float(flux[i])
        run = float(np.mean(flux[max(0, i - window):i])) if i else 0.0
        t = (i + 1) * hop / sr
        if f > 3.0 * max(run, 1e-12) and t - last >= 0.04:
            out.append(t)
            last = t
    return out


# ── self-checks: the scorer proves itself before anyone trusts it ────────────

def _selftests() -> list[tuple[str, list[str]]]:
    """The behaviour the scorer must be shown to have, in fixed synthetic
    cases where the expected numbers are known by hand."""
    results: list[tuple[str, list[str]]] = []

    def check(name: str, problems: list[str]) -> None:
        results.append((name, problems))

    # 1. a perfect detector — returning the truth verbatim — scores F=1.0
    truth = SCENARIOS["four_on_floor"]()[2]
    m = evaluate(lambda s, sr: list(truth), scenarios=["four_on_floor"])["four_on_floor"]
    check(
        "perfect detector (truth verbatim) scores F=1.0",
        [] if m["f"] == 1.0 and m["tp"] == len(truth) and m["mae"] == 0.0
        else [f"f={m['f']}, tp={m['tp']}/{len(truth)}, mae={m['mae']}"],
    )

    # 2. an empty detector scores F=0.0 on a track that has onsets
    m = evaluate(lambda s, sr: [], scenarios=["four_on_floor"])["four_on_floor"]
    check(
        "empty detector scores F=0.0",
        [] if m["f"] == 0.0 and m["fn"] == len(truth)
        else [f"f={m['f']}, fn={m['fn']}/{len(truth)}"],
    )

    # 3. 40 ms off is still a match at 50 ms tolerance
    m = _metrics([0.5], [0.54], TOLERANCE)
    check(
        "40 ms offset still matches at 50 ms tolerance",
        [] if m["tp"] == 1 and abs(m["mae"] - 0.04) < 1e-6
        else [f"tp={m['tp']}, mae={m['mae']}"],
    )

    # 4. 80 ms off is not a match
    m = _metrics([0.5], [0.58], TOLERANCE)
    check(
        "80 ms offset does not match",
        [] if m["tp"] == 0 and m["f"] == 0.0 else [f"tp={m['tp']}, f={m['f']}"],
    )

    # 5. 9 detections per truth (8 decoys each): recall holds at 1.0 while
    # precision collapses — the decoys stay more than 50 ms from any truth
    # other than their own, so they can only pile up as false positives
    click = SCENARIOS["click"]()[2]
    decoys = (0.01, -0.02, 0.03, -0.04, 0.25, -0.25, 0.125, -0.125)
    det = sorted(t + d for t in click for d in (0.0,) + decoys)
    m = _metrics(click, det, TOLERANCE)
    check(
        "10x detections: high recall, low precision",
        [] if m["recall"] == 1.0 and m["precision"] < 0.15
        else [f"recall={m['recall']}, precision={m['precision']}"],
    )

    # 6. two detections 10 ms apart on one truth: one TP and one FP, never
    # two TPs — the one-to-one rule
    m = _metrics([1.0], [1.0, 1.01], TOLERANCE)
    check(
        "two detections on one truth: one TP, one FP",
        [] if m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 0
        else [f"tp={m['tp']}, fp={m['fp']}, fn={m['fn']}"],
    )

    # 7. the pooled total must not fabricate matches the scenarios lack
    m = evaluate(lambda s, sr: [], scenarios=["silence", "noise"])["total"]
    check(
        "empty detections on empty truths are not pooled false positives",
        [] if m["tp"] == 0 and m["fp"] == 0 and m["f"] == 1.0
        else [f"tp={m['tp']}, fp={m['fp']}, f={m['f']}"],
    )

    return results


def _corpus_check() -> list[str]:
    """Every scenario must be internally consistent: well-formed audio and a
    ground truth that an identity detector reproduces exactly (F = 1.0,
    zero timing error). A scenario its own generator disagrees with would
    poison every detector graded on it, so this runs every time."""
    problems: list[str] = []
    for name, fn in SCENARIOS.items():
        samples, sr, truth = fn()
        if samples.ndim != 2 or samples.shape[1] != 2:
            problems.append(f"{name}: samples shape {samples.shape}, want (N, 2)")
        if samples.dtype != np.float32:
            problems.append(f"{name}: dtype {samples.dtype}, want float32")
        if not np.isfinite(samples).all():
            problems.append(f"{name}: non-finite samples")
        if sr != SR:
            problems.append(f"{name}: sr {sr}, want {SR}")
        dur = len(samples) / sr
        if not all(0.0 <= t <= dur for t in truth):
            problems.append(f"{name}: truth time outside the track")
        if len(truth) != len(set(truth)):
            problems.append(f"{name}: duplicate truth times")
        m = evaluate(lambda s, sr, want=truth: list(want), scenarios=[name])[name]
        want_mae = "0.0" if truth else "None"
        got_mae = m["mae"] if m["mae"] is None else round(m["mae"], 6)
        if m["f"] != 1.0 or (truth and m["mae"] != 0.0):
            problems.append(
                f"{name}: identity scores f={m['f']}, mae={got_mae}, "
                f"want 1.0 and {want_mae} — the ground truth disagrees "
                "with the audio"
            )
    return problems


# ── the standalone table ─────────────────────────────────────────────────────

def _print_table() -> None:
    rows = evaluate(_reference_detect)
    print(f"{'scenario':<18} {'truth':>5} {'det':>5} {'tp':>5} {'fp':>5} "
          f"{'fn':>5} {'precision':>9} {'recall':>9} {'f':>9} {'mae':>7}")

    def row(label: str, m: dict) -> str:
        mae = f"{m['mae']:7.3f}" if m["mae"] is not None else "      -"
        return (f"{label:<18} {m['tp'] + m['fn']:>5} {m['tp'] + m['fp']:>5} "
                f"{m['tp']:>5} {m['fp']:>5} {m['fn']:>5} "
                f"{m['precision']:>9.3f} {m['recall']:>9.3f} {m['f']:>9.3f} "
                f"{mae:>7}")

    for name in SCENARIOS:
        print(row(name, rows[name]))
    print("-" * 79)
    print(row("total", rows["total"]))
    print("\nmae is the mean absolute timing error over matched pairs only;")
    print("'-' means nothing matched, so there are no pairs to measure.")


def _main() -> int:
    failures = 0
    print("onset-eval self-checks")
    for name, problems in _selftests():
        mark = "ok  " if not problems else "FAIL"
        print(f"  [{mark}] {name}")
        for p in problems:
            print(f"         {p}")
        failures += len(problems)

    print("\ncorpus self-consistency (identity detector must score F=1.0)")
    problems = _corpus_check()
    if problems:
        for p in problems:
            print("  [FAIL] " + p)
            failures += 1
    else:
        print(f"  [ok  ] all {len(SCENARIOS)} scenarios agree with their "
              "ground truth")

    print("\nevaluation table (reference detector: mean spectral flux, "
          "1024/512, adaptive threshold)")
    _print_table()

    print("\n" + ("all good" if not failures else f"{failures} problems"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
