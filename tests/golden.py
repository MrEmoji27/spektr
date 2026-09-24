"""Golden output for every built-in mode.

Every built-in mode is run over the same seeded input — three signals, three
sizes, a light theme and the quadrant cell geometry — and what it draws is
fingerprinted: the arrays the mode returns on a few sampled frames, and the
strips ``make_strips`` builds from them. The fingerprints are recorded in
``tests/golden/modes.json`` and ``test_golden_modes.py`` fails on any change.

This is the check that lets 0.5.5 move every file without changing a picture.
A change that is meant to alter what a mode draws regenerates the file with

    python tests/golden.py --update

and says so in its commit message. The update runs everything twice and
refuses to write if the two runs disagree, so a mode that is not
deterministic is caught here rather than as a flaky test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spektr.modes as M  # noqa: E402
from spektr import render  # noqa: E402
from spektr.analysis import N_BANDS, WAVE_POINTS  # noqa: E402
from spektr.modes import Ctx  # noqa: E402
from spektr.palette import BUILTIN, Palette  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden" / "modes.json"

#: Frames run per case, and the frames whose output is fingerprinted. The
#: first shows the cold start, the others show state carried between frames.
FRAMES = 16
SAMPLED = (0, 7, 15)
DT = 1.0 / 60.0


@dataclass(frozen=True)
class Case:
    signal: str
    w: int
    h: int
    theme: str = "gruvbox"
    cells: str = "octant"

    @property
    def id(self) -> str:
        return f"{self.signal}-{self.w}x{self.h}-{self.theme}-{self.cells}"


CASES: tuple[Case, ...] = tuple(
    Case(signal, w, h)
    for (w, h) in ((80, 24), (200, 50), (400, 100))
    for signal in ("quiet", "loud", "beat")
) + (
    Case("beat", 80, 24, theme="flexoki-light"),
    Case("beat", 80, 24, cells="quadrant"),
    # Digital silence, with the noise gate shut — a state the other signals
    # never reach, and one modes get wrong in their own way. Dither Storm
    # drew five times its documented rest density here and no test saw it.
    Case("silent", 80, 24),
    Case("silent", 200, 50),
)


def _bands(signal: str, t: float) -> np.ndarray:
    n = N_BANDS
    if signal == "silent":
        return np.zeros(n)
    if signal == "quiet":
        return np.full(n, 0.04)
    if signal == "loud":
        b = np.full(n, 0.15)
        c = int((np.sin(t * 1.1) * 0.5 + 0.5) * (n - 1))
        b[max(0, c - 4):c + 5] = 0.95
        return b
    if signal == "beat":
        b = np.full(n, 0.35)
        kick = n // 8
        b[:kick] = 0.25 + 0.7 * max(0.0, 1.0 - (t % 0.5) * 8.0)
        lo, hi = n // 4, (n * 13) // 16
        b[lo:hi] = 0.55 + 0.2 * np.sin(np.arange(hi - lo) * 1.7 + t * 3)
        return b
    raise ValueError(f"unknown signal {signal!r}")


def _ctx(case: Case, i: int, palette: Palette, state: dict, onset_seq: int) -> tuple[Ctx, int]:
    t = (i + 1) * DT
    b = _bands(case.signal, t)
    level = float(b.mean())
    wave = np.sin(np.linspace(0.0, 40.0, WAVE_POINTS) + t * 10.0) * level
    rhythm = {}
    if case.signal == "beat":
        # a hit every half second (120 bpm), on the frame the kick restarts
        onsets = 1 if (t % 0.5) < DT else 0
        onset_seq += onsets
        beat = int(t / 0.5) % 4
        chroma = np.asarray(b[:24], dtype=np.float32).reshape(12, 2).mean(axis=1)
        rhythm = dict(
            onset_seq=onset_seq,
            onsets=onsets,
            onset_strength=0.8,
            flux=float(b[0]),
            tempo_bpm=120.0,
            beat_phase=(t % 0.5) / 0.5,
            # the 0.6.0 analysis: a backbeat in a known bar, in a known key
            drums=({"kick": 0.9, "snare": 0.0, "hat": 0.2} if beat % 2 == 0
                   else {"kick": 0.0, "snare": 0.85, "hat": 0.3}),
            bar_phase=((t % 2.0) / 2.0),
            beat_in_bar=beat,
            bar_confidence=0.8,
            chroma=chroma / max(float(chroma.max()), 1e-6),
            key="A minor",
            key_confidence=0.8,
        )
    ctx = Ctx(
        w=case.w, h=case.h, bands=b, peaks=b, bands_l=b, bands_r=b,
        wave=wave, stereo=np.stack((wave, wave), axis=1),
        frame=i, t=t, dt=DT, energy=level, silent=case.signal == "silent",
        palette=palette, state=state, **rhythm,
    )
    return ctx, onset_seq


def _feed(h: "hashlib._Hash", arr: np.ndarray) -> None:
    a = np.ascontiguousarray(arr)
    a = a.astype(np.int64) if a.dtype.kind in "iub" else a.astype(np.float64)
    h.update(str(a.shape).encode())
    h.update(a.tobytes())


def _feed_strips(h: "hashlib._Hash", out: tuple, palette: Palette) -> None:
    codes, cidx = out[0], out[1]
    bidx = out[2] if len(out) == 3 else None
    for strip in render.make_strips(codes, cidx, palette, bidx, None):
        h.update(b"|")
        for seg in strip:
            h.update(seg.text.encode("utf-8"))
            h.update(b"\x00")
            h.update(str(seg.style).encode("utf-8"))


#: Key under which the recording platform is stored in the golden file.
PLATFORM_KEY = "_platform"

#: How much of a picture may differ before a foreign platform calls it a
#: change, as a share of its cells.
#:
#: Exact hashes cannot hold across machines. A mode that decides a cell by
#: comparing a float against a threshold lands on the other side of it when
#: the maths library rounds the last bit differently, and Windows and Linux
#: do. Measured on the CI runner against a file recorded here: seven modes
#: differed, all of them trig-heavy (Radial, Crosscurrent, the Kaleidoscope
#: family, JP Pulse), while the same modes are stable to a 1e-7 nudge of
#: their input on one machine.
#:
#: So the machine that recorded the file still checks every cell, which is
#: where a release is cut. Everywhere else the picture is compared by its
#: shape: how much is lit, where the colour sits, and how the ramp is spread.
#: A real change moves those far past this bar — a 1% louder signal moves 17%
#: of the colours — while a rounding difference moves a fraction of a percent.
TOLERANCE = 0.02


def builtin_modes() -> list:
    return [m for m in M.MODES if m.plugin is None]


def key(mode_name: str, case: Case) -> str:
    return f"{mode_name}|{case.id}"


def _summary(arrays: tuple) -> list:
    """A picture's shape, in numbers a rounding difference cannot move.

    Lit cells, then for each layer its mean and a sixteen-bucket histogram of
    the values in it. Enough to catch a mode drawing something else; blind to
    a handful of cells landing either side of a threshold.
    """
    out: list[float] = []
    for arr in arrays:
        a = np.asarray(arr).astype(np.float64).ravel()
        out.append(float(np.count_nonzero(a)) / a.size)
        # Shares only, never raw values: a mean over codepoints moves by half
        # a unit when twenty cells land either side of a threshold, which is
        # exactly the difference this comparison exists to ignore.
        lo, hi = float(a.min()), float(a.max())
        hist = np.histogram(a, bins=16, range=(lo, hi if hi > lo else lo + 1.0))[0]
        out.extend((hist / a.size).tolist())
    return [round(v, 6) for v in out]


def differs(recorded: list, measured: list) -> float:
    """How far two summaries are apart, on their worst number."""
    if len(recorded) != len(measured):
        return 1.0
    return max(abs(a - b) for a, b in zip(recorded, measured))


def measure(mode, case: Case) -> tuple[str, list]:
    """What ``mode`` draws on the sampled frames of ``case``.

    Returns the exact fingerprint and the tolerant summary beside it.
    """
    render.set_cell_mode(case.cells)
    try:
        palette = Palette(BUILTIN[case.theme])
        h = hashlib.sha256()
        shape: list = []
        state: dict = {}
        onset_seq = 0
        for i in range(FRAMES):
            ctx, onset_seq = _ctx(case, i, palette, state, onset_seq)
            try:
                out = mode.fn(ctx)
            except Exception as exc:  # a crash is an output too
                return f"error: {type(exc).__name__}", []
            if i in SAMPLED:
                for arr in out:
                    _feed(h, arr)
                _feed_strips(h, out, palette)
                shape.extend(_summary(out))
        return h.hexdigest(), shape
    finally:
        render.set_cell_mode("octant")


def run_all() -> dict:
    out: dict = {PLATFORM_KEY: sys.platform}
    for m in builtin_modes():
        for case in CASES:
            digest, shape = measure(m, case)
            out[key(m.name, case)] = [digest, shape]
    return out


def load_golden() -> dict[str, str]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update", action="store_true",
                        help="rewrite tests/golden/modes.json from the current code")
    args = parser.parse_args()
    if not args.update:
        parser.print_help()
        return 0
    first, second = run_all(), run_all()
    unstable = sorted(k for k in first if k != PLATFORM_KEY
                      and first[k][0] != second[k][0])
    if unstable:
        print("not deterministic — fix these before recording:")
        for k in unstable:
            print("  ", k)
        return 1
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(first, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(first)} fingerprints for {len(builtin_modes())} modes to {GOLDEN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
