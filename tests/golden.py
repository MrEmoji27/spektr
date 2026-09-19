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
)


def _bands(signal: str, t: float) -> np.ndarray:
    n = N_BANDS
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
        rhythm = dict(
            onset_seq=onset_seq,
            onsets=onsets,
            onset_strength=0.8,
            flux=float(b[0]),
            tempo_bpm=120.0,
            beat_phase=(t % 0.5) / 0.5,
        )
    ctx = Ctx(
        w=case.w, h=case.h, bands=b, peaks=b, bands_l=b, bands_r=b,
        wave=wave, stereo=np.stack((wave, wave), axis=1),
        frame=i, t=t, dt=DT, energy=level, silent=False,
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


#: Modes whose glyphs are an ordered threshold over a gradient, where a cell
#: sits exactly on the threshold and the last bit of a float32 decides which
#: subcells light. That bit is not the same on every CPU, so the pattern is
#: reproducible on one machine but not across machines: measured here, a
#: relative change of 1e-7 in the input moves 9% of the glyph cells while
#: moving no colour at all.
#:
#: They are still pinned whole — glyphs included — on the machine that
#: recorded the file, which is where a release is checked. Anywhere else,
#: only their colours are compared: coarser, because in Ultra the colour
#: comes from the flat field and the glyphs from the interpolated one, so a
#: change to the antialiasing alone would not show. A real change does show:
#: 1% more level moves 17% of the colours.
PLATFORM_SENSITIVE = {"Kaleidoscope Ultra (o)"}

#: Key under which the recording platform is stored in the golden file.
PLATFORM_KEY = "_platform"


def builtin_modes() -> list:
    return [m for m in M.MODES if m.plugin is None]


def key(mode_name: str, case: Case, colours_only: bool = False) -> str:
    return f"{mode_name}|{case.id}" + ("|colours" if colours_only else "")


def fingerprint(mode, case: Case, colours_only: bool = False) -> str:
    """SHA-256 over what ``mode`` draws on the sampled frames of ``case``."""
    render.set_cell_mode(case.cells)
    try:
        palette = Palette(BUILTIN[case.theme])
        h = hashlib.sha256()
        state: dict = {}
        onset_seq = 0
        for i in range(FRAMES):
            ctx, onset_seq = _ctx(case, i, palette, state, onset_seq)
            try:
                out = mode.fn(ctx)
            except Exception as exc:  # a crash is an output too
                return f"error: {type(exc).__name__}"
            if i in SAMPLED:
                if colours_only:
                    h.update(str(np.asarray(out[0]).shape).encode())
                    for arr in out[1:]:
                        _feed(h, arr)
                else:
                    for arr in out:
                        _feed(h, arr)
                    _feed_strips(h, out, palette)
        return h.hexdigest()
    finally:
        render.set_cell_mode("octant")


def run_all() -> dict[str, str]:
    out = {PLATFORM_KEY: sys.platform}
    for m in builtin_modes():
        for case in CASES:
            out[key(m.name, case)] = fingerprint(m, case)
            if m.name in PLATFORM_SENSITIVE:
                out[key(m.name, case, True)] = fingerprint(m, case, True)
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
    unstable = sorted(k for k in first if first[k] != second[k])
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
