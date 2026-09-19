"""What a frame costs, end to end, with the audio held fixed.

The frame budget in ``bench.py`` times a mode's own numpy work. This times the
whole app: the mode, the strip builder, and everything Textual does with the
result. It exists because a profile of a running app put about 80% of the CPU
inside Textual rather than in spektr's own code, and any change to how frames
reach the terminal has to be judged on this number, not on the mode's.

Audio is replayed rather than captured, so two runs see exactly the same
signal. With no file it replays a synthetic loop (a four-to-the-floor kick
under a mid wall), which is what the mode reactivity checks already use:

    python tests/render_cost.py                       # every default case
    python tests/render_cost.py --mode Terra --size 400x100
    python tests/render_cost.py --wav track.wav --seconds 20

It reports, per case: process CPU as a share of one core, frame time at p50,
p95 and p99, and how many frames the app missed against the rate it asked for.
The terminal's own CPU is not visible from here — a headless run has no
terminal — so a comparison between two render paths has to add the terminal
process's CPU separately before drawing a conclusion.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spektr import config  # noqa: E402
from spektr.analysis import N_BANDS  # noqa: E402

CASES = (("Bars", 200, 50), ("Bars", 400, 100), ("Terra", 200, 50), ("Terra", 400, 100))
SAMPLERATE = 48000


def synth_blocks(seconds: float, block: int = 256):
    """A steady kick under a mid wall, as stereo float32 blocks."""
    total = int(seconds * SAMPLERATE)
    t = np.arange(total, dtype=np.float32) / SAMPLERATE
    kick = np.sin(2 * np.pi * 55.0 * t) * np.exp(-8.0 * (t % 0.5)).astype(np.float32)
    wall = 0.25 * np.sin(2 * np.pi * 440.0 * t) + 0.15 * np.sin(2 * np.pi * 1320.0 * t)
    mono = (0.6 * kick + wall).astype(np.float32) * 0.5
    stereo = np.stack((mono, mono), axis=1)
    for start in range(0, total - block, block):
        yield stereo[start:start + block]


def wav_blocks(path: Path, seconds: float, block: int = 256):
    """The first ``seconds`` of a WAV file, as stereo float32 blocks."""
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise SystemExit(f"{path}: only 16-bit WAV files are supported")
        channels = wav.getnchannels()
        frames = wav.readframes(int(seconds * wav.getframerate()))
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    data = data.reshape(-1, channels)
    stereo = data if channels == 2 else np.repeat(data[:, :1], 2, axis=1)
    for start in range(0, len(stereo) - block, block):
        yield stereo[start:start + block]


async def _feed(ring, blocks, block: int = 256) -> None:
    """Push blocks into the ring at the rate the audio really plays."""
    period = block / SAMPLERATE
    next_at = time.perf_counter()
    for chunk in blocks:
        ring.push(chunk)
        next_at += period
        delay = next_at - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)


async def measure(mode: str, w: int, h: int, fps: int, seconds: float, blocks) -> dict:
    from spektr.app import Spektr

    settings = config.Settings(mode=mode, fps=fps, bands=N_BANDS)
    app = Spektr(settings=settings)
    app.notify = lambda *a, **k: None  # type: ignore[method-assign]
    frames: list[float] = []

    async with app.run_test(size=(w, h)) as pilot:
        viz = app.viz
        viz.set_mode(mode)
        # Live capture would mix the machine's real audio into the replay, so
        # the device is closed and the ring is fed by hand instead. The
        # analyser reads the ring either way.
        viz.capture.stop()
        await pilot.pause()
        feeder = asyncio.create_task(_feed(viz.capture.ring, blocks))

        # Count real frames by wrapping the build, rather than watching
        # _build_ms: two frames that cost the same would look like one.
        build = viz._build

        def timed_build(*a, **k):
            t0 = time.perf_counter()
            out = build(*a, **k)
            frames.append(1000.0 * (time.perf_counter() - t0))
            return out

        viz._build = timed_build  # type: ignore[method-assign]
        cpu0, wall0 = time.process_time(), time.perf_counter()
        # Sleep between samples: a tight await loop would spend this process's
        # own CPU and land in the number being measured.
        while time.perf_counter() - wall0 < seconds:
            await asyncio.sleep(0.05)
        cpu = time.process_time() - cpu0
        wall = time.perf_counter() - wall0
        viz._build = build  # type: ignore[method-assign]
        feeder.cancel()

    drawn = len(frames)
    asked = int(fps * wall)
    return {
        "mode": mode, "size": f"{w}x{h}", "fps": fps,
        "cpu": 100.0 * cpu / wall,
        "p50": statistics.median(frames) if frames else float("nan"),
        "p95": statistics.quantiles(frames, n=20)[18] if len(frames) > 20 else float("nan"),
        "p99": max(frames) if frames else float("nan"),
        "missed": max(0, asked - drawn),
        "asked": asked,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", help="one mode instead of the default cases")
    ap.add_argument("--size", help="WIDTHxHEIGHT, with --mode")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--wav", type=Path, help="16-bit WAV to replay instead of the synthetic loop")
    ap.add_argument("--repeats", type=int, default=1,
                    help="run each case this many times and report the median; "
                         "anything measured on a busy machine needs 3 or more")
    args = ap.parse_args()

    if args.mode:
        w, h = (int(x) for x in (args.size or "200x50").split("x"))
        cases = ((args.mode, w, h),)
    else:
        cases = CASES

    print(f"{'mode':<10} {'size':>8} {'fps':>4} {'cpu%core':>9} "
          f"{'p50 ms':>7} {'p95 ms':>7} {'p99 ms':>7} {'missed':>7}")
    for mode, w, h in cases:
        runs = []
        for _ in range(max(1, args.repeats)):
            blocks = (wav_blocks(args.wav, args.seconds + 2) if args.wav
                      else synth_blocks(args.seconds + 2))
            runs.append(asyncio.run(measure(mode, w, h, args.fps, args.seconds, blocks)))
        med = {k: statistics.median([r[k] for r in runs])
               for k in ("cpu", "p50", "p95", "p99", "missed", "asked")}
        spread = max(r["cpu"] for r in runs) - min(r["cpu"] for r in runs)
        note = f"  (spread {spread:.0f}pp over {len(runs)} runs)" if len(runs) > 1 else ""
        print(f"{mode:<10} {w}x{h:<5} {args.fps:>4} {med['cpu']:>8.1f}% "
              f"{med['p50']:>7.2f} {med['p95']:>7.2f} {med['p99']:>7.2f} "
              f"{med['missed']:>4.0f}/{med['asked']:.0f}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
