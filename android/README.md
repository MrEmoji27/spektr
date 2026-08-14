# android/ — prep only, not yet a building app

**Status: preparation.** Started 2026-08-15 and deliberately stopped before the
Gradle app is complete, to be picked up in a dedicated session. Nothing here has
been compiled or run. Do not read a file in this directory as working code.

What exists: `app/src/main/python/spektr_android.py`, the Python half of the
bridge. It is the piece worth having first because it is the only new *Python*
in the port and it pins the contract the Kotlin side has to meet.

## What this machine can and cannot do

Checked 2026-08-15, so the next session does not rediscover it:

| | |
|---|---|
| Android SDK | `C:\Users\mremo\AppData\Local\Android\Sdk`, `ANDROID_HOME` set |
| Platforms | android-30, 31, 33, 34, 35, 36 |
| Build tools | 34.0.0, 35.0.0, 36.0.0, 36.1.0 |
| NDK | 27.0.12077973, 27.1.12297006 (not needed — v1 is Kotlin + numpy) |
| JDK | 17 (Adoptium), `JAVA_HOME` set |
| `gradle` on PATH | **no** — use the wrapper |
| Device on adb | **none attached** |

So the toolchain can compile and assemble. It **cannot** verify the capture
path, because that needs the tablet on adb — and capture is the one part of the
design with real risk. Plan the session around that: everything else can be
proven on the desktop, `AudioPlaybackCapture` cannot.

**Still unanswered:** the tablet's Android API level, which pins the
foreground-service rules for `CaptureService`. The bridge assumes nothing about
it. `minSdk` must be at least 29 regardless, since that is where
`AudioPlaybackCapture` begins.

## Two things the design document does not account for

Both found while writing the bridge against the actual engine, and both change
the Kotlin contract, so they are worth knowing before any Kotlin is written.

**1. `frame()` returns two arrays in the design and three in the code.**

`docs/android-port.md` describes modes returning `(codes, cidx)`. That was true
when it was written. Modes drawing through the half-block `▀` trick return
`(codes, cidx, bidx)` — a *background* ramp index per cell as well — and
`tests/bench.py` has handled the three-tuple for some time. Several of the
best-looking modes use it, and as of 2026-08-15 Kaleidoscope does too.

Packing only two planes would silently drop the background of every half-block
mode: they would not crash, they would render half-wrong, which is the worst
kind of failure to inherit. The wire format therefore carries a **plane count**
and Kotlin must honour it rather than assuming two.

**2. `capture.py` is "replaced", but `RingBuffer` lives inside it.**

The layer table marks `capture.py` as replaced wholesale by Kotlin
`AudioRecord`. But `Analyser` takes a `RingBuffer` in its constructor, and
`RingBuffer` is defined in `capture.py` — a pure-numpy circular buffer with no
audio device anywhere in it.

So the split is not file-shaped. Either that one class moves somewhere shared,
or the port keeps `capture.py` importable for it alone. Decide this before
writing the Kotlin, because it determines what "ships unchanged" actually means
for the audio path.

## The wire format

One crossing per frame — the design names this boundary as the port's main
measured risk, so nothing here returns Python lists across it.

```
header  (little-endian)
  magic   4s   "SPKT"
  version H    1
  planes  H    2 or 3
  w       H
  h       H
then, each C-contiguous with no padding:
  codes   int32[h*w]   Unicode codepoints (braille U+2800+, block elements lower)
  cidx    uint8[h*w]   foreground ramp index, < 64
  bidx    uint8[h*w]   background ramp index — only when planes == 3
```

`codes` is int32 because codepoints run past what a narrower type holds. The two
index planes are bytes because `tests/bench.py` asserts they stay under 64.

Kotlin should refuse an unrecognised `version` rather than read a stale layout
as though it were current.

## Next session

1. Decide the `RingBuffer` question above.
2. Confirm the tablet's API level, then pin `minSdk`/`targetSdk`.
3. Gradle + Chaquopy skeleton; get `spektr_android.Engine` constructed from
   Kotlin and one frame packed. That alone proves Chaquopy hosts numpy.
4. Compose renderer against the wire format — run-length draw per row.
5. `CaptureService` last, with the tablet attached, since it is the only part
   that cannot be proven without it.
