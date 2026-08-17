# android/ — v1 Kotlin shell

**Status: v1 milestone built.** `gradlew assembleDebug` completes from a clean
checkout. The v1 flow — consent → capture → grid — is written end to end, with
one hardcoded mode (Kaleidoscope) and one hardcoded theme (gruvbox), exactly
the design's build order. Nothing after this file's "unproven" section has been
seen running: the only build machine has no device on adb.

## What builds

- `app/src/main/kotlin/dev/spektr/` — the whole v1 Kotlin shell:
  - `MainActivity` — the consent flow: start button → OS screen-capture prompt
    → foreground service.
  - `CaptureService` — `mediaProjection` foreground service doing
    `AudioPlaybackCapture` (usages MEDIA/GAME/UNKNOWN) into the engine's ring
    buffer as float32 stereo at 48 kHz.
  - `EngineManager` — one Python thread owns every Chaquopy call: engine
    construction, pushes from the capture thread, and a ~30 fps render loop
    feeding the Compose grid.
  - `PyEngine` + `FrameBuf` — the bridge: `Engine.render(w, h)` returns one
    packed buffer; `FrameBuf` parses the header, honours the plane count and
    refuses an unrecognised magic/version/length.
  - `GridView` — a Compose `Canvas` that measures its own cell metrics from
    the grid font, reports the cell count to the engine each frame, and draws
    run-length-encoded: one rect per background run, one `drawText` per
    foreground run. Ramp colours come from Python (`Palette.hexes`) — nothing
    is hardcoded in Kotlin.
- `app/src/main/python/spektr/` — a vendored copy of the engine package, and
  `spektr_android.py` next to it. See "Why vendored" below.
- The grid font is DejaVu Sans, bundled under `res/font/` (license in
  `font-license/`). DejaVu Sans Mono was checked and **does not** contain
  braille; DejaVu Sans does, and so do all block elements the half-block modes
  draw.

## What this machine can and cannot do

| | |
|---|---|
| Android SDK | `C:\Users\mremo\AppData\Local\Android\Sdk`, `ANDROID_HOME` set |
| Platforms | android-30, 31, 33, 34, 35, 36 |
| Build tools | 34.0.0, 35.0.0, 36.0.0, 36.1.0 |
| NDK | 27.0.12077973, 27.1.12297006 (not needed — v1 is Kotlin + numpy) |
| JDK | 17 (Adoptium), `JAVA_HOME` set |
| `gradle` on PATH | **no** — use `gradlew` |
| Device on adb | **none attached** |

Stack pinned in the build files: Gradle 8.9 (wrapper), AGP 8.7.2, Kotlin 2.0.21
+ Compose BOM 2024.12.01, Chaquopy 17.0.0 hosting CPython 3.13 with numpy
1.26.2, rich and textual from Chaquopy's pip. minSdk 29, targetSdk/compileSdk
35.

## What is proven here

- **`gradlew assembleDebug` passes from a clean checkout.** `app-debug.apk`
  (~79 MB) contains, verified by inspection: CPython 3.13 native libs for
  arm64-v8a and x86_64, numpy 1.26.2 + OpenBLAS, rich, textual and their pure
  Python dependency trees, and the whole `spektr` package (every mode)
  compiled to bytecode. **The risky JNI thing — Chaquopy hosting numpy in an
  APK — is proven to the extent a build can prove it.**
- **The bridge tests still pass** — `python -m pytest tests/` (229 tests,
  including `test_android_bridge.py`). They import the exact vendored copy the
  APK ships, so the Python in the APK is the Python that was tested.
- **The wire format is parsed as written** — plane counts, sizes, magic and
  version are all checked; a wrong buffer is refused, not misread.
- **spektr runs on numpy 1.26** — the Android wheel is 1.26.2, and the source
  uses nothing numpy 2.0 removed (checked statically; the suite itself runs on
  a newer numpy).

## What still needs the tablet

All of it, until adb has a device:

- **Capture** — `AudioPlaybackCapture` is the one part of the design with real
  risk, and it cannot be exercised off-device. The service, the consent token
  plumbing and the FGS rules are written to the API surface but none of it has
  run.
- **Frame rate** — the render loop is paced at ~30 fps, but whether a tablet
  SoC can hold that (the design's main risk, the per-frame JNI crossing) is
  unmeasured. The code is built so a slower result just shows fewer frames,
  not a wrong picture.
- **The Compose renderer** — compiled, not seen. Cell metrics, run-length
  drawing and the astral-codepoint path (some modes emit past U+FFFF; the run
  builder uses `appendCodePoint`, never `Char.toChar`) are all unrun code.
- **Blocked-source copy** — silence renders as a flat grid in v1; naming the
  blocking app is v3's notification-listener grant.
- **The tablet's API level** — still unknown; the manifest targets the
  Android 14+ FGS rules and is written to run down to API 29.

## Decisions the design left open

- **The `RingBuffer` question** — resolved in favour of shipping `capture.py`
  unchanged: it is pure numpy with no audio device in it, so the whole package
  goes in the APK and Kotlin feeds its `RingBuffer` via `Engine.push`. No
  file moves on `main`.
- **Font** — DejaVu Sans (see above). Noto Sans Symbols 2 was checked and has
  braille but **no** block elements, so it cannot serve half-block modes.
- **Why spektr is vendored, not pip-installed** — pip-installing the package
  would resolve its desktop-only dependencies (sounddevice, soundcard, winrt,
  dbus-next), none of which have Android wheels, and Chaquopy cannot scope
  `--no-deps` to a single requirement (one pip invocation per build, options
  apply to all). So `scripts/sync-python.ps1` copies the package from the
  checkout root; run it when the engine moves on `main`, commit the result.

## Still true of the design

`docs/android-port.md` remains the architecture. v1 changed nothing about it:
consent → capture → grid, one mode, one theme; pickers, settings, media
controls and ambient behaviour stay v2/v3, and nothing in v1's code precludes
them. The one thing v1 cannot claim is that it proves the three risky things —
that is the tablet's job, and the tablet is not attached.
