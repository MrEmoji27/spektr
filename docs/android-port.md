# spektr on Android — Kotlin shell design

**Date:** 2026-08-08. **Last checked against the code: 2026-08-15.**
**Status:** **Shipped and merged.** This is the design the port was built to;
the app now runs on hardware and the branch has been merged into `main`. See
`CHANGELOG.md` for what actually landed and where it differs from the plan,
Planning — scope, build order, risks, open questions — is kept out of this
file; what remains is how the thing is put together and why.
The *Interim work* both landed on `main` on 2026-08-15, and the counts in this
file were corrected against the code the same day — they had drifted within a
week of being written. Recount before quoting any figure here.
**Branch:** none any more. The port lived on `android-port` while it was being
built and was merged into `main`; everything Android is now in the main tree
under `android/`, and `android/README.md` is the port's own front page. The
counts below are the ones this document was written against and several have
moved since — the engine has grown modes and Flipbook has been removed
outright rather than merely cut from the port. Recount before quoting any
figure here; `android/README.md` and `CHANGELOG.md` are the current ones.
**Target:** the author's 11.3" Android OLED tablet first. Personal build,
sideloaded — no Play Store, so no policy review and no fragmentation matrix.
**Goal:** parity with the desktop app, not a reduced mobile edition. See
*Parity*.
**Release posture:** when it eventually goes on GitHub it ships labelled
experimental, with the working conditions stated up front. See
*Release labelling*.

---

## The constraint that shaped everything

Android has no general "tap the system mix" API. Three routes exist and two are
shut:

| Route | Status |
|---|---|
| `AudioPlaybackCapture` (API 29+) | Works, but only for apps that have not opted out |
| `Visualizer` on audio session 0 | Deprecated; the framework disables global effects when per-player effects are present |
| Audio HAL / AudioFlinger tap | Root only |

`AudioPlaybackCapture` is therefore the only supported path, and it is real
system audio — clean, no room noise. Its hole is that an app can set
`android:allowAudioPlaybackCapture="false"`, and a blocked app yields **silence
rather than an error**.

The mic was considered and rejected: it captures the room, not the output, which
is a different product.

### What that means for this device

Checked against the RootlessJamesDSP compatibility list — the same API, the same
problem, documented in the field rather than in theory:

- **YouTube Music — the primary source here — is not blocked.** No patch, no
  root, works as-is.
- Amazon Music, Deezer, Apple Music, Poweramp, Substreamer, Twitch: also fine.
- **Spotify blocks capture.** The workaround is the ReVanced Manager
  "Remove screen capture restriction" patch, applied once to the Spotify APK.
  Personal-device move: it breaks Play Store updates for that app, needs
  re-patching on upgrade, and is against Spotify's ToS. Nothing in spektr
  depends on it.
- Chrome and SoundCloud are blocked and **cannot** be patched — the ReVanced
  patch does not reach apps that play through the native AAudio C++ API.

Because silence is indistinguishable from a blocked source, the app must say so
rather than draw a flat line. See *Blocked-source detection*.

---

## Architecture

The premise that makes this cheap: spektr's modes already return
`(codes, cidx)` — a `(h, w)` grid of Unicode codepoints and a matching grid of
palette ramp indices. They never touch Rich and never see a colour value. That
is already a renderer-agnostic interface, so Kotlin can consume it directly and
no mode code forks.

The boundary is that **return type**, not a ban on strings inside a mode. An
earlier draft of this file said modes "never build strings", and that is not
true: `Readout` in `modes/spectrum.py` composes a ticker tape with `"".join`
and then converts it with `[ord(c) for c in tape]`. What matters is that what
crosses the boundary is an int array — a mode may build whatever it likes
internally. Stated the wrong way, the rule would fail its first review against
the actual code. See `docs/architecture.md` on `main`, which records the same
boundary for the desktop line.

| Layer | Lines | On Android |
|---|---|---|
| `capture.py` | 1069 | **Replaced** by Kotlin `AudioRecord` + MediaProjection |
| `analysis.py` | 527 | **Ships unchanged** (pure numpy) |
| `modes/` | ~3500 | **Ships unchanged** — every mode the picker offers. (Written when Flipbook was cut from the port only; it has since been removed from spektr entirely, along with `asciiart.py`.) |
| `palette.py` | ~900 | **Ships unchanged**; Kotlin reads `Palette.hexes` |
| `render.py` | 281 | **Partly** — keep `pack_braille`, `cell_max`; drop `make_strips` (Rich `Segment`) |
| `widget.py`, `app.py` | ~1400 | **Dropped** — Textual |

About 5000 lines of tuned Python ship unchanged and stay shared with the desktop
build. Kotlin owns the audio path and the view.

**Python host:** Chaquopy. MIT-licensed and free since 12.0.1 (Aug 2022), ships
prebuilt numpy for Android. No licensing blocker.

**No C/C++ in v1.** Kotlin's `AudioRecord` handles capture, numpy does the FFT,
Compose does the drawing — native code has no job here yet. The one place it
could earn a place is the per-frame bridge, and only if measurement demands it
(see *Risks*). Carrying an NDK toolchain from day one for a hypothetical is not
worth it.

**Frame rate target: 30 fps, not the desktop's 60.** Two reasons. The panel is
being watched from across a room rather than at a desk, where 30 reads as
smooth; and every frame costs a JNI crossing, so halving the rate halves the one
cost this design is most exposed to. Nothing in the engine cares — spektr's
motion is integrated in seconds, not frames, and `test_audit.py` enforces that.
The analyser is unaffected either way: it runs on its own clock at HOP 256,
~188 analyses/sec, so new spectra still arrive faster than frames consume them.

---

## Components

### 1. `CaptureService` (Kotlin, foreground service)

`mediaProjection` service type — mandatory on API 29+ and enforced harder since
Android 14. Flow: `MediaProjectionManager.createScreenCaptureIntent()` consent →
`AudioRecord` with `AudioPlaybackCaptureConfiguration` (usages MEDIA, GAME,
UNKNOWN) → 48 kHz float ring buffer.

Owns the ring buffer and nothing else. Publishes 256-sample hops — the same
`HOP` the desktop analyser uses, so `analysis.py` needs no change.

### 2. `Engine` (Kotlin ↔ Chaquopy boundary)

Single entry point per frame:

```
fun frame(hopPcm: FloatArray, w: Int, h: Int): ByteBuffer
```

Calls into Python, which runs the existing analyser and the selected mode, and
returns **one packed direct `ByteBuffer`** rather than two Python lists — see
*Risks*. Layout per cell: `uint32` codepoint + `uint8` ramp index.

Holds the mode name, theme name and sensitivity. Nothing else crosses the
boundary.

### 3. `GridView` (Compose `Canvas`)

Reads the buffer and draws runs of same-coloured cells with one `drawText` per
run — the same run-length idea `make_strips` uses, reimplemented in Kotlin
against `Palette.hexes`. Requires a font with full braille coverage bundled in
the APK (system fonts cannot be relied on for U+2800–U+28FF).

Grid size derives from measured cell metrics and the view size, and is passed
*into* `frame()` — the Python side stays resolution-agnostic exactly as it is on
desktop.

### 4. `AmbientController`

Long listening sessions on OLED. Four pieces, all in the view layer, none
touching Python:

1. **Keep-screen-on** — `FLAG_KEEP_SCREEN_ON` while capturing. Overrides the
   device's 20-minute timeout without a permission.
2. **Auto-hiding chrome** — mode name and controls fade after ~5 s, tap to
   reveal. The single biggest burn-in factor, since the braille field is
   constantly changing and self-mitigates but static UI does not.
3. **Pixel shift** — translate the whole grid a few pixels on a ~60 s cycle. A
   Canvas translation, not a re-render.
4. **Idle auto-dim** — after ~10 min untouched, scale the ramp's peak luminance
   down.

A `#000000` background on OLED is pixels genuinely off, drawing no current and
ageing not at all, so a near-black theme is the default.

### 5. `MediaControls` (footer)

Transport and volume, so the tablet does not have to be picked up mid-session.

**No permission required for any of it:**

- Previous / play-pause / next — `AudioManager.dispatchMediaKeyEvent()` with
  `KEYCODE_MEDIA_PREVIOUS` / `PLAY_PAUSE` / `NEXT`. Goes to whichever app
  currently holds media-button focus, which is the one being captured. Each key
  needs a matching `ACTION_DOWN` and `ACTION_UP`.
- Volume down / up — `AudioManager.adjustStreamVolume(STREAM_MUSIC, …)`.
  Hardware keys keep working regardless; these exist so the whole surface is
  reachable by touch.

`dispatchMediaKeyEvent` is **fire-and-forget** — it sends, it cannot read back.
So the play-pause button cannot show true state without the optional grant
below, and should render as a neutral toggle rather than lying about state.

The footer is chrome, so it **participates in auto-hide** (§4). A permanently
visible row of buttons is exactly the static element OLED burn-in punishes.

### 6. Session read-back — optional, one permission

`MediaSessionManager.getActiveSessions()` needs either `MEDIA_CONTENT_CONTROL`
— privileged, not grantable to a normal app — or **notification-listener
access**, which the user grants once in system settings.

It is optional, and it buys three things at once:

1. True play/pause state and the current track for the footer.
2. Now-playing text, matching the desktop header.
3. **Naming the app that is blocking capture** (below).

Without it everything still works; the UI is just less specific.

### 7. Blocked-source detection

If the captured stream is exactly zero for ~2 s while audio is playing, say so
rather than drawing a flat line, and point at the ReVanced workaround.

With the §6 grant this names the offending app. **Without it, it cannot** — it
can only report that the current source blocks capture, because identifying the
app requires reading the active session. Worth stating plainly in the UI copy
either way; the failure is the same, only the specificity differs.

---

## Data flow

```
MediaProjection consent
  -> CaptureService: AudioRecord -> ring buffer (48 kHz float)
  -> 256-sample hop
  -> Engine.frame(hop, w, h)
       -> Python: analysis.py -> bands/wave
       -> Python: modes.get(name).fn(ctx) -> (codes, cidx)
       -> pack -> direct ByteBuffer
  -> GridView: run-length draw onto Compose Canvas
  -> AmbientController: offset, dim, chrome visibility
```

---

## Parity

The goal is that this behaves like spektr on Windows or Linux, not like a
stripped mobile port. That is largely achievable, and the reason is one
function: **every user-facing feature routes through `palette.config_dir()`**.

```
config_dir()/config.json      settings
config_dir()/presets.json     presets
config_dir()/themes/*.toml    user themes + the theme editor's output
config_dir()/plugins/         plugin modules
```

Point that at Android app storage and the whole persistence layer transfers with
no other change. Plugins in particular work: `plugins.py` loads them with
`importlib.util.spec_from_file_location` + `exec_module`, which is plain CPython
and available under Chaquopy, so any pure-Python/numpy plugin runs unmodified.

| Feature | Android |
|---|---|
| every offered mode, 55 themes | Unchanged — they are data and numpy. (Flipbook was the one mode cut here; it has since been removed from spektr entirely.) |
| Settings, user themes | Unchanged once the config root is redirected — **done on `main`**, see *Interim work* |
| Presets, plugins, theme editor | Possible by the same mechanism, but deliberately desktop-only — see *Scope* |

| Mode/theme pickers, the three settings | Re-implemented in Compose |
| Now-playing | *Easier* here — `MediaSessionManager`, behind one optional permission (§6) |
| Media transport + volume | **New on Android**, no desktop equivalent (§5) |
| Keybindings | Same keys when a keyboard is attached — plausible on an 11.3" tablet — plus touch equivalents |
| `--fps unlimited` | Transfers; `Display.getRefreshRate()` is the Android equivalent of `display.py` |

**What cannot transfer**, and why it is not a loss:

- `--devices`, `--device N`, `--diagnose`, `--monitor`, `d`/`D` source cycling.
  Android has exactly one capture source — the MediaProjection — so there is
  nothing to enumerate or choose between. The concept does not exist rather
  than being unimplemented.
- The CLI itself. A launcher icon has no argv. Everything `--mode`/`--theme`
  did is reachable from the UI.
- `--mic`. Deliberately dropped; see the constraint section.

So the divergence is confined to audio-device selection, which Android removes
by construction. Everything a user actually interacts with can match.

### Interim work — **done on `main`, 2026-08-15**

The port is cheap in proportion to how clean the engine/frontend boundary is,
and that boundary used to be implicit. Both changes named here have now landed
on `main`, so the port starts from a cleaner seam than this document assumed:

1. **Make `config_dir()` injectable** — done. The config root is an optional
   `config_dir=` parameter threaded through `config`, `presets`, `plugins`,
   `asciiart` and `palette`, with an `Spektr(config_dir=...)` app seam that
   reaches the widget's theme list. Commits `35604d1`, `b3454a0`, `e425170`.
   It defaults through the module attribute at call time, so nothing changes
   for a user who injects nothing. The one deliberate exception: the CLI entry
   points still read the platform default, because there is no `--config-dir`
   flag yet.

   For the port this is the piece that matters — redirecting the config root to
   Android's app-private storage is now a parameter, not a monkey-patch.
2. **Name the split** — done, as `docs/architecture.md` on `main`. Read that
   file rather than the summary here; it was written against the code and
   corrects this document in two places (see *Architecture* above and the
   counts below).

**Standing rule this file learned the hard way:** the numbers in this document
were written on 2026-08-08 and were wrong by 2026-08-15 — modes, themes and the
test count had all moved. Recount before quoting any of them; do not trust a
figure here without checking it against the code.
