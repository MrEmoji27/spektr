# spektr for Android

The Android port is part of spektr's main repository and is merged into `main`.
It puts the same audio visualiser engine on a tablet or another spare Android
screen. The APK carries its own version line: Android v0.2.0 ships inside the
spektr 0.4.0 release, and its derived Android `versionCode` is 200.

This is an ambient display for a big screen, not a tool. It is meant to sit on a
desk or stand and be watched while another device does the work. It visualises
the device it runs on: Android cannot read the audio playing on a different
computer, so a tablet running spektr can show the tablet's audio, not the audio
coming from your PC.

## What it looks like

<table>
<tr>
<td width="50%"><img src="https://raw.githubusercontent.com/MrEmoji27/spektr/main/assets/android-home.jpg"
     alt="the home screen, with the mode and theme pickers" width="100%"></td>
<td width="50%"><img src="https://raw.githubusercontent.com/MrEmoji27/spektr/main/assets/android-tablet.jpg"
     alt="spektr running on a tablet" width="100%"></td>
</tr>
<tr>
<td><sub>Retro on <code>hot-pink</code>. The picture behind the buttons is the
engine running, not a still.</sub></td>
<td><sub>Chladni Extreme on <code>ice</code>, on the kind of screen this is
for.</sub></td>
</tr>
</table>

The home screen renders a live preview of the selected mode before capture
consent, so choosing a mode does not require going through Android's permission
dialog first. The mode picker offers 53 of the engine's 65 modes, and the theme
picker contains all 55 themes. The twelve `(o)` mode variants are not offered:
they use Unicode 16 octants, which no Android font currently provides. They
remain available by name to the underlying engine, but the picker does not list
them because they would render as missing glyphs.

The settings sheet contains four display controls. **True black** changes the
background to `#000000` and fades the bottom of the ramp into it, which lets an
OLED pixel switch off. **Smooth** draws the mode's field as a picture instead of
typesetting its glyphs. **Detail** chooses how many rows of cells fit on the
screen, which is the main control over how coarse a cell-based mode looks.
**Sensitivity** applies the same analyser trim as the desktop build.

The normal renderer draws a grid of cells using the mode's Unicode codepoints
and the theme's colour ramp. Smooth mode removes the terminal-cell constraint
and blits the field as a bitmap, so continuous shapes such as Chladni lines can
be drawn as curves rather than staircases. Rendering is paced at about 30 fps
by design: the screen is watched from a distance and every frame crosses the
Python/Kotlin boundary.

The home screen also has a `what's new` view backed by the repository's
`CHANGELOG.md`, so the changelog shown in the app is the same one shipped with
the release.

## Install

Download `spektr-android-*-arm64-v8a.apk` from the
[spektr releases page](https://github.com/MrEmoji27/spektr/releases) and
sideload it. The release APK is about 52 MB, targets 64-bit ARM only, and needs
Android 10 or newer (`minSdk 29`). The first install may require allowing the
browser or file manager to install an APK.

The APK reports `v0.2.0`, even when downloaded from the spektr 0.4.0 release.
Those numbers are intentionally separate: the desktop release and the Android
port do not move at the same pace.

## How it works

Android owns audio capture and drawing; the Python visualiser remains the
engine. Chaquopy hosts CPython and numpy in the APK and runs the desktop engine
package unmodified. Importing CPython and numpy takes about 0.5 seconds. After
startup, Kotlin feeds captured audio to the Python analyser, and one JNI call
per frame carries a packed grid of Unicode codepoints and colour indices back
to the Compose renderer. The packed frame keeps the bridge from passing one
object per cell.

Capture runs in a foreground service using `AudioPlaybackCapture`. The audio is
read as the device's playback stream, sent to the engine's ring buffer, and
discarded after analysis. The Kotlin renderer then draws either the packed cell
grid or the field frame produced by the Android adapter. This keeps mode logic,
analysis, and palette data shared with the desktop build while the Android side
handles the platform-specific audio and screen.

## Permissions and privacy

Starting a session requires both Android's screen-capture consent and the
`RECORD_AUDIO` permission. This is not a request to record the display: Android
uses the screen-capture consent for `AudioPlaybackCapture`, and it has no
separate audio-only consent. The foreground service also keeps an ongoing
notification while capture is running; on Android 13 and newer, Android may ask
for notification permission for that notification.

The app captures only what the same device is playing. An app that opts out of
playback capture produces silence rather than audio that spektr can inspect.
The capture is consumed by the analyser; spektr does not save or upload the
audio, and nothing leaves the device.

## Build it

From the repository root, build a debug APK with the Gradle wrapper:

```text
cd android && ./gradlew :app:assembleDebug
```

The Android module uses Gradle, Kotlin, Jetpack Compose, and Chaquopy. The
Python package under `app/src/main/python/spektr/` is a vendored copy of the
engine used by the APK. When the engine changes in the repository root, update
that copy with `scripts/sync-python.ps1` before building so the Android bridge
and the engine stay in step.

Release builds are produced by the repository's Android workflow and attach an
arm64-v8a APK to the matching spektr release. The Android version is read from
`gradle.properties`; its `versionCode` is calculated from that version rather
than maintained as a second hand-edited number.

## Where it is going

The current port already has a field renderer: every mode can provide a scalar
field that Android draws as a bitmap when Smooth is enabled. The next technical direction is to move that existing field onto the GPU and
displace a mesh as a height field. That
would make the existing modes look three-dimensional without creating a
second, Android-only mode system. It is an exploration of the next renderer,
not a promise that the current APK is a 3D application.

## Directory layout

The Android project is intentionally a small platform shell around the shared
engine:

```text
android/
  app/src/main/kotlin/dev/spektr/  Capture service, engine bridge, and Compose UI
  app/src/main/python/spektr/      Vendored Python engine package
  app/src/main/python/spektr_android.py
                                  Android entry point and packed-frame adapter
  app/src/main/res/font/            Bundled DejaVu Sans grid fonts
  scripts/sync-python.ps1           Refreshes the vendored engine copy
  font-license/                     License for the bundled font
  gradle.properties                 Android port version
  gradlew                            Gradle wrapper used for builds
```

The design and implementation notes live in [`docs/android-port.md`](../docs/android-port.md). The release history, including the Android-specific
version sections, is in [`CHANGELOG.md`](../CHANGELOG.md).
