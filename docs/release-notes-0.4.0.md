<!--
The body of the GitHub release for v0.4.0, as published.

It has to exist before the tag is pushed: all three build workflows attach
with append_body, so they add a download section to whatever is already there
and never write the notes themselves.

    gh release create v0.4.0 --draft --title "spektr 0.4.0"       --notes-file docs/release-notes-0.4.0.md
-->

**Windows, Linux and Android**, all prebuilt and attached below — no Python
needed on any of them. That is the first release where every platform ships a
binary; macOS still runs from source.

44 modes → 65. 49 themes → 55.

### Onset detection

Before this release, anything that reacted to a beat was reacting to bass
energy, because that was the only thing available. A kick and a sustained low
note look the same to a level meter.

`spektr/analysis.py` now has a real detector: spectral flux, log compression,
adaptive median/MAD thresholding, per-sub-band peak picking, refractory,
adaptive whitening, two region gates, and a rescue arm for hits masked by the
drum in front of them.

Scored against an 11-scenario corpus at ±50 ms with one-to-one matching:
**P 1.000, R 0.940, F 0.969**. The build fails if that drops. Precision
matters more than recall here — flashing on a beat that did not happen is
worse than missing a quiet one.

Modes read it through `ctx.onsets` (beats since the mode's last frame),
`ctx.onset_strength` (how hard), `ctx.pulse` (a beat-locked decay, for modes
that would otherwise freeze between hits) and `ctx.drive` (how percussive the
signal is, per hop, safe at any frame rate).

### Subcell rendering

A terminal cell is one character in one colour. The Unicode octant block gives
eight lit regions per cell in two colours, so a curve can land inside a cell
instead of on its boundary.

Twelve modes have an `(o)` variant that uses it: Scope, ECG, Sonar, Radial,
Plasma, Maelstrom, Valentine, Kaleidoscope, Kaleidoscope Ultra, and the three
Chladni modes.

Fonts are patchy about the block, which is why this is opt-in. Turn it on in
Settings (`c`). `spektr --glyph-test` prints every glyph the app can emit so
you can check yours first, the eight patterns fonts most often lack are never
emitted, and there is a `(q)` quadrant fallback for fonts with no octants at
all.

### Android

Same engine, same modes, same themes, on a tablet. It lives in this repository
under `android/` and the APK is attached below. It carries its own version
number — 0.2.0, because the port has had two versions — and ships inside this
release.

It visualises the device it runs on. Android has no way for one device to read
another's audio, so a tablet running spektr shows the tablet, not your PC.

### Modes and themes

New: Shooting Star, Snow, Valentine, Locket, Kaleidoscope, Tunnel In, Dither,
Dither Storm, Dither Storm Extreme. Removed: Flipbook.

New themes: emerald, sapphire, amethyst, citrine, tangerine, indigo.

### Performance

`make_strips` encodes the whole grid in one pass instead of row by row. Ten
modes were rebuilt around what was actually costing: Radial, Dune, Ember,
Vinyl, Arcs, Pulse, Auroras, Tunnel, Scatter, Kaleidoscope. Four moved to
float32. The colour block coarsens sooner so a large terminal stays playable.

`tests/bench.py` can fail now. It had printed every number needed to catch a
mode sitting at 10.8 ms against a 16.7 ms budget for the project's whole life,
and nothing read them.

Known: `Chladni Extreme (o)` measures 16.5–17.2 ms at 400x100, so at a
maximised window on a large screen it will not hold 60 fps. It is an opt-in
variant and the widget reuses frames past 11 ms, so it drops frame rate rather
than stalling.

### Fixed

- The analysis hop rate was tied to the capture block size, so changing audio
  device changed the rhythm reading.
- Non-finite samples reached the FFT. Zeroed at the ring buffer now.
- `d` leaked a thread on every press.
- The status line's audio gate disagreed with the analyser's, so the app could
  report silence while drawing.
- The goniometer's geometry was wrong.
- A 1-D stereo buffer crashed the scope and stereo modes.
- `make_strips` divided by a zero-width grid; Flame divided by a zero flame
  width.

### Also

`h` opens a help panel generated from the key bindings. The mode picker lists
names instead of prose. The theme editor can pick a colour, not just nudge
one. Tests run on every push, and tagging a release builds the Windows exe,
the installer, the Linux binary and the APK from one tag.

[Full changelog](https://github.com/MrEmoji27/spektr/blob/main/CHANGELOG.md)

---

### Which file

| you have | download |
|---|---|
| Windows, no Python | `spektr.exe` — portable, double-click |
| Windows, want a Start Menu entry | `spektr-0.4.0.0-setup.exe` |
| Linux, no Python | `spektr` — `chmod +x` and run |
| Android | `spektr-android-0.2.0-arm64-v8a.apk` — Android 10+, 64-bit ARM |
| Python already | clone and `pip install -e .` — not on PyPI yet |

The Windows and Linux builds are unsigned. SmartScreen will warn on first run:
More info → Run anyway.
