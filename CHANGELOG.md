# Changelog

What changed, and why — kept in the same spirit as the commit log: the reason
matters more than the list, because the reason is what tells you whether a
change applies to you.

Dates are the day the work landed. Versions follow the desktop app; the
Android port carries its own `android-vN` tags until it merges.

## Unreleased — Android v2

### The picker release

**Modes and themes are pickable on the device.** v1 shipped one hardcoded mode
and one hardcoded theme, because the risk worth retiring first was whether the
engine ran at all, not whether the menu was nice. It runs, so: 52 modes and 54
themes, chosen from the screen, remembered across launches.

The theme picker draws colours rather than names. Fifty-four names is a list,
not a choice — nobody knows what `ayu-mirage` looks like, and finding out by
selecting each in turn is the whole afternoon.

The picker offers 52 of the engine's 64 modes. The twelve `(o)` variants draw
through Unicode 16 octants (U+1CD00 and up), which no font on Android has yet;
listing them would mean twelve entries that render as tofu. They are still
selectable by name, so a saved setting naming one keeps working.

**A settings sheet**, behind a button on the home screen:

- **true black** — background goes `#000000` and the ramp's bottom third fades
  into it, in linear light. An OLED pixel showing black is switched off, so on
  that panel this is not a darker theme, it is less screen.
- **smooth** — draws the picture instead of the glyphs (see below).
- **sensitivity** — the same 0.15–8 trim as the desktop's `[` and `]`.

**The home screen previews the selected mode**, live, before any capture
consent. Better for picking a mode, and it is also the reason the renderer
could be tested at all — the grid previously drew nothing until the OS
screen-capture dialog had been accepted.

Also: prev/next mode buttons, chrome that hides on a tap over the picture, and
a "made by zemo" footer.

### Fixed: a dozen modes drew sheared

The bug was in the renderer, not in any mode. It measures one cell from U+2588
and draws a whole run of identical glyphs as a single string — which only
lands on the grid if every glyph advances by exactly one cell, and in DejaVu
Sans almost none of them do. Braille is 0.7324 em against the full block's
0.7690, so each braille glyph in a run sits 4.8% of a cell left of the one
before it and a hundred-cell row finishes five cells adrift.

Locket, Tunnel, Dither Storm, Gonio, Valentine, Vinyl, Radial and the rest
were all affected. The block-element modes were not — which is exactly why
Kaleidoscope looked perfect in v1 and hid this for a whole version.

Fixed with tracking rather than scaling: `letterSpacing` pads each glyph's
advance out to a cell without touching its shape. Measured afterwards on the
tablet, the dot lattice has period 13.474 px at the left edge and 13.474 px at
the right — 0.00% drift across 1968 px.

### Fixed: Matrix would have shipped as tofu

`Matrix` draws entirely in halfwidth katakana (U+FF71–FF9D) and DejaVu Sans,
the only font the APK carries, has none of it. Invisible from the Python side,
which was perfectly happy. Anything the bundled font lacks now goes to a
second Paint on the platform's fallback chain, and `tests/test_android_font.py`
renders every offered mode and checks its codepoints against the shipped
font's cmap, so the next one fails in CI instead of on a tablet.

### New: smooth rendering

A cell is not a pixel. Chladni computes a smooth nodal field and then picks
one half-block to stand for each cell, so on a 118×34 grid you see a mosaic of
something with four times the detail in it. With **smooth** on, the mode runs
at a higher grid and the field it actually computed is sent whole, blitted and
filtered — continuous curves and thin streaks of light instead of staircases.

Same renderer, not a second one: the field is exactly what the braille dots
and half-block halves would have shown. Work is capped by total cells rather
than by a fixed multiplier, so a bigger screen gets a smaller multiplier
instead of a dropped frame. Measured at 29.5 fps on the tablet at 4×.

### Performance

- The grid renderer built a fresh `TextStyle` per run and called Compose's
  `drawText`, which is a full measure-and-shape pass each time. Replaced with
  one reusable `android.graphics.Paint` drawing to the native canvas:
  **93 ms → 19 ms** at the 50th percentile, 125 → 25 at the 90th.
- The render loop slept a whole frame *after* the work, making the period
  render + 33 ms. Paced by deadline instead: **10.7 fps → 29.7 fps**.

### Removed

- **Flipbook**, and with it `spektr/asciiart.py`, the `ascii_reel` and
  `ascii_fx` settings, its two settings rows and its tests.

### Diagnostics

Debug builds log fps, frame time, energy, onsets/s, peak band and raw sample
peak once a second. A port has no window onto itself, and every number that
decides how a mode behaves lives on the Python side of the boundary.

## Earlier — Android v1

First build that ran on hardware. Chaquopy hosting CPython and numpy
(~0.5 s to import the whole engine), `AudioPlaybackCapture` feeding the
analyser, and a Compose renderer drawing the engine's own glyphs and colours.
One mode, one theme, deliberately.

Fixed on the way: a `NullPointerException` before the first frame — Chaquopy's
`PyObject.get` is *attribute* access, so `BUILTIN.get("gruvbox")` asked a dict
for an attribute of that name and returned null. `RECORD_AUDIO` was missing
from the manifest, without which `AudioPlaybackCapture` cannot be built at
all. And the Python bridge had rotted 101 commits behind the engine it wrapped,
calling an `Analyser.snapshot()` that never existed.
