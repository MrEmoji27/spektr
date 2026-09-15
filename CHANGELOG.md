# Changelog

What changed, and why — kept in the same spirit as the commit log: the reason
matters more than the list, because the reason is what tells you whether a
change applies to you.

Dates are the day the work landed. The Android port keeps its own version
line because it moves at its own pace, and ships inside a spektr release: the
APK carries the release's version number, and the heading below says which
port version that is.

## spektr 0.5.0

Four new modes, and a release mostly about making the pictures that already
existed look the way they were meant to: themes that are the theme's colour
in any terminal, a tunnel drawn as lines rather than bands, a night sky that
stays dark until the music says otherwise, and frames that are never shown
half-painted. And, if you want it back, the see-through terminal.

57 modes to 61. Themes stay at 55.

### Crosscurrent

A new tunnel, next to Tunnel In. Round rings travel out toward you and
eight-sided rings travel in toward the vanishing point, in the same wireframe
at the same time, and the whole wireframe turns.

The streams are told apart by shape, not colour, and the crossing is the
mechanic: wherever a round ring and an octagon coincide the round stroke
flares, and because a circle and an octagon only meet at some angles, a
crossing is a spark sliding round the ring. The lower half of the spectrum
pushes the outbound stream and the upper half pulls the inbound one, each
against its own recent level — absolute levels do not work on real music,
where the bottom quarter of the bands had a median level of 0.00 and 0.02
across a minute of each of two tracks.

The turn follows the music. A detected tempo sets the pace at about a spoke
gap a beat (96 bpm without one), and activity scales it on a curve, from a
sixth of that for a steady pad to three and a half times it when everything is
going. It eases toward its target over half a second, so the tunnel visibly
accelerates into a busy passage and coasts out of it, and silence settles it
to a slow turn. A band that jumps pulses the spokes in its sector, and a
pulsing spoke strobes on the beat's sixteenth notes between a solid line and a
dashed one whose gaps move every strobe, so the light stutters along the line
instead of only changing colour. A strong hit sends the pulse sweeping out from
the sector that jumped most.

A turning stroke cannot be cached, and computing the whole grid every frame
cost two thirds more than a static tunnel at 400x100. Each frame works only on
the dots that can be on a stroke, found from sorted angle and depth indices,
with output identical frame for frame to the whole-grid form; it is now
cheaper than the static version was.

### The JP family

Three modes inspired by Japanese car audio hardware — the segmented LED level
meters on in-dash head units and equalisers — drawn as that panel: one shared
LED ladder, with the unlit bulbs drawn as well as the lit ones, which is the whole
difference between these and `Bars` — a bar chart with gaps in it against a
panel with the power on. Each is that ladder blended with a mechanic already
somewhere else in the app, and the family is held to that rule: a JP mode that
cannot be named "Bars × something that exists" is a clone and does not ship.

* **JP Bars** takes the note roll: the crest of a rising bar sheds bulbs
  that climb the ladder and fade, so the panel holds the level and the last
  couple of seconds of it at once. A held note sheds nothing.
* **JP Drift** takes `Dune`'s sandpile. Columns fill with the square of
  the level, and past the angle of repose they wait for a beat strong enough
  to topple them, then pour into their neighbours over a fifth of a second — so
  bass-heavy music brings columns down on the kicks and a cascade steps across
  the spectrum in time.
* **JP Pulse** bends the ladder into a dial, a spoke of bulbs per band,
  with a chaser launched on onsets.

The panels are drawn in four ink weights — lit bulb, peak, trail, unlit dot —
with the unlit colour chosen by measured contrast against the theme's
background, because on a theme like gruvbox the bottom of the ramp is the
brightest colour it has. In silence they go properly dark.

Two earlier members were cut before release rather than kept to make up the
number: a Sweep that was a scan light over a bar chart, which `Sonar` already
does, and a Keys whose piano laid 32 log-spaced bands out on a keyboard, where
a lit key never meant that note.

The family is not final. Any of the three may be changed, improved, reworked
or removed, and more modes will join it in the next major release.

### The theme's background, in every cell

In 0.4.5 the visualizer never actually drew the theme's background. Textual
draws a line-API widget's cells exactly as they are handed over, the widget's
background only fills padding, and every cell went out as a bare foreground
colour — so the terminal painted its own scheme behind them. Logged at the
byte level on Windows Terminal running Catppuccin Mocha: gruvbox rendered on
`#1e1e2e`, and flexoki-light on a dark ground instead of cream. Every contrast
decision a mode made was made against a colour that was not on screen. Every
cell now carries the theme's background.

### A see-through background, if you want it

That fix is what makes a translucent terminal look solid, because a terminal
only draws a cell translucent when the cell names no background of its own.
So it is a setting: the settings panel's **background** row, or
`--background terminal`, leaves empty cells to the terminal and lets Windows
Terminal's opacity or acrylic show through. `theme (solid)` stays the default,
because it is the only way the theme is guaranteed to be what you see. The
cost is stated in the setting: a light theme over a dark terminal is pale
lines on a dark ground, and modes still judge contrast against the theme's
background.

Two-colour modes paint the floor of their ramp where there is nothing — the
dark between Radial's wedges, the plate between Chladni's lines — so those
cells are cleared too, and a run of cells is never allowed to carry a colour
across a cleared one. Real field colours keep their backgrounds, and the
header, footer and panels stay solid. It applies live, with no restart.

### Tunnel and Tunnel In, redrawn

In a real terminal the sixteen spokes broke up exactly where they should read
cleanest, converging on the centre, and at larger sizes the whole corridor
read as a mess. Measured on the rendered braille, it was built from bands of
stretched polar coordinates rather than from lines: an undocumented depth
twist bent the spokes up to 18 degrees near the centre, their width stepped
between one and four dots, a loud ring was a filled annulus with 79% of its
dots in solid blocks, and a per-frame dither punched holes in everything and
left sixty-odd specks near the centre.

Both are now a wireframe of strokes with half-widths in real dots. The
distance to a ring comes from its depth error over the local gradient, so an
elliptical ring is the same weight all the way round, and the distance to a
spoke is an exact perpendicular. Rings thicken with their band's level but
never more than a seventh of the way to the next; a ring too close to its
neighbour to resolve is not drawn. There is no dither — depth is carried by
colour. The spokes run on into the vanishing point as hairlines that stop in a
hierarchy, each where it would touch its neighbour, and fade to near the
background as they go, so sixteen lines read as meeting at one dim point
rather than as a starburst or a ring of stubs. At 188x50 the picture went from
63-71 connected pieces to one and from 60-65 specks to none, and both modes got
cheaper.

### The cosmos family, reworked

The family's bargain is a mostly dark sky with the music arriving as events,
and inspected live every mode broke it. Faintness was a low ramp index, and a
theme's ramp is a hue gradient, so faint stars and fading trails came out as
loud as meteors. The sky is now coloured by measured contrast against the
theme's background, so faint recedes on every theme.

* **Shooting Star**: a hard hit always breaks a cluster loose, a medium one
  sometimes throws a single meteor, quiet onsets throw nothing — no meteors
  in a pad or after the music stops, where there had been. A cluster's
  trailing fragments had been culled on their first frame.
* **Constellations** grows a figure only to one of its nearest unused stars
  and ends it at six lines, instead of tangling into a knot that outlived the
  track.
* **Star Trails** stopped being a wall of arcs: no blur at rest, and arcs
  capped against the gap to the next star. About half the screen coverage and
  half the cost.
* **Supernova** waits for a hit to ask for one. The first nova had come off a
  four-to-eight-second timer and landed in the quiet opening.

### No more torn frames on Windows

Breaks that looked like rendering bugs in Tunnel's fastest ring were the
terminal showing a frame half-painted: a 60 fps capture matched back to the
recorded renders had 95 of 720 frames stitched from two. Textual only brackets
a repaint in synchronized-output markers after the terminal answers a query,
and its Windows driver never sends the query. spektr now asks on Windows;
Windows Terminal answers, every repaint is bracketed, and captures show no
stitched frames. A terminal without the mode never answers and nothing
changes.

### Under it

* `Pulse` and `Chladni (o)` came off the benchmark's over-budget list; neither
  was over budget by design, and one bad run recording them there is how a
  real regression hides.
* New test files hold each of these to what it promises: the JP family,
  the cosmos family, the tunnels, Crosscurrent, synchronized output, and the
  theme background in both its solid and see-through forms. 729 tests.

## spektr 0.4.5

Five new modes, a new mark, and a change to how you get at the modes at all:
`l` now picks which of them the app offers you, because fifty-seven is more
than anyone wants to cycle through to reach their four.

### One line to install on Windows

```powershell
irm https://github.com/MrEmoji27/spektr/releases/latest/download/spektr.exe -OutFile spektr.exe; ./spektr.exe
```

`releases/latest/download/<asset>` is a GitHub redirect that always resolves
to the newest published release, so the line does not carry a version and
does not go stale on the next tag. It fetches the portable build rather than
the installer: nothing is registered, nothing is written outside the working
directory, and deleting the exe is the uninstall — which is the right default
for a one-liner someone pastes to try a thing once.

`irm` rather than `curl.exe` because `Invoke-RestMethod` is a PowerShell
builtin on every supported Windows and needs no aliasing, and rather than
`iwr` because `Invoke-WebRequest` builds a parsed response object it then
throws away. Both work; this is the shortest that is also honest about what
it does.

Not a `pip install`, which is still pending: `spektr` on PyPI belongs to an
unrelated project, so that one arrives with a package name to announce.

### A new logo

spektr has a proper mark, drawn by **Roshan (RRDOJ)**. It is a terminal
prompt — `>` over `_` — built out of spectrum bars, with the gradient running
cyan through violet to red the way the frequency range does. The old icon was
the Bars mode in a rounded square, which said "audio meter" and nothing about
where it runs; this one says both halves of what spektr is in one shape.

It ships everywhere the old one did and in a few places it did not: the
README, the Windows `.ico`, and a full Android adaptive icon set — every
density, plus the monochrome layer Android 13 themes the launcher with.
`packaging/make_icon.py` was rewritten to generate them from the one source.

### Loadouts replace presets

`l` now opens a loadout: a checklist of every mode, where what you pick is
what `v`, `m` and shuffle will offer you from then on. Spektr ships fifty-odd
modes and nobody wants all of them in one rotation — cycling past the
forty-six you did not want to reach the four you did was the whole problem.
Picking everything is the same as picking nothing, so it does nothing until
you narrow it, and an existing config is unaffected.

Press `s` in that panel to name what you have picked and keep it. Saved
loadouts appear as `★` rows in the same list; `space` loads one into the
ticks so you can adjust it before applying, `d` deletes it. They live in
`loadouts.json` and hold modes only.

This replaces the `l`/`L` preset pair, which is a removal, so: a preset
bundled mode, theme, frame rate, bands, sensitivity and gate into one named
snapshot, which meant loading one moved four things you could already see and
change in the settings panel and had not asked it to touch. Splitting "choose
a set" from "name the set you just chose" across two keys did not help either
— you had to know both, and neither showed you what the other had done. A
loadout is one kind of thing on one key. Your `presets.json` is left on disk
rather than migrated or deleted: an entry holds a single `mode`, not a set,
so there is nothing in one that answers "which modes should be offered", and
inventing a one-mode loadout out of each would be worse than ignoring them.

`h` was reworked alongside it. It now explains what the things *are* — what a
loadout is, what shuffle actually swaps, what the `(o)`/`(q)` suffixes and the
`·plugin` marker mean, why a mode sometimes disappears — instead of only
listing keys, and every panel's own keys are spelled out. A key and a
two-word description do not tell you what the thing does.

### Five modes, and the terrain family

`Swell` and `Terra` render a height field rather than a bar chart — the band
plan becomes a landscape and the music moves it. `Constellations`, `Star
Trails` and `Supernova` join the cosmos family: one draws a figure that grows
with the beat, one is a long exposure of a turning sky, and one spends its
whole budget on a single event and waits for it.

`Shooting Star` was reworked at the same time — fragments now enter at the
edge of the display and cross it, rather than appearing partway through their
own run, and how near a meteor is decides its speed, tail and brightness
together instead of three independent rolls. Vertical separation in the band
columns got an adaptive gutter, so a wide terminal no longer smears
neighbouring bars into each other.

52 modes to 57. Nothing was removed.

### Star Trails stopped flooding the screen

Three compounding errors, all reading as a screen of solid white. The motion
blur weighted four neighbours at half each, so the kernel summed to more than
one and a single lit dot filled the field in ten seconds with no stars drawn
at all. The exposure length was fixed while the spin was not, so on ordinary
percussive material every star swept a 110-degree arc and they fused into a
disc; the exposure is now cut short against the measured spacing between
stars, which is what a photographer does anyway. And the blur was applied per
frame rather than per second, so the same passage came out twice as dense at
240 fps as at 24.

`Maelstrom` had the same per-frame bug in its dissipation, left behind when
its forcing terms were converted: the dye kept 93% of itself per second at 24
fps against 49% at 240, so the smoke lingered nearly twice as long on a slow
display.

### Motion profiles

The settings panel has a new `motion` row with two stops. `snappy` is the
tuning everything was calibrated against and stays the default; `glide` is
the slower, cava-like character — bars rise lazily, sink for most of a
second, energy leans into neighbouring bars, and transients arrive already
softened. Side by side on the same track, `glide` is the one that reads as
"busier in the centre": cava's smoothing rounds off bass hits while
sustained mids accumulate, which is most of why cava looks centre-weighted
next to spektr's faster attack.

None of it touches the analysis. The band plan, the EQ tilt, autosens and
the onset detector are upstream of the switch and identical under both
profiles — this changes how a measurement is animated, never what was
measured. And it is dt-correct throughout, deliberately not a port of
cava's gravity/integral filters: those are framerate-dependent by
construction (`framerate_mod = 66/framerate`), which is the exact bug
`motion.py` exists to keep out. `glide` reproduces the feel — slower spring
constants, a temporal pre-blend standing in for noise reduction, and a
monstercat-style neighbour spread standing in for cava's filter of the same
name — without importing the frame-rate coupling.

The Android engine takes the same row in its own settings screen, driven by
the same Python profile table, so both platforms move the same way.

## spektr 0.4.0

First release with a prebuilt binary for every platform: Windows exe and
installer, Linux binary, Android APK. macOS still runs from source.

44 modes → 65. 49 themes → 55. Two things underneath the app are new — onset
detection and subcell rendering — and most of the rest follows from them. The
Android port lives here too; it has its own version line and its own section
below.

### Onset detection

Anything that reacted to a beat before this release was reacting to bass
energy, because that was the only thing available. A kick and a sustained low
note look the same to a level meter, and modes built on one were driven by the
other about as often.

`spektr/analysis.py` now carries a real detector: half-wave rectified spectral
flux, log compression, adaptive median-plus-MAD thresholding, peak picking per
sub-band, a refractory period, adaptive whitening, two region gates, and a
rescue arm for hits masked by the drum before them.

`tests/onset_eval.py` is an 11-scenario
MIREX-style corpus with ±50 ms one-to-one matching, written against the
problem rather than against the implementation, and `tests/onset_score.py`
fails the build if the score drops:

```
scenario               P       R       F
click              1.000   0.875   0.933
four_on_floor      1.000   0.938   0.968
kick_snare         1.000   0.938   0.968
breakbeat          1.000   0.938   0.968
pad_under_kick     1.000   0.938   0.968
swing              1.000   0.938   0.968
tempo_ramp         1.000   0.974   0.987
quiet              1.000   0.938   0.968
silence            1.000   1.000   1.000
noise              1.000   1.000   1.000
note_stream        1.000   1.000   1.000
total              1.000   0.940   0.969
```

Precision is 1.000 everywhere, which matters more than recall here: flashing
on a beat that did not happen is worse than missing a quiet one. Most
remaining misses are the first event of a track, where there is no history to
compare against yet.

Breakbeat took the longest. It sat at F 0.720, and every missed hit died in
the flux peak test below the track's own median flux — the kick owned the
single scalar peak picking ran on, so no threshold could have found them. The
fix was to peak-pick per sub-band and merge.

Modes read this through four fields on `Ctx`:

* `onsets` — beats since this mode's previous frame. Differenced once, so a
  mode that skips a frame does not miss a beat or see it twice.
* `onset_strength` — how hard the hit was, which is a different question from
  how loud the track is. They come apart exactly where it matters: a quiet
  track with a crisp snare.
* `pulse` — a beat-locked swell, 1.0 on the beat and decaying through the
  bar. Onsets are discrete and only exist on the frames the peak picker
  committed on, so a mode driven only by them coasts in between; on a slow
  track that reads as a still picture jerking four times a bar. This fills the
  gaps, and returns 0.0 rather than a permanent full swell when there is no
  tempo — which is the trap the raw beat phase sets.
* `drive` — how percussive the signal is right now, per hop, safe to read at
  any frame rate. Answers "how much attack is in the signal" rather than "was
  there a hit this frame", so it is the one to drive rates with.

### New: sub-cell rendering

A terminal cell is one character in one colour. Everything spektr draws has
had to fit that, and the usual escape is braille — 4×2 dots in a cell, but all
eight the same colour.

The octant block (U+1CD00–1CDE5) gives 2×4 lit regions with a foreground and a
background colour, so a cell can hold two colours and eight sub-regions at
once. Twelve modes now have an `(o)` variant that uses it: the same mode,
drawn as a surface instead of as a lattice.

* `Scope (o)`, `ECG (o)` — the trace as a continuous stroke rather than a
  column of dots.
* `Radial (o)`, `Sonar (o)`, `Maelstrom (o)`, `Plasma (o)` — solid fields.
* `Chladni (o)`, `Chladni Extreme (o)`, `Chladni Flow (o)` — the plate figures
  at dot-grid resolution.
* `Kaleidoscope (o)`, `Kaleidoscope Ultra (o)` — the latter antialiases the
  seams between facets.
* `Valentine (o)`.

Fonts are the catch, and the reason this is opt-in rather than automatic. The
octant block is new enough that most fonts have part of it or none of it, and
a missing glyph draws as a tofu box, which is worse than the staircase it
replaced. So:

* spektr never emits the eight octant patterns fonts most often lack. Each
  is widened to the nearest shape that is a Block Element or a safer octant —
  an isolated subcell grows to its quadrant, three-quarters fills. Growing
  rather than dropping, because at a 4x4-pixel subcell the difference is
  invisible where a hole in an outline is not.
* `Quadrant` cells (2×2, from a much older Unicode block) are the fallback for
  fonts without octants at all.
* `spektr --glyph-test` prints every glyph the app can emit, so you can see
  what your font has before choosing.
* The variants are off the menu by default. Turn them on in settings.

The base modes are unchanged and still the default. `(o)` and `(q)` name the
geometry — octant and quadrant — rather than claiming to be "Fine", which the
earlier naming did and could not always deliver.

### New modes

Nine, plus the twelve variants above.

* **Shooting Star** — a night sky, mostly empty and mostly still, with meteors
  thrown from a drifting radiant on the beat. Built on the opposite bargain
  from every other mode here: the music arrives as events rather than as a
  level being redrawn, which is only possible now that events are detected.
* **Snow** — Rain's sibling and deliberately its opposite. Three depth planes
  of crystals, no radiant, nothing in the background.
* **Valentine** — a heart with a seamless interior depth built from a radius
  table.
* **Locket** — a heart-shaped tunnel that emits pulses from its outline, each
  on a path of its own.
* **Kaleidoscope** — a mirrored tube. `Kaleidoscope Ultra (o)` antialiases the
  seams.
* **Tunnel In** — Tunnel with the rings travelling the other way. Rings are
  visible from the frame they spawn on.
* **Dither** — a one-bit field with directional waves and absolute-tiled Bayer
  ordering.
* **Dither Storm** — the reactive one.
* **Dither Storm Extreme** — the saturating variant, kept on purpose.

**Removed: Flipbook.** It was a frame sequencer rather than a visualiser and
never earned its slot.

### New themes

Six saturated single-hue ramps: `emerald`, `sapphire`, `amethyst`, `citrine`,
`tangerine`, and `indigo`.

They fill measured gaps rather than crowding the set. There was no saturated
violet at all; the greens were muted or scientific; the blues were all
atmospheric and cold where sapphire is jewel-bright; and no theme was a plain
vivid orange, only heat ramps passing through it on the way to yellow.
`indigo` was the last hole — bucketing every theme's mid anchor by hue leaves
240–269° the only empty range, with sapphire stopping at 216 and amethyst
starting at 271.

The theme editor can now pick a colour rather than only nudge one.

### Performance

The frame budget is 16.7 ms at 60 fps and the whole app has to fit in it.

* `make_strips` run-length encodes the whole grid in one pass instead of row
  by row.
* Ten modes were rewritten around what was actually costing rather than what
  looked expensive: Radial, Dune, Ember, Vinyl, Arcs, Pulse, Auroras, Tunnel,
  Scatter and Kaleidoscope. Most were rebuilding something every frame that
  does not move.
* Four modes moved to float32.
* The colour block coarsens sooner, so a large terminal stays playable.
* `tests/bench.py` can now fail. It ratchets each mode's cost against a
  recorded multiple of the median mode, and it gates on the statistic that
  survives a second run — the earlier one drifted with machine phase and was
  red on a clean tree. It printed every number needed to catch a mode sitting
  at 10.8 ms for the whole of the project's life. Nothing read them.

### Fixed

* **The analysis hop rate was tied to the capture block size.** Change the
  device and the rhythm reading changed with it. Decoupled, and pinned by a
  test.
* **Non-finite samples reached the FFT.** Zeroed at the ring buffer instead.
* **`d` leaked a settle thread on every press.** Reopening the device now
  cleans up after itself.
* **The status line's audio gate disagreed with the analyser's**, so the app
  could report silence while drawing, or the reverse.
* **The goniometer's geometry was wrong**, along with three claims in comments
  around it that described what it was supposed to do.
* **A 1-D stereo buffer crashed the scope and stereo modes.** Rendered as mono
  now.
* **`make_strips` divided by a zero-width grid**, and **Flame divided by a
  zero flame width**.
* Edge cells are cut rather than stippled, coloured by each side's mean rather
  than by its extremes, and only dithered when there is actually an edge in
  them.

### The app

* **`h` opens a help panel**, generated from the key bindings so it cannot go
  stale. It shipped broken twice — first taking the app down on a `[` in a key
  label, then opening an invisible panel because it appeared in no CSS rule —
  and the tests now check that rows actually paint rather than that a widget
  mounted.
* The mode picker lists names rather than prose.
* Each ramp index drifts as far as its own colours allow, rather than every
  index sharing one limit.
* The settings panel is held to the invariants that keep it openable.
* `spektr --glyph-test`.

### Under it

* The engine/frontend boundary is written down, and the config directory is
  handed in rather than reached for. This is what made the Android port
  possible: the port supplies its own frontend and runs the same engine
  unmodified.
* Modes can declare which mode they belong behind, so the picker orders itself.
* The shared polar geometry lives in `modes/__init__.py` rather than being
  re-derived in each mode that needs it.
* Tests run on every push and pull request, and the whole suite runs rather
  than four files out of fifteen.
* Tagging a release now builds and attaches the Windows exe, the Windows
  installer, the Linux binary and the Android APK from one tag.

## Android v0.4.0 — ships in spektr 0.5.0

### The new engine

The port runs spektr's engine unmodified, so everything in 0.5.0 that is
drawn rather than terminal-specific arrives here too. The picker offers
**Crosscurrent** and the three **JP** modes, 62 of the engine's 74, and
Tunnel, Tunnel In and the cosmos family draw with their reworked geometry and
contrast.

The terminal-only changes do not apply: the see-through background and
synchronized output are about how a terminal paints cells, and the app paints
its own.

The version moves because the APK's contents did: a different engine under
the same `versionCode` (300) could not be told apart from the last build. It
is 400 now.

## Android v0.3.0 — ships in spektr 0.4.5

### The scene view

The height-mapped terrain renderer is gone and four raymarched scenes have
taken its place — **Metaball**, **Wormhole**, **Monolith** and **Lattice** —
drawn entirely in a fragment shader. This is a change of kind, not of degree.

The old renderer asked Python for a picture: a float per pixel, every frame,
about 33 ms of numpy on the tablet, and then displaced a grid mesh by it. What
it could ever be was therefore one thing — a lit sheet — and on a device it
read as a strip of landscape floating in a black frame, which is exactly what
it was.

The scene view ships no picture at all. A frame is forty floats: energy, bass,
mid, treble, an onset envelope, the beat phase, and twenty-four bands. The
shader builds the world from them. Three things follow. It fills the screen by
construction, because there is no mesh to frame and no camera distance to
solve — every pixel is in the scene, and the ray through it either hits
something or hits the sky, which is drawn in the theme's colours too. It costs
almost nothing on the Python side: 172 bytes and no arrays, against a mode's
worth of numpy per frame. And shape is free — fusing metaballs, an infinite
tunnel, a fractured solid and a lattice running to the horizon are four `map`
functions rather than four geometry pipelines.

Colour still comes from spektr's 64-entry ramp, uploaded as a texture, so all
fifty-odd themes work here unchanged. The surface renders at reduced
resolution and is scaled up by the display hardware: a raymarcher is
fill-bound and the panel is 1536x2560, so at native resolution the cost is
four megapixels of sphere tracing, which no phone GPU does at thirty frames.
Half is invisible on an organic scene and four times cheaper.

### The launcher icon

The new mark, as a proper adaptive icon: every density, a foreground and
background layer, and the monochrome layer Android 13 uses to theme the
launcher with the wallpaper. The old build shipped a single flat drawable.

### Also

The motion row from the desktop settings panel is here too, driven by the same
Python profile table, so both platforms move the same way.

## Android v0.2.0 — ships in spektr 0.4.0

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

### New mode: Shooting Star

The first of a cosmology family, and built on a different bargain from
everything else: the picture is mostly empty and mostly still, and the music
arrives as events rather than as a level being redrawn. A mode that is 98%
dark has to earn its reactivity from timing, so the onset detector *is* the
mode here rather than a garnish on it.

The meteors come from a **radiant** — the one point a real shower appears to
diverge from, because meteors travel parallel and only look otherwise. It
drifts, so a long session does not put every streak on the same diagonal. A
harder onset throws a brighter, longer, faster one from further out; that is
`onset_strength` rather than `energy`, because a fireball should answer to how
sharp the hit was and not to how loud the bed under it is.

The base spawn rate is deliberately low. There is one, so silence is not a
still image, but a mode built on events is ruined by a steady supply of them.

Written into `spektr/modes/cosmos.py`, which the family will share.

### New mode: Snow

Rain's sibling, and deliberately its opposite in how it moves. A raindrop
falls fast enough that its own motion is the shape, which is why Rain draws
each drop as a streak. A snowflake has almost no terminal velocity and a great
deal of air resistance, so it hangs, sways and arrives — drawn as a crystal
that drifts on its own sine and is pushed around by a shared wind.

Three depth planes: a five-dot crystal in front falling fastest and swaying
widest, a diagonal speck in the middle, a single dim dot far back. Without
that parallax a screen of white dots is noise; with it, it is depth.

No bokeh behind the glass. Rain's blurred circles are lights seen through a
wet window — a thing that happens indoors looking out — and snow is the
weather itself.

Snow also lies: flakes reaching the bottom add to a per-column depth that
slumps sideways and melts back slowly, so a loud passage leaves drifts along
the floor for a while after it has gone. Mid band sets how thickly it falls,
energy how fast, and `ctx.drive` gusts the wind sideways, so percussive
material blows the fall about rather than only thickening it.

### Fixed: the cell was the wrong shape, so nothing round was round

Every mode in the engine assumes a terminal cell — about twice as tall as it
is wide. That assumption is load-bearing: it is why Radial's rings are round,
why Kaleidoscope's symmetry is symmetric, and why the modes that halve a
vertical velocity or a horizontal delta do so at all.

DejaVu Sans in a Compose line box is **1.53:1**, not 2:1. So every one of those
corrections was over-correcting by a quarter, on every mode, since v1.

The cell is now forced to exactly 2:1 and the glyph squeezed horizontally to
fill it (measured on the tablet: 14.5×29.0 px, squeeze 0.763). Squeezing
rather than padding, because a block element has to keep tiling — U+2588 fills
its advance box, so scaling the box scales the fill and solid areas stay
solid. Adding leading instead would put a gap between every row.

### Fixed: Fireworks burst low and sideways

Two faults, both measured rather than guessed, and both in the mode rather
than the port.

**Bursts were 2.4 to 3.6 times wider than tall.** The shower's vertical
velocity was being halved, which reads as an aspect correction and is not one:
a terminal cell is twice as tall as it is wide and braille puts four dot rows
and two dot columns in it, so a dot is already square. Halving it a second
time made every shell an ellipse. Now isotropic, and only gravity bends it —
measured at 1.20–1.77, the remainder being the fall, which is real.

**Rockets never used the top of the screen.** Burst height was mapped from
`ctx.energy`, which is the mean over every band — and with the analyser's
autosens normalising the loudest band to about 1.0, a busy track means a mean
of 0.25 to 0.35, not 0.8. The old mapping wanted 0.77 before it would send a
shell high, so in practice rockets burst at 56–65% of the screen height and
the sky above them was never used. Recalibrated: 75–89% across the range music
actually produces.

Both apply to the desktop too — the mode was mis-calibrated on both, and the
tablet's wide screen is only what made it obvious.

### The changelog is in the app

On the home screen, behind **what's new**, with each version collapsible —
the newest open, the rest shut, because what you want from a changelog is what
changed *this* time. It is the same `CHANGELOG.md` you are reading now, copied
into the APK at build time rather than kept as a second copy, so the shipped
one cannot quietly fall behind.

### New: a detail setting

How many rows of cells fit on the screen — the app's resolution, and the one
number that decides how coarse everything looks.

It matters most for the modes that draw at cell resolution rather than into
braille dots. Needle's dial and pointer are built from whole cells, so at 40
rows on a 2560-wide tablet one cell is a 21×42 pixel pixel and the meter reads
as broken rather than as coarse. At 72 rows the same dial is a fine arc.

24 to 72 rows, in the settings sheet. More rows is a finer picture and more
cells to compute and draw; which trade is right depends on the screen and on
how far away you are sitting, which is why it is a setting.

### Fixed: ECG drew a rule across the middle of itself

Its scroll buffer starts as zeros, and zero is the centre line — so every
column the trace had not reached yet drew a dot exactly halfway up, and a
screen's worth of them drew a hard horizontal line through the middle of the
mode. It reads as part of the display rather than as an absence of data, and
it appeared every time the mode was selected, on every resize, and for the
whole of any silence — which is exactly when there is nothing else to look at.

Unwritten columns now draw nothing. Measured before and after through the full
Android engine: the centre dot-row went from covering 100% of the width on
every single frame to covering only what the arriving signal actually puts
there.

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

Measured on the tablet rather than guessed at. The Python render turned out to
be 3.5 ms a frame against a 33 ms budget — the cost was all on the UI thread,
which is not where anyone would have looked first.

- **Spectro's draw: 16 ms → 10 ms** at the 50th percentile (90th 17 → 11, jank
  87% → 24%). Two causes. The foreground pass built a `StringBuilder` per run
  to hold what was usually a single character — thousands of allocations a
  frame on any mode with fine detail. And the background pass ran for
  two-plane modes, which have no per-cell background, so every rect it drew
  was the surface's own colour painted over itself.
- **The field multiplier adapts to the mode.** At 4x the tablet's grid Chladni
  costs 1.4 ms a frame and Tunnel In 11.5 — same cell count, but braille
  carries eight picture pixels per cell where half-blocks carry two. A fixed
  multiplier is wasted detail on one or a dropped frame on the other, so it is
  a control loop on the measured render time, driven by a running average
  because individual frames scatter too much to steer on.
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

## Android v0.1.0 — the first build that ran on hardware

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
