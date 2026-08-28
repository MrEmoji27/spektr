<!--
The body of the GitHub release for v0.4.5, as published.

It has to exist before the tag is pushed: all three build workflows attach
with append_body, so they add a download section to whatever is already there
and never write the notes themselves.

    gh release create v0.4.5 --draft --title "spektr 0.4.5" \
      --notes-file docs/release-notes-0.4.5.md
-->

Five new modes, a new mark, and a change to how you reach the modes at all.

52 modes → 57. Themes unchanged at 55.

### A loadout, on `l`

Fifty-seven modes is more than anyone wants to cycle through to reach their
four. `l` opens a checklist of every mode; what you pick is what the picker,
the `m` key and shuffle offer you from then on. Picking everything is the same
as picking nothing, so it does nothing until you narrow it, and an existing
config is unaffected.

Press `s` in that panel to name what you picked and keep it. Saved loadouts
appear as `★` rows in the same list — `space` loads one into the ticks so you
can adjust it before applying, `d` deletes it. They hold modes only, so
loading one never moves your theme or settings.

This replaces the `l`/`L` presets, which bundled mode, theme, frame rate,
bands, sensitivity and gate into one snapshot: loading one moved four things
you could already see in the settings panel and had not asked it to touch.
Your `presets.json` is left on disk, just no longer read.

`h` was reworked alongside it and now explains what things *are* — what a
loadout is, what shuffle actually swaps, what the `(o)`/`(q)` suffixes and the
`·plugin` marker mean — instead of only listing keys.

### A new logo

spektr has a proper mark, drawn by **Roshan (RRDOJ)**: a terminal prompt, `>`
over `_`, built out of spectrum bars with the gradient running cyan through
violet to red the way the frequency range does. The old icon was the Bars mode
in a rounded square, which said "audio meter" and nothing about where it runs.

It ships in the README, the Windows `.ico`, and a full Android adaptive icon
set — every density, plus the monochrome layer Android 13 themes the launcher
with.

### Five modes

`Swell` and `Terra` render a height field rather than a bar chart: the band
plan becomes a landscape and the music moves it. `Constellations`, `Star
Trails` and `Supernova` join the cosmos family — one draws a figure that grows
with the beat, one is a long exposure of a turning sky, and one spends its
whole budget on a single event and waits for it.

`Shooting Star` was reworked at the same time: fragments enter at the edge of
the display and cross it rather than appearing partway through their own run,
and how near a meteor is now decides its speed, tail and brightness together.

### Fixes

**Star Trails** was flooding the screen white — three compounding errors, each
enough on its own. The motion blur weighted four neighbours at half each, so
the kernel summed to more than one and a single lit dot filled the field in
ten seconds with no stars drawn at all. The exposure length was fixed while
the spin was not, so on ordinary percussive material every star swept a
110-degree arc and they fused into a disc. And the blur was applied per frame
rather than per second, so the same passage came out twice as dense at 240 fps
as at 24.

**Maelstrom** had the same per-frame bug in its dissipation, left behind when
its forcing terms were converted: the dye kept 93% of itself per second at 24
fps against 49% at 240, so the smoke lingered nearly twice as long on a slow
display.

**Band columns** got an adaptive gutter, so a wide terminal no longer smears
neighbouring bars into each other.

### Motion profiles

The settings panel has a new `motion` row. `snappy` is the tuning everything
was calibrated against and stays the default; `glide` is the slower,
cava-like character — bars rise lazily, sink for most of a second, and energy
leans into neighbouring bars.

None of it touches the analysis: the band plan, the EQ tilt, autosens and the
onset detector are upstream of the switch and identical under both. It is
dt-correct throughout, deliberately not a port of cava's gravity filters,
which are frame-rate-dependent by construction.

### Android

The APK carries its own version number — still **v0.2.0**, because the port
has had two versions. Putting 0.4.5 on it would claim five versions of
something that has had two.
