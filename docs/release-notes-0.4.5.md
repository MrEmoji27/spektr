<!--
The body of the GitHub release for v0.4.5, as published.

It has to exist before the tag is pushed: all three build workflows attach
with append_body, so they add a download section to whatever is already there
and never write the notes themselves.

    gh release create v0.4.5 --draft --title "spektr 0.4.5" \
      --notes-file docs/release-notes-0.4.5.md
-->

spektr now lets you choose which visualisers you actually want, has five new
ones, four 3D scenes on Android, and a logo.

**52 modes → 57.** Themes stay at 55.

### Pick the ones you like — press `l`

Fifty-seven modes is a lot to press `m` through when you only love four of
them. `l` opens a list of every mode with a tickbox next to each. Tick the
ones you want; from then on those are the only ones the app offers you — the
picker, the `m` key and shuffle all follow it.

Press `s` in that list to give your selection a name and keep it. Named sets
show up as `★` rows at the top; `space` loads one back, `d` deletes it. They
remember modes only, so loading one never changes your theme or your settings.

If you tick everything, nothing is restricted — which is how it starts, so
your setup works exactly as it did until you narrow it yourself.

This replaces the old preset keys. A preset used to save your mode, theme,
frame rate, bands, sensitivity and gate all together, which meant loading one
quietly changed four things you hadn't asked it to. Your old `presets.json`
is left alone on disk, just no longer used.

Press `h` for help — it was rewritten to say what things *are*, not just list
keys.

### A logo, by Roshan (RRDOJ)

spektr has a proper mark now, drawn by **Roshan (RRDOJ)**: a terminal prompt —
`>` above `_` — built out of spectrum bars, coloured cyan through violet to
red the way the frequency range runs.

It replaces the old icon everywhere, including a full set of Android launcher
icons that theme themselves with your wallpaper.

### Five new modes

- **Swell** and **Terra** draw a landscape instead of a bar chart, and the
  music moves the terrain.
- **Star Trails** is a long exposure of a night sky turning overhead.
- **Constellations** joins stars into a figure that grows as the track goes on.
- **Supernova** waits, mostly dark, and spends everything on one hit.

**Shooting Star** was reworked too. Meteors now come in from the edge of the
screen and cross it, instead of appearing halfway through their own flight.

### Android: four 3D worlds

The Android build has four new scenes — **Metaball**, **Wormhole**,
**Monolith** and **Lattice** — real 3D, drawn by the GPU, filling the whole
screen. They replace the old terrain view, which only ever managed a strip of
landscape floating in a black frame.

They also run far lighter than what came before, and every one of spektr's
themes colours them.

The APK has its own version number, **v0.3.0**, because the port has had three
versions and the desktop app has had more. It ships inside this release.

### Fixes

- **Star Trails** used to fill the screen with solid white after a few
  seconds. Three separate bugs, all pulling the same way. Fixed.
- **Maelstrom** looked much thicker on a slow display than a fast one. Now it
  looks the same on both.
- The bars no longer smear into each other on a very wide terminal.

### A slower feel, if you want one

Settings (`c`) has a new **motion** row. `snappy` is what spektr has always
done and stays the default. `glide` is lazier — bars rise slowly, fall for
about a second, and lean into their neighbours. It only changes how the
picture moves, never what spektr hears.

It works the same way on Android.
