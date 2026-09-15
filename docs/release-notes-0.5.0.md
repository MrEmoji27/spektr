<!--
The body of the GitHub release for v0.5.0.

It has to exist before the tag is pushed: all three build workflows attach
with append_body, so they add a download section to whatever is already there
and never write the notes themselves.

    gh release create v0.5.0 --draft --title "spektr 0.5.0" \
      --notes-file docs/release-notes-0.5.0.md
-->

spektr 0.5.0 adds a spinning tunnel and a family of car-audio meters, draws every
theme in its real colours, and — if you want it — lets your terminal's
transparency show through again.

**57 modes → 61.** Themes stay at 55.

### Crosscurrent — a tunnel going both ways

Round rings rush out toward you while eight-sided rings are drawn in toward
the centre, in the same tunnel at the same time. Where the two meet, the ring
flares — a spark that slides around it.

The whole tunnel spins, and the music drives the spin: it turns faster with
the tempo and speeds up when the track gets busy, then coasts back down when
it calms. When a part of the spectrum jumps, the spokes in that part flicker
on the beat — the lines really break into dashes and back, not just change
colour — and a hard hit sends that flicker sweeping around the tunnel.

Bass pushes the outward rings, treble pulls the inward ones. In silence it
slows to a gentle drift.

### Three JP meters

A new family inspired by Japanese car audio hardware: the segmented LED level
meters on in-dash head units and equalisers, with every LED drawn — lit or
not.

- **JP Bars** — a segmented meter whose tops break off and float up as
  the bars rise.
- **JP Drift** — the meter as a sandpile. Bulbs pile up, and on the beat
  the tall columns collapse and pour into their neighbours.
- **JP Pulse** — the meter bent into a dial, with a light that chases
  around it on the beat.

All three go dark when the music stops.

> **Note:** the JP modes are new and not final. They may be changed, improved,
> reworked or removed, and more modes will join this family in the next major
> release.

### Every theme in its real colours

In 0.4.5 the empty space in the visualizer showed your terminal's own
background instead of the theme's — so gruvbox sat on whatever colour your
terminal uses, and a light theme like flexoki-light could end up on a dark
screen. Now the theme's background fills the whole picture, in any terminal.

### Want your see-through terminal back? Settings → background

That fix makes a transparent terminal look solid. If you liked seeing your
desktop through spektr, open settings (`c`) and set **background** to
`terminal (see-through)`, or start spektr with `--background terminal`. It
switches instantly and is remembered.

It is off by default for a reason: a light theme over a dark terminal will
look pale, because the theme no longer decides what is behind the lines. Modes
that fill the screen with colour, like Plasma or Chladni, keep their colours
and only their empty areas turn see-through. The header and footer stay solid
either way.

### Tunnel and Tunnel In, cleaned up

The tunnels are now drawn as clean, even lines. The spokes run straight into
the centre and fade out as they meet, instead of bending, breaking into dots,
or leaving a messy blob in the middle. The rings stay thin lines even when
they are loud. They also run faster.

### A darker, calmer night sky

The four cosmos modes were reworked so the sky stays dark and the music shows
up as events.

- **Shooting Star** throws meteors on hard hits, not on every hi-hat, and
  nothing flies across a quiet part or after the music stops.
- **Constellations** draws neat figures of a few lines instead of tangled
  knots, and old figures fade sooner.
- **Star Trails** no longer fills the screen with arcs.
- **Supernova** saves its explosion for a real hit instead of going off on a
  timer at the start of a song.

Faint stars and fading trails now actually look faint on every theme.

### No more torn frames on Windows

On Windows Terminal, spektr could sometimes show the top half of one frame
with the bottom half of the next, which looked like lines breaking. spektr now
asks the terminal to show each frame only once it is complete, so that is
gone.

### Android

The APK is now **v0.4.0**. It carries the new engine, so **Crosscurrent** and
the three **JP** meters are in the picker, and the tunnels and night-sky
modes have their new look. (The see-through background and the torn-frame fix
are terminal features, so they do not apply to the app.)

The APK keeps its own version number because the Android port has had fewer
releases than the desktop app.

### For plugin authors

If your mode returns a background colour array, background index `0` now
means "nothing here": with the see-through background on, those cells are left
to the terminal. Every other index keeps its colour. See
[docs/plugins.md](https://github.com/MrEmoji27/spektr/blob/main/docs/plugins.md).

[Full changelog](https://github.com/MrEmoji27/spektr/blob/main/CHANGELOG.md)

---

### Which file

| you have | download |
|---|---|
| Windows, no Python | `spektr.exe` — portable, double-click |
| Windows, one line in PowerShell | `irm https://github.com/MrEmoji27/spektr/releases/latest/download/spektr.exe -OutFile spektr.exe; ./spektr.exe` |
| Windows, want a Start Menu entry | `spektr-0.5.0.0-setup.exe` |
| Linux, no Python | `spektr` — `chmod +x` and run |
| Android | `spektr-android-0.4.0-arm64-v8a.apk` — Android 10+, 64-bit ARM |
| Python already | clone and `pip install -e .` — not on PyPI yet |

The Windows and Linux builds are unsigned. SmartScreen will warn on first run:
More info → Run anyway.
