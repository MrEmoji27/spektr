# Writing a spektr visualizer

A plugin is a Python file that registers a render mode. It appears in the `v`
picker alongside the built-ins and is indistinguishable from them — they use the
same decorator and the same contract.

```
~/.config/spektr/plugins/heartbeat.py     →  mode "Heartbeat"
```

On Windows that's `%APPDATA%\spektr\plugins\`. `spektr plugins path` prints it.

## Trust

**Plugins are Python and run with your privileges.** spektr cannot sandbox
them — a Python process can't meaningfully contain arbitrary Python. Rather than
pretend otherwise, the trust decision is made explicit:

```console
$ spektr plugins list
  heartbeat   untrusted  —

$ spektr plugins trust heartbeat

  plugin  heartbeat
  source  /home/you/.config/spektr/plugins/heartbeat.py
  sha256  9f2c1a04b7e3...e40b
  size    41 lines

  This is Python. It runs with your privileges — it can read your files
  and reach the network. spektr cannot sandbox it. Read it first.

  Trust this plugin? [y/N] y
  trusted heartbeat (9f2c1a04b7e3…)
```

The approval records a SHA-256 in `plugins/.trust.json`. Editing the file
changes its hash and disables it until you approve again. Read plugins before
you trust them — the same care you'd give any script you downloaded.

`--yes` skips the prompt for scripted installs. Use it only when you've reviewed
the same contents another way.

## The contract

```python
from spektr.api import mode

@mode("Heartbeat", blurb="pulses on the kick drum")
def heartbeat(ctx):
    ...
    return codes, cidx
```

Return two `(h, w)` integer arrays:

| | |
|---|---|
| `codes` | Unicode codepoints — what character goes in each cell |
| `cidx` | index into the active palette ramp, `0` (cool) to `RAMP_STEPS-1` (hot) |

Optionally a third array of background ramp indices, if you want to colour whole
cells — see [Plasma](#two-colours-per-cell) below.

The decorator takes three optional arguments: `group` (which family the mode
belongs to), `blurb` (one line, shown while the picker previews it), and
`after`, which places your mode directly behind a named one in the picker
instead of at the end:

```python
@mode("Heartbeat Extreme", after="Heartbeat", blurb="the same, with no brakes")
```

A variant belongs next to what it varies. Naming a mode that does not exist is
not an error — it just appends, so a plugin keeps loading if a later spektr
renames the built-in it pointed at.

You never see a colour value. The user picked a theme; you say how *hot* each
cell is and spektr resolves it. That's why every plugin works with all thirty
themes for free.

Declare the API version you were written against, and spektr will refuse to run
your plugin against an incompatible build rather than failing obscurely:

```python
SPEKTR_API = 1
```

## What's in `ctx`

| | |
|---|---|
| `ctx.w`, `ctx.h` | terminal cells available |
| `ctx.dot_rows`, `ctx.dot_cols` | braille sub-cell grid — `h*4` by `w*2` |
| `ctx.bands` | smoothed spectrum, each `0..1` |
| `ctx.n_bands` | how many. **Don't assume 32** |
| `ctx.peaks` | peak-hold markers, same length |
| `ctx.bands_l`, `ctx.bands_r` | per-channel |
| `ctx.wave` | smoothed mono trace, roughly `-1..1` |
| `ctx.stereo` | raw `(N, 2)` L/R sample pairs |
| `ctx.t` | seconds since start |
| `ctx.dt` | seconds since the last frame |
| `ctx.frame` | monotonic counter |
| `ctx.energy` | mean band level |
| `ctx.silent` | True when the noise gate is shut |
| `ctx.palette` | active palette, for `.index(0..1)` |

Helpers:

| | |
|---|---|
| `ctx.range(lo, hi)` | mean level across a slice given as fractions — `ctx.range(0, 0.15)` is the kick |
| `ctx.display_bands(n)` | resample to exactly `n` bands |
| `ctx.ramp(field)` | float field `0..1` → ramp indices, vectorised |
| `ctx.scratch(key, factory)` | per-mode state that survives frames, cleared on resize |

**Write resolution-independent code.** Use `ctx.range()` and
`ctx.display_bands(n)` rather than slicing `ctx.bands` at fixed indices. The
internal band count is not promised to stay at 32, and plugins that hard-code it
will break when it changes.

**Don't write into the arrays you're given** — they're the live smoothing
buffers. Copy first if you need to modify.

## Toolkit

From `spektr.api`:

| | |
|---|---|
| `pack_braille(dots)` | `(h*4, w*2)` bool → `(h, w)` braille codepoints |
| `cell_max(field)`, `cell_mean(field)` | reduce a dot-resolution field to one value per cell |
| `noise(shape, seed)` | cheap deterministic hash noise in `[0, 1)` |
| `blocks_from_levels(levels, h)` | column heights `0..1` → partial block characters |
| `resample_bands(bands, n)` | area-averaged resample |
| `band_columns(w, n)` | column → band map with gutters |
| `spread(levels, w)` | interpolate band levels across every column |
| `polar_grid(ctx)` | `(dist, turn, max_r)` over the dot grid, aspect-corrected and cached per size |
| `angular_bands(ctx, turn, n, spin)` | every dot's angle → its band level, blended, one gather |
| `empty(w, h)` | a blank `(codes, cidx)` pair |
| `BLOCKS_UP`, `BLOCKS_LEFT`, `SHADES`, `SPACE` | character sets |

### Rhythm

`ctx` carries the raw analyser fields — `onsets`, `onset_strength`, `flux`,
`tempo_bpm`, `beat_phase` — and two derived ones that are usually what you
want:

| | |
|---|---|
| `ctx.pulse` | `0..1` beat-locked swell, `1.0` on the beat, decaying through the bar |
| `ctx.drive` | `0..1` how percussive the signal is right now |

Prefer these to the raw fields unless you need something they don't express.
`beat_phase` is `0.0` whenever `tempo_bpm` is `0.0`, and `0.0` is *on the
beat*, so the obvious `1 - ctx.beat_phase` reads as a permanent
full-strength swell through silence and the opening seconds of every track.
`ctx.pulse` has that gate built in and returns `0.0` until there is a tempo.

Use `ctx.pulse` for swells and `ctx.drive` for rates — how fast something
scrolls, spins or spawns. `ctx.onsets` is still the right call for discrete
events; it is the count for *this frame*, so never difference `onset_seq`
yourself (scratch survives a mode switch, and a private counter replays every
beat that played while your mode was not drawing).

## Example: a dot-grid mode

```python
# ~/.config/spektr/plugins/nightrider.py
"""Scanning red eye, KITT-style, swept by the beat."""

import numpy as np
from spektr.api import mode, pack_braille, cell_max

SPEKTR_API = 1


@mode("Nightrider", group="scenes", blurb="scanning eye, swept by the beat")
def nightrider(ctx):
    dr, dc = ctx.dot_rows, ctx.dot_cols

    # sweep speed rides the low end, so it lunges on the kick
    speed = 0.5 + ctx.range(0.0, 0.15) * 2.5
    pos = (np.sin(ctx.t * speed) * 0.5 + 0.5) * (dc - 1)

    x = np.arange(dc)[None, :]
    y = np.arange(dr)[:, None]

    # a horizontal bar, brightest at the head and trailing off behind it
    thickness = dr * (0.12 + ctx.energy * 0.25)
    centre = (dr - 1) / 2.0
    band = np.abs(y - centre) < thickness

    glow = np.clip(1.0 - np.abs(x - pos) / (dc * 0.18), 0.0, 1.0)
    field = np.where(band, glow ** 1.6, 0.0)

    dots = field > 0.10
    return pack_braille(dots), ctx.ramp(cell_max(field))
```

## Example: keeping state between frames

Anything you need across frames goes in `ctx.scratch()`. It's keyed per mode and
thrown away on resize, so you never have to check whether your buffer still
matches the terminal.

```python
# ~/.config/spektr/plugins/embers.py
"""Falling sparks that pile up and fade."""

import numpy as np
from spektr.api import mode, pack_braille, cell_max, noise

SPEKTR_API = 1


@mode("Embers", group="particles", blurb="sparks that fall and fade")
def embers(ctx):
    dr, dc = ctx.dot_rows, ctx.dot_cols
    field = ctx.scratch("embers", lambda: np.zeros((dr, dc), dtype=np.float32))

    # decay in real time, so the trail length doesn't change with frame rate
    field *= float(np.exp(-ctx.dt / 0.55))

    # everything drifts down one dot row
    field[1:] = np.maximum(field[1:], field[:-1] * 0.85)
    field[0] = 0.0

    # seed new sparks along the top, denser where the spectrum is loud
    if not ctx.silent:
        from spektr.api import spread
        density = spread(ctx.display_bands(), dc) ** 2
        born = noise((1, dc), ctx.frame) < density * 0.35
        field[0] = np.where(born[0], 1.0, field[0])

    np.clip(field, 0.0, 1.0, out=field)
    return pack_braille(field > 0.08), ctx.ramp(cell_max(field))
```

## Two colours per cell

Returning a third array colours the background as well. Combined with `▀`, that
gives you two independently coloured half-cells and doubles your vertical
resolution:

```python
@mode("Gradient", group="fields")
def gradient(ctx):
    rows2 = ctx.h * 2
    field = np.linspace(0.0, 1.0, rows2)[:, None] * np.ones(ctx.w)[None, :]
    idx = ctx.ramp(field * ctx.energy)
    codes = np.full((ctx.h, ctx.w), ord("▀"), dtype=np.int32)
    return codes, idx[0::2], idx[1::2]
```

## Performance

You have a **16.7 ms** frame budget at 60 fps, and every built-in mode fits in
under **7 ms** at 240×60. Anything slower than **11 ms** has its previous frame
reused on alternate ticks — the visual keeps moving, at half the sampling rate,
rather than stuttering the whole UI.

Two things dominate, and both have a fast path already written for you.
``pack_braille`` and ``cell_max`` are the per-dot reductions; use them rather
than rolling your own, because the obvious ``reshape(h, 4, w, 2).max(axis=(1,3))``
reduces over non-contiguous axes and is **16x slower** than the strided version
in ``spektr.render``. And colour: ``make_strips`` run-length encodes each row,
so a smooth field costs a handful of segments while a per-cell-random one costs
one segment per cell — a 20x difference at 240×60. Smooth fields are cheap;
noise fields are not.

The way to stay fast is to express a frame as whole-array numpy operations. A
loop over dots is roughly a hundred times slower than the equivalent broadcast:
the two built-in modes that still had Python loops cost 26 and 30 ms per frame,
and vectorising them brought both under 5 ms.

Check yours with:

```console
$ python tests/bench.py
```

## When it goes wrong

A mode that raises is retried twice, then quarantined — removed from the picker,
with the app carrying on. You'll get a notification, and:

```console
$ spektr plugins doctor

heartbeat  [error]
  path    /home/you/.config/spektr/plugins/heartbeat.py
  sha256  9f2c1a04b7e3d5f1…  (41 lines)
  modes   Heartbeat
  error:
    Traceback (most recent call last):
      File ".../heartbeat.py", line 22, in heartbeat
        return codes, cidx
    ValueError: cidx has shape (30, 119), expected (30, 120)
```

Plugin output is shape-checked and range-clipped before it reaches the renderer,
so a mistake here names your plugin instead of crashing somewhere unrelated.

Run with `--no-plugins` to start clean if something is badly misbehaving.

## Distribution

There's no remote installer yet. Share a `.py` file; people drop it in their
plugins folder and run `spektr plugins trust <name>`. If plugins take off, the
plan is cliamp's convention — a `spektr-plugin-<name>` repo with `<name>.py` at
the root, installable with `spektr plugins install user/repo`.

## Command reference

```
spektr plugins list             what's installed, and whether it's trusted
spektr plugins trust <name>     review and approve
spektr plugins untrust <name>   revoke approval
spektr plugins remove <name>    delete from disk
spektr plugins doctor           full diagnostics for every plugin
spektr plugins path             print the plugins folder
spektr --no-plugins             skip loading them this run
```
