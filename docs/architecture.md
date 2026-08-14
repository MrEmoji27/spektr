# Architecture — the engine/frontend boundary

spektr is two layers with one contract between them. Nothing needs to move
today; this file exists so the next feature knows which side it is on before
it starts, because a feature that reaches across the boundary is what makes a
second renderer expensive later.

## The split

**Engine** — everything that produces or consumes `(codes, cidx)` arrays:

- `analysis.py` — FFT, band plan, sensitivity loop. Pure numpy, fed by the
  capture ring buffer.
- `modes/` — the visualisers. Each is `fn(ctx) -> (codes, cidx)`. Plugin
  modes register through the same decorator and obey the same shape
  (`api.py`, `plugins.py`), so the plugin loader is engine-side too.
- `palette.py` — theme model, ramp maths, and the float→ramp-index lookup
  (`Palette.index`/`indices`). Its `rgb`/`hexes` tables are pure numpy; the
  `colors`/`styles`/`bg_styles`/`pair_styles` tables build Rich objects and
  exist only for the frontend's strip pipeline.
- `render.py`, array half — `pack_braille`, `cell_max`, `cell_mean`, `noise`,
  `frac`, `blocks_from_levels`, `row_gradient`, `broadcast_rows`, `blank`:
  pure-numpy grid transforms. The other half of the file — `make_strips` and
  the `_rle_*` merges — is frontend: it turns the pair into Rich `Segment`s
  and Textual `Strip`s, and `widget.py` is its only runtime caller.
- `asciiart.py` — frame decoding for the Flipbook mode. Data in, arrays out.

**Frontend** — one Textual screen over the engine:

- `widget.py` — drives the analyser and the active mode each frame, feeds the
  output to `make_strips`, and owns the live palette/theme state.
- `app.py`, `pickers.py` — the CLI, keybindings, overlays, and the persisted
  state (config, presets, themes, plugins, ascii reels) through the config
  directory.
- `capture.py`, `motion.py`, `display.py`, `nowplaying.py` — platform
  services the widget composes; they are not part of the portable engine.

The dependency arrow points frontend → engine, and never the other way.

## The contract

A mode returns `(codes, cidx)`: a `(h, w)` grid of Unicode codepoints and a
matching grid of palette ramp indices (`0..RAMP_STEPS-1`, cool to hot).
Optionally a third `(h, w)` array of background ramp indices, for modes that
colour whole cells. That pair is the whole interface — verified against the
code, not taken on faith:

- **Modes never import Rich or Textual.** There is no `rich`/`textual`
  import anywhere under `modes/`; the only cross-module imports are
  `analysis.resample_bands`, palette constants, the array half of `render`,
  and (for Flipbook) `asciiart`.
- **Modes never see a colour value.** Colour is reached only through
  `ctx.ramp(...)` and `ctx.palette.index(...)`, which return ramp indices. No
  mode reads a theme's hex strings, even though the `Palette` object on
  `ctx` technically exposes them.
- **Modes hand back integer arrays, not strings.** One mode (Readout,
  `modes/spectrum.py`) assembles its digit tape as a Python string before
  encoding it to codepoints — the boundary is the *return type*, int arrays,
  not what happens inside a mode.

That is what makes the engine reusable: any renderer that can draw a grid of
codepoints coloured by ramp index can consume it. The Android port is one
such renderer; `tests/bench.py` is another, driving modes headlessly with no
terminal in sight.

## Reaching across the boundary

The mistake to recognise before making it:

- A mode importing `rich`/`textual`, building `Segment`/`Style` objects, or
  returning strings instead of int arrays — it no longer runs on any renderer
  but the Textual one.
- A mode choosing its own colours from `palette.hexes` or a theme's fields —
  it now depends on the theme model rather than the ramp contract.
- A mode calling `make_strips` — turning the pair into Strips is the
  frontend's job.
- `analysis.py`/`modes/` importing `widget`, `app`, `pickers`, or capture
  internals — the dependency arrow must keep pointing frontend → engine.
- A plugin doing any of the above. `plugins.validate` enforces the return
  shape at load; nothing enforces the imports, so review is the gate.

If a change wants to live in the engine, it must survive being driven by a
headless loop that calls `mode(ctx)` and reads the arrays back. That loop is
what the port will be.