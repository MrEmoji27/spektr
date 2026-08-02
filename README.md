<div align="center">

# spektr

**A terminal spectrum analyser for whatever your speakers are doing.**

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-00c853)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20·%20Linux%20·%20macOS-546e7a)](#how-it-captures-audio)
[![Modes](https://img.shields.io/badge/render%20modes-27-ff6d00)](#modes)
[![Themes](https://img.shields.io/badge/themes-30-7c4dff)](#themes)
[![Built with Textual](https://img.shields.io/badge/built%20with-Textual-5e35b1)](https://textual.textualize.io/)

</div>

```
                                        ──────
───────                         ─────── ▅▅▅▅▅▅
▇▇▇▇▇▇▇                         ███████ ██████
███████                         ███████ ██████
███████                         ███████ ██████ ▄▄▄▄▄▄
███████ ───────         ─────── ███████ ██████ ██████
███████ ▆▆▆▆▆▆▆         ▆▆▆▆▆▆▆ ███████ ██████ ██████
███████ ███████         ███████ ███████ ██████ ██████ ▂▂▂▂▂▂
███████ ███████ ▄▄▄▄▄▄▄ ███████ ███████ ██████ ██████ ██████ ▁▁▁▁▁▁ ▅▅▅▅▅▅
███████ ███████ ███████ ███████ ███████ ██████ ██████ ██████ ██████ ██████
███████ ███████ ███████ ███████ ███████ ██████ ██████ ██████ ██████ ██████
███████ ███████ ███████ ███████ ███████ ██████ ██████ ██████ ██████ ██████
```


Point it at nothing. Play music anywhere — Spotify, a browser tab, a game, a call — and
spektr draws it: an overlapped 2048-point FFT across 32 log-spaced bands from 20 Hz to
20 kHz, rendered with braille sub-characters so the picture moves at four times the
vertical resolution of a text cell.

**Twenty-seven render modes. Thirty themes. Locked 60 fps.**

```bash
pip install spektr
spektr
```

Or from source:

```bash
git clone https://github.com/MrEmoji27/spektr
cd spektr
pip install -e .
spektr
```

No configuration, no file to point it at, no music service to log into. It finds your
output device, taps it, and draws.

---

**Contents** · [Gallery](#gallery) · [Modes](#modes) · [Themes](#themes) ·
[Plugins](#plugins) · [Keys](#keys) · [Command line](#command-line) ·
[Audio capture](#how-it-captures-audio) · [How it works](#how-it-works) ·
[Development](#development)

---

## Gallery

Snapshots below are the real render path with the colour stripped out — in the terminal
every cell carries one of 64 gradient steps from the active theme.

<details open>
<summary><b>Flame</b> — fire, licking upward from each band</summary>

```
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣤⡀⠀⠀⠀⠀⠀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣶⡆⠀⠀⠀⠀⠀⢠⣤⣤
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣾⣿⡇⠀⠀⠀⠀⠠⣿⣿⠆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣹⣿⣯⠀⠀⠀⠀⠀⣼⣿⡯
⠀⠀⢀⣀⠀⠀⠀⠀⠀⣶⣿⣿⠆⠀⠀⠀⠀⢘⣿⣿⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣤⡄⠀⠀⠀⣿⣿⣷⡀⠀⠀⠀⠰⣿⣿⣗
⠀⠀⣸⣿⣧⡀⠀⠀⠸⣾⣿⣿⡃⠀⠀⠀⠀⢸⣿⣿⣧⠀⠀⠀⢀⣶⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⣿⡟⠀⠀⠀⣹⣿⣿⣯⠀⠀⠀⣾⣿⣿⡯⠀⠀⠀⠀⠀⢰⣶⣶
⠀⠀⣻⣿⣿⡧⠀⠀⣺⣿⣿⣿⢆⠀⠀⠀⠀⣿⣿⣿⣟⡄⠀⠀⢈⣿⣿⡧⠀⠀⠀⢀⣀⣀⠀⠀⠀⠀⠀⣜⣿⣿⡧⠀⠀⠀⣹⣿⣿⣿⣧⠀⢨⣿⣿⣿⣷⠀⠀⠀⠀⠀⣿⣿⣇
⠀⠀⢿⣿⣿⣿⡄⠀⣿⣿⣿⣿⡾⠀⠀⠀⣰⣿⣿⣿⡏⡄⠀⠀⠸⣿⣿⣷⢆⠀⠀⢿⣿⣷⡄⠀⠀⠀⠐⣽⣿⣿⣯⡄⠀⠀⣺⣿⣿⣿⣷⡄⣏⣿⣿⣿⡯⡇⠀⠀⢀⣸⣿⣿⡏⡄
⠀⠐⣻⣿⣿⣿⣾⢸⣹⣿⣿⣿⣧⡇⠀⡰⣼⣿⣿⣿⣷⡃⠀⠀⣫⣿⣿⣿⡟⡄⢘⣿⣿⣿⣷⠀⠀⠠⣶⣿⣿⣿⡟⡃⠀⠀⣽⣿⣿⣿⣿⠠⡹⣿⣿⣿⣿⡇⠀⠀⣴⣿⣿⣿⣷⠃⠀⠀⠀⠀⣶⣶⡄
⠀⢰⣽⣿⣿⣿⡇⣇⢿⣿⣿⣿⣿⣲⡀⣫⣿⣿⣿⣿⣯⡁⠀⢸⢾⣿⣿⣿⣷⢦⢨⣿⣿⣿⣷⣧⠀⣚⣿⣿⣿⣿⣿⡁⠀⢰⣼⣿⣿⣿⣿⡷⡧⣿⣿⣿⣿⣷⡂⢘⣿⣿⣿⣿⣿⡅⠀⠀⠀⣻⣿⣿⣇⡀
⢀⣾⣿⣿⣿⣿⣿⣗⣿⣿⣿⣿⣿⣿⣥⣼⣿⣿⣿⣿⡇⠆⢀⡗⣿⣿⣿⣿⣿⢚⡹⣿⣿⣿⣿⣷⣃⣽⣿⣿⣿⣿⣧⡇⠀⢎⣿⣿⣿⣿⣿⣓⣏⣿⣿⣿⣿⣿⡽⢨⣿⣿⣿⣿⣇⡆⠀⣀⣹⣿⣿⣿⣿⣎
```

</details>

<details>
<summary><b>Arcs</b> — hollow rings, one per band, pushed out by level</summary>

```
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣠⣤⣤⠤⠶⠶⠶⠒⠒⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠓⠒⠲⠶⠶⠦⢤⣤⣤⣀⣀⡀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣤⡴⠶⠛⠛⠉⠉⠁⠀⠀⠀⢀⣀⣀⣠⣤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⢤⣤⣀⣀⣀⡀⠀⠀⠀⠉⠉⠙⠛⠳⠶⣤⣄⣀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⣶⠾⠛⠉⠁⠀⠀⠀⣀⣠⡤⠶⠒⠛⠉⠉⠁⠀⠀⢀⣀⣀⣀⣀⡀⠀⠀⠀⣀⣀⣀⣀⣀⠀⠀⠀⠉⠉⠙⠓⠲⠦⣤⣀⡀⠀⠀⠀⠉⠙⠻⢶⣦⣄
⠀⠀⠀⠀⠀⠀⠀⣠⣾⡿⠋⠀⠀⠀⠀⢀⣤⡶⠛⠉⠀⠀⠀⢀⣠⠤⠖⢒⣉⡭⠥⠴⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠲⠤⠭⣍⣑⠒⠦⢤⣀⠀⠀⠀⠈⠙⠳⣦⣄⠀⠀⠀⠀⠈⠻⣿⣦⡀
⠀⠀⠀⠀⠀⢀⣾⣿⠋⠀⠀⠀⠀⢀⣴⡿⠁⠀⠀⠀⠀⣤⠞⠋⣠⡶⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠳⣦⡈⠛⢦⡄⠀⠀⠀⠀⠹⣷⣄⠀⠀⠀⠀⠈⢻⣿⣆
⠀⠀⠀⠀⠀⢸⣿⡇⠀⠀⠀⠀⠀⣾⣿⠀⠀⠀⠀⠀⢼⡏⠀⢼⣟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢘⣿⡆⠈⣿⠄⠀⠀⠀⠀⢸⣿⡆⠀⠀⠀⠀⠀⣿⣿
⠀⠀⠀⠀⠀⠘⣿⣷⡀⠀⠀⠀⠀⠘⢿⣦⠀⠀⠀⠀⠈⠿⣄⡈⠻⣦⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⡾⠋⣀⡼⠏⠀⠀⠀⠀⢠⣾⠟⠀⠀⠀⠀⠀⣰⣿⡟
⠀⠀⠀⠀⠀⠀⠈⠻⣿⣦⡀⠀⠀⠀⠀⠙⠷⣦⣀⠀⠀⠀⠈⠙⠲⠤⣍⣙⠒⠦⠤⢤⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣠⠤⠤⠖⢚⣉⡥⠴⠚⠉⠀⠀⠀⢀⣠⡶⠟⠁⠀⠀⠀⠀⣠⣾⡿⠋
⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠿⣶⣤⣀⠀⠀⠀⠀⠉⠛⠲⠦⣤⣀⣀⠀⠀⠀⠉⠉⠑⠒⠒⠒⠒⠒⠀⠀⠐⠒⠒⠒⠒⠒⠉⠉⠁⠀⠀⢀⣀⣠⡤⠶⠚⠋⠁⠀⠀⠀⢀⣠⣴⡾⠟⠉
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠛⠷⢦⣤⣀⣀⠀⠀⠀⠀⠉⠉⠛⠓⠒⠲⠶⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠴⠶⠒⠒⠛⠋⠉⠁⠀⠀⠀⢀⣀⣠⣤⠶⠟⠛⠉
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠛⠛⠳⠶⠶⠤⣤⣤⣤⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣠⣤⣤⡤⠴⠶⠶⠛⠛⠋⠉⠁
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠁
```

</details>

<details>
<summary><b>Skyline</b> — a city at night, windows lit by their band</summary>

```
                        ╹╹╹╹╹╹╹ ╹╹╹╹╹╹╹
                        ███████ ███████
                        ████▀██ █████▀█
                        ███████ ███████
                        ███████ ███████
                ███████ ███████ ███████ ██████
                ▀██▀███ █▀██▀██ ██▀████ ███▀██
███████         ███████ ███████ ███████ ██████ ██████ ██████ ██████
████▀██ ███████ ███████ █▀█████ █████▀█ ▀█████ ██████ █▀████ ██████ ██████
███████ ███████ ███████ ███████ ███████ ██████ ██████ ██████ ██████ ██████
███████ ███████ ███▀██▀ █▀█████ ██▀██▀█ ███▀██ █████▀ ████▀█ ██████ ██▀███
███████ ███████ ███████ ███████ ███████ ██████ ██████ ██████ ██████ ██████
```

</details>

<details>
<summary><b>Matrix</b> — digital rain, falling faster when it's loud</summary>

```
   ﾆ    ﾆ      ﾍ   *    ｶ      ｽ            ﾎ 6 ﾏ    4   7         ﾒ 8ｳ94>
   ﾀ         ｾ2ﾏ        3      ﾕ  1         ﾜ ｳ ﾁ  ﾃ ﾉ   ﾂ        ﾐ3 >ﾎﾁｸﾐ
   =     2<  ﾜｴﾊ    ｾ ﾍ           ﾕ         < ﾒｹﾈ ﾗﾑ ﾄ ﾗ ｻ        ﾛ3ｾﾆﾓ|ｶﾗ
   ﾒ     ﾋﾁ  ﾝﾚﾄ    ﾆ ｼ           8ﾝ        ﾓ ﾄ1ｳ ﾆﾚ ﾘ ﾈ ｹ        ﾑ ﾍ ﾒ6ｿﾍ
   ﾈ     ﾉﾖ  ｻﾚﾁ    ｳ4ﾀ           ﾌﾜ 1      4 ﾆﾐﾓ ﾎ| ｺ ﾖ        ﾊｾ6 ﾃ 6> ﾐ
       ﾅ 6=  ﾌ>5    ﾒ5ｾ   ｾ       ｵｿ |      ｲ ﾙｿﾊ ｼﾆ ﾒ 8  >     ﾊﾛｷ ﾐ ﾅﾃ 5
       ﾔ ﾎ   ﾁ 4    ﾊﾌｳ   3       69 ｿ     <ﾁ ｷﾎ5 8ﾎ   ﾔ  ｺ     5ﾃｶ ｼ ｷﾒ ｿ
      ﾘｵ ﾙ   8 ﾚ  0 5ﾆ7   ﾈﾆ      ﾕ5 ﾏ     ﾆﾉ ﾝｸｴ 8ｵ      ﾚ     ｽﾙ  ﾍ  5 ﾝ
 =  ﾊｿﾈ  ﾐ   ｸ   ﾃﾁ ﾄﾖ6   ﾁﾗ         ｳ    7ﾚﾙ ｿﾚ3ﾇﾅｶﾈ     *     2ｷ  ﾅ  ｽ
 ﾍ  ﾕﾋﾋ  ﾙ  ﾁｻ   ｱｽ 6ﾋﾉ   ﾃﾅ          ﾎ  ||ﾗﾒ ｼﾜ ｱｹｷ2           4|  ﾇ
5ﾚ  ﾒｴ   ﾙ  ﾌ    ﾈﾂ ﾁ*9ｻ  ｸ9     ｽ    ｸ  ﾁﾚﾀ     82ﾔﾊ           ｾﾗ
ｸﾊﾂ ﾇﾀ   ﾊ  ﾙ    |ﾈ ﾃﾂｾ|  ﾓﾅﾁ   62    ｲ  ｸﾚﾖ     ﾜﾈ<ﾀ           ｲｻ
ﾝ ﾖ ｾ+   | ｱｾ    ﾆｲ  = 7  +ﾈ3ﾑ  ｺﾁ    ﾃ  ﾌﾚ4      ｶﾝﾗ ﾛ         ﾁｹ
```

</details>

## Modes

Press `v` for a filterable picker that previews each one live as you arrow through it.
Listed in the order the picker cycles them.

| | | | |
|---|---|---|---|
| **Bars** | the classic — bars with peak markers | **Bubbles** | bubbles from the low end, popping at the top |
| **Bricks** | chunky, no partial cells | **Radial** | the spectrum wrapped into a circle |
| **Columns** | gapless, interpolated across the full width | **Retro** | sunset grid, with the spectrum as the horizon |
| **Ladder** | segmented LED stack | **Auroras** | light curtains, billowing on the treble |
| **Mirror** | grows out from the centre line | **Skyline** | a city at night, windows lit by their band |
| **Stereo** | per-band L/R meters, mirrored from centre | **Tunnel** | flying down a pipe, ribbed by the beat |
| **Wave** | smoothed waveform | **Warp** | starfield, accelerating with the music |
| **Scope** | trigger-synced oscilloscope — the trace holds still | **Matrix** | digital rain, falling faster when it's loud |
| **ECG** | scrolling trace, like a heart monitor | **Spectro** | scrolling waterfall — frequency up, time across |
| **Strings** | plucked strings, bowed by their own band | **Plasma** | solid colour field, warped by the spectrum |
| **Gonio** | stereo phase scope with a phosphor trail | **Needle** | analogue VU — one sweeping needle, one red zone |
| **Scatter** | density sparkle, thicker where it's loud | **VU** | big L/R LED meters with peak hold |
| **Flame** | fire, licking upward from each band | **Arcs** | hollow rings, one per band, pushed out by level |
| **Pulse** | radial pulse with shockwaves | | |

A twenty-eighth entry, **None**, is registered as the off switch — it draws nothing.
That is why the test output counts 28 modes against the twenty-seven listed here.

## Themes

Thirty built in, previewed live from the `t` picker: `classic`, `gruvbox`, `catppuccin`
(+`-latte`), `dracula`, `nord`, `tokyo-night`, `rose-pine`, `everforest`, `kanagawa`,
`ayu-mirage`, `monokai`, `solarized`, `nightfox`, `oxocarbon`, `miasma`, `osaka-jade`,
`ristretto`, `flexoki-light`, `hackerman`, `ember`, `ethereal`, `synthwave`,
`blade-runner`, `nostromo`, `plasma`, `viridis`, `ice`, `matte-black`, `vantablack` —
plus `auto`, which derives a ramp from whatever Textual theme your terminal is wearing.

Gradients are blended in linear light rather than straight sRGB, so the midpoint of a
ramp doesn't go muddy the way naive hex interpolation does.

### Custom themes

Drop a TOML file in `~/.config/spektr/themes/` (`%APPDATA%\spektr\themes\` on Windows).
The filename becomes the theme name; press `r` to reload without restarting.

```toml
# ~/.config/spektr/themes/solarized.toml
low    = "#859900"   # bottom of the spectrum ramp
mid    = "#b58900"
high   = "#dc322f"   # top
bg     = "#002b36"
fg     = "#839496"
accent = "#268bd2"
```

cliamp's `green`/`yellow`/`red`/`bright_fg` key names are accepted as aliases, so themes
port across without editing.

## Plugins

Your own visualizers, in `~/.config/spektr/plugins/`. They appear in the `v` picker
alongside the built-ins, because they use the same decorator and the same contract:

```python
# ~/.config/spektr/plugins/nightrider.py
import numpy as np
from spektr.api import mode, pack_braille, cell_max

@mode("Nightrider", blurb="scanning eye, swept by the beat")
def nightrider(ctx):
    speed = 0.5 + ctx.range(0.0, 0.15) * 2.5      # lunges on the kick
    pos = (np.sin(ctx.t * speed) * 0.5 + 0.5) * (ctx.dot_cols - 1)
    x = np.arange(ctx.dot_cols)[None, :]
    y = np.arange(ctx.dot_rows)[:, None]
    band = np.abs(y - (ctx.dot_rows - 1) / 2) < ctx.dot_rows * (0.12 + ctx.energy * 0.25)
    glow = np.clip(1.0 - np.abs(x - pos) / (ctx.dot_cols * 0.18), 0.0, 1.0)
    field = np.where(band, glow ** 1.6, 0.0)
    return pack_braille(field > 0.10), ctx.ramp(cell_max(field))
```

You return codepoints and *heat* — never colours — so every plugin works with all
thirty themes for free.

> [!WARNING]
> **Plugins are Python and run with your privileges.** spektr can't sandbox them, and
> won't pretend to. Instead the trust decision is explicit: a plugin doesn't run until
> you've approved its exact contents, and any edit invalidates that.

```console
$ spektr plugins list
  nightrider   untrusted  —

$ spektr plugins trust nightrider
  sha256  f28ceb19b2d630bd…   (31 lines)
  This is Python. It runs with your privileges. Read it first.
  Trust this plugin? [y/N] y
```

Failure *is* contained: a plugin that raises gets quarantined after a few attempts
rather than taking the app down, one that renders too slowly has its previous frame
reused, and its output is shape-checked so a mistake names the plugin instead of
crashing somewhere unrelated. `spektr plugins doctor` explains anything that didn't
load; `--no-plugins` starts clean.

Full guide, including the whole of `ctx` and the drawing toolkit: **[docs/plugins.md](docs/plugins.md)**.

## Keys

| Key | Action | | Key | Action |
|---|---|---|---|---|
| `v` | Visualizer picker — live preview, `/` filter | | `s` | What am I listening to? |
| `t` | Theme picker — live preview, `/` filter | | `d` | Try the next audio source |
| `m` / `space` | Next mode (`M` for previous) | | `[` `]` | Sensitivity down / up |
| `T` | Next theme | | `g` `G` | Noise gate down / up |
| `f` | Hide header and footer — full-screen visual | | `r` | Reload themes and plugins from disk |
| `p` | Frame time and FPS | | `q` | Quit |

Mode, theme, sensitivity and gate are remembered between runs.

## Command line

```
spektr --diagnose       probe every source: is audio arriving, and how loud?
spektr --devices        list every audio device
spektr --device 7       force a capture device by index
spektr --mode Retro     start in a given visualiser
spektr --theme gruvbox  start with a given theme
spektr --fps 30         cap the frame rate
spektr --mic            allow the microphone as an automatic source
spektr --list-modes     print visualiser names
spektr --list-themes    print theme names
spektr --no-plugins     skip loading plugins this run

spektr plugins list     what's installed, and whether it's trusted
spektr plugins trust    review and approve a plugin
spektr plugins doctor   why isn't mine loading?
spektr plugins path     print the plugins folder
```

## How it captures audio

spektr listens to your **output** device via loopback, so it visualises whatever is
already playing — it never needs a file, a stream, or a music service. Stereo is
preserved end to end, which is what the Stereo, VU, Needle and Gonio modes read.

| Platform | Status |
|---|---|
| **Windows** | WASAPI loopback via `soundcard` — works out of the box |
| **Linux** | PulseAudio / PipeWire monitor via `soundcard`, or a monitor input |
| **macOS** | Needs a loopback device (BlackHole, Soundflower) |

**Loopback comes from `soundcard`, not `sounddevice`.** PortAudio has no WASAPI loopback
flag, so sounddevice cannot capture system audio at *any* version — a detail that costs
people a lot of time, because the failure looks like a missing upgrade. `soundcard` talks
to WASAPI directly and sets `AUDCLNT_STREAMFLAGS_LOOPBACK` itself. It's a hard dependency
for that reason; sounddevice is still used to enumerate monitors and the mic.

If `spektr --diagnose` shows no `loopback:` entries, the environment block at the top says
which library is missing or unusable.

**It will never pick your microphone on its own.** A loopback tap reporting silence is
telling the truth — nothing is playing — whereas a mic always has *something* on it, so
selecting whichever source has signal picks the room every time you start spektr with the
music paused. The mic is only used automatically if no output tap can be opened at all,
and it says so in red when that happens. Press `d` to cycle onto it deliberately, or start
with `--mic`.

If a tap opens but stays silent, spektr holds it, watches for audio, and rotates onto the
next tap if none arrives — so launching before you press play, or output going to a
non-default endpoint, both sort themselves out.

Press `s` at any time for the current source and the input level against the noise gate.
When nothing adds up, `spektr --diagnose` opens every candidate in turn and prints the
measured RMS and peak for each, which settles it:

```
  source                                       rms     peak   x gate  verdict
  loopback: Speakers (Realtek)            3.41e-02    0.412    426.6  AUDIO
  loopback: HDMI Output                   1.00e-12    0.000      0.0  silent
  microphone (NOT system audio)           8.80e-04    0.021     11.0  audio (microphone!)
```

## How it works

Three details do most of the work.

**Analysis runs on its own clock.** A 2048-sample window advances by a 512-sample hop —
75% overlap, about 94 analyses per second at 48 kHz. Sampling the FFT from the frame
timer instead (23 blocks/sec read by a 30–60 fps loop) produces beat-rate aliasing that
no amount of easing can hide, because the target sequence itself is stepped.

**The easing is expressed in seconds, not frames.** Bands are driven by a damped spring
integrated with sub-stepping, and peak markers hold for a duration rather than a frame
count. The animation feels identical at 15 fps and 120 fps, which means the frame rate
can adapt to load without the motion changing character.

**Modes emit arrays, not strings.** Every mode returns a `(h, w)` grid of codepoints and
a matching grid of palette indices; the widget run-length encodes those into Textual
`Strip`s from `render_line`. Nothing goes through a Rich console render, and a smooth
field costs a handful of segments per row instead of one per cell.

## Development

```bash
python tests/bench.py        # shape checks + per-mode render benchmark
python tests/test_audit.py   # logic audit: mutation, animation, reactivity, leaks
python tests/test_audio.py   # analysis, gating, frame-rate independence
python tests/test_app.py     # headless UI smoke test (no audio device needed)
python tests/test_plugins.py # discovery, trust, loading, quarantine
python tests/perf.py all     # analyser cost, strip scaling, memory, headroom
```

`bench.py` prints build and strip time for every mode at 120×16, 200×50 and 240×60.
`test_audit.py` is the one that catches logic errors rather than crashes — a mode that
writes into the shared band buffer, or renders the same picture regardless of the audio,
passes every shape check ever written.

Measured on one core: the analyser costs **1.5%** of a core continuously, the audio
callback **0.02%**, and the heaviest mode at 240×60 takes **7.5 ms** against a 16.7 ms
budget. Nothing exceeds budget even at 400×100.

## Why it exists

It began as the visualiser inside a terminal music client. It turned out to be the most
interesting part of that project and the only part that didn't depend on anyone's API,
so it moved out and got its own name.

## License

MIT © zemo — see [LICENSE](LICENSE).
