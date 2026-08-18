"""Theme registry and colour ramps.

A theme is six colours. Three of them (low/mid/high) are the spectrum anchors
that get blended into a ``RAMP_STEPS``-step ramp; the other three (bg/fg/accent)
dress the UI. A theme may declare an explicit ``ramp`` of extra anchors — the blend then
walks them in order instead of just low→mid→high, for palettes like rainbow
that want more than three riding points. Built-ins live in a dict — no file IO
on startup — and user themes are read from ``~/.config/spektr/themes/*.toml``
at first use.

Blending happens in linear light rather than straight sRGB. Interpolating hex
values directly darkens the midpoint of a gradient noticeably (mixing #00ff41
and #ff3300 in sRGB gives you a muddy olive); going through gamma keeps the
midtones where the eye expects them.
"""

from __future__ import annotations

import colorsys
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.color import Color
from rich.style import Style

from . import palette

#: Distinct colour steps in the ramp. Tried 256, to give an animated theme's
#: per-column spread more buckets for a smoother sweep — measured instead:
#: make_strips run-length-encodes each row, so its cost tracks how often
#: adjacent cells land in *different* buckets, not raw cell count. A coarser
#: ramp is what let a continuous field (Plasma, Scatter, Tunnel, Pulse,
#: Radial — anything colouring by a smooth per-cell value rather than a
#: per-row/per-band one) alias neighbouring cells into the same bucket and
#: merge into one Segment; at 256 they mostly didn't, and make_strips cost on
#: a smooth field at 400x100 went from 12.8 ms to 22.2 ms — those modes blew
#: the 16.7 ms/frame budget on every theme, not just the animated one. Back
#: at 64. The per-column spread still got smoother (see
#: AudioVisualizer._animate_ramp): rounding the column shift instead of
#: flooring it was the fix that mattered, not the bucket count.
RAMP_STEPS = 64
_GAMMA = 2.2

#: Worst per-channel error, out of 255, that ``make_strips`` may introduce by
#: merging near-identical colours into one Segment. About 4% of the channel
#: range — under the ~2% JND for a large flat patch only in the sense that the
#: merged cells are adjacent and the gradient is already quantised to 64 steps,
#: which is the bound this has to live inside. Raising it starts to show as
#: banding on smooth full-screen fields.
_RLE_MAX_RGB = 10
#: Never merge more than this many ramp steps regardless of how gentle the
#: ramp is — beyond a few steps the run is no longer "the same colour", it is
#: a decision to draw a different picture than the mode asked for.
_RLE_MAX_TOL = 3


# ── colour helpers ───────────────────────────────────────────────────────────


def hex_to_rgb(value) -> tuple[int, int, int]:
    text = str(value).strip().lstrip("#")
    if len(text) == 3:
        text = "".join(c * 2 for c in text)
    try:
        return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    except (ValueError, IndexError):
        return 255, 255, 255


def rgb_to_hex(rgb) -> str:
    r, g, b = (max(0, min(255, int(round(c)))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.power(np.clip(rgb, 0, 255) / 255.0, _GAMMA)


def _to_srgb(lin: np.ndarray) -> np.ndarray:
    return np.power(np.clip(lin, 0.0, 1.0), 1.0 / _GAMMA) * 255.0


def mix(a, b, t: float) -> str:
    """Blend two colours in linear light. t=0 is a, t=1 is b."""
    x = _to_linear(np.array(hex_to_rgb(a), dtype=np.float64))
    y = _to_linear(np.array(hex_to_rgb(b), dtype=np.float64))
    return rgb_to_hex(_to_srgb(x + (y - x) * t))


def _luminance(colour) -> float:
    r, g, b = _to_linear(np.array(hex_to_rgb(colour), dtype=np.float64))
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)


def hex_to_hsl(colour) -> tuple[float, float, float]:
    """``(hue 0..1, saturation 0..1, lightness 0..1)``.

    HSL rather than RGB because this exists for the theme editor, and nudging
    a colour is something people do in terms of "more blue" and "paler", not
    in terms of a red channel. Round-trips exactly through ``hsl_to_hex`` for
    every colour that came from an 8-bit-per-channel value.
    """
    r, g, b = (c / 255.0 for c in hex_to_rgb(colour))
    hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    return hue, saturation, lightness


def hsl_to_hex(hue: float, saturation: float, lightness: float) -> str:
    hue = hue % 1.0
    saturation = min(1.0, max(0.0, saturation))
    lightness = min(1.0, max(0.0, lightness))
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return rgb_to_hex((r * 255.0, g * 255.0, b * 255.0))


def rgb_distance(a, b) -> float:
    """Straight-line distance in unit RGB, 0..sqrt(3)."""
    x = np.array(hex_to_rgb(a), dtype=np.float64) / 255.0
    y = np.array(hex_to_rgb(b), dtype=np.float64) / 255.0
    return float(np.sqrt(((x - y) ** 2).sum()))


def contrast_ratio(a, b) -> float:
    """WCAG contrast ratio, 1..21."""
    la, lb = _luminance(a), _luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


#: A compact set of named colours for the theme editor's colour picker.
#: Keys are what the picker searches and what ``resolve_colour`` accepts —
#: CSS-flavoured names with familiar hexes, biased toward colours that read
#: as anchors on a dark background (the picker's real audience is the three
#: ramp slots), with a few light neutrals for fg/bg.
NAMED_COLOURS: dict[str, str] = {
    "black": "#000000", "grey": "#808080", "silver": "#c0c0c0", "white": "#ffffff",
    "brown": "#8b4513", "maroon": "#800000", "olive": "#808000",
    "red": "#ff0000", "crimson": "#dc143c", "scarlet": "#ff2400", "rust": "#b7410e",
    "tomato": "#ff6347", "coral": "#ff7f50", "salmon": "#fa8072",
    "orange": "#ff7f00", "amber": "#ffb000", "gold": "#ffd700", "khaki": "#c3b091",
    "yellow": "#ffff00", "lime": "#7fff00", "chartreuse": "#ccff00",
    "green": "#00cc33", "emerald": "#00c853", "mint": "#00ff7f",
    "teal": "#00b8a8", "turquoise": "#40e0d0", "cyan": "#00ffff",
    "sky": "#3aa0ff", "azure": "#1e90ff", "royal": "#4169e1", "blue": "#0000ff",
    "navy": "#000080", "indigo": "#4b0082", "violet": "#8a2be2",
    "purple": "#a020f0", "orchid": "#da70d6", "magenta": "#ff00ff",
    "pink": "#ff69b4", "hot pink": "#ff1493", "rose": "#ff2e88", "wine": "#722f37",
    "lavender": "#b57edc", "slate": "#708090", "champagne": "#f7e7ce",
}


def resolve_colour(text: str) -> str | None:
    """Free text -> a canonical ``#rrggbb``, or None if it is not a colour.

    Accepts a name from :data:`NAMED_COLOURS`, ``#rrggbb``, ``#rgb``, or bare
    ``rrggbb``/``rgb``. This is the hex entry's validator — None keeps the
    prompt open rather than committing garbage. Unlike ``hex_to_rgb`` there
    is no fallback colour: a typo is a typo, not a new favourite white.
    """
    t = str(text).strip()
    if not t:
        return None
    low = t.lower()
    if low in NAMED_COLOURS:
        return NAMED_COLOURS[low]
    if t.startswith("#"):
        t = t[1:]
    if len(t) in (3, 6) and all(c in "0123456789abcdefABCDEF" for c in t):
        return "#" + (t if len(t) == 6 else "".join(c * 2 for c in t)).lower()
    return None


#: A ramp anchor closer than this to its own background renders as background.
#: ``infrared`` shipped at 0.18 against every other theme's 0.27 or higher.
MIN_ANCHOR_DISTANCE = 0.18
#: WCAG AA for body text, which fg-on-bg has to clear.
MIN_FG_CONTRAST = 4.5


def theme_visibility_problems(theme: "Theme") -> list[str]:
    """Why this theme would be hard to see, as human-readable lines.

    One definition of "visible", used in two places that must not drift: the
    audit test that guards the built-ins, and the theme editor, which has to
    warn about the same thing *live* rather than letting someone save a theme
    the test suite would reject.
    """
    bad = []
    for label, colour in (("low", theme.low), ("mid", theme.mid), ("high", theme.high)):
        d = rgb_distance(colour, theme.bg)
        if d < MIN_ANCHOR_DISTANCE:
            bad.append(
                f"{label} anchor {colour} is only {d:.2f} from bg {theme.bg} — nearly invisible"
            )
    ratio = contrast_ratio(theme.fg, theme.bg)
    if ratio < MIN_FG_CONTRAST:
        bad.append(f"fg/bg contrast is {ratio:.2f}, below WCAG AA's {MIN_FG_CONTRAST}")
    return bad


def derive_bg(low: str, mid: str, high: str) -> str:
    """A background for a theme whose ramp anchors are already chosen.

    Built in HSL — ``low``'s hue at half its saturation and a fixed low
    lightness — rather than by blending ``low`` toward black. Blending is the
    obvious approach and it does not work here: ``mix`` interpolates in linear
    light, which is right for a gradient and wrong for this, because 94% of
    the way to black in linear light is still #004706. A background has to be
    dark *perceptually*, and lightness is the axis that means that.

    Steps darker if the result is too close to any anchor, and ends at pure
    black. A near-black ``low`` on a near-black bg still fails after that, and
    should: that is the ``infrared`` bug, and the editor warns rather than
    silently choosing a colour the user did not ask for.
    """
    hue, sat, _ = hex_to_hsl(low)
    for lightness in (0.07, 0.05, 0.03):
        candidate = hsl_to_hex(hue, sat * 0.5, lightness)
        if all(
            rgb_distance(c, candidate) >= MIN_ANCHOR_DISTANCE + 0.04
            for c in (low, mid, high)
        ):
            return candidate
    return "#000000"


def derive_fg(high: str, bg: str) -> str:
    """Readable body text for a derived theme: a pale tint of the hot anchor."""
    hue, sat, _ = hex_to_hsl(high)
    for lightness in (0.86, 0.90, 0.94, 0.98):
        candidate = hsl_to_hex(hue, sat * 0.25, lightness)
        if contrast_ratio(candidate, bg) >= MIN_FG_CONTRAST + 0.5:
            return candidate
    return "#ffffff"


# ── theme model ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Theme:
    name: str
    low: str
    mid: str
    high: str
    bg: str = "#000000"
    fg: str = "#ffffff"
    accent: str = ""
    #: an animated theme drifts its colour ramp over time + position, so the
    #: spectrum flows across the bands instead of sitting still.
    animated: bool = False
    #: optional extra ramp anchors, walked in order after low→mid→high. Used
    #: by rainbow, which needs a yellow and a blue riding point that a three
    #: stop ramp can never reach without turning to mud.
    ramp: tuple[str, ...] | None = None


def _t(name, low, mid, high, bg, fg, accent="", animated=False, ramp=None) -> tuple[str, Theme]:
    return name, Theme(name, low, mid, high, bg, fg, accent, animated, ramp)


BUILTIN: dict[str, Theme] = dict(
    [
        # the original — winamp's green/amber/red
        _t("classic", "#00ff41", "#ffb000", "#ff3300", "#000000", "#e0e0e0", "#ffb000"),
        _t("gruvbox", "#b8bb26", "#fabd2f", "#fb4934", "#282828", "#ebdbb2", "#83a598"),
        _t(
            "catppuccin",
            "#a6e3a1",
            "#f9e2af",
            "#f38ba8",
            "#1e1e2e",
            "#cdd6f4",
            "#cba6f7",
        ),
        _t(
            "catppuccin-latte",
            "#40a02b",
            "#df8e1d",
            "#d20f39",
            "#eff1f5",
            "#4c4f69",
            "#8839ef",
        ),
        _t("dracula", "#50fa7b", "#f1fa8c", "#ff5555", "#282a36", "#f8f8f2", "#bd93f9"),
        _t("nord", "#a3be8c", "#ebcb8b", "#bf616a", "#2e3440", "#d8dee9", "#88c0d0"),
        _t(
            "tokyo-night",
            "#9ece6a",
            "#e0af68",
            "#f7768e",
            "#1a1b26",
            "#c0caf5",
            "#7aa2f7",
        ),
        _t(
            "rose-pine",
            "#9ccfd8",
            "#f6c177",
            "#eb6f92",
            "#191724",
            "#e0def4",
            "#c4a7e7",
        ),
        _t(
            "everforest",
            "#a7c080",
            "#dbbc7f",
            "#e67e80",
            "#2d353b",
            "#d3c6aa",
            "#7fbbb3",
        ),
        _t(
            "kanagawa", "#98bb6c", "#e6c384", "#e46876", "#1f1f28", "#dcd7ba", "#7e9cd8"
        ),
        _t(
            "ayu-mirage",
            "#bae67e",
            "#ffcc66",
            "#f28779",
            "#1f2430",
            "#cbccc6",
            "#73d0ff",
        ),
        _t("monokai", "#a6e22e", "#e6db74", "#f92672", "#272822", "#f8f8f2", "#66d9ef"),
        _t(
            "solarized",
            "#859900",
            "#b58900",
            "#dc322f",
            "#002b36",
            "#839496",
            "#268bd2",
        ),
        _t(
            "nightfox", "#81b29a", "#dbc074", "#c94f6d", "#192330", "#cdcecf", "#719cff"
        ),
        _t(
            "oxocarbon",
            "#3ddbd9",
            "#33b1ff",
            "#be95ff",
            "#161616",
            "#f2f4f8",
            "#ee5396",
        ),
        _t("miasma", "#78834b", "#bb7744", "#e0a363", "#222222", "#c2c2b0", "#8f6f5f"),
        _t(
            "osaka-jade",
            "#43a58a",
            "#8ec07c",
            "#e06c75",
            "#111c18",
            "#c1c8c4",
            "#549e6a",
        ),
        _t(
            "ristretto",
            "#adda78",
            "#f9cc6c",
            "#fd6883",
            "#2c2525",
            "#e6d9db",
            "#f38d70",
        ),
        _t(
            "flexoki-light",
            "#66800b",
            "#ad8301",
            "#af3029",
            "#fffcf0",
            "#100f0f",
            "#205ea6",
        ),
        # ── moods rather than editor ports ──
        _t(
            "hackerman",
            "#005f11",
            "#00cc33",
            "#00ff41",
            "#000000",
            "#00ff41",
            "#00ff41",
        ),
        _t("ember", "#6b2d00", "#ff7a18", "#ffd166", "#17110d", "#f0e0d0", "#ff7a18"),
        _t(
            "ethereal", "#6affc2", "#7ab8ff", "#d3a4ff", "#0f1020", "#e8e8ff", "#9d7aff"
        ),
        _t(
            "synthwave",
            "#03edf9",
            "#ff7edb",
            "#fede5d",
            "#241b2f",
            "#f4eee4",
            "#ff7edb",
        ),
        _t(
            "blade-runner",
            "#00e5ff",
            "#b14aed",
            "#ff2e88",
            "#0b0c17",
            "#d6deeb",
            "#00e5ff",
        ),
        _t(
            "nostromo", "#4a2600", "#ff8c00", "#ffd08a", "#0d0700", "#ffb454", "#ff8c00"
        ),
        _t("plasma", "#0d0887", "#cc4778", "#f0f921", "#0a0612", "#f0e8ff", "#cc4778"),
        _t("viridis", "#440154", "#21918c", "#fde725", "#0b0a12", "#e8f0e8", "#21918c"),
        _t("ice", "#0a2a5e", "#3aa0ff", "#e8f6ff", "#050b16", "#cfe6ff", "#3aa0ff"),
        _t(
            "matte-black",
            "#4a4a4a",
            "#8a8a8a",
            "#d5d5d5",
            "#121212",
            "#bcbcbc",
            "#eaeaea",
        ),
        _t(
            "vantablack",
            "#333333",
            "#888888",
            "#ffffff",
            "#000000",
            "#ffffff",
            "#ffffff",
        ),
        # ── more editor ports ──
        _t(
            "nightfly", "#a1cd5e", "#e3d18a", "#fc514e", "#011627", "#c3ccdc", "#82aaff"
        ),
        _t(
            "material", "#c3e88d", "#ffcb6b", "#f07178", "#263238", "#eeffff", "#82aaff"
        ),
        _t("gotham", "#2aa889", "#edb443", "#d26937", "#0c1014", "#98d1ce", "#195466"),
        _t("oceanic", "#99c794", "#fac863", "#ec5f67", "#1b2b34", "#d8dee9", "#6699cc"),
        _t(
            "gruvbox-light",
            "#79740e",
            "#b57614",
            "#9d0006",
            "#fbf1c7",
            "#3c3836",
            "#076678",
        ),
        _t(
            "tokyo-night-day",
            "#587539",
            "#8f5e15",
            "#f52a65",
            "#e1e2e7",
            "#3760bf",
            "#2e7de9",
        ),
        # ── more moods ──
        _t(
            "vaporwave",
            "#00f0c0",
            "#ff77e9",
            "#ff2e88",
            "#1a0b2e",
            "#f2e6ff",
            "#b967ff",
        ),
        # low was #3a0000 on a #0d0000 bg — a perceptual RGB distance of 0.18,
        # against every other theme's 0.27+ (see test_theme_visibility). The
        # cold end of a thermal gradient was rendering as flat background, not
        # a colour, which defeats a theme whose entire point is that the cold
        # parts are still visible. Deepened the low anchor and darkened bg to
        # match, rather than lightening bg — a thermal reading should still
        # start near black.
        _t(
            "infrared", "#4d0000", "#c22800", "#ffd000", "#050000", "#ffb4a2", "#ff4800"
        ),
        _t(
            "deep-sea", "#0a3d62", "#12cbc4", "#a5f3ff", "#04141f", "#b8e0e6", "#12cbc4"
        ),
        _t("magma", "#2c115f", "#b73779", "#fcfdbf", "#0b0417", "#f5e3ff", "#fe9f6d"),
        # ── new, not ports or palette-family reshuffles ──
        # amber phosphor — the other CRT colour. classic already covers green
        # phosphor (hackerman is its monochrome extreme); this is the IBM 5151
        # amber monitor's actual single hue, dark rust to a near-white-hot
        # glow, rather than green with amber merely borrowed for the mid stop.
        _t("phosphor-amber", "#3d1f00", "#b36b00", "#ffcc33", "#0a0600", "#ffb000", "#ff9500"),
        _t(
            "sakura",
            "#7a2848",
            "#f7a8c4",
            "#fff0f5",
            "#1a1014",
            "#f7d9e3",
            "#e191b3",
        ),
        # hazard yellow-green rather than hackerman's pure green — the same
        # family everything else in "moods" avoids doubling up on gets a
        # second, deliberately different entry because the two read nothing
        # alike once the mid/high stops turn it sickly and saturated instead
        # of clean.
        _t("toxic", "#1f4d00", "#7fff00", "#e6ff33", "#050800", "#ccff66", "#aaff00"),
        _t("copper", "#3d1f0f", "#b87333", "#ffd699", "#0a0603", "#e8c9a3", "#cd7f32"),
        # bright and cold rather than moody-dark like ice/deep-sea — those two
        # are a night ocean; this is open sky and glacier, meant to read as
        # the brightest theme in the set next to something like vantablack at
        # the other end.
        _t("polar", "#3fa9dc", "#8ee8d4", "#eafcff", "#061826", "#eafcff", "#9fe8ff"),
        _t("bubblegum", "#7bf0ff", "#c07bff", "#ff5ec4", "#150a1f", "#ffe3fb", "#ff9de2"),
        # Pink already appears four times, and not once as *the* colour: sakura
        # is deliberately pale and desaturated, while bubblegum, vaporwave and
        # blade-runner all only arrive at pink after starting from cyan or
        # teal, so the ramp spends most of its length not being pink at all.
        # This one never leaves the hue: dark magenta-plum through CSS
        # deeppink (#ff1493, the canonical hot pink) to a pale highlight, so
        # every band from quietest to loudest reads as the same colour getting
        # hotter. Staying monochromatic is also what keeps it smooth — a ramp
        # that crosses hues has to pass through whatever sits between them,
        # which is where bubblegum's cyan->purple leg goes momentarily grey.
        _t("hot-pink", "#6b0f47", "#ff1493", "#ffb3dd", "#14060f", "#ffd9ee", "#ff69b4"),
        # The set had no red that stays red. infrared and ember are both heat
        # ramps that climb out of red into orange and then yellow, and
        # classic only touches red at its very top stop. A gemstone is the
        # opposite idea: dark garnet in the shadows, true ruby (#9b111e) at
        # the middle, and the bright crimson "fire" a cut stone throws at its
        # facets as the peak — deep and saturated across the whole range
        # rather than brightening by shifting hue toward yellow.
        _t("ruby", "#5c0714", "#9b111e", "#ff2b4d", "#0d0206", "#ffd3da", "#e0115f"),
        # Four more cut stones, on ruby's construction rather than its hue.
        #
        # The rule that makes a gemstone read as one is that the hue holds from
        # shadow to peak and only saturation and lightness move. Most of the
        # older ramps here brighten by *travelling*: ember and infrared and
        # magma all climb red into orange into yellow, which reads as heat, not
        # as a stone. So each of these keeps one hue and spends its range on
        # depth — a dark inclusion at the bottom, the stone's own colour in the
        # middle, and at the top the bright flash a facet throws back.
        #
        # They fill real gaps rather than crowding the set. There was no
        # saturated violet at all; the greens were muted (osaka-jade, everforest)
        # or scientific (viridis); the blues were all atmospheric — deep-sea,
        # ice, polar, oceanic are cold and desaturated where sapphire is
        # jewel-bright; and nothing was a plain vivid orange, only heat ramps
        # passing through it on the way to yellow.
        _t("emerald", "#05532a", "#0f9b5a", "#3ff08d", "#02120a", "#cdf5e0", "#2ee6a0"),
        _t("sapphire", "#071a4a", "#0f52ba", "#4d9bff", "#030818", "#d0e0ff", "#2f7fe0"),
        _t("amethyst", "#2a0b47", "#7b2fbe", "#c77dff", "#100320", "#ecd9ff", "#9d4edd"),
        _t("citrine", "#4a2c04", "#c68a12", "#ffd447", "#140c02", "#fff0cc", "#e8a317"),
        # Not a gemstone and deliberately so — the one warm hue the set never
        # held still. Every other orange here is a waypoint in a heat ramp.
        _t("tangerine", "#4a1c02", "#e2560d", "#ff9f45", "#150701", "#ffe2cc", "#ff6b1a"),
        # The one hole left in the set, and it was found by measuring rather
        # than by eye: sorting every theme's mid anchor into 30-degree hue
        # buckets leaves 240-269 the only empty one, while 24 of the other 54
        # crowd into 30-59. Sapphire stops at 216 and amethyst starts at 271,
        # so the blue that is neither — the one dyers mean by the word — had
        # no theme at all.
        #
        # Kept at about half saturation, which is what separates it from the
        # two it sits between. Both of those are jewel-bright by design and
        # a third of the same would read as a variant of them; this is dye in
        # cloth, dense and slightly grey, and it lands 0.184 from amethyst
        # where the median gap between neighbours in this set is 0.208 and the
        # closest existing pair is 0.059.
        _t("indigo", "#232258", "#554fc4", "#bcb6ec", "#080a16", "#d8ddff", "#6b7bff"),
        # red -> yellow -> green -> blue -> violet -> magenta, and back to red:
        # a closed loop around the colour wheel. Three anchors can't do this —
        # the red→green seam dries to olive and the green→violet one washes
        # through grey instead of blue — and an animating rainbow needs the loop
        # closed with magenta too, or the wrap from the last colour back to the
        # first reads as a hard tear sweeping across the bands. ``animated``
        # makes the palette rotate around this loop over time (see
        # Palette.set_phase).
        _t(
            "rainbow",
            "#ff0033",
            "#22e022",
            "#7a3aff",
            "#0d0d1a",
            "#ffffff",
            "#ff9500",
            animated=True,
            ramp=("#ff0033", "#ffd400", "#22e022", "#0088ff", "#7a3aff", "#ff2bc8"),
        ),
    ]
)

#: Pseudo-theme: derive the ramp from whatever Textual theme is active.
AUTO = "auto"


# ── user themes ──────────────────────────────────────────────────────────────


def config_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming"
        return Path(base) / "spektr"
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "spektr"


def _root(config_dir: Path | None = None) -> Path:
    """An injected config root, or the platform default via :func:`config_dir`.

    Resolved through the module rather than a bound name — tests patch
    ``config_dir`` and every default must follow the patch, while a caller
    that knows the root passes it straight through.
    """
    return config_dir if config_dir is not None else palette.config_dir()


def load_user_themes(config_dir: Path | None = None) -> dict[str, Theme]:
    """Read ``<config>/themes/*.toml``. A malformed file is skipped, not fatal.

    ``config_dir`` overrides where ``<config>`` points; None means the
    platform default from :func:`config_dir`.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # 3.10
        try:
            import tomli as tomllib  # type: ignore
        except ModuleNotFoundError:
            return {}

    try:
        folder = _root(config_dir) / "themes"
        if not folder.is_dir():
            return {}
        candidates = sorted(folder.glob("*.toml"))
    except OSError:
        # an unreadable config directory should cost you custom themes, not
        # the whole application
        return {}

    found: dict[str, Theme] = {}
    for path in candidates:
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        # accept both spektr's names and cliamp's, so themes port across
        low = data.get("low") or data.get("green")
        mid = data.get("mid") or data.get("yellow")
        high = data.get("high") or data.get("red")
        if not (low and mid and high):
            continue
        found[path.stem] = Theme(
            name=path.stem,
            low=low,
            mid=mid,
            high=high,
            bg=data.get("bg") or data.get("background") or "#000000",
            fg=data.get("fg") or data.get("bright_fg") or "#ffffff",
            accent=data.get("accent") or "",
        )
    return found


def all_themes(config_dir: Path | None = None) -> dict[str, Theme]:
    """Built-ins, with user themes of the same name taking priority."""
    merged = dict(BUILTIN)
    merged.update(load_user_themes(config_dir))
    return dict(sorted(merged.items()))


# ── the live palette ─────────────────────────────────────────────────────────


class Palette:
    """The active theme, plus everything derived from it, prebuilt.

    Every renderer works in *ramp indices* (0..RAMP_STEPS-1) rather than colour
    strings, so a frame never touches a hex value or parses a style. Swapping
    theme rebuilds these tables once and the renderers carry on unchanged.
    """

    __slots__ = (
        "_phase",
        "bg_styles",
        "colors",
        "hexes",
        "name",
        "note",
        "pair_styles",
        "rgb",
        "rle_budget",
        "rle_tol",
        "styles",
        "theme",
    )

    def __init__(self, theme: Theme | None = None):
        self.set(theme or BUILTIN["classic"])

    def set(self, theme: Theme) -> None:
        self.theme = theme
        self.name = theme.name
        self.note = f"{theme.name} — {theme.low} → {theme.high}"
        self._phase = 0.0
        self._build(0.0)

    def set_phase(self, phase: float) -> None:
        """Rotate an animated theme's colour loop to ``phase`` (in cycles).

        A no-op for static themes. Called once a frame for animated themes, so
        the ramp — and every style derived from it — is rebuilt at the current
        point on the loop. That is what makes the flow smooth: the colours move
        by a fraction of a step each frame instead of jumping a whole step at a
        time.
        """
        if not self.theme.animated:
            return
        self._phase = phase % 1.0
        self._build(self._phase)

    def _build(self, phase: float = 0.0) -> None:
        th = self.theme
        anchors = th.ramp or (th.low, th.mid, th.high)
        n = len(anchors)
        lin = _to_linear(np.array([hex_to_rgb(c) for c in anchors], dtype=np.float64))

        if th.animated:
            # A closed loop rather than an open ramp. RAMP_STEPS samples are
            # taken around the colour wheel — at i/RAMP_STEPS, endpoint excluded
            # — and ``phase`` rotates the wheel, so index 0 and index 63 sit one
            # step apart on the loop and the wrap the animation relies on is
            # seamless. Blending in linear light keeps the midtones clean as the
            # loop turns.
            t = (np.arange(RAMP_STEPS, dtype=np.float64) / RAMP_STEPS + phase) % 1.0
            pos = t * n
            i = np.floor(pos).astype(np.int64)
            seg = pos - i
            seg = seg * seg * (3.0 - 2.0 * seg)
            j = (i + 1) % n
            out = lin[i] + (lin[j] - lin[i]) * seg[:, None]
        else:
            t = np.linspace(0.0, 1.0, RAMP_STEPS)
            # piecewise low->mid->high (or the declared ramp, in order), smooth-
            # stepped so the anchors don't crease. For the classic three-stop ramp
            # this reduces to the same two segments as before; extra anchors just
            # add more riding points between them.
            pos = t * (n - 1)
            i = np.clip(np.floor(pos).astype(np.int64), 0, n - 2)
            seg = pos - i
            seg = seg * seg * (3.0 - 2.0 * seg)
            out = lin[i] + (lin[i + 1] - lin[i]) * seg[:, None]

        self.rgb = _to_srgb(out)
        rgb_int = np.clip(np.rint(self.rgb), 0, 255).astype(np.int64)
        self.hexes = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in rgb_int]
        self.colors = [Color.from_rgb(int(r), int(g), int(b)) for r, g, b in rgb_int]
        self.styles = [Style.from_color(color=c) for c in self.colors]
        self.bg_styles = [Style.from_color(bgcolor=c) for c in self.colors]
        # Combined fg-on-bg styles, filled on demand and dropped when the ramp
        # changes. Building all RAMP_STEPS**2 of them up front would be 4096
        # Style objects per theme swap to serve the handful of pairs a frame
        # actually uses; caching here instead of in make_strips is the point,
        # because that cache was rebuilt sixty times a second. For an animated
        # theme the ramp moves every frame, so the cache is dropped with it —
        # the indices it keys on now name different colours.
        self.pair_styles = {}

        # How far ``make_strips`` may let a colour run drift from its start
        # before it must split, derived from *this* ramp rather than fixed.
        #
        # A flat tolerance cannot be right for every theme, because "two ramp
        # steps" is not a fixed amount of colour. Across the built-ins the
        # largest channel move over two steps ranges from about 4/255 on the
        # gentle ramps to 34/255 on ``classic`` — the default — and 72/255 on
        # ``rainbow``, which walks a whole hue wheel. Merging at a flat two
        # steps is invisible on the first and obvious banding on the last.
        #
        # So: take the largest drift whose worst-case channel error stays
        # under ``_RLE_MAX_RGB``. Gentle ramps get the full merge and the
        # Segment count that comes with it; aggressive ones fall back toward
        # exact RLE, which is what they always had.
        self.rle_tol = 0
        for t in range(1, _RLE_MAX_TOL + 1):
            if len(rgb_int) <= t:
                break
            if int(np.abs(rgb_int[t:] - rgb_int[:-t]).max()) > _RLE_MAX_RGB:
                break
            self.rle_tol = t

        # ...and the same question asked per index rather than once for the
        # whole ramp, which is what ``make_strips`` actually uses.
        #
        # One number for the ramp is decided by its *worst* segment, and
        # steepness is local. ``classic`` has a stretch where a single step
        # moves more than ``_RLE_MAX_RGB``, so the rule above returns 0 and the
        # merge never fires — on the default theme — even though the average
        # index could safely drift nearly two steps. Measured on Plasma at
        # 400x100: 10,579 colour runs against 2,203, which is 80% of the cost
        # of the mode, given away to protect a handful of indices that needed
        # protecting.
        #
        # Each index gets the drift *it* can afford. The guarantee is unchanged
        # and still the one the constant names: no cell is more than
        # ``_RLE_MAX_RGB`` per channel from the colour it asked for. A steep
        # index still gets 0 and still splits exactly as before.
        # Symmetric, which the first version was not: a run starting at index
        # i absorbs values both above and below it, so i's budget has to hold
        # for rgb[i-t] as well as rgb[i+t]. Checking only forward let `classic`
        # drift 16/255 against a bound of 10 — caught by measuring the colour
        # actually emitted for every cell rather than by trusting the
        # construction.
        n = len(rgb_int)
        budget = np.zeros(n, dtype=np.int32)
        for i in range(n):
            t = 0
            while t < _RLE_MAX_TOL:
                nxt = t + 1
                lo, hi = i - nxt, i + nxt
                ok = True
                if lo >= 0:
                    ok = int(np.abs(rgb_int[lo] - rgb_int[i]).max()) <= _RLE_MAX_RGB
                if ok and hi < n:
                    ok = int(np.abs(rgb_int[hi] - rgb_int[i]).max()) <= _RLE_MAX_RGB
                if not ok:
                    break
                t = nxt
            budget[i] = t
        self.rle_budget = budget

    # ── lookups ──
    def pair_style(self, key: int) -> Style:
        """Style for a packed ``fg * RAMP_STEPS + bg`` index."""
        st = self.pair_styles.get(key)
        if st is None:
            st = Style.from_color(
                color=self.colors[key // RAMP_STEPS],
                bgcolor=self.colors[key % RAMP_STEPS],
            )
            self.pair_styles[key] = st
        return st

    def index(self, norm: float) -> int:
        i = int(norm * (RAMP_STEPS - 1))
        return 0 if i < 0 else (RAMP_STEPS - 1 if i >= RAMP_STEPS else i)

    def indices(self, norm: np.ndarray) -> np.ndarray:
        """Vectorised float field -> ramp index array (int32)."""
        return np.clip((norm * (RAMP_STEPS - 1)).astype(np.int32), 0, RAMP_STEPS - 1)

    def style(self, norm: float) -> Style:
        return self.styles[self.index(norm)]


def _C(hex_value: str):
    from rich.color import Color

    return Color.parse(hex_value)


# ── deriving a ramp from a Textual theme ─────────────────────────────────────


def theme_from_textual(app) -> Theme | None:
    """Build a spektr theme out of whatever Textual theme the app is wearing.

    Kept from the original design because it genuinely does harmonise with the
    surrounding chrome — but it is now one entry in the theme list rather than
    the entire theme system.
    """
    t = getattr(app, "current_theme", None)
    if t is None:
        try:
            t = app.get_theme(app.theme)
        except Exception:
            return None
    if t is None:
        return None

    primary = getattr(t, "primary", None) or "#ffb000"
    bg = getattr(t, "background", None) or "#000000"
    fg = getattr(t, "foreground", None) or "#ffffff"
    accent = getattr(t, "accent", None) or getattr(t, "secondary", None) or primary
    return Theme(
        name=f"auto:{getattr(t, 'name', 'theme')}",
        low=mix(primary, bg, 0.55),
        mid=primary,
        high=mix(accent, fg, 0.35),
        bg=bg,
        fg=fg,
        accent=accent,
    )


# ── the theme editor's draft model ───────────────────────────────────────────


#: Slots the editor exposes by default. A theme has six; four is what someone
#: who has not read the ramp documentation can pick meaningfully, and bg/fg are
#: derived from these well enough that most people never need the other two.
BASIC_SLOTS = ("low", "mid", "high", "accent")
#: What the advanced toggle unlocks, in addition to the four above.
ADVANCED_SLOTS = ("bg", "fg")


class ThemeDraft:
    """A theme being edited, in HSL, with bg/fg derived until they are not.

    Held in HSL rather than as hex strings because that is the axis the editor
    nudges along, and repeatedly round-tripping a hue through 8-bit RGB
    quantises it: nudge hue down and back up ten times through hex and you do
    not return to where you started, so the control feels like it is slipping.
    Hex is generated on demand instead.

    ``advanced`` unlocks bg and fg for direct editing. While it is off they are
    recomputed from the ramp on every change, which is what makes a four-colour
    pick produce a usable six-slot theme.
    """

    def __init__(self, base: "Theme | None" = None, name: str = "custom"):
        src = base or BUILTIN["classic"]
        self.name = name
        self.advanced = False
        self._hsl = {
            "low": hex_to_hsl(src.low),
            "mid": hex_to_hsl(src.mid),
            "high": hex_to_hsl(src.high),
            "accent": hex_to_hsl(src.accent or src.mid),
            "bg": hex_to_hsl(src.bg),
            "fg": hex_to_hsl(src.fg),
        }

    @property
    def slots(self) -> tuple[str, ...]:
        return BASIC_SLOTS + ADVANCED_SLOTS if self.advanced else BASIC_SLOTS

    def hex_of(self, slot: str) -> str:
        if slot in ADVANCED_SLOTS and not self.advanced:
            low, mid, high = (self.hex_of(s) for s in ("low", "mid", "high"))
            bg = derive_bg(low, mid, high)
            return bg if slot == "bg" else derive_fg(high, bg)
        return hsl_to_hex(*self._hsl[slot])

    def component(self, slot: str, which: str) -> float:
        return self._hsl[slot]["hsl".index(which)]

    def set_advanced(self, on: bool) -> None:
        """Unlock bg and fg for editing, or hand them back to the derivation.

        Turning it on seeds both from the values that *were* being derived, so
        flipping the toggle changes nothing on screen — it only makes two more
        colours reachable. Seeding from the original base theme instead would
        make the picture jump the moment the toggle was touched, which reads
        as the editor having lost the user's work.
        """
        on = bool(on)
        if on and not self.advanced:
            derived = {slot: self.hex_of(slot) for slot in ADVANCED_SLOTS}
            for slot, colour in derived.items():
                self._hsl[slot] = hex_to_hsl(colour)
        self.advanced = on

    def nudge(self, slot: str, which: str, delta: float) -> None:
        """Move one HSL component of one slot. Hue wraps; the others clamp."""
        h, s, lum = self._hsl[slot]
        i = "hsl".index(which)
        values = [h, s, lum]
        values[i] = (values[i] + delta) % 1.0 if i == 0 else min(1.0, max(0.0, values[i] + delta))
        self._hsl[slot] = (values[0], values[1], values[2])

    def set_slot(self, slot: str, colour: str) -> None:
        """Replace one slot outright from a hex value — the picker's path.

        The editor is otherwise all nudges; a picked colour has to land
        somewhere, and converting once here means the nudge rows pick up from
        exactly the chosen colour (see the docstring for why the draft lives
        in HSL rather than hex).
        """
        self._hsl[slot] = hex_to_hsl(colour)

    def to_theme(self, name: str | None = None) -> "Theme":
        return Theme(
            name=name or self.name,
            low=self.hex_of("low"),
            mid=self.hex_of("mid"),
            high=self.hex_of("high"),
            bg=self.hex_of("bg"),
            fg=self.hex_of("fg"),
            accent=self.hex_of("accent"),
        )

    def problems(self) -> list[str]:
        return theme_visibility_problems(self.to_theme())


def sanitise_theme_name(name: str) -> str:
    """A filename-safe, lowercase theme name. Empty input becomes ``custom``."""
    kept = [c if (c.isalnum() or c in "-_") else "-" for c in str(name).strip().lower()]
    cleaned = "".join(kept).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned[:32] or "custom"


def available_theme_name(name: str, config_dir: Path | None = None) -> str:
    """``name``, suffixed until it collides with nothing that already exists.

    Suffix rather than reject. A built-in name is a *good* name for a variant
    on it — someone editing ``gruvbox`` and calling the result ``gruvbox`` is
    being reasonable — and silently shadowing the built-in would remove it
    from the picker with no way back short of deleting a file by hand.
    """
    base = sanitise_theme_name(name)
    taken = set(all_themes(config_dir))
    if base not in taken:
        return base
    for n in range(2, 100):
        candidate = f"{base}-{n}"
        if candidate not in taken:
            return candidate
    return f"{base}-{os.getpid()}"


def save_user_theme(theme: "Theme", config_dir: Path | None = None) -> Path:
    """Write a theme to ``<config>/themes/<name>.toml``, and return the path.

    Hand-rolled TOML: the standard library reads it and cannot write it, and
    six quoted hex strings do not justify a dependency. Every value here is a
    ``#rrggbb`` produced by ``rgb_to_hex``, so there is nothing to escape.
    """
    folder = _root(config_dir) / "themes"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{theme.name}.toml"
    body = "\n".join(
        [
            f"# {theme.name} — written by spektr's theme editor",
            f'low    = "{theme.low}"',
            f'mid    = "{theme.mid}"',
            f'high   = "{theme.high}"',
            f'bg     = "{theme.bg}"',
            f'fg     = "{theme.fg}"',
            f'accent = "{theme.accent}"',
            "",
        ]
    )
    path.write_text(body, encoding="utf-8")
    return path
