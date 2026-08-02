"""Theme registry and colour ramps.

A theme is six colours. Three of them (low/mid/high) are the spectrum anchors
that get blended into a 64-step ramp; the other three (bg/fg/accent) dress the
UI. Built-ins live in a dict — no file IO on startup — and user themes are read
from ``~/.config/spektr/themes/*.toml`` at first use.

Blending happens in linear light rather than straight sRGB. Interpolating hex
values directly darkens the midpoint of a gradient noticeably (mixing #00ff41
and #ff3300 in sRGB gives you a muddy olive); going through gamma keeps the
midtones where the eye expects them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.style import Style

RAMP_STEPS = 64
_GAMMA = 2.2


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


def _t(name, low, mid, high, bg, fg, accent="") -> tuple[str, Theme]:
    return name, Theme(name, low, mid, high, bg, fg, accent)


BUILTIN: dict[str, Theme] = dict(
    [
        # the original — winamp's green/amber/red
        _t("classic", "#00ff41", "#ffb000", "#ff3300", "#000000", "#e0e0e0", "#ffb000"),
        _t("gruvbox", "#b8bb26", "#fabd2f", "#fb4934", "#282828", "#ebdbb2", "#83a598"),
        _t("catppuccin", "#a6e3a1", "#f9e2af", "#f38ba8", "#1e1e2e", "#cdd6f4", "#cba6f7"),
        _t("catppuccin-latte", "#40a02b", "#df8e1d", "#d20f39", "#eff1f5", "#4c4f69", "#8839ef"),
        _t("dracula", "#50fa7b", "#f1fa8c", "#ff5555", "#282a36", "#f8f8f2", "#bd93f9"),
        _t("nord", "#a3be8c", "#ebcb8b", "#bf616a", "#2e3440", "#d8dee9", "#88c0d0"),
        _t("tokyo-night", "#9ece6a", "#e0af68", "#f7768e", "#1a1b26", "#c0caf5", "#7aa2f7"),
        _t("rose-pine", "#9ccfd8", "#f6c177", "#eb6f92", "#191724", "#e0def4", "#c4a7e7"),
        _t("everforest", "#a7c080", "#dbbc7f", "#e67e80", "#2d353b", "#d3c6aa", "#7fbbb3"),
        _t("kanagawa", "#98bb6c", "#e6c384", "#e46876", "#1f1f28", "#dcd7ba", "#7e9cd8"),
        _t("ayu-mirage", "#bae67e", "#ffcc66", "#f28779", "#1f2430", "#cbccc6", "#73d0ff"),
        _t("monokai", "#a6e22e", "#e6db74", "#f92672", "#272822", "#f8f8f2", "#66d9ef"),
        _t("solarized", "#859900", "#b58900", "#dc322f", "#002b36", "#839496", "#268bd2"),
        _t("nightfox", "#81b29a", "#dbc074", "#c94f6d", "#192330", "#cdcecf", "#719cff"),
        _t("oxocarbon", "#3ddbd9", "#33b1ff", "#be95ff", "#161616", "#f2f4f8", "#ee5396"),
        _t("miasma", "#78834b", "#bb7744", "#e0a363", "#222222", "#c2c2b0", "#8f6f5f"),
        _t("osaka-jade", "#43a58a", "#8ec07c", "#e06c75", "#111c18", "#c1c8c4", "#549e6a"),
        _t("ristretto", "#adda78", "#f9cc6c", "#fd6883", "#2c2525", "#e6d9db", "#f38d70"),
        _t("flexoki-light", "#66800b", "#ad8301", "#af3029", "#fffcf0", "#100f0f", "#205ea6"),
        # ── moods rather than editor ports ──
        _t("hackerman", "#005f11", "#00cc33", "#00ff41", "#000000", "#00ff41", "#00ff41"),
        _t("ember", "#6b2d00", "#ff7a18", "#ffd166", "#17110d", "#f0e0d0", "#ff7a18"),
        _t("ethereal", "#6affc2", "#7ab8ff", "#d3a4ff", "#0f1020", "#e8e8ff", "#9d7aff"),
        _t("synthwave", "#03edf9", "#ff7edb", "#fede5d", "#241b2f", "#f4eee4", "#ff7edb"),
        _t("blade-runner", "#00e5ff", "#b14aed", "#ff2e88", "#0b0c17", "#d6deeb", "#00e5ff"),
        _t("nostromo", "#4a2600", "#ff8c00", "#ffd08a", "#0d0700", "#ffb454", "#ff8c00"),
        _t("plasma", "#0d0887", "#cc4778", "#f0f921", "#0a0612", "#f0e8ff", "#cc4778"),
        _t("viridis", "#440154", "#21918c", "#fde725", "#0b0a12", "#e8f0e8", "#21918c"),
        _t("ice", "#0a2a5e", "#3aa0ff", "#e8f6ff", "#050b16", "#cfe6ff", "#3aa0ff"),
        _t("matte-black", "#4a4a4a", "#8a8a8a", "#d5d5d5", "#121212", "#bcbcbc", "#eaeaea"),
        _t("vantablack", "#333333", "#888888", "#ffffff", "#000000", "#ffffff", "#ffffff"),
        # ── more editor ports ──
        _t("nightfly", "#a1cd5e", "#e3d18a", "#fc514e", "#011627", "#c3ccdc", "#82aaff"),
        _t("material", "#c3e88d", "#ffcb6b", "#f07178", "#263238", "#eeffff", "#82aaff"),
        _t("gotham", "#2aa889", "#edb443", "#d26937", "#0c1014", "#98d1ce", "#195466"),
        _t("oceanic", "#99c794", "#fac863", "#ec5f67", "#1b2b34", "#d8dee9", "#6699cc"),
        _t("gruvbox-light", "#79740e", "#b57614", "#9d0006", "#fbf1c7", "#3c3836", "#076678"),
        _t("tokyo-night-day", "#587539", "#8f5e15", "#f52a65", "#e1e2e7", "#3760bf", "#2e7de9"),
        # ── more moods ──
        _t("vaporwave", "#00f0c0", "#ff77e9", "#ff2e88", "#1a0b2e", "#f2e6ff", "#b967ff"),
        _t("infrared", "#3a0000", "#c22800", "#ffd000", "#0d0000", "#ffb4a2", "#ff4800"),
        _t("deep-sea", "#0a3d62", "#12cbc4", "#a5f3ff", "#04141f", "#b8e0e6", "#12cbc4"),
        _t("magma", "#2c115f", "#b73779", "#fcfdbf", "#0b0417", "#f5e3ff", "#fe9f6d"),
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


def load_user_themes() -> dict[str, Theme]:
    """Read ``<config>/themes/*.toml``. A malformed file is skipped, not fatal."""
    try:
        import tomllib
    except ModuleNotFoundError:  # 3.10
        try:
            import tomli as tomllib  # type: ignore
        except ModuleNotFoundError:
            return {}

    try:
        folder = config_dir() / "themes"
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


def all_themes() -> dict[str, Theme]:
    """Built-ins, with user themes of the same name taking priority."""
    merged = dict(BUILTIN)
    merged.update(load_user_themes())
    return dict(sorted(merged.items()))


# ── the live palette ─────────────────────────────────────────────────────────

class Palette:
    """The active theme, plus everything derived from it, prebuilt.

    Every renderer works in *ramp indices* (0..RAMP_STEPS-1) rather than colour
    strings, so a frame never touches a hex value or parses a style. Swapping
    theme rebuilds these tables once and the renderers carry on unchanged.
    """

    __slots__ = (
        "theme", "name", "hexes", "colors", "styles", "bg_styles", "rgb",
        "note", "pair_styles",
    )

    def __init__(self, theme: Theme | None = None):
        self.set(theme or BUILTIN["classic"])

    def set(self, theme: Theme) -> None:
        self.theme = theme
        self.name = theme.name
        self._build()

    def _build(self) -> None:
        th = self.theme
        anchors = [hex_to_rgb(th.low), hex_to_rgb(th.mid), hex_to_rgb(th.high)]
        lin = _to_linear(np.array(anchors, dtype=np.float64))

        t = np.linspace(0.0, 1.0, RAMP_STEPS)
        # piecewise low->mid->high, smoothstepped so the anchors don't crease
        seg = np.where(t < 0.5, t * 2.0, (t - 0.5) * 2.0)
        seg = seg * seg * (3.0 - 2.0 * seg)
        first = t < 0.5
        a = np.where(first[:, None], lin[0][None, :], lin[1][None, :])
        b = np.where(first[:, None], lin[1][None, :], lin[2][None, :])
        out = a + (b - a) * seg[:, None]

        self.rgb = _to_srgb(out)
        self.hexes = [rgb_to_hex(c) for c in self.rgb]
        self.colors = [_C(h) for h in self.hexes]
        self.styles = [Style.from_color(color=c) for c in self.colors]
        self.bg_styles = [Style.from_color(bgcolor=c) for c in self.colors]
        # Combined fg-on-bg styles, filled on demand and dropped when the theme
        # changes. Building all RAMP_STEPS**2 of them up front would be 4096
        # Style objects per theme swap to serve the handful of pairs a frame
        # actually uses; caching here instead of in make_strips is the point,
        # because that cache was rebuilt sixty times a second.
        self.pair_styles = {}
        self.note = f"{th.name} — {th.low} → {th.high}"

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
        return np.clip(
            (norm * (RAMP_STEPS - 1)).astype(np.int32), 0, RAMP_STEPS - 1
        )

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
