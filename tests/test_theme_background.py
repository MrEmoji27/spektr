"""Every visualizer cell has to name the theme's background colour.

Textual draws the lines a Line-API widget returns exactly as they are handed
over; the widget's own ``styles.background`` only fills padding. So a segment
whose style has no bgcolor goes to the terminal as "default background", and
the terminal paints its own scheme there. On Windows Terminal running
Catppuccin Mocha at 80% opacity, gruvbox rendered on #1e1e2e and flexoki-light
on a dark ground instead of cream — logged at the byte level as foreground-only
SGR sequences for every cell below the header.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spektr.palette import BUILTIN, Palette  # noqa: E402
from spektr.render import SPACE, make_strips  # noqa: E402


@pytest.mark.parametrize("theme", sorted(BUILTIN))
def test_every_cell_carries_the_theme_background(theme):
    pal = Palette(BUILTIN[theme])
    codes = np.full((6, 30), SPACE, dtype=np.int32)
    codes[2, 3:9] = ord("█")
    codes[4, 10] = ord("·")
    cidx = np.tile(np.arange(30, dtype=np.int32) * 2, (6, 1))
    want = BUILTIN[theme].bg.lower()
    for strip in make_strips(codes, cidx, pal):
        for seg in strip:
            assert seg.style is not None and seg.style.bgcolor is not None, "a cell with no background"
            assert seg.style.bgcolor.triplet.hex == want


def test_an_animated_theme_keeps_its_background_as_the_ramp_turns():
    pal = Palette(BUILTIN["rainbow"])
    codes = np.full((2, 8), ord("█"), dtype=np.int32)
    cidx = np.tile(np.arange(8, dtype=np.int32) * 8, (2, 1))
    for phase in (0.0, 0.37, 0.81):
        pal.set_phase(phase)
        for strip in make_strips(codes, cidx, pal):
            for seg in strip:
                assert seg.style.bgcolor.triplet.hex == BUILTIN["rainbow"].bg.lower()


# ── transparent: the terminal's background instead ───────────────────────────
#
# The inverse, for ``Settings.transparent_background``. A terminal draws a cell
# translucent only when the cell names no background, so with the setting on no
# segment may carry the theme's background, on either path through make_strips.

from spektr import config  # noqa: E402
from spektr.palette import RAMP_STEPS  # noqa: E402


def _pair_grid():
    """A two-colour grid with every kind of cell the pair path sees: floor
    background under a lit glyph, uniform floor, and real field colours."""
    h, w = 6, 40
    codes = np.full((h, w), ord("▀"), dtype=np.int32)
    codes[1, :] = ord("█")
    codes[2, 5:15] = SPACE
    cidx = np.tile((np.arange(w, dtype=np.int32) * 3) % RAMP_STEPS, (h, 1))
    bidx = np.tile((np.arange(w, dtype=np.int32) * 5 + 2) % RAMP_STEPS, (h, 1))
    bidx[0, :] = 0                      # floor behind lit cells
    cidx[3, 10:30] = 0
    bidx[3, 10:30] = 0                  # uniform floor: nothing to draw
    bidx[4, ::2] = 0                    # floor alternating with colour, to tempt the run merge
    bidx[5, :] = np.arange(w) % 3       # floor next to its nearest neighbours
    return codes, cidx, bidx


def _cells(strips):
    """Per cell: (text, fg hex or None, bg hex or None)."""
    out = []
    for strip in strips:
        row = []
        for seg in strip:
            fg = seg.style.color.triplet.hex if seg.style and seg.style.color else None
            bg = seg.style.bgcolor.triplet.hex if seg.style and seg.style.bgcolor else None
            row.extend((ch, fg, bg) for ch in seg.text)
        out.append(row)
    return out


def _rgb(hex_value):
    return np.array([int(hex_value[i:i + 2], 16) for i in (1, 3, 5)])


@pytest.mark.parametrize("theme", sorted(BUILTIN))
def test_transparent_cells_never_name_the_theme_background(theme):
    pal = Palette(BUILTIN[theme], transparent=True)
    want = BUILTIN[theme].bg.lower()
    codes = np.full((6, 30), SPACE, dtype=np.int32)
    codes[2, 3:9] = ord("█")
    codes[4, 10] = ord("·")
    cidx = np.tile(np.arange(30, dtype=np.int32) * 2, (6, 1))
    for strip in make_strips(codes, cidx, pal):
        for seg in strip:
            assert seg.style.bgcolor is None, "a foreground-only cell carries a background"
    p_codes, p_cidx, p_bidx = _pair_grid()
    for strip in make_strips(p_codes, p_cidx, pal, p_bidx):
        for seg in strip:
            assert seg.style.bgcolor is None or seg.style.bgcolor.triplet.hex != want


@pytest.mark.parametrize("theme", ["classic", "flexoki-light", "gruvbox", "rainbow"])
def test_transparent_pairs_clear_exactly_the_floor(theme):
    """Cell by cell against what was asked for: a floor background goes to the
    terminal, every other background is its ramp colour, a uniform floor cell
    is a space, and no merged run carries a colour across the floor."""
    codes, cidx, bidx = _pair_grid()
    pal = Palette(BUILTIN[theme], transparent=True)
    got = _cells(make_strips(codes, cidx, pal, bidx))
    for y in range(codes.shape[0]):
        for x in range(codes.shape[1]):
            ch, fg, bg = got[y][x]
            if bidx[y, x] == 0:
                assert bg is None, f"floor background painted at {y},{x}"
                if cidx[y, x] == 0:
                    assert ch == " ", f"uniform floor cell drew {ch!r} at {y},{x}"
            else:
                assert bg is not None, f"a field colour was cleared at {y},{x}"
                drift = np.abs(_rgb(pal.hexes[bidx[y, x]]) - _rgb(bg)).max()
                assert drift <= 10, f"background drifted {drift} at {y},{x}"
            if ch != " ":
                drift = np.abs(_rgb(pal.hexes[cidx[y, x]]) - _rgb(fg)).max()
                assert drift <= 10, f"foreground drifted {drift} at {y},{x}"


def test_solid_pairs_keep_every_background():
    """The default path must not route through the cleared-marker packing."""
    codes, cidx, bidx = _pair_grid()
    pal = Palette(BUILTIN["gruvbox"])
    got = _cells(make_strips(codes, cidx, pal, bidx))
    for y in range(codes.shape[0]):
        for x in range(codes.shape[1]):
            assert got[y][x][2] is not None, "solid mode left a pair cell without a background"
            assert got[y][x][0] == chr(codes[y, x])


def test_an_animated_theme_stays_transparent_as_the_ramp_turns():
    """Both paths, across phases, with the pair indices shifted by column the
    way the widget does for animated themes — the floor decided beforehand."""
    from spektr.widget import AudioVisualizer

    pal = Palette(BUILTIN["rainbow"], transparent=True)
    want = BUILTIN["rainbow"].bg.lower()
    fg_codes = np.full((2, 8), ord("█"), dtype=np.int32)
    fg_cidx = np.tile(np.arange(8, dtype=np.int32) * 8, (2, 1))
    for phase in (0.0, 0.37, 0.81):
        pal.set_phase(phase)
        for strip in make_strips(fg_codes, fg_cidx, pal):
            for seg in strip:
                assert seg.style.bgcolor is None
        codes, cidx, bidx = _pair_grid()
        clear = bidx == 0
        cidx, bidx = AudioVisualizer._animate_ramp(None, codes, cidx, bidx, codes.shape[1])
        got = _cells(make_strips(codes, cidx, pal, bidx, clear))
        for y in range(codes.shape[0]):
            for x in range(codes.shape[1]):
                bg = got[y][x][2]
                assert bg != want
                assert (bg is None) == bool(clear[y, x]), f"phase {phase}: wrong floor at {y},{x}"


def test_real_two_colour_modes_clear_only_their_floor():
    """The modes that hand make_strips a background index, rendered transparent."""
    import spektr.modes as M
    from spektr.modes import Ctx

    b = np.linspace(0.8, 0.2, 32)
    pal = Palette(BUILTIN["flexoki-light"], transparent=True)
    want = BUILTIN["flexoki-light"].bg.lower()
    for name in ("Radial (o)", "Chladni", "Maelstrom (o)", "Plasma", "Kaleidoscope", "Swell"):
        st: dict = {}
        for f in range(12):
            ctx = Ctx(w=40, h=12, bands=b, peaks=b, bands_l=b, bands_r=b, wave=np.sin(np.arange(512) / 9),
                      stereo=np.zeros((512, 2)), frame=f, t=f / 60, dt=1 / 60, energy=0.5, silent=False,
                      palette=pal, state=st, onsets=int(f % 6 == 0), onset_strength=0.9)
            out = M.get(name).fn(ctx)
        codes, cidx, bidx = out[:3]
        got = _cells(make_strips(codes, cidx, pal, bidx))
        for y in range(codes.shape[0]):
            for x in range(codes.shape[1]):
                bg = got[y][x][2]
                assert bg != want, f"{name}: theme background at {y},{x}"
                assert (bg is None) == bool(bidx[y, x] == 0), f"{name}: wrong cells cleared at {y},{x}"


def test_toggling_rebuilds_the_styles_with_nothing_stale():
    pal = Palette(BUILTIN["gruvbox"])
    want = BUILTIN["gruvbox"].bg.lower()
    codes, cidx, bidx = _pair_grid()
    make_strips(codes, cidx, pal, bidx)            # fill the pair cache under solid
    fg_codes = np.full((2, 8), ord("█"), dtype=np.int32)
    fg_cidx = np.tile(np.arange(8, dtype=np.int32) * 8, (2, 1))
    for on, expect_bg in ((True, None), (False, want), (True, None)):
        pal.set_transparent(on)
        assert pal.transparent is on
        for strip in make_strips(fg_codes, fg_cidx, pal):
            for seg in strip:
                got = seg.style.bgcolor.triplet.hex if seg.style.bgcolor else None
                assert got == expect_bg, f"transparent={on}: stale style"
        cells = _cells(make_strips(codes, cidx, pal, bidx))
        floor = [cells[0][x][2] for x in range(codes.shape[1])]
        assert all((c is None) == on for c in floor), f"transparent={on}: stale pair styles"


def test_the_widget_toggles_live_and_the_setting_survives_a_restart(tmp_path):
    from spektr.app import Spektr
    from spektr.widget import AudioVisualizer

    app = Spektr(settings=config.Settings())
    viz = AudioVisualizer(settings=app.settings)
    assert viz.palette.transparent is False
    viz._strips = ["stale"]
    assert viz.set_transparent_background(True) is True
    assert viz.palette.transparent is True and app.settings.transparent_background is True
    assert viz._strips is None, "a frame built under solid styles survived the toggle"
    assert all(s.bgcolor is None for s in viz.palette.styles)
    viz.set_transparent_background(False)
    assert all(s.bgcolor is not None for s in viz.palette.styles)
    viz.set_transparent_background(True)

    config.save(app.settings, config_dir=tmp_path)
    back = config.load(config_dir=tmp_path)
    assert back.transparent_background is True
    reborn = AudioVisualizer(settings=back)
    assert reborn.palette.transparent is True, "a restart came back solid"
    config.save(config.Settings(), config_dir=tmp_path)
    assert config.load(config_dir=tmp_path).transparent_background is False


def test_the_setting_is_clamped_to_a_bool():
    assert config.Settings(transparent_background=1).clamp().transparent_background is True
    assert config.Settings(transparent_background=None).clamp().transparent_background is False
