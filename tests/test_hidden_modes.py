"""Hidden modes: out of the interface, still in the app.

The twelve subcell variants — the ``Fine`` modes and ``Kaleidoscope Ultra (o)`` —
are opt-in. They draw the same pictures as the originals at four times the
subcell resolution, but they need a font with Unicode 16 octants, they cost
roughly twice as much, and listing both halves of every pair doubles the menu.
So the originals are what the interface offers.

"Not offered" has to mean exactly that and nothing more: they are still
registered, still rendered, still tested by the audit, and still selectable by
name, because a config file or a ``--mode`` flag naming one has to keep
working. Hiding a mode must not quietly change what someone's setup does.
"""

from __future__ import annotations

import numpy as np
import pytest

import spektr.modes as M


def test_something_is_hidden_and_most_things_are_not():
    hidden = [m.name for m in M.MODES if m.hidden]
    assert hidden, "nothing is hidden — did the flag stop being applied?"
    assert len(M.listed()) > len(hidden), "more modes are hidden than shown"
    assert len(M.listed()) + len(hidden) == len(M.MODES)


def test_hidden_modes_are_still_registered_and_reachable():
    for m in M.MODES:
        if m.hidden:
            assert M.get(m.name) is m, f"{m.name} is hidden and also unreachable"
            assert m.name in M.names(), f"{m.name} vanished from names()"


def test_listed_excludes_exactly_the_hidden_ones():
    assert [m.name for m in M.listed()] == [m.name for m in M.MODES if not m.hidden]


def test_the_hidden_set_is_exactly_the_subcell_variants():
    """Nothing else may drift into being opt-in without saying so here."""
    assert {m.name for m in M.MODES if m.hidden} == {
        "Scope (o)", "ECG (o)", "Radial (o)", "Sonar (o)", "Plasma (o)",
        "Chladni (o)", "Chladni Flow (o)", "Chladni Extreme (o)",
        "Kaleidoscope (o)", "Kaleidoscope Ultra (o)", "Valentine (o)",
        "Maelstrom (o)",
    }


@pytest.mark.parametrize("name", [m.name for m in M.MODES if m.hidden])
def test_every_hidden_variant_has_its_original_on_the_menu(name):
    """A mode may only be hidden because something else stands in for it."""
    base = name.removesuffix(" (o)").removesuffix(" Ultra")
    original = M.get(base)
    assert original is not None, f"{name} is hidden and {base} does not exist"
    assert not original.hidden, f"{name} is hidden and so is {base} — nothing is offered"


@pytest.mark.parametrize("name", ["Radial", "Maelstrom", "Plasma", "Chladni"])
def test_the_originals_are_what_the_interface_offers(name):
    m = M.get(name)
    assert m is not None and not m.hidden


def _viz(**settings):
    from spektr.config import Settings
    from spektr.widget import AudioVisualizer

    return AudioVisualizer(settings=Settings(**settings))


def test_the_setting_puts_the_variants_on_the_menu():
    off, on = _viz(), _viz(fine_modes=True)
    hidden = {m.name for m in M.MODES if m.hidden}

    assert not (hidden & set(off.mode_names)), "a hidden mode is being offered"
    assert hidden <= set(on.mode_names), "fine_modes did not bring them in"
    assert len(on.mode_names) == len(off.mode_names) + len(hidden)


def test_the_setting_flips_live():
    """The settings panel toggles the attribute; the list follows immediately."""
    viz = _viz()
    assert "Chladni (o)" not in viz.mode_names
    viz.show_fine = True
    assert "Chladni (o)" in viz.mode_names
    viz.show_fine = False
    assert "Chladni (o)" not in viz.mode_names


def test_the_originals_are_always_offered_either_way():
    for viz in (_viz(), _viz(fine_modes=True)):
        for name in ("Chladni", "Plasma", "Scope", "Valentine"):
            assert name in viz.mode_names, f"{name} went missing from the menu"


def test_a_hidden_mode_can_still_be_selected_while_it_is_hidden():
    """--mode and a saved config name one; hiding must not break either."""
    viz = _viz()
    viz.set_mode("Chladni (o)")
    assert viz.mode_name == "Chladni (o)"


def test_the_setting_survives_a_round_trip(tmp_path):
    from spektr import config

    s = config.Settings(fine_modes=True, cells="quadrant")
    config.save(s, config_dir=tmp_path)
    back = config.load(config_dir=tmp_path)
    assert back.fine_modes is True
    assert back.cells == "quadrant"


def test_junk_in_the_config_falls_back_rather_than_raising():
    from spektr.config import Settings

    s = Settings(fine_modes="yes please", cells="sextants").clamp()
    assert s.fine_modes is True            # any truthy string is "on"
    assert s.cells == "octant"             # not a geometry spektr has


def test_a_hidden_mode_still_draws():
    """The point of hiding rather than deleting."""
    from spektr.analysis import N_BANDS
    from spektr.modes import Ctx
    from spektr.palette import BUILTIN, Palette

    pal = Palette(BUILTIN["gruvbox"])
    bands = np.linspace(0.2, 0.9, N_BANDS).astype(np.float32)
    hidden = [m for m in M.MODES if m.hidden]
    assert hidden

    for m in hidden:
        state: dict = {}
        for frame in range(3):
            out = m.fn(
                Ctx(
                    w=80, h=24, bands=bands, peaks=bands, bands_l=bands, bands_r=bands,
                    wave=np.zeros(512, dtype=np.float32),
                    stereo=np.zeros((512, 2), dtype=np.float32),
                    frame=frame, t=frame / 60.0, dt=1 / 60.0, energy=0.5,
                    silent=False, palette=pal, state=state, bars=len(bands),
                )
            )
        assert out[0].shape == (24, 80), f"{m.name} stopped rendering when hidden"


# ── the (o)/(q) naming ───────────────────────────────────────────────────────

def test_the_variants_are_named_for_the_geometry_not_for_being_finer():
    """They were "Fine", which said better rather than which glyphs."""
    for m in M.MODES:
        if m.hidden:
            assert m.name.endswith(" (o)"), f"{m.name} is a variant without a geometry suffix"
        else:
            assert not m.name.endswith((" (o)", " (q)")), f"{m.name} claims a geometry"


@pytest.mark.parametrize("old, new", sorted(M._RENAMED.items()))
def test_every_name_these_modes_ever_had_still_resolves(old, new):
    """A config or --mode flag from an earlier version must not fall to Bars.

    A mode that stops resolving is silent: ``set_mode`` ignores it and the app
    carries on with whatever it had, so the failure looks like the setting
    being forgotten rather than like an error.
    """
    assert M.get(old) is M.get(new) is not None


@pytest.mark.parametrize("name", [m.name for m in M.MODES if m.hidden])
def test_the_quadrant_spelling_resolves_too(name):
    """The picker shows (q) in quadrant mode, so (q) is a name people will type."""
    assert M.get(name[:-4] + " (q)") is M.get(name)


def test_the_label_follows_the_cell_setting():
    """The suffix reports the geometry actually being drawn, which is a setting."""
    import spektr.render as R

    before = R.CELL_MODE
    try:
        R.set_cell_mode("octant")
        assert M.label("Chladni (o)") == "Chladni (o)"
        R.set_cell_mode("quadrant")
        assert M.label("Chladni (o)") == "Chladni (q)"
        assert M.label("Bars") == "Bars", "a plain mode must not grow a suffix"
    finally:
        R.set_cell_mode(before)


def test_no_suffixed_name_is_visible_until_the_setting_is_on():
    """The (o)/(q) names belong to opt-in modes and must not leak.

    With subcell modes off, nothing in the interface should mention a geometry
    — the picker, the cycle keys and shuffle all read the same list, so one
    check covers all three.
    """
    off, on = _viz(), _viz(fine_modes=True)

    leaked = [n for n in off.mode_names if n.endswith((" (o)", " (q)"))]
    assert not leaked, f"geometry suffixes shown while the setting is off: {leaked}"

    shown = [n for n in on.mode_names if n.endswith((" (o)", " (q)"))]
    assert len(shown) == len([m for m in M.MODES if m.hidden])


def test_cycling_never_lands_on_a_suffixed_mode_while_it_is_off():
    """`m`/space walk the same list, so this is the keybinding's guarantee."""
    viz = _viz()
    seen = {viz.cycle_mode(1) for _ in range(len(viz.mode_names) + 5)}
    assert not [n for n in seen if n.endswith((" (o)", " (q)"))]
