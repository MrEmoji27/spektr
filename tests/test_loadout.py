"""The loadout: which modes the interface offers at all.

Spektr registers far more modes than anyone wants in a shuffle rotation, and
cycling past fifty to reach the four you like is the problem this solves. The
loadout is a set of names in the settings, edited in a modal on ``V``, and it
narrows one thing: :attr:`AudioVisualizer.mode_names`. That property is the
single list the picker, the ``m``/space cycle keys and shuffle all read, which
is why the filter lives there and not in three places.

Two properties matter more than the plumbing, and both are about not trapping
someone in a state they cannot get out of:

* An empty loadout means *no restriction*, not "no modes". It is the default,
  so a config that has never seen the modal behaves exactly as before.
* The loadout is a filter over the offered modes, never a source of them. A
  name in it that is quarantined, hidden or no longer registered stays out —
  and if that leaves nothing at all, the full list comes back rather than an
  interface with no modes in it.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spektr import config  # noqa: E402
from spektr.app import Spektr  # noqa: E402
from spektr.pickers import LoadoutPicker  # noqa: E402
from spektr.widget import AudioVisualizer  # noqa: E402


def _viz(**kw) -> AudioVisualizer:
    return AudioVisualizer(settings=config.Settings(**kw).clamp())


# ── the filter ───────────────────────────────────────────────────────────────

def test_an_empty_loadout_restricts_nothing():
    """The default. Nothing changes for a config that never opened the modal."""
    assert _viz(loadout=[]).mode_names == _viz().mode_names
    assert len(_viz(loadout=[]).mode_names) > 1


def test_a_loadout_narrows_what_is_offered():
    names = _viz(loadout=["Bars", "Star Trails"]).mode_names
    assert names == ["Bars", "Star Trails"]


def test_the_loadout_filters_and_never_invents():
    """A name that is not on offer does not come back just for being listed.

    The loadout says which of the available modes you want, so it can only
    ever subtract. Otherwise a typo, a removed mode, or a plugin that failed
    to load would put an unrunnable name into the cycle.
    """
    names = _viz(loadout=["Bars", "No Such Mode", "Star Trails"]).mode_names
    assert names == ["Bars", "Star Trails"]


def test_a_loadout_that_survives_nothing_falls_back_to_everything():
    """Never leave the interface with no modes to offer.

    Reachable without the user doing anything wrong: quarantine can empty a
    small loadout on its own. An empty offer list would mean a dead picker,
    a cycle key that does nothing and a shuffle that cannot move.
    """
    names = _viz(loadout=["Nope", "Also Nope"]).mode_names
    assert len(names) == len(_viz().mode_names)


def test_cycling_stays_inside_the_loadout():
    """Including when the mode playing right now is outside it.

    Switching modes out from under someone the moment they save is a worse
    surprise than one mode that lingers, so the running mode is left alone
    and the next cycle steps into the loadout.
    """
    viz = _viz(mode="Bars", loadout=["Flame", "Star Trails"])
    seen = {viz.cycle_mode() for _ in range(6)}
    assert seen == {"Flame", "Star Trails"}


# ── the setting survives a hand-edited file ──────────────────────────────────

def test_clamp_keeps_only_usable_names():
    """A config file is hand-editable, so every shape has to be survivable."""
    assert config.Settings(loadout="Bars").clamp().loadout == ["Bars"]
    assert config.Settings(loadout=42).clamp().loadout == []
    assert config.Settings(loadout=None).clamp().loadout == []
    # junk dropped, order kept, duplicates collapsed
    messy = ["Bars", None, 7, "", "Bars", "Flame"]
    assert config.Settings(loadout=messy).clamp().loadout == ["Bars", "Flame"]


# ── the modal ────────────────────────────────────────────────────────────────

def test_the_modal_starts_with_everything_ticked_when_unrestricted():
    """"No restriction" and "all ticked" are the same picture.

    Opening the modal on a fresh config and seeing nothing ticked would read
    as though the modes had been lost.
    """
    p = LoadoutPicker("loadout", ["Bars", "Flame", "Snow"], chosen=[])
    assert p._chosen == {"Bars", "Flame", "Snow"}


def test_the_modal_drops_a_stale_name_rather_than_re_saving_it():
    p = LoadoutPicker("loadout", ["Bars", "Flame"], chosen=["Bars", "Gone"])
    assert p._chosen == {"Bars"}


def test_saving_everything_is_stored_as_no_restriction():
    """Ticking all is recorded as ``[]``, not as today's mode list.

    Freezing the list would silently exclude any mode added later — a new
    release, or a plugin — from a loadout the user believes says "all".
    """
    got: list = []
    p = LoadoutPicker("loadout", ["Bars", "Flame"], chosen=["Bars"],
                      on_done=got.append)
    p._chosen = {"Bars", "Flame"}
    p.action_choose()
    assert got == [[]]


def test_cancelling_is_distinct_from_saving_an_empty_set():
    """Cancel returns None so the caller can tell it from "no restriction"."""
    got: list = []
    p = LoadoutPicker("loadout", ["Bars", "Flame"], on_done=got.append)
    p.action_cancel()
    assert got == [None]


def test_the_tickbox_survives_the_markup_parser():
    """A ticked row has to actually show a tick.

    ``[x]`` is a well-formed Textual tag, so the parser swallowed the whole
    box and a ticked mode rendered as a bare name. ``[ ]`` came through
    untouched — the space makes it invalid markup — so exactly the ticked
    half of the list lost its box, which reads as a rendering quirk rather
    than as the unescaped markup it is. Asserted against the *rendered* text,
    because the raw string looked correct the whole time.
    """
    from rich.text import Text

    p = LoadoutPicker("loadout", ["Bars", "Flame"], chosen=["Bars"])
    assert "[x]" in Text.from_markup(p._label_for("Bars")).plain
    assert "[ ]" in Text.from_markup(p._label_for("Flame")).plain


def test_a_mode_named_with_a_bracket_cannot_take_the_panel_down():
    """Mode names come from plugins, so one can contain a bracket.

    Unparseable markup raises out of the compositor — a crash on layout, not
    a row that looks wrong.
    """
    from rich.text import Text

    p = LoadoutPicker("loadout", ["Bars [dim]", "[/]"], chosen=["Bars [dim]"])
    for name in ("Bars [dim]", "[/]"):
        assert name in Text.from_markup(p._label_for(name)).plain


# ── end to end, through real keystrokes ──────────────────────────────────────

async def _drive() -> list[str]:
    problems: list[str] = []
    # Same two harness concessions test_app.py makes: a 60 fps repaint never
    # lets Pilot see an idle app, and a toast holds it busy for its timeout.
    app = Spektr(settings=config.Settings(fps=15))
    app.notify = lambda *a, **k: None  # type: ignore[method-assign]
    app._save_settings = lambda *a, **k: None  # type: ignore[method-assign]

    async with app.run_test(size=(120, 32)) as pilot:
        offered = list(app.viz.mode_names)

        await pilot.press("V")
        await pilot.pause()
        panel = app.query(LoadoutPicker)
        if not panel:
            return ["V did not open the loadout modal"]

        # 'n' clears, space ticks the row under the cursor, enter saves.
        await pilot.press("n")
        await pilot.pause()
        await pilot.press("space")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        if app.query(LoadoutPicker):
            problems.append("enter did not close the modal")
        if len(app.settings.loadout) != 1:
            problems.append(f"expected one mode saved, got {app.settings.loadout}")
        if app.viz.mode_names != app.settings.loadout:
            problems.append(
                f"offer list {app.viz.mode_names} does not match the saved "
                f"loadout {app.settings.loadout}")

        # Reopening must show the whole roster again, or a mode could never
        # be added back once it had been dropped.
        await pilot.press("V")
        await pilot.pause()
        panel = app.query_one(LoadoutPicker)
        if list(panel._items) != offered:
            problems.append("the modal was built from the narrowed list, so a "
                            "dropped mode could never be re-added")
        # Escape leaves the saved loadout alone.
        saved = list(app.settings.loadout)
        await pilot.press("escape")
        await pilot.pause()
        if app.settings.loadout != saved:
            problems.append("escape changed the loadout")

    return problems


def test_the_modal_saves_and_the_offer_list_follows():
    problems = asyncio.run(_drive())
    assert not problems, "; ".join(problems)
