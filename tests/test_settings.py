"""The settings panel, held to the things that would break it.

The panel is built fresh every time it opens, from a list of rows plus a dict
of current values. Two invariants decide whether it opens at all, and neither
is visible from any single row — they are properties of the *list*:

* ``SettingsPanel._row_text`` subscripts the values dict, so a row with
  neither a ``live`` callback nor an entry there is a ``KeyError`` on open;
* ``Setting.index_of`` falls back to ``min(choices)`` for a numeric value, so
  a row with neither a ``step`` callback nor a non-empty ``choices`` raises.

Both hold by construction today. That is exactly the kind of thing that stops
holding when someone adds a row, and a settings panel that raises when you
press ``c`` is not a small bug.

The rest is the other half of "won't break": every value the panel can be
stepped onto has to be one the config layer will keep. A row offering a
choice that ``clamp`` rewrites means picking it, closing the panel, and
finding it did not stick.
"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spektr import config  # noqa: E402
from spektr.app import Spektr  # noqa: E402
from spektr.config import Settings  # noqa: E402
from spektr.widget import AudioVisualizer  # noqa: E402

FIELDS = {f.name for f in fields(Settings)}


def _rows(mode: str = "Bars", **settings):
    """The panel as it would be built, without running the app."""
    app = Spektr(settings=Settings(**settings))
    viz = AudioVisualizer(settings=app.settings)
    viz.mode_name = mode
    return app, viz, *app._settings_rows(viz, app.settings)


@pytest.fixture
def panel():
    _, _, rows, values = _rows()
    return rows, values


# ── the two that decide whether it opens at all ──────────────────────────────

@pytest.mark.parametrize("mode", ["Bars", "Kaleidoscope"])
def test_every_row_can_show_a_value(mode):
    """No row may reach ``values[key]`` without an entry there."""
    _, _, rows, values = _rows(mode)
    for r in rows:
        assert r.live is not None or r.key in values, (
            f"row {r.key!r} has no live() and no values entry — KeyError on open"
        )


@pytest.mark.parametrize("mode", ["Bars", "Kaleidoscope"])
def test_every_row_can_be_stepped(mode):
    """No row may fall through to ``min([])``."""
    _, _, rows, _ = _rows(mode)
    for r in rows:
        assert r.step is not None or len(r.choices) > 0, (
            f"row {r.key!r} has no step() and no choices — ValueError on arrow key"
        )


def test_the_panel_is_the_same_whatever_mode_is_showing():
    """Flipbook was the only mode that added rows of its own, and it is gone.

    Asserted rather than dropped with it: a per-mode row is a panel whose
    *shape* depends on state, which is how a row ends up with no value or no
    step behind it. If one comes back, it should come back deliberately.
    """
    keys = [{r.key for r in _rows(m)[2]} for m in ("Bars", "Kaleidoscope", "Snow")]
    assert keys[0] == keys[1] == keys[2]


# ── the values dict ──────────────────────────────────────────────────────────

def test_every_value_is_a_real_setting(panel):
    """A key that is not a field is a value the panel shows and never saves."""
    _, values = panel
    unknown = sorted(set(values) - FIELDS)
    assert not unknown, f"values the config cannot persist: {unknown}"


def test_the_panel_opens_on_what_is_actually_set():
    """Off-by-one here means the row opens on the wrong stop and steps oddly."""
    app, _, rows, values = _rows(sensitivity=2.0, bands=24, cells="quadrant",
                                 fine_modes=True, chrome=False)
    for key, shown in values.items():
        assert shown == getattr(app.settings, key), (
            f"{key}: panel opens on {shown!r}, settings hold {getattr(app.settings, key)!r}"
        )


def test_index_of_finds_every_choice(panel):
    """Each stop must resolve to its own index, whatever its type."""
    rows, _ = panel
    for r in rows:
        for i, choice in enumerate(r.choices):
            assert r.index_of(choice) == i, f"{r.key}: {choice!r} resolved to another stop"


def test_index_of_survives_the_current_value(panel):
    """Opening the panel calls this with whatever is in the config."""
    rows, values = panel
    for r in rows:
        if r.key in values:
            i = r.index_of(values[r.key])
            assert 0 <= i < max(1, len(r.choices))


def test_render_never_raises(panel):
    """The row's label is built before anything is stepped."""
    rows, _ = panel
    for r in rows:
        for choice in r.choices:
            assert isinstance(r.render(choice), str)
        if r.live is not None:
            assert isinstance(r.render(r.live()), str)


# ── every offered value has to survive the config layer ──────────────────────

def test_every_choice_survives_clamp_unchanged(panel):
    """A stop that clamp rewrites is a stop that silently does not stick.

    ``bands`` is the one with a real trap in it: clamp floors a non-zero band
    count at 8, so a 4 in the choice list would read back as 8 the next time
    the panel opened.
    """
    rows, _ = panel
    bad = []
    for r in rows:
        if r.key not in FIELDS:
            continue
        for choice in r.choices:
            s = Settings()
            setattr(s, r.key, choice)
            s.clamp()
            got = getattr(s, r.key)
            if got != choice:
                bad.append(f"{r.key}: offered {choice!r}, clamp made it {got!r}")
    assert not bad, "the panel offers values the config rejects:\n" + "\n".join(bad)


def test_every_choice_round_trips_through_the_file(tmp_path, panel):
    """Picked, closed, reopened — the same value has to come back."""
    rows, _ = panel
    bad = []
    for r in rows:
        if r.key not in FIELDS:
            continue
        for choice in r.choices:
            s = Settings()
            setattr(s, r.key, choice)
            config.save(s.clamp(), config_dir=tmp_path)
            back = getattr(config.load(config_dir=tmp_path), r.key)
            if back != choice:
                bad.append(f"{r.key}: saved {choice!r}, loaded {back!r}")
    assert not bad, "settings that do not survive a restart:\n" + "\n".join(bad)


def test_a_config_full_of_junk_still_builds_a_panel():
    """The panel is built from settings; settings can come off disk corrupt."""
    junk = Settings(
        fps="soon", bands=None, sensitivity=[], gate="low", cells="sextants",
        fine_modes="yes", chrome=None, shuffle_scope="everything",
        mode=None, theme=42,
    ).clamp()
    app = Spektr(settings=junk)
    viz = AudioVisualizer(settings=app.settings)
    rows, values = app._settings_rows(viz, app.settings)
    assert rows and values
    for r in rows:
        if r.key in values:
            assert 0 <= r.index_of(values[r.key]) < max(1, len(r.choices))
            assert isinstance(r.render(values[r.key]), str)


# ── the setters actually reach the object that gets saved ────────────────────

#: Rows whose setter needs a running Textual app: ``fps`` starts a timer and
#: ``chrome`` walks the screen stack. Excluded from the end-to-end check below
#: and pinned separately, so that if either starts failing for some *other*
#: reason it is not mistaken for the lifecycle they legitimately need.
NEEDS_A_LIVE_APP = {"fps", "chrome"}


def test_the_app_and_the_widget_share_one_settings_object():
    """Four rows are set through the widget and all of them are saved from the
    app, so a copy anywhere between them loses fps, bands, sensitivity and gate
    on exit while appearing to work for the whole session."""
    app = Spektr(settings=Settings())
    viz = AudioVisualizer(settings=app.settings)
    assert viz.settings is app.settings
    viz.set_gate(1e-4)
    assert app.settings.gate == viz.settings.gate


def test_every_setter_writes_the_value_it_was_given(monkeypatch):
    """Settings are written to disk once, on unmount, from ``self.settings``.

    So a row that changes the live app without recording it there works
    perfectly until you restart, which is the worst way for a preference to
    fail. This walks every stop of every row and checks it landed.
    """
    app = Spektr(settings=Settings())
    viz = AudioVisualizer(settings=app.settings)
    monkeypatch.setattr(Spektr, "viz", property(lambda self: viz))
    rows, _ = app._settings_rows(viz, app.settings)

    bad = []
    for r in rows:
        if r._apply is None or r.key in NEEDS_A_LIVE_APP or r.key not in FIELDS:
            continue
        for choice in r.choices:
            r.apply(choice)
            got = getattr(app.settings, r.key)
            if got != choice:
                bad.append(f"{r.key}: applied {choice!r}, settings hold {got!r}")
            else:
                app.settings.clamp()
                if getattr(app.settings, r.key) != choice:
                    bad.append(f"{r.key}: {choice!r} did not survive clamp after apply")
    assert not bad, "settings that would be lost on restart:\n" + "\n".join(bad)


@pytest.mark.parametrize("key", sorted(NEEDS_A_LIVE_APP))
def test_the_two_live_app_rows_fail_only_for_the_expected_reason(monkeypatch, key):
    """Pins *why* they are excluded above, so the exclusion cannot rot.

    If one of these starts raising something else — an AttributeError from a
    renamed method, say — that is a real break hiding behind a known one.
    """
    app = Spektr(settings=Settings())
    viz = AudioVisualizer(settings=app.settings)
    monkeypatch.setattr(Spektr, "viz", property(lambda self: viz))
    rows, _ = app._settings_rows(viz, app.settings)
    row = next(r for r in rows if r.key == key)

    try:
        row.apply(row.choices[0])
    except Exception as exc:                          # noqa: BLE001 — the point
        assert type(exc).__name__ in {"RuntimeError", "ScreenStackError"}, (
            f"{key} now fails with {type(exc).__name__}: {exc}"
        )


# ── the README has to agree with the app ─────────────────────────────────────

README = Path(__file__).resolve().parent.parent / "README.md"


def test_the_readme_names_every_opt_in_mode():
    """Twelve variants documented, and named, so searching for one finds it."""
    import spektr.modes as M

    text = README.read_text(encoding="utf-8")
    missing = [m.name for m in M.MODES if m.hidden and m.name not in text]
    assert not missing, f"opt-in modes the README never names: {missing}"


def test_the_readme_lists_every_mode_the_picker_offers():
    """The table is the only place the default set is written down."""
    import spektr.modes as M

    text = README.read_text(encoding="utf-8")
    missing = [
        m.name for m in M.listed()
        if m.name != "None" and f"**{m.name}**" not in text
    ]
    assert not missing, f"modes offered but undocumented: {missing}"


def test_the_readme_documents_every_command_line_flag():
    """The usage text is printed by --help; the README is read first."""
    from spektr.app import _USAGE

    text = README.read_text(encoding="utf-8")
    flags = {
        line.strip().split()[0]
        for line in _USAGE.splitlines()
        if line.strip().startswith("--")
    }
    missing = sorted(f for f in flags if f not in text)
    assert not missing, f"flags in --help but not in the README: {missing}"


def test_no_stale_fine_naming_survives_anywhere_user_visible():
    """The rename is only done if nothing still calls them "Fine"."""
    from spektr.app import _USAGE
    import spektr.modes as M

    assert "Fine" not in README.read_text(encoding="utf-8")
    assert "Fine" not in _USAGE
    assert not [m for m in M.MODES if m.name.endswith(" Fine")]


#: The headline spells its numbers out, so the test has to as well or it
#: checks nothing. Only the range the counts plausibly occupy is covered;
#: past it the test fails loudly rather than passing quietly on a KeyError.
_TENS = {20: "Twenty", 30: "Thirty", 40: "Forty", 50: "Fifty",
         60: "Sixty", 70: "Seventy", 80: "Eighty", 90: "Ninety"}
_UNITS = ("", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def _spelled(n: int) -> str:
    assert 20 <= n < 100, f"no spelling for {n}; extend _spelled if the app grew"
    tens, unit = divmod(n, 10)
    return _TENS[tens * 10] + (f"-{_UNITS[unit]}" if unit else "")


def test_the_readme_counts_agree_with_the_registry():
    """The badges and the headline are the first numbers anyone reads.

    They have drifted before and by a lot — the README claimed 49 themes while
    the app shipped 54, and the mode badge said 53 against a headline of 52 in
    the same block of eight lines. Nothing about adding a mode or a theme
    touches this file, so the only thing that keeps it honest is a test.

    "Render modes" excludes ``None``, which is offered by the picker but does
    not render anything. Both places have to use the same convention or one of
    them is wrong whichever number is chosen.
    """
    import spektr.modes as M
    from spektr.palette import BUILTIN

    text = README.read_text(encoding="utf-8")
    modes = len([m for m in M.listed() if m.name != "None"])
    themes = len(BUILTIN)

    expected = {
        f"render%20modes-{modes}-": "the mode badge",
        f"themes-{themes}-": "the theme badge",
        f"{_spelled(modes)} render modes": "the headline mode count",
        f"{_spelled(themes)} themes": "the headline theme count",
    }
    missing = sorted(f"{what} ({needle!r})"
                     for needle, what in expected.items() if needle not in text)
    assert not missing, (
        f"the app has {modes} render modes and {themes} themes; the README "
        f"does not say so in: {missing}"
    )
