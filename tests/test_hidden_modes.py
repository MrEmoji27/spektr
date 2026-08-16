"""Hidden modes: out of the interface, still in the app.

Nine modes were superseded by their octant variants and taken out of the
picker. "Taken out of the picker" has to mean exactly that and nothing more —
they are still registered, still rendered, still tested by the audit, and
still selectable by name, because a config file or a ``--mode`` flag naming
one has to keep working. Hiding a mode must not quietly change what someone's
setup does.
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


@pytest.mark.parametrize(
    "superseded, replacement",
    [
        ("Scope", "Scope Fine"),
        ("ECG", "ECG Fine"),
        ("Sonar", "Sonar Fine"),
        ("Plasma", "Plasma Fine"),
        ("Chladni", "Chladni Fine"),
        ("Chladni Flow", "Chladni Flow Fine"),
        ("Chladni Extreme", "Chladni Extreme Fine"),
        ("Kaleidoscope", "Kaleidoscope Fine"),
        ("Valentine", "Valentine Fine"),
    ],
)
def test_nothing_is_hidden_without_a_replacement_on_the_menu(superseded, replacement):
    """A mode may only be hidden because something else took its place."""
    old, new = M.get(superseded), M.get(replacement)
    assert old is not None and new is not None
    assert old.hidden, f"{superseded} was expected to be superseded by {replacement}"
    assert not new.hidden, f"{replacement} replaced {superseded} and is itself hidden"


@pytest.mark.parametrize("name", ["Radial", "Maelstrom"])
def test_these_two_keep_their_place(name):
    """Kept on the menu beside their variants, deliberately — they read as
    different pictures rather than as the same one drawn better."""
    m = M.get(name)
    assert m is not None and not m.hidden


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
