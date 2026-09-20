"""Mode metadata stays cheap while implementations load only when needed."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _probe(source: str) -> dict:
    run = subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def test_importing_the_registry_loads_no_mode_implementation():
    got = _probe(
        "import json, sys; import spektr.modes; "
        "print(json.dumps(sorted(m for m in sys.modules "
        "if m.startswith('spektr.modes.') and not m.endswith('._shared'))))"
    )
    assert got == []


def test_get_loads_only_the_requested_modes_catalogue_module():
    got = _probe(
        "import json, sys; import spektr.modes as modes; "
        "mode = modes.get('Swell'); "
        "loaded = sorted(m for m in sys.modules if m.startswith('spektr.modes.')); "
        "print(json.dumps({'name': mode.name, 'module': mode.fn.__module__, "
        "'loaded': loaded}))"
    )
    assert got["name"] == "Swell"
    assert got["module"] == "spektr.modes.terrain"
    assert got["loaded"] == ["spektr.modes._shared", "spektr.modes.terrain"]


def test_catalogue_metadata_keeps_the_existing_mode_arrangement():
    import spektr.modes as modes
    from spektr import catalog

    assert [m.name for m in modes.MODES] == [row[0] for row in catalog.CATALOG]
    assert [m.group for m in modes.MODES] == [row[1] for row in catalog.CATALOG]
    assert [m.hidden for m in modes.MODES] == [row[3] for row in catalog.CATALOG]


def test_moved_shared_helpers_forward_live(monkeypatch):
    import spektr.modes as modes
    from spektr.modes import _shared

    sentinel = object()
    monkeypatch.setattr(_shared, "band_columns", sentinel)
    assert modes.band_columns is sentinel


def test_old_fields_path_forwards_family_attributes_live(monkeypatch):
    from spektr.modes import field_spectral, fields

    sentinel = object()
    monkeypatch.setattr(field_spectral, "spectrogram", sentinel)
    assert fields.spectrogram is sentinel
