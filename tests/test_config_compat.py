"""Config written by 0.5.0 loads in this version with nothing lost, and keys
this version does not know survive a save."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spektr import config  # noqa: E402


def _write(folder: Path, data: dict) -> None:
    (folder / "config.json").write_text(json.dumps(data), encoding="utf-8")


def _read(folder: Path) -> dict:
    return json.loads((folder / "config.json").read_text(encoding="utf-8"))


def test_unknown_keys_survive_a_load_and_a_save(tmp_path):
    _write(tmp_path, {"mode": "Tunnel", "a_later_setting": "auto", "from_the_future": [1, 2]})
    settings = config.load(tmp_path)
    assert settings.mode == "Tunnel"
    config.save(settings, tmp_path)
    saved = _read(tmp_path)
    assert saved["a_later_setting"] == "auto"
    assert saved["from_the_future"] == [1, 2]
    assert saved["mode"] == "Tunnel"


def test_a_known_setting_is_saved_from_its_field_not_from_the_extras(tmp_path):
    _write(tmp_path, {"mode": "Tunnel", "a_later_setting": "auto"})
    settings = config.load(tmp_path)
    settings.mode = "Bars"
    config.save(settings, tmp_path)
    assert _read(tmp_path)["mode"] == "Bars"


def test_settings_built_in_code_save_exactly_their_fields(tmp_path):
    config.save(config.Settings(mode="Flame"), tmp_path)
    saved = _read(tmp_path)
    assert saved["mode"] == "Flame"
    assert set(saved) == {f for f in config.Settings.__dataclass_fields__}


import shutil  # noqa: E402

import pytest  # noqa: E402

import spektr.modes as registry  # noqa: E402
from spektr import loadouts, palette, plugins  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "config-0.5.0"


@pytest.fixture
def folder(tmp_path):
    """A copy, so a test that saves never edits the fixture."""
    dst = tmp_path / "spektr"
    shutil.copytree(FIXTURE, dst)
    return dst


def test_every_0_5_0_setting_comes_through(folder):
    raw = json.loads((FIXTURE / "config.json").read_text(encoding="utf-8"))
    settings = config.load(folder)
    for name, value in raw.items():
        assert getattr(settings, name) == value, name


def test_saving_a_0_5_0_config_loses_nothing(folder):
    before = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    config.save(config.load(folder), folder)
    after = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    for name, value in before.items():
        assert after[name] == value, name


def test_every_mode_a_0_5_0_config_names_still_exists(folder):
    settings = config.load(folder)
    named = {settings.mode, *settings.loadout}
    for modes in loadouts.load(folder).values():
        named.update(modes)
    missing = sorted(n for n in named if registry.get(n) is None)
    assert not missing, f"0.5.0 configs name modes that are gone: {missing}"


def test_the_0_5_0_loadouts_load(folder):
    assert loadouts.load(folder) == {
        "calm": ["Bars", "Auroras", "Star Trails"],
        "loud": ["Tunnel", "Crosscurrent", "JP Drift", "Fireworks"],
    }


def test_a_0_5_0_user_theme_loads(folder):
    themes = palette.all_themes(folder)
    assert "solarized" in themes
    assert themes["solarized"].low.lower() == "#859900"


def test_a_0_5_0_plugin_loads_once_trusted(folder):
    ok, message = plugins.trust("nightrider", folder)
    assert ok, message
    try:
        loaded = {p.name: p for p in plugins.load_all(folder)}
        assert loaded["nightrider"].loaded, loaded["nightrider"].error
        assert registry.get("Nightrider") is not None
    finally:
        registry.unregister_plugin("nightrider")
