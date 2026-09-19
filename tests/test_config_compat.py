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
    _write(tmp_path, {"mode": "Tunnel", "eco": "auto", "from_the_future": [1, 2]})
    settings = config.load(tmp_path)
    assert settings.mode == "Tunnel"
    config.save(settings, tmp_path)
    saved = _read(tmp_path)
    assert saved["eco"] == "auto"
    assert saved["from_the_future"] == [1, 2]
    assert saved["mode"] == "Tunnel"


def test_a_known_setting_is_saved_from_its_field_not_from_the_extras(tmp_path):
    _write(tmp_path, {"mode": "Tunnel", "eco": "auto"})
    settings = config.load(tmp_path)
    settings.mode = "Bars"
    config.save(settings, tmp_path)
    assert _read(tmp_path)["mode"] == "Bars"


def test_settings_built_in_code_save_exactly_their_fields(tmp_path):
    config.save(config.Settings(mode="Flame"), tmp_path)
    saved = _read(tmp_path)
    assert saved["mode"] == "Flame"
    assert set(saved) == {f for f in config.Settings.__dataclass_fields__}
