"""The widget's theme read path honours an injected config directory.

The config root is a seam (see ``palette.config_dir``): the write paths
already receive it as a parameter, and the widget's theme list must read
through the same seam — with a config dir injected, the list shows that
directory's user themes and none from the real user directory.
"""

from __future__ import annotations

import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# cp1252 consoles cannot encode this file's output characters.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from spektr.palette import BUILTIN, load_user_themes  # noqa: E402
from spektr.widget import AudioVisualizer  # noqa: E402


def _scratch_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="spektr-widget-theme-test-"))


def _write_theme(folder: Path, name: str) -> None:
    theme_dir = folder / "themes"
    theme_dir.mkdir(exist_ok=True)
    (theme_dir / f"{name}.toml").write_text(
        'low = "#00ff41"\nmid = "#ffb000"\nhigh = "#ff3300"\n',
        encoding="utf-8",
    )


def test_widget_reads_theme_list_from_the_injected_config_dir() -> None:
    tmp = _scratch_dir()
    # a unique name, so it can never collide with a real user theme
    name = f"injected-{uuid.uuid4().hex[:8]}"
    _write_theme(tmp, name)

    viz = AudioVisualizer(config_dir=tmp)

    user_themes = set(viz._themes) - set(BUILTIN)
    assert name in user_themes, f"injected theme {name} not in the widget's list"
    assert user_themes == {name}, f"real user themes leaked in: {user_themes}"
    assert name in viz.theme_names

    real_user = set(load_user_themes())
    assert not (user_themes & real_user), "the real user directory leaked through"


def test_widget_reload_themes_rerreads_the_injected_dir() -> None:
    tmp = _scratch_dir()
    first = f"injected-a-{uuid.uuid4().hex[:8]}"
    second = f"injected-b-{uuid.uuid4().hex[:8]}"
    _write_theme(tmp, first)

    viz = AudioVisualizer(config_dir=tmp)
    assert first in viz.theme_names

    _write_theme(tmp, second)
    viz.reload_themes()
    assert second in viz.theme_names


def test_widget_apply_theme_miss_refetches_from_the_injected_dir() -> None:
    tmp = _scratch_dir()
    name = f"injected-{uuid.uuid4().hex[:8]}"

    viz = AudioVisualizer(config_dir=tmp)
    assert name not in viz.theme_names

    _write_theme(tmp, name)
    note = viz.apply_theme(name)
    assert viz.theme_name == name
    assert note