"""The generated catalogue matches the real registry, and needs no mode code."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spektr.modes as registry  # noqa: E402
from spektr import catalog  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _live() -> list[tuple]:
    return [
        (m.name, m.group, m.blurb, m.hidden, m.fn.__module__.split(".")[-1])
        for m in registry.MODES
        if m.plugin is None
    ]


def test_the_catalogue_matches_the_registry():
    assert list(catalog.CATALOG) == _live(), (
        "catalogue is stale: run python tools/build_catalog.py"
    )


def test_every_mode_names_a_module_that_exists():
    for name, _group, _blurb, _hidden, module in catalog.CATALOG:
        path = ROOT / "spektr" / "modes" / f"{module}.py"
        assert path.exists() or module == "modes", f"{name}: no module {module}"


def test_module_for_finds_a_mode_and_misses_a_stranger():
    assert catalog.module_for("Bars") == "spectrum"
    assert catalog.module_for("Not A Mode") is None


def test_reading_the_catalogue_imports_no_mode_module():
    """The point of the file: a picker can list modes without loading them."""
    code = (
        "import sys; from spektr import catalog; "
        "loaded = [m for m in sys.modules if m.startswith('spektr.modes')]; "
        "print(len(catalog.CATALOG), loaded)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    count, loaded = out.stdout.strip().split(" ", 1)
    assert int(count) == len(catalog.CATALOG)
    assert loaded == "[]", f"importing the catalogue pulled in mode modules: {loaded}"
