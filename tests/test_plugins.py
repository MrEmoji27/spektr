"""Plugin system test — discovery, trust, loading, quarantine.

Uses a temporary config directory so it never touches your real plugins.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# cp1252 consoles cannot encode this file's output characters.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from spektr import modes as registry  # noqa: E402
from spektr import palette, plugins   # noqa: E402

GOOD = '''
"""A well-behaved plugin."""
import numpy as np
from spektr.api import mode, pack_braille, cell_max

SPEKTR_API = 1

@mode("TestGood", group="scenes", blurb="a test mode")
def good(ctx):
    field = np.full((ctx.dot_rows, ctx.dot_cols), ctx.energy, dtype=np.float32)
    return pack_braille(field > 0.001), ctx.ramp(cell_max(field))
'''

RAISES = '''
from spektr.api import mode

@mode("TestRaises")
def boom(ctx):
    raise RuntimeError("this mode is broken")
'''

BAD_SHAPE = '''
import numpy as np
from spektr.api import mode

@mode("TestBadShape")
def wrong(ctx):
    n = np.zeros((ctx.h, ctx.w - 1), dtype=np.int32)
    return n, n
'''

BAD_RANGE = '''
import numpy as np
from spektr.api import mode

@mode("TestBadRange")
def hot(ctx):
    codes = np.full((ctx.h, ctx.w), ord("#"), dtype=np.int32)
    cidx = np.full((ctx.h, ctx.w), 9999, dtype=np.int32)   # way past the ramp
    return codes, cidx
'''

IMPORT_ERROR = '''
import a_module_that_does_not_exist
from spektr.api import mode
'''

NO_MODES = '''
x = 1 + 1     # registers nothing
'''

COLLIDES = '''
from spektr.api import mode

@mode("Bars")
def shadow(ctx):
    return None
'''

WRONG_API = '''
from spektr.api import mode
SPEKTR_API = 99

@mode("TestFuture")
def future(ctx):
    return None
'''


def write(folder: Path, name: str, body: str) -> None:
    (folder / f"{name}.py").write_text(body, encoding="utf-8")


def make_ctx(w=60, h=20):
    from spektr.modes import Ctx
    from spektr.palette import BUILTIN, Palette

    n = 32
    bands = np.linspace(0.2, 0.9, n)
    return Ctx(
        w=w, h=h, bands=bands, peaks=bands, bands_l=bands, bands_r=bands,
        wave=np.zeros(512), stereo=np.zeros((512, 2)),
        frame=1, t=1.0, dt=1 / 60, energy=0.5, silent=False,
        palette=Palette(BUILTIN["gruvbox"]), state={},
    )


def check_plugin_system() -> list[str]:
    bad: list[str] = []
    tmp = Path(tempfile.mkdtemp(prefix="spektr-test-"))
    original = palette.config_dir
    palette.config_dir = lambda: tmp          # redirect the whole config tree

    folder = tmp / "plugins"
    folder.mkdir(parents=True)

    try:
        for name, body in [
            ("good", GOOD), ("raises", RAISES), ("badshape", BAD_SHAPE),
            ("badrange", BAD_RANGE), ("importerror", IMPORT_ERROR),
            ("nomodes", NO_MODES), ("collides", COLLIDES), ("wrongapi", WRONG_API),
        ]:
            write(folder, name, body)

        # ── discovery finds everything, trusts nothing ──
        found = {p.name: p for p in plugins.discover()}
        if len(found) != 8:
            bad.append(f"discover found {len(found)} plugins, expected 8")
        if any(p.trusted for p in found.values()):
            bad.append("a freshly written plugin was already trusted")
        print(f"    discovered {len(found)}, all untrusted")

        # ── nothing loads until trusted ──
        plugins.load_all()
        if registry.get("TestGood") is not None:
            bad.append("an untrusted plugin was loaded")
        print("    untrusted plugins refused to load")

        # ── trust and load ──
        for name in found:
            plugins.trust(name)
        loaded = {p.name: p for p in plugins.load_all()}

        if not loaded["good"].loaded:
            bad.append(f"good plugin failed: {loaded['good'].error}")
        if loaded["good"].modes != ["TestGood"]:
            bad.append(f"mode attribution wrong: {loaded['good'].modes}")
        m = registry.get("TestGood")
        if m is None or m.plugin != "good":
            bad.append("TestGood was not attributed to its plugin")
        print("    trusted plugin loaded and attributed")

        # ── bad plugins report rather than crash ──
        for name, expect in [
            ("importerror", "no module"),
            ("nomodes", "registered no modes"),
            ("collides", "already registered"),
            ("wrongapi", "api"),
        ]:
            err = (loaded[name].error or "").lower()
            if not err:
                bad.append(f"{name}: expected an error, got none")
            elif expect not in err:
                bad.append(f"{name}: error was {err.splitlines()[-1]!r}, wanted {expect!r}")
        if registry.get("Bars").plugin is not None:
            bad.append("a colliding plugin overwrote the built-in Bars")
        if registry.get("TestFuture") is not None:
            bad.append("a plugin declaring the wrong API version still registered")
        print("    broken plugins reported cleanly, built-ins intact")

        # ── edit invalidates trust ──
        write(folder, "good", GOOD + "\n# edited\n")
        p = next(x for x in plugins.discover() if x.name == "good")
        if p.trusted:
            bad.append("editing a plugin did not invalidate its approval")
        plugins.trust("good")
        print("    editing invalidated approval, re-trust worked")

        # ── output validation ──
        ctx = make_ctx()
        shape_mode = registry.get("TestBadShape")
        try:
            plugins.validate(shape_mode.fn(ctx), ctx.w, ctx.h)
            bad.append("a wrong-shaped return was accepted")
        except plugins.BadModeOutput as exc:
            if "shape" not in str(exc):
                bad.append(f"shape error was unhelpful: {exc}")

        range_mode = registry.get("TestBadRange")
        codes, cidx = plugins.validate(range_mode.fn(ctx), ctx.w, ctx.h)
        if cidx.max() >= palette.RAMP_STEPS:
            bad.append(f"out-of-range ramp index was not clipped: {cidx.max()}")
        print("    output validated: bad shape rejected, bad range clipped")

        # ── quarantine ──
        q = plugins.Quarantine(limit=3)
        disabled_at = None
        for i in range(5):
            if q.record_failure("TestRaises", "boom") and disabled_at is None:
                disabled_at = i + 1
        if disabled_at != 3:
            bad.append(f"quarantine triggered after {disabled_at} failures, expected 3")
        if not q.is_disabled("TestRaises"):
            bad.append("mode was not disabled")
        q.clear("TestRaises")
        if q.is_disabled("TestRaises"):
            bad.append("clear() did not re-enable")

        q2 = plugins.Quarantine(limit=3)
        q2.record_failure("Flaky", "x")
        q2.record_success("Flaky")
        q2.record_failure("Flaky", "x")
        q2.record_failure("Flaky", "x")
        if q2.is_disabled("Flaky"):
            bad.append("a success did not reset the failure count")
        print(f"    quarantine after {disabled_at} failures, reset on success")

        # ── unload ──
        removed = registry.unregister_plugin("good")
        if removed != ["TestGood"] or registry.get("TestGood") is not None:
            bad.append(f"unregister_plugin left modes behind: {removed}")
        print("    unregister removed only the plugin's modes")

    finally:
        for name in list(found):
            registry.unregister_plugin(name)
        palette.config_dir = original
        shutil.rmtree(tmp, ignore_errors=True)

    return bad


# ── pytest entry point ────────────────────────────────────────────────────────
#
# Before this, the checker was named ``run()`` — not ``test_*`` — so ``pytest
# tests/`` collected zero tests from this file. Not a warning, not a silent
# pass: nothing. This is the plugin trust/quarantine boundary, the one part of
# the app the README calls out as security-sensitive ("plugins are Python and
# run with your privileges"), and it had no pytest coverage at all. CI never
# noticed because it runs this file directly as a script instead — but anyone
# running the standard ``pytest`` workflow got nothing here.
def test_plugin_system() -> None:
    bad = check_plugin_system()
    assert not bad, "\n".join(bad)


if __name__ == "__main__":
    print("  plugin system…")
    problems = check_plugin_system()
    for p in problems:
        print(f"    FAIL {p}")
    print("\nall good" if not problems else f"\n{len(problems)} failures")
    raise SystemExit(1 if problems else 0)
