"""Every image the documentation points at exists, and is worth its weight.

A broken image is invisible from inside the repository. It renders as a torn
icon on the project's front page, to everyone except the person who added it,
and nothing in a normal test run goes anywhere near it.

The size checks are here for a different reason: these files are the first
thing a visitor loads and the last thing a clone can ever drop. The demo
recordings came off the recorder at 50 fps and 29 MB for four clips, which is
most of a release's download for six seconds of video each.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

#: Both spellings are used: a relative path, and the raw.githubusercontent URL
#: that a README needs when the same file is rendered from a mirror repo.
_REF = re.compile(r'(?:assets/|/main/assets/)([A-Za-z0-9._-]+)')

#: A GIF nobody scrolls past should not cost more than a small application.
#: The hero is allowed more because it is the one that has to autoplay above
#: the fold; the rest are re-encoded at 10 fps and 640 wide.
_MAX_MB = {"hero.gif": 5.0}
_DEFAULT_MAX_MB = 4.5


def _markdown() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    return [ROOT / line for line in out.stdout.split()]


def test_every_referenced_asset_exists():
    missing = []
    for md in _markdown():
        text = md.read_text(encoding="utf-8", errors="replace")
        for name in set(_REF.findall(text)):
            if not (ASSETS / name).exists():
                missing.append(f"{md.relative_to(ROOT).as_posix()} -> assets/{name}")
    assert not missing, "documentation points at files that are not there: " + \
        ", ".join(sorted(missing))


def test_no_asset_is_committed_and_never_used():
    """An unreferenced binary is weight every clone pays for nothing."""
    referenced = set()
    for md in _markdown():
        referenced |= set(_REF.findall(md.read_text(encoding="utf-8", errors="replace")))
    tracked = subprocess.run(["git", "ls-files", "assets"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
    orphans = [p for p in tracked if Path(p).name not in referenced]
    assert not orphans, "committed but referenced nowhere: " + ", ".join(sorted(orphans))


def test_the_demo_recordings_stay_small():
    oversized = []
    for f in sorted(ASSETS.glob("*")):
        if not f.is_file():
            continue
        mb = f.stat().st_size / 1e6
        cap = _MAX_MB.get(f.name, _DEFAULT_MAX_MB)
        if mb > cap:
            oversized.append(f"{f.name} is {mb:.1f} MB, over {cap} MB")
    assert not oversized, (
        "re-encode before committing — 10 fps, 640 wide, 64 colours, "
        "`dither=none` and `stats_mode=diff`, which is what the rest were "
        "cut to: " + "; ".join(oversized)
    )


def test_the_asset_names_survive_a_url():
    """A space in a filename becomes %20 in some renderers and a break in others.

    One of these arrived as `last few themes.gif`.
    """
    bad = [f.name for f in ASSETS.iterdir()
           if f.is_file() and not re.fullmatch(r"[a-z0-9._-]+", f.name)]
    assert not bad, "not safe in a URL: " + ", ".join(sorted(bad))
