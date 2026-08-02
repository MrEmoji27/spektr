# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build definition for spektr.

Two shapes, one spec:

    pyinstaller packaging/spektr.spec                  -> dist/spektr.exe   (onefile)
    SPEKTR_ONEDIR=1 pyinstaller packaging/spektr.spec  -> dist/spektr/      (onedir)

Onefile is the thing you hand to a beginner: one download, double-click, done.
It self-extracts to a temp directory on every launch, which costs a couple of
seconds before the first frame — so the installer in ``installer.iss`` ships
the onedir build instead, where startup is immediate.

**console=True is not optional.** spektr is a TUI: Textual draws inside a
terminal. A windowed (``console=False``) build has no console to draw in and
exits immediately.
"""

import importlib.util
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).parent          # noqa: F821 — SPECPATH is injected
sys.path.insert(0, str(ROOT))

from spektr import __version__        # noqa: E402

ONEDIR = os.environ.get("SPEKTR_ONEDIR") == "1"
ICON = str(Path(SPECPATH) / "spektr.ico")            # noqa: F821

datas, binaries, hiddenimports = [], [], []


def pull(package: str) -> None:
    """Collect a dependency's data files, libraries and submodules.

    Wrapped because not every name here is a package on every platform:
    ``_sounddevice_data`` only exists in the Windows and macOS wheels, and
    ``sounddevice`` is a single module rather than a package. A miss is a
    printed note, not a failed build.
    """
    global datas, binaries, hiddenimports
    try:
        d, b, h = collect_all(package)
    except Exception as exc:                          # noqa: BLE001
        print(f"[spektr.spec] nothing collected for {package!r}: {exc}")
        return
    datas += d
    binaries += b
    hiddenimports += h
    print(f"[spektr.spec] {package}: {len(d)} data, {len(b)} binaries, {len(h)} submodules")


# textual ships tree-sitter query files (*.scm) next to its modules;
# soundcard reads <backend>.py.h at import time and would fail without it —
# it ships its own PyInstaller hook that says so, and this repeats it in case
# the hook entry point is not picked up.
for package in ("textual", "soundcard", "sounddevice", "_sounddevice_data"):
    pull(package)

hiddenimports += [
    # cffi ABI-mode shim that sounddevice imports, plus its C backend
    "_sounddevice",
    "_cffi_backend",
    # soundcard picks its backend with a platform test at import time. The
    # import is static so the analysis finds it, but naming them is free and
    # makes a cross-built exe obvious if one ever goes missing.
    "soundcard.mediafoundation",
    "soundcard.coreaudio",
    "soundcard.pulseaudio",
]

# Python 3.10 has no tomllib, so user themes fall back to tomli there. Only
# name it when it is both needed and present — otherwise every build logs a
# missing-module warning for a dependency that is deliberately optional.
if sys.version_info < (3, 11) and importlib.util.find_spec("tomli") is not None:
    hiddenimports.append("tomli")

# Nothing here is imported by spektr; excluding them keeps the exe from
# swallowing a GUI toolkit and a plotting stack it will never touch.
excludes = [
    "tkinter", "matplotlib", "PIL", "pandas", "scipy",
    "pytest", "IPython", "notebook", "sphinx", "setuptools._distutils",
]


# ── Windows version metadata ────────────────────────────────────────────────
# Written into build/ (git-ignored) rather than the source tree, because it is
# derived from spektr.__version__ and should never drift from it by hand.
parts = [int(p) for p in __version__.split(".")[:3]] + [0]
vtuple = tuple(parts[:4])

version_file = ROOT / "build" / "version_info.txt"
version_file.parent.mkdir(parents=True, exist_ok=True)
version_file.write_text(
    f"""VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={vtuple}, prodvers={vtuple},
    mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        '040904B0',
        [StringStruct('CompanyName', 'zemo'),
         StringStruct('FileDescription', 'spektr — terminal spectrum analyser'),
         StringStruct('FileVersion', '{__version__}'),
         StringStruct('InternalName', 'spektr'),
         StringStruct('LegalCopyright', 'MIT licensed'),
         StringStruct('OriginalFilename', 'spektr.exe'),
         StringStruct('ProductName', 'spektr'),
         StringStruct('ProductVersion', '{__version__}')])
    ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
""",
    encoding="utf-8",
)


a = Analysis(
    [str(Path(SPECPATH) / "entry.py")],               # noqa: F821
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)                                      # noqa: F821

common = dict(
    name="spektr",
    icon=ICON,
    version=str(version_file),
    console=True,          # see the module docstring — do not flip this
    disable_windowed_traceback=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,             # UPX-packed exes trip more AV heuristics than they save space
)

if ONEDIR:
    exe = EXE(                                         # noqa: F821
        pyz, a.scripts, [], exclude_binaries=True, **common
    )
    coll = COLLECT(                                    # noqa: F821
        exe, a.binaries, a.datas, strip=False, upx=False, name="spektr"
    )
else:
    exe = EXE(                                         # noqa: F821
        pyz, a.scripts, a.binaries, a.datas, [], runtime_tmpdir=None, **common
    )
