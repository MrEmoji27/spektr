# Copies the spektr package from the checkout root into the Chaquopy source
# tree, so the APK ships exactly what the desktop runs — one copy, no fork.
# Run this whenever the engine moves on main; commit the result.
#
# spektr is vendored rather than pip-installed because pip-installing it would
# resolve its desktop-only dependencies (sounddevice, soundcard, winrt, ...),
# none of which have Android wheels, and Chaquopy cannot scope --no-deps to a
# single requirement. The copy is small (pure Python + numpy) and the tests in
# tests/test_android_bridge.py exercise the exact files this tree ships.
$ErrorActionPreference = "Stop"

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent   # android/ -> repo root
$src = Join-Path $root "spektr"
$dst = Join-Path $PSScriptRoot "..\app\src\main\python\spektr"

if (-not (Test-Path -LiteralPath (Join-Path $src "__init__.py"))) {
    throw "spektr package not found at $src"
}

Remove-Item -LiteralPath $dst -Recurse -Force -ErrorAction SilentlyContinue
Copy-Item -LiteralPath $src -Destination $dst -Recurse
Write-Host "synced spektr/ -> $dst"
