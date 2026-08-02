<#
.SYNOPSIS
    Build spektr.exe on Windows. One command, from a clean checkout.

.DESCRIPTION
    Creates an isolated build venv (so your day-to-day environment is never
    touched), installs spektr's dependencies plus PyInstaller, and runs the
    spec in packaging/spektr.spec.

    ASCII only, on purpose. Windows PowerShell 5.1 reads .ps1 files as ANSI
    unless they carry a UTF-8 BOM, and a UTF-8 em dash decoded as cp1252 ends
    in 0x94 - a curly quote, which the parser treats as a string delimiter.
    The result is a cascade of "missing closing brace" errors pointing at
    lines that are perfectly fine. This file is saved with a BOM as well, but
    staying inside ASCII means it cannot come back.

.PARAMETER OneDir
    Build dist/spektr/ (a folder) instead of a single exe. Starts instantly
    because there is nothing to self-extract; this is what the installer ships.

.PARAMETER Installer
    After building, compile packaging/installer.iss with Inno Setup into
    dist/installer/. Implies -OneDir. Needs Inno Setup 6 installed.

.PARAMETER Clean
    Delete build/, dist/ and the build venv first. The venv is recreated
    automatically when -Python switches interpreters, so this is only for a
    fully clean slate.

.PARAMETER Python
    Which interpreter to build with - either a version resolved through the py
    launcher ("3.12") or a full path to python.exe. Use this when the newest
    Python on the machine is ahead of the published numpy wheels.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File packaging\build_exe.ps1
    powershell -ExecutionPolicy Bypass -File packaging\build_exe.ps1 -Installer
    powershell -ExecutionPolicy Bypass -File packaging\build_exe.ps1 -Python 3.12 -Clean
#>
[CmdletBinding()]
param(
    [switch]$OneDir,
    [switch]$Installer,
    [switch]$Clean,
    [string]$Python
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ($Installer) { $OneDir = $true }

function Step($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }

# ---- python -----------------------------------------------------------------
# Probed with literal calls rather than an array of arguments. @(...) is the
# array operator, NOT splatting - splatting needs a bare variable name (@args)
# - so building the argument list dynamically silently passes one mangled
# token and every probe fails. Two explicit branches cannot go wrong that way.
#
# Every attempt is recorded, so a failure says what it tried and what came
# back instead of the useless "no Python found".
$attempts = @()
$pythonExe = $null
$pythonArgs = @()
$version = ""

function Test-Interpreter($exe, $flag) {
    $label = if ($flag) { "$exe $flag" } else { $exe }
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) {
        $script:attempts += "  $label  ->  not on PATH"
        return $false
    }
    try {
        # 2>&1 turns a native command's stderr into error records, and under
        # ErrorActionPreference=Stop that is a *terminating* error - so an
        # interpreter that prints a harmless warning would be rejected. Relax
        # it for the probe only.
        $ErrorActionPreference = "Continue"
        $global:LASTEXITCODE = 0
        $out = if ($flag) { & $exe $flag --version 2>&1 } else { & $exe --version 2>&1 }
        $out = ($out | Out-String).Trim()
        if ($LASTEXITCODE -eq 0 -and $out -match "Python\s+3\.(\d+)") {
            # Free-threaded ("nogil") builds print the same --version as the
            # regular build, so they need a direct probe. cffi - a sounddevice
            # dependency - has no wheels for them, so pip falls back to
            # compiling it and dies in the C toolchain. sys._is_gil_enabled
            # only exists on 3.13+; older interpreters are never free-threaded,
            # so an erroring probe is treated as "fine".
            $global:LASTEXITCODE = 0
            $gil = if ($flag) { & $exe $flag -c "import sys; print(sys._is_gil_enabled())" 2>&1 } else { & $exe -c "import sys; print(sys._is_gil_enabled())" 2>&1 }
            $gil = ($gil | Out-String).Trim()
            if ($LASTEXITCODE -eq 0 -and $gil -eq "False") {
                $script:attempts += "  $label  ->  $out  (free-threaded build, skipped: cffi has no wheels for it)"
                return $false
            }
            $script:attempts += "  $label  ->  $out"
            $script:pythonExe = $exe
            $script:pythonArgs = if ($flag) { @($flag) } else { @() }
            $script:version = $out
            return $true
        }
        $script:attempts += "  $label  ->  exit $LASTEXITCODE, $out"
    } catch {
        $script:attempts += "  $label  ->  $($_.Exception.Message)"
    }
    return $false
}

# -Python takes either a version ("3.12", resolved through the py launcher) or
# a full path to an interpreter. Needed when the newest Python on the machine
# is ahead of the numpy wheels - see the wheel check further down.
if ($Python) {
    if ($Python -match '^\d+\.\d+$') {
        [void](Test-Interpreter "py" "-$Python")
    } else {
        [void](Test-Interpreter $Python $null)
    }
} else {
    if (-not (Test-Interpreter "py" "-3")) {
        [void](Test-Interpreter "python" $null)
    }
}

if (-not $pythonExe) {
    Write-Host ""
    Write-Host "  Tried:" -ForegroundColor Yellow
    $attempts | ForEach-Object { Write-Host $_ -ForegroundColor Yellow }
    Write-Host ""
    if ($attempts -match "free-threaded") {
        Write-Host "  Free-threaded builds (the t suffix in py -0p) are skipped on" -ForegroundColor Yellow
        Write-Host "  purpose: cffi, which sounddevice needs, has no wheels for them." -ForegroundColor Yellow
        Write-Host "  Pick the regular build of the same version." -ForegroundColor Yellow
        Write-Host ""
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        Write-Host "  Interpreters the py launcher knows about:" -ForegroundColor Yellow
        try { $ErrorActionPreference = "Continue"; & py -0p 2>&1 | Write-Host } catch { }
        Write-Host ""
        Write-Host "  Pick one of those:  ...\build_exe.ps1 -Python 3.12 -Clean" -ForegroundColor White
    } else {
        Write-Host "  Install Python 3.12 from https://www.python.org/downloads/" -ForegroundColor Yellow
        Write-Host "  and tick 'Add python.exe to PATH' in the installer." -ForegroundColor Yellow
    }
    Write-Host ""
    if ($Python) { throw "could not run the interpreter you asked for: $Python" }
    throw "no usable Python found"
}
Step "using $pythonExe $($pythonArgs -join ' ')  ($version)"

if ($Clean) {
    Step "cleaning build/, dist/ and .venv-build/"
    Remove-Item -Recurse -Force build, dist, .venv-build -ErrorAction SilentlyContinue
}

# ---- isolated build environment ---------------------------------------------
$venv = Join-Path $root ".venv-build"
$venvPython = Join-Path $venv "Scripts\python.exe"

# Deliberately free of quote characters. PowerShell 5.1 re-quotes arguments on
# their way to a native command and drops embedded double quotes, so a -c
# payload containing a string literal arrives at Python as a syntax error.
# The stamp ends with the GIL state: a free-threaded build reports the same
# 3.13.0 as the regular one, and the venv is tied to whichever interpreter
# made it - so a stamp mismatch means the venv must be recreated.
$describe = 'import sys, platform; print(sys.version.split()[0], platform.machine(), 64 if sys.maxsize > 2**32 else 32, sys._is_gil_enabled())'
$target = (& $pythonExe @pythonArgs -c $describe 2>&1 | Out-String).Trim()
if (-not $target) { $target = "unknown" }
$fields = $target -split '\s+'
$bits = $fields[2]
Step "build environment: Python $($fields[0]) $($fields[1]) ($($bits), gil=$($fields[3]))"

if (Test-Path $venvPython) {
    $stamp = (& $venvPython -c $describe 2>&1 | Out-String).Trim()
    if ($stamp -ne $target) {
        Step "existing build venv was made by a different interpreter - recreating it"
        Remove-Item -Recurse -Force $venv
    }
}
if (-not (Test-Path $venvPython)) {
    Step "creating build venv at .venv-build"
    # real splatting this time: @name on a plain variable, not @(expression)
    & $pythonExe @pythonArgs -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "could not create the build venv" }
}

& $venvPython -m pip install --upgrade pip --quiet

# ---- wheel check ------------------------------------------------------------
# numpy is the one dependency with no usable fallback: if pip cannot find a
# wheel it tries to *compile* numpy, goes looking for a C toolchain, and either
# grinds for ten minutes or dies on whatever ancient MinGW is on PATH. That
# happens whenever the interpreter is newer than the newest numpy release, or
# on 32-bit and some ARM builds. --only-binary turns a confusing ten-minute
# failure into an instant, honest one.
Step "checking for prebuilt wheels"
& $venvPython -m pip install --only-binary=:all: --upgrade numpy --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Warning "No prebuilt numpy wheel exists for Python $target-bit."
    Write-Host ""
    if ($bits -eq "32") {
        # The likely answer whenever the interpreter is a supported version.
        Write-Host "  That interpreter is 32-bit. NumPy 2.x does not publish 32-bit" -ForegroundColor Yellow
        Write-Host "  Windows wheels at all, so pip can only try to compile it." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  Install the 64-bit build of Python (the default download on" -ForegroundColor Yellow
        Write-Host "  python.org is 64-bit; 'Windows installer (64-bit)') and re-run" -ForegroundColor Yellow
        Write-Host "  with -Clean. Everything else here is fine." -ForegroundColor Yellow
    } else {
        Write-Host "  This is not a spektr problem - numpy simply has not published a" -ForegroundColor Yellow
        Write-Host "  Windows wheel for that interpreter yet (it usually lags a new" -ForegroundColor Yellow
        Write-Host "  Python release by a few months), so pip fell back to building it" -ForegroundColor Yellow
        Write-Host "  from source, which needs a full C toolchain." -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "  Point the build at an older interpreter instead:" -ForegroundColor Yellow
    Write-Host "      powershell -ExecutionPolicy Bypass -File packaging\build_exe.ps1 -Python 3.12 -Clean" -ForegroundColor White
    Write-Host ""
    Write-Host "  Installed on this machine:" -ForegroundColor Yellow
    try { & py -0p } catch { Write-Host "      (the py launcher is not available)" }
    Write-Host ""
    Write-Host "  If none of those are 3.10-3.13, install one from" -ForegroundColor Yellow
    Write-Host "  https://www.python.org/downloads/  and re-run with -Python 3.12 -Clean" -ForegroundColor Yellow
    throw "no numpy wheel for Python $target"
}

Step "installing dependencies (this takes a minute the first time)"
& $venvPython -m pip install --upgrade "pyinstaller>=6.3" --quiet
& $venvPython -m pip install --upgrade -e . --quiet
if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }

# ---- build ------------------------------------------------------------------
if ($OneDir) {
    $env:SPEKTR_ONEDIR = "1"
    Step "building dist\spektr\ (onedir, fast startup)"
} else {
    Remove-Item Env:\SPEKTR_ONEDIR -ErrorAction SilentlyContinue
    Step "building dist\spektr.exe (onefile, one portable download)"
}

& $venvPython -m PyInstaller --noconfirm --clean "packaging\spektr.spec"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$relative = if ($OneDir) { "dist\spektr\spektr.exe" } else { "dist\spektr.exe" }
$artifact = Join-Path $root $relative
if (-not (Test-Path $artifact)) { throw "expected $relative, but it is not there" }
$mb = [math]::Round((Get-Item $artifact).Length / 1MB, 1)

# ---- smoke test -------------------------------------------------------------
# --version exits without opening the UI, so it is safe to run headless and it
# proves the bundle can import spektr and its dependencies at all.
Step "smoke test: $relative --version"
$reported = & $artifact --version
if ($LASTEXITCODE -ne 0) { throw "the built exe failed to run" }
Write-Host "    $reported" -ForegroundColor Green

# ---- installer --------------------------------------------------------------
if ($Installer) {
    $iscc = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1

    if (-not $iscc) {
        Write-Warning "Inno Setup 6 not found - skipping the installer."
        Write-Warning "Get it from https://jrsoftware.org/isdl.php, then re-run with -Installer."
    } else {
        Step "compiling the installer with Inno Setup"
        & $iscc "packaging\installer.iss"
        if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }
    }
}

Write-Host "`ndone." -ForegroundColor Green
Write-Host "  $relative  ($mb MB)"
if ($Installer -and (Test-Path "dist\installer")) {
    Get-ChildItem "dist\installer\*.exe" | ForEach-Object {
        $size = [math]::Round($_.Length / 1MB, 1)
        Write-Host "  dist\installer\$($_.Name)  ($size MB)"
    }
}
Write-Host "`nTest it on a machine with no Python installed - that is the whole point."
