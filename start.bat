@echo off
REM ---------------------------------------------------------------------------
REM  spektr - double-click to run from source.
REM
REM  For people who already have Python. If you do not, grab spektr.exe from
REM  the Releases page instead: https://github.com/MrEmoji27/spektr/releases
REM
REM  First run builds a private environment in .venv and installs what spektr
REM  needs. That takes a minute. Every run after is instant.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

REM Prefer py -3, but only when it is the regular build. A free-threaded
REM interpreter (the launcher marks it with a t, and it is some machines'
REM default) cannot install cffi - a sounddevice dependency - so it would
REM fail on first run with a confusing build error.
set PY=python
where py >nul 2>&1 && (
    py -3 -c "import sys; print(sys._is_gil_enabled())" 2>nul | findstr /b "True" >nul
    if not errorlevel 1 set PY=py -3
)

%PY% --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo   Python is not installed.
    echo.
    echo   Either install it from https://www.python.org/downloads/
    echo   ^(tick "Add python.exe to PATH" in the installer^)
    echo   or download spektr.exe, which needs nothing at all:
    echo   https://github.com/MrEmoji27/spektr/releases
    echo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo   First run - setting up. This takes a minute, once.
    echo.
    %PY% -m venv .venv || goto :failed
    ".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
    ".venv\Scripts\python.exe" -m pip install -e . || goto :failed
    echo.
    echo   Ready. Starting spektr - press q to quit, v to change the visual.
    echo.
)

".venv\Scripts\python.exe" -m spektr.app %*
if errorlevel 1 goto :failed
exit /b 0

:failed
echo.
echo   Something went wrong. Run this for a report of what spektr can hear:
echo       .venv\Scripts\python.exe -m spektr.app --diagnose
echo.
pause
exit /b 1
