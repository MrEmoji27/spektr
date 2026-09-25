# spektr installer for Windows.
#
#   irm https://raw.githubusercontent.com/MrEmoji27/spektr/main/install.ps1 | iex
#
# Puts spektr.exe in %LOCALAPPDATA%\spektr, adds it to your PATH and the Start
# Menu, and checks it runs. Run it again to update. To remove it, run
# spektr-uninstall. No Python needed, and no admin rights.
#
# Settings, as environment variables, all optional:
#   SPEKTR_VERSION      a release to install, like v0.6.0 (default: the latest)
#   SPEKTR_DIR          where to install (default: %LOCALAPPDATA%\spektr)
#   SPEKTR_FROM         install this local spektr.exe instead of downloading
#   SPEKTR_NO_PATH      leave PATH alone
#   SPEKTR_NO_SHORTCUT  no Start Menu entry
#   SPEKTR_NO_LAUNCH    do not offer to start spektr at the end
#   NO_COLOR            plain text
#
# Written for Windows PowerShell 5.1, which is what `irm | iex` runs in, and
# kept to ASCII: the glyphs are built from code points, so the script reads the
# same whatever encoding it was fetched in.

& {
Set-StrictMode -Version 2
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Repo = 'MrEmoji27/spektr'
$Dir = if ($env:SPEKTR_DIR) { $env:SPEKTR_DIR } else { Join-Path $env:LOCALAPPDATA 'spektr' }
$Exe = Join-Path $Dir 'spektr.exe'
$State = @{ Step = '' }   # the step on screen, for Done to rewrite

# -- output ------------------------------------------------------------------

$oldEncoding = [Console]::OutputEncoding
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch {}
$Live = -not [Console]::IsOutputRedirected
$Color = $Live -and -not $env:NO_COLOR -and $Host.UI.SupportsVirtualTerminal
$E = [char]27

$Blocks = @(0x2581, 0x2582, 0x2583, 0x2584, 0x2585, 0x2586, 0x2587, 0x2588) | ForEach-Object { [string][char]$_ }
$Full = [string][char]0x2588
$Dot = [string][char]0x00B7
$Tick = [string][char]0x2713
$Cross = [string][char]0x2717
$Arrow = [string][char]0x25B8

# the spektr sunset: violet, magenta, orange, gold
$Stops = @(@(123, 47, 247), @(241, 7, 163), @(255, 109, 0), @(255, 208, 0))

function Blend([double]$u) {
    $u = [Math]::Min(1.0, [Math]::Max(0.0, $u))
    $x = $u * ($Stops.Count - 1)
    $i = [Math]::Min([int][Math]::Floor($x), $Stops.Count - 2)
    $f = $x - $i
    $a = $Stops[$i]; $b = $Stops[$i + 1]
    return @(0, 1, 2 | ForEach-Object { [int]($a[$_] + ($b[$_] - $a[$_]) * $f) })
}

function Paint([string]$text, $rgb) {
    if (-not $Color) { return $text }
    return "$E[38;2;$($rgb[0]);$($rgb[1]);$($rgb[2])m$text$E[0m"
}

function Dim([string]$text) { if ($Color) { "$E[2m$text$E[0m" } else { $text } }
function Bold([string]$text) { if ($Color) { "$E[1m$text$E[0m" } else { $text } }

function Say([string]$text) { [Console]::Out.Write($text) }

function Step([string]$what) {
    if ($Live) { Say ('  ' + (Paint $Arrow (Blend 0.66)) + ' ' + $what.PadRight(24)) }
}

function Done([string]$detail) {
    Say ("`r" + '  ' + (Paint $Tick @(80, 220, 120)) + ' ')
    # rewrite the label in place, then the detail after it
    Say ($State.Step.PadRight(24) + (Dim $detail) + "`n")
}

function Fail([string]$why, [string]$hint) {
    Say ("`n  " + (Paint $Cross @(255, 80, 80)) + ' ' + (Bold $why) + "`n")
    if ($hint) { Say ('    ' + $hint + "`n") }
    Say "`n"
    try { [Console]::OutputEncoding = $oldEncoding } catch {}
    throw 'spektr was not installed.'
}

function Begin([string]$what) { $State.Step = $what; Step $what }

# One frame of a spectrum, $width bars, moving with $t. Used for the intro
# and for anything that makes you wait.
function Spectrum([int]$width, [double]$t, [int]$lit) {
    $out = New-Object Text.StringBuilder
    for ($i = 0; $i -lt $width; $i++) {
        if ($i -ge $lit) { [void]$out.Append((Dim $Dot)); continue }
        $h = 0.5 + 0.5 * [Math]::Sin($t * 7.0 + $i * 0.55) * [Math]::Cos($t * 3.1 - $i * 0.23)
        $g = $Blocks[[int][Math]::Round($h * ($Blocks.Count - 1))]
        [void]$out.Append((Paint $g (Blend ($i / [Math]::Max(1, $width - 1)))))
    }
    return $out.ToString()
}

# -- the banner ----------------------------------------------------------------

$Logo = @(
    ' ####  #####  ###### #    # ###### ##### ',
    '#      #    # #      #   #    #    #    #',
    ' ####  #####  #####  ####     #    ##### ',
    '     # #      #      #   #    #    #  #  ',
    ' ####  #      ###### #    #   #    #   # '
)

Say "`n"
foreach ($row in $Logo) {
    $line = New-Object Text.StringBuilder
    [void]$line.Append('  ')
    for ($i = 0; $i -lt $row.Length; $i++) {
        if ($row[$i] -eq '#') {
            [void]$line.Append((Paint $Full (Blend ($i / ($row.Length - 1)))))
        } else {
            [void]$line.Append(' ')
        }
    }
    Say ($line.ToString() + "`n")
}
Say ('  ' + (Dim 'a music visualiser for your terminal') + "`n`n")
if ($Live) {
    # a second of the thing itself
    $w = $Logo[0].Length
    $clock = [Diagnostics.Stopwatch]::StartNew()
    while ($clock.Elapsed.TotalSeconds -lt 1.1) {
        Say ("`r  " + (Spectrum $w $clock.Elapsed.TotalSeconds $w))
        Start-Sleep -Milliseconds 40
    }
    Say ("`r  " + (' ' * $w) + "`r")
}

# -- 1. the system -----------------------------------------------------------------

Begin 'Checking your system'
$os = [Environment]::OSVersion.Version
if ($os.Major -lt 10) {
    Fail 'spektr needs Windows 10 or newer.' ''
}
$arch = $env:PROCESSOR_ARCHITECTURE
$archNote = switch ($arch) {
    'AMD64' { 'x64' }
    'ARM64' { 'ARM64, running the x64 build' }
    default { Fail "spektr has no build for $arch Windows." 'Try the Python install: pip install spektr-audio' }
}
$winName = if ($os.Build -ge 22000) { 'Windows 11' } else { 'Windows 10' }
Done "$winName ($archNote), PowerShell $($PSVersionTable.PSVersion.Major).$($PSVersionTable.PSVersion.Minor)"

# -- 2. which release ------------------------------------------------------------

[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$tag = $env:SPEKTR_VERSION
$source = $env:SPEKTR_FROM
if ($source) {
    Begin 'Using a local build'
    if (-not (Test-Path $source)) { Fail "There is no file at $source." '' }
    Done $source
} else {
    Begin 'Finding the release'
    if (-not $tag) {
        try {
            $rel = Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest" -Headers @{ 'User-Agent' = 'spektr-installer' }
            $tag = $rel.tag_name
        } catch {
            $tag = $null   # rate-limited or offline: the download link below still works
        }
    }
    if ($tag) { Done $tag } else { Done 'the latest' }
}

# -- 3. the download, with something to watch --------------------------------------

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("spektr-" + [Guid]::NewGuid().ToString('N') + '.exe')
if ($source) {
    Copy-Item $source $tmp
} else {
    $url = if ($tag) { "https://github.com/$Repo/releases/download/$tag/spektr.exe" }
           else { "https://github.com/$Repo/releases/latest/download/spektr.exe" }
    Begin 'Downloading spektr.exe'
    Add-Type -AssemblyName System.Net.Http
    $client = New-Object Net.Http.HttpClient
    $client.DefaultRequestHeaders.Add('User-Agent', 'spektr-installer')
    try {
        $resp = $client.GetAsync($url, [Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
        if (-not $resp.IsSuccessStatusCode) {
            Fail "The download failed ($([int]$resp.StatusCode))." "Check the release exists: https://github.com/$Repo/releases"
        }
        $total = $resp.Content.Headers.ContentLength
        $in = $resp.Content.ReadAsStreamAsync().Result
        $out = [IO.File]::Create($tmp)
        $buf = New-Object byte[] 65536
        $got = 0L
        $clock = [Diagnostics.Stopwatch]::StartNew()
        $drawn = -1.0
        try {
            while (($n = $in.Read($buf, 0, $buf.Length)) -gt 0) {
                $out.Write($buf, 0, $n)
                $got += $n
                $now = $clock.Elapsed.TotalSeconds
                if ($Live -and ($now - $drawn) -ge 0.05) {
                    $drawn = $now
                    $share = if ($total) { $got / $total } else { 0.0 }
                    $mb = '{0:N1}' -f ($got / 1MB)
                    $of = if ($total) { ' / {0:N1} MB' -f ($total / 1MB) } else { ' MB' }
                    $rate = if ($now -gt 0) { '  {0:N1} MB/s' -f ($got / 1MB / $now) } else { '' }
                    Say ("`r  " + (Paint $Arrow (Blend 0.66)) + ' ' + 'Downloading spektr.exe'.PadRight(24) +
                         (Spectrum 28 $now ([int](28 * $share))) + ' ' + (Dim "$mb$of$rate") + '   ')
                }
            }
        } finally {
            $out.Close(); $in.Close()
        }
    } catch {
        if ($_.Exception.Message -eq 'spektr was not installed.') { throw }
        Fail 'The download failed.' "$($_.Exception.GetBaseException().Message)"
    } finally {
        $client.Dispose()
    }
    if ($Live) { Say ("`r" + (' ' * 90) + "`r") }
    Done ('{0:N1} MB in {1:N1} s' -f ((Get-Item $tmp).Length / 1MB), $clock.Elapsed.TotalSeconds)
}

# a Windows program starts with MZ; anything else is an error page
$head = New-Object byte[] 2
$fs = [IO.File]::OpenRead($tmp); [void]$fs.Read($head, 0, 2); $fs.Close()
if ((Get-Item $tmp).Length -lt 1MB -or $head[0] -ne 0x4D -or $head[1] -ne 0x5A) {
    Remove-Item $tmp -Force
    Fail 'What came down is not spektr.exe.' "Try again, or download it from https://github.com/$Repo/releases"
}

# -- 4. install ------------------------------------------------------------------

Begin 'Installing'
$running = Get-Process spektr -ErrorAction SilentlyContinue | Where-Object {
    try { $_.Path -eq $Exe } catch { $false }
}
if ($running) {
    Remove-Item $tmp -Force
    Fail 'spektr is running.' 'Close it, then run this again.'
}
New-Item -ItemType Directory -Force -Path $Dir | Out-Null
Move-Item -Force $tmp $Exe

# an uninstaller next to it, on the PATH as spektr-uninstall
$uninstall = @'
$ErrorActionPreference = 'SilentlyContinue'
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$path = [Environment]::GetEnvironmentVariable('Path', 'User')
$keep = ($path -split ';' | Where-Object { $_ -and ($_.TrimEnd('\') -ne $dir.TrimEnd('\')) }) -join ';'
[Environment]::SetEnvironmentVariable('Path', $keep, 'User')
Remove-Item (Join-Path ([Environment]::GetFolderPath('Programs')) 'spektr.lnk') -Force
Write-Host 'spektr is uninstalled. Your settings are kept in your config folder.'
Start-Process cmd.exe -ArgumentList '/c', "timeout /t 1 >nul & rmdir /s /q `"$dir`"" -WindowStyle Hidden
'@
Set-Content -Path (Join-Path $Dir 'uninstall.ps1') -Value $uninstall -Encoding UTF8
Set-Content -Path (Join-Path $Dir 'spektr-uninstall.cmd') -Encoding ASCII -Value `
    '@powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0uninstall.ps1"'
Done $Dir

# -- 5. PATH and the Start Menu ----------------------------------------------------

$pathNote = ''
if (-not $env:SPEKTR_NO_PATH) {
    Begin 'Adding to PATH'
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $parts = @($userPath -split ';' | Where-Object { $_ })
    if ($parts | Where-Object { $_.TrimEnd('\') -eq $Dir.TrimEnd('\') }) {
        Done 'already there'
    } else {
        [Environment]::SetEnvironmentVariable('Path', (($parts + $Dir) -join ';'), 'User')
        $pathNote = 'new terminals'
        Done 'done, for new terminals'
    }
    if (-not (($env:Path -split ';') -contains $Dir)) { $env:Path = "$env:Path;$Dir" }
}
if (-not $env:SPEKTR_NO_SHORTCUT) {
    Begin 'Start Menu entry'
    try {
        $lnk = (New-Object -ComObject WScript.Shell).CreateShortcut(
            (Join-Path ([Environment]::GetFolderPath('Programs')) 'spektr.lnk'))
        $lnk.TargetPath = $Exe
        $lnk.WorkingDirectory = $Dir
        $lnk.IconLocation = "$Exe,0"
        $lnk.Description = 'A music visualiser for your terminal'
        $lnk.Save()
        Done 'spektr'
    } catch {
        Done 'skipped, Windows would not make one'
    }
}

# -- 6. check it runs, with something to watch while the first start unpacks --------

function Watch([string]$label, [string[]]$argv) {
    $State.Step = $label
    $outFile = [IO.Path]::GetTempFileName()
    $p = Start-Process -FilePath $Exe -ArgumentList $argv -NoNewWindow -PassThru `
        -RedirectStandardOutput $outFile -RedirectStandardError "$outFile.err"
    # read the handle now: without it Windows PowerShell never learns the exit code
    $null = $p.Handle
    $clock = [Diagnostics.Stopwatch]::StartNew()
    while (-not $p.HasExited) {
        if ($Live) {
            Say ("`r  " + (Paint $Arrow (Blend 0.66)) + ' ' + $label.PadRight(24) + (Spectrum 16 $clock.Elapsed.TotalSeconds 16))
        }
        if ($clock.Elapsed.TotalSeconds -gt 120) { $p.Kill(); break }
        Start-Sleep -Milliseconds 40
    }
    $p.WaitForExit()
    $text = (Get-Content $outFile -Raw -ErrorAction SilentlyContinue)
    Remove-Item $outFile, "$outFile.err" -Force -ErrorAction SilentlyContinue
    if ($Live) { Say ("`r" + (' ' * 60) + "`r") }
    if ($p.ExitCode -ne 0 -or -not $text) { return $null }
    return $text.Trim()
}

$version = Watch 'Checking it starts' @('--version')
if (-not $version) { Fail 'spektr.exe is installed but would not start.' "Try running it yourself: $Exe" }
Done $version
# --check-modes arrived in 0.5.5; an older build would start the visualiser
$v = [version]'0.0'
[void][version]::TryParse(($version -replace '^spektr\s+', ''), [ref]$v)
if ($v -ge [version]'0.5.5') {
    $modes = Watch 'Loading every mode' @('--check-modes')
    if (-not $modes) { Fail 'Some of spektr''s modes failed to load.' "Run $Exe --check-modes to see which." }
    Done $modes
}

# -- done -------------------------------------------------------------------------

$lines = @(
    (Bold "$version is installed."),
    '',
    ('Start it       ' + (Paint 'spektr' (Blend 0.66))),
    ('Every option   ' + (Paint 'spektr --help' (Blend 0.66))),
    ('Uninstall      ' + (Paint 'spektr-uninstall' (Blend 0.66)))
)
if ($pathNote) { $lines += ''; $lines += (Dim 'Open a new terminal first, so it can find spektr.') }
$plain = @($lines | ForEach-Object { $_ -replace "$E\[[0-9;]*m", '' })
$wide = ($plain | Measure-Object -Property Length -Maximum).Maximum + 4
$h = [string][char]0x2500
Say ("`n  " + (Paint ([string][char]0x256D + ($h * $wide) + [string][char]0x256E) (Blend 0.1)) + "`n")
for ($i = 0; $i -lt $lines.Count; $i++) {
    $pad = ' ' * ($wide - 4 - $plain[$i].Length)
    Say ('  ' + (Paint ([string][char]0x2502) (Blend 0.1)) + '  ' + $lines[$i] + $pad + '  ' +
         (Paint ([string][char]0x2502) (Blend 0.9)) + "`n")
}
Say ('  ' + (Paint ([string][char]0x2570 + ($h * $wide) + [string][char]0x256F) (Blend 0.9)) + "`n`n")

try { [Console]::OutputEncoding = $oldEncoding } catch {}

if ($Live -and -not $env:SPEKTR_NO_LAUNCH -and -not [Console]::IsInputRedirected) {
    $answer = Read-Host '  Start spektr now? [Y/n]'
    if ($answer -notmatch '^[nN]') { & $Exe }
}
}
