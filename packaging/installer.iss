; Inno Setup script for spektr - https://jrsoftware.org/isdl.php
;
; Build the onedir bundle first, then compile this:
;
;     powershell -ExecutionPolicy Bypass -File packaging\build_exe.ps1 -Installer
;
; Per-user by design: PrivilegesRequired=lowest means no UAC prompt, no admin
; account, and the install lands in %LOCALAPPDATA%\Programs\spektr. spektr
; already keeps its settings, themes and plugins in %APPDATA%\spektr, so an
; ordinary user can install, run and uninstall it without asking anyone.

#define AppName        "spektr"
#define AppPublisher   "zemo"
#define AppURL         "https://github.com/MrEmoji27/spektr"
#define AppExe         "spektr.exe"

; Version is read out of the built exe, so it can never disagree with the code.
#define AppVersion GetVersionNumbersString("..\dist\spektr\spektr.exe")

[Setup]
AppId={{8F3C1E5A-3D2B-4A7E-9C61-5E2B0A7D4F11}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\dist\installer
OutputBaseFilename=spektr-{#AppVersion}-setup
SetupIconFile=spektr.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "addtopath"; Description: "Add spektr to &PATH (so you can type 'spektr' in any terminal)"; GroupDescription: "Shortcuts:"

[Files]
; The whole onedir bundle. DestDir keeps the layout PyInstaller produced -
; spektr.exe expects its _internal folder to sit beside it.
Source: "..\dist\spektr\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\README.md"; DestDir: "{app}"; DestName: "README.md"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\{#AppName} - what am I listening to?"; Filename: "{app}\{#AppExe}"; Parameters: "--diagnose"; Comment: "Probe every audio source and report what it delivers"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Registry]
; Per-user PATH. Inno rewrites the whole value, so this appends rather than
; replaces - check first that it is not already there.
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; \
    ValueData: "{olddata};{app}"; Tasks: addtopath; Check: NeedsPath(ExpandConstant('{app}'))

[Run]
Filename: "{app}\{#AppExe}"; Description: "Run {#AppName} now"; Flags: nowait postinstall skipifsilent

[Code]
function NeedsPath(Dir: string): Boolean;
var
  Existing: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', Existing) then
  begin
    Result := True;
    exit;
  end;
  { pad with semicolons so a partial match (...\spektr-old) cannot fool it }
  Result := Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Existing) + ';') = 0;
end;
