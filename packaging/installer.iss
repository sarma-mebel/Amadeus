; Inno Setup script for the Amadeus desktop app (Windows).
;
; Build order:
;   1. scripts\build_desktop.ps1              -> dist\AmadeusKurisu\
;   2. iscc packaging\installer.iss           -> dist\AmadeusKurisu-Setup.exe
;
; Requires Inno Setup 6 (https://jrsoftware.org/isinfo.php).
; The app needs the Microsoft WebView2 Runtime; Windows 11 ships it,
; and this installer bootstraps it on machines that lack it when
; MicrosoftEdgeWebView2Setup.exe (the evergreen bootstrapper, free from
; https://developer.microsoft.com/microsoft-edge/webview2/) is placed
; next to this script before compiling.

#define AppName "Amadeus - Kurisu"
#define AppVersion "0.1.0"
#define AppExe "AmadeusKurisu.exe"

[Setup]
AppId={{8B1C64F3-2B7A-4C57-9C55-AMADEUSKURISU}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Amadeus Contributors
DefaultDirName={autopf}\AmadeusKurisu
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=AmadeusKurisu-Setup
SetupIconFile=assets\amadeus.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; Per-user install: no admin prompt, mirrors how game launchers install.
PrivilegesRequired=lowest

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\AmadeusKurisu\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion
#ifexist "MicrosoftEdgeWebView2Setup.exe"
Source: "MicrosoftEdgeWebView2Setup.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall
#endif

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
#ifexist "MicrosoftEdgeWebView2Setup.exe"
Filename: "{tmp}\MicrosoftEdgeWebView2Setup.exe"; Parameters: "/silent /install"; \
  StatusMsg: "Installing WebView2 Runtime..."; Check: not IsWebView2Installed
#endif
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; \
  Flags: nowait postinstall skipifsilent

[Code]
function IsWebView2Installed: Boolean;
var
  Version: String;
begin
  { Evergreen WebView2 runtime writes its version here (per-machine or per-user). }
  Result :=
    RegQueryStringValue(HKLM,
      'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}',
      'pv', Version) or
    RegQueryStringValue(HKCU,
      'Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}',
      'pv', Version);
end;
