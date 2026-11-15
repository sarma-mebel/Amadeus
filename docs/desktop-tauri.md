# Desktop Tauri Packaging

> This page is an overview of how the desktop app is packaged. The full build
> procedure is in [desktop-release.md](desktop-release.md).

This path packages Amadeus Kurisu as a Windows-first desktop app:

- Tauri hosts the native window and OS keychain bridge.
- A PyInstaller-built Python backend runs as a Tauri sidecar.
- The backend serves the built React UI, `/api`, `/api/chat/stream`, `/static`, and `/assets`.
- Gemini API keys are stored in Windows Credential Manager through Tauri and passed only as `GEMINI_API_KEY` to the sidecar process.
- Mutable data lives in the app data directory: `config.yaml`, logs, transcripts, and memory files.

## Required Bundled Assets

The package validates these resources at backend startup:

- `models/kurisu_voice_onnx/model.onnx`
- `models/kurisu_voice_onnx/tokens.txt`
- `models/kurisu_voice_onnx/lexicon.txt`
- `static/models/Kurisu.vrm`
- `static/environment/steinsgate_-_future_gadget_lab.glb`
- `static/environment/steinsgate_time_machine.glb`
- `assets/kurisu_isms.txt`

The large training checkpoints in `models/kurisu_voice/*.pth` are intentionally not bundled.

## Build On Windows

From the repo root:

```powershell
.\scripts\package_desktop_windows.ps1
```

Outputs:

- Tauri NSIS installer under `frontend\src-tauri\target\release\bundle\nsis`
- Portable zip at `dist-desktop\Amadeus-Kurisu-portable-windows.zip`

If the backend sidecar alone needs to be rebuilt:

```powershell
.\scripts\package_backend_windows.ps1
```

Tauri requires sidecar binaries to include the target triple suffix, so the script copies PyInstaller output to:

```text
frontend\src-tauri\binaries\amadeus-backend-x86_64-pc-windows-msvc.exe
```

