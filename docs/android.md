# Android app

The Android build is the same web client in a Capacitor shell. Unlike the
desktop app it ships no backend: on first launch it asks for the address of
your Amadeus server, checks it, and connects everything (chat, streaming
voice, the VRM avatar, settings) to it.

## Run a server it can reach

On the machine that runs Amadeus:

```bash
uvicorn amadeus.web.app:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` is the part people forget: the default binding only accepts
connections from the same machine. Find the machine's LAN IP (`ipconfig` on
Windows, `ip addr` on Linux) and enter it in the app as
`http://192.168.x.x:8000`. A hosted HTTPS backend works too, and pairs with
the optional API token (Settings, Access).

The connect screen probes `/healthz` before saving, and the URL can be
changed later in Settings, Access, Backend URL (use "Reconnect now" after
changing it).

## Install the app

Build the APK (see below) and sideload it
(Android will ask you to allow installs from your browser or file manager).
The APK is debug-signed for now, which is fine for sideloading; store-grade
signing is future work.

## Build it yourself

Requirements: Node 20+, JDK 17, and the Android SDK (platform 34,
build-tools 34). Then:

```bash
cd frontend
npm ci
npm run build
npx cap sync android
cd android && ./gradlew assembleDebug
# APK lands in app/build/outputs/apk/debug/app-debug.apk
```

If Gradle can't find the SDK, write `sdk.dir=/path/to/android-sdk` into
`frontend/android/local.properties` (that file stays local, it is
gitignored).

The CI release workflow builds the same APK on every `v*` tag and attaches
it to the GitHub release.

## What is and isn't wired

- Chat, WebSocket streaming with ordered audio, the 3D avatar with lip sync
  and gestures, memory panel, settings: all work against your server. The
  avatar and animation files stream from the backend's `/static`, so the
  first load needs a decent connection to it.
- Mic input works through the same push-to-talk/hands-free path as the
  browser; Android will prompt for microphone permission on first use.
- Plain-HTTP LAN backends are allowed by the app's network security config;
  it also works against HTTPS.
- Not yet: a bundled on-device backend, store distribution/signing, and
  battery-aware tuning of the 3D scene (consider closing the app rather
  than leaving it in the foreground all day).
