# Hosted public demo (Fly.io)

A public "meet Kurisu in 30 seconds" instance. The repo already contains
everything except the secrets: `Dockerfile` builds the frontend and installs
the `[web,gemini]` extras, `fly.toml` serves it on port 8080, and
`config/config.fly.yaml` (baked into the image as the active config) runs
Gemini for chat with demo hardening switched on.

## What demo hardening does

`config.fly.yaml` sets three `web:` keys that exist for exactly this use case:

- `demo_mode: true` locks every settings and persona mutation endpoint with a
  403, so anonymous visitors cannot rewrite providers, paste their own keys
  into the server config, or edit the persona. `/api/status` reports
  `demo_mode: true` and the web UI hides the settings gear.
- `rate_limit_per_minute: 8` caps chat/TTS/STT turns per client IP (sliding
  60s window, `Fly-Client-IP` aware) so one visitor cannot drain the shared
  Gemini quota. Visitors who hit it get a polite 429 and a working retry a
  minute later.
- `session_ttl_hours: 1.0` evicts idle anonymous sessions quickly.

Long-term memory is off in the demo config; each visitor gets short-term
conversation context only.

## Deploy

One-time setup (needs a Fly.io account and `flyctl`):

```bash
fly launch --no-deploy          # or reuse the existing app name in fly.toml
fly secrets set GEMINI_API_KEY=<key from https://aistudio.google.com/app/apikey>
```

Then, for every deploy:

```bash
fly deploy
```

CI can do this automatically: `.github/workflows/fly-deploy.yml` deploys on
push once a `FLY_API_TOKEN` repo secret exists (`fly tokens create deploy`).

## Sizing and voice notes

- `fly.toml` currently requests a 256 MB shared-CPU VM. That is enough for
  text chat. If TTS is enabled (`providers.tts: huggingface_inference`
  offloads synthesis to the HF Space, so the VM itself stays light), bump to
  512 MB if you see OOM restarts during startup.
- The VRM avatar loads client-side from `/static`. The heavyweight runtime
  assets (`Kurisu.vrm`, environment `.glb`s) are gitignored but present in a
  full local checkout, and `fly deploy` ships the local build context. So
  deploy from a checkout that has your own avatar and environment files in
  place, or the demo will run without the 3D avatar.
- The demo shares one Gemini key across all visitors. Free-tier quota is
  fine for light traffic; watch `fly logs` for 429s from Google and consider
  lowering `rate_limit_per_minute` if they appear.

## Smoke test after deploy

```bash
curl -s https://<app>.fly.dev/healthz
curl -s https://<app>.fly.dev/api/status | python -m json.tool | head
# demo_mode must be true, llm.is_dummy must be false
```
