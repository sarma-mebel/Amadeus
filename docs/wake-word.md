# Wake word

Say the wake phrase and she starts a call. No button, no push-to-talk. The
listener streams mic audio to your own backend, a small openWakeWord ONNX
model scores each 80 ms hop locally, and a detection kicks off the normal
full-duplex call (or the classic hands-free call if duplex can't run).

Nothing is stored and nothing leaves the machine. The wake socket never
transcribes audio and never reaches the LLM or TTS. It only ever answers
with "wake" frames.

## Setup

1. Install the extra and fetch a model (one time):

   ```bash
   pip install "amadeus-kurisu[wake]"
   python scripts/fetch_wake_word.py
   ```

   This downloads the default `hey_jarvis_v0.1` model plus the two shared
   feature-extractor graphs into `models/wake_word/`.

2. Turn it on server-side in your config:

   ```yaml
   wake_word:
     enabled: true
   ```

3. Turn it on in the browser: Settings -> Features -> Wake word. Both
   switches are required. The browser toggle is per-device and defaults off,
   same as full-duplex calls.

Then say "hey Jarvis" while the tab is open and the mic is allowed.

## Custom phrases

There is no pretrained "Kurisu" or "Amadeus" model. openWakeWord ships a few
community phrases (hey jarvis, hey mycroft, alexa), and you can train your
own phrase with their synthetic-training notebook in about an hour on a free
Colab GPU: https://github.com/dscripka/openWakeWord#training-new-models

Drop the resulting `.onnx` into `models/wake_word/` (or point
`wake_word.model` at it) and restart the backend. If several models sit in
the directory the first one alphabetically wins; feature-extractor files are
skipped automatically.

## Tuning

```yaml
wake_word:
  enabled: true
  model: null              # explicit .onnx path; null = auto-discover
  threshold: 0.5           # raise if it false-triggers, lower if it misses
  refractory_seconds: 2.0  # ignore repeat hits inside this window
```

## How it degrades

Everything fails soft and loud, following the repo's degrade-loudly rule:

- Package or model missing: the socket answers one `wake_unavailable` error
  frame, the client shows a hint and stops trying, and the backend counts
  `web.wake_word_unavailable`.
- Server flag off: `wake_disabled` frame, same client behavior.
- Mic permission denied: the client reports it and latches off until you
  re-toggle the setting.
- Backend restart mid-listen: the client retries quietly every 15 seconds.
- Repeated inference errors: the detector disables itself for that
  connection after 3 consecutive failures (`web.wake_word_errors`).

Counters (`web.wake_sessions`, `web.wake_detections`, plus the two above)
show up in `GET /api/status` and Settings -> Diagnostics.

## Notes

- The wake listener releases the mic while any call is active. The call owns
  the audio pipeline; the listener resumes when the call ends.
- The wake word is deliberately not part of the presence presets
  (Quiet/Balanced/Lively). An always-on microphone is a privacy opt-in, same
  category as screen watching.
- Pet mode does not run the wake listener, and there is no plan to add it: a
  duplex call is one click away, which covers the same use. The wake word
  feature is complete as shipped.
