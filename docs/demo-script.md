# 60-second demo storyboard

The pitch is memory and presence, not features. Every competing "VTuber chatbot"
video shows an avatar answering a question; almost none can show the assistant
remembering last week. Lead with that.

Record at 1080p with system audio. The web UI in VN (immersive) mode looks best on
camera. Use the GPU voice if recording on the desktop build, ONNX voice otherwise.

## Shot list

**0:00 - 0:08 | Cold open on the voice.**
Desktop app already open. Press the mic, ask by voice: "Morning, Kurisu." She answers
out loud, in Japanese with English subtitles, avatar lip-syncing and gesturing. No
title cards yet, let the first seconds do the work.

**0:08 - 0:20 | She remembers yesterday.**
Ask: "What was that thing I was worried about yesterday?" She references the actual
topic from a previous session (seed this the day before recording: tell her about an
upcoming exam, interview, or deadline). Overlay caption: "Layered memory. She
remembers across sessions."

**0:20 - 0:30 | She noticed how you've been.**
Show a greeting where she checks in based on tone history ("You've sounded worn out
lately. Did something change?"). If the live line doesn't fire naturally, show the
Memory panel instead: relationship gauge, tone trend, the synapse map of facts she
holds. Caption: "Relationship arc. Trust builds over weeks, not messages."

**0:30 - 0:40 | The avatar acts.**
One expressive exchange: tease her (call her Christina) and catch the indignant
reaction, expression change, and gesture. Caption: "200+ vetted animations,
emotion-driven."

**0:40 - 0:50 | Pet mode.**
Toggle pet mode from the tray. She shrinks to a small transparent window over your
IDE/desktop. Ask her something by voice while "working". Caption: "Pet mode. She
stays with you while you work."

**0:50 - 1:00 | The honest close.**
Say "I should get back to work." She gives one warm send-off line and lets you go.
Caption over the goodbye: "No guilt-tripping. No engagement tricks. Self-hosted, Apache 2.0
licensed." End card: repo URL.

## Recording notes

- Seed memory the day before: one real conversation mentioning a concrete upcoming
  event, so the 0:08 beat has something true to recall.
- `web.streaming_inner_thoughts: false` for the recording; the ghost line reads as
  clutter at video pace.
- Idle chatter and poke reactions have cooldowns; if a beat doesn't fire, move on and
  splice a second take.
- Keep her replies short for pacing: the length policy already adapts, but short
  questions get short answers.
- Trim silences between beats; the target is a fast 60 seconds, not real time.
