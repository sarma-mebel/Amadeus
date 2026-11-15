# Bring your own character

Amadeus ships as Kurisu, but the engine underneath is character-agnostic:
layered memory, the relationship arc, presence features (greetings, pokes,
idle chatter, farewells), the daily diary, streaming TTS with lip sync, and
the tagged bilingual output all work for any persona. A character pack swaps
the identity on top of that engine through configuration alone.

The fastest way in is a character pack: one folder (or zip) with everything,
installed by one command. The sections after that describe what the installer
does, for anyone wiring it by hand.

## Character packs

A pack looks like this:

```
rin/
  character.yaml    required
  prompt.txt        required (the persona prompt)
  avatar.vrm        optional VRM avatar (0.x or 1.0)
  voice_ref.wav     optional voice clip for the zero-shot clone
```

`character.yaml`:

```yaml
name: Rin
full_name: Rin Asakura
description: Retired pro gamer turned late-night streamer.
author: you
license: CC0
prompt_file: prompt.txt
avatar: avatar.vrm          # omit if the pack has no avatar
spoken_language: english    # or japanese
birthday: "03-14"           # optional; greeting milestone on that day
# optional; 1-3 sentence register spec used by the short satellite prompts
# (pokes, idle chatter, greetings, morning brief, diary, reach-out)
voice_style: "calm, bookish, endlessly patient; never sarcastic"
voice:                      # omit if the pack has no voice
  ref_audio: voice_ref.wav
  ref_text: "The exact transcript of the reference clip."
  language: English
```

Install it:

```bash
python -m amadeus.characters path/to/rin --apply
```

This copies the avatar into `static/models/` (and registers it in
`models.json` so it shows up in the model picker), the voice clip into
`models/characters/<id>/`, the prompt into `characters/installed/<id>/`, and
merges the activating config into `config.local.yaml` (your previous config
is kept as `config.local.yaml.bak-character`). Without `--apply` it prints
the YAML for you to paste instead. Restart the app to meet them.

A complete example pack lives in [`characters/examples/rin/`](../characters/examples/rin/)
(prompt only; add a VRM and a voice clip to make it fully embodied).

## What a pack sets, by hand

A full character swap has four parts. Only the first is required.

## 1. Persona (required)

Add a `character:` section to your config:

```yaml
character:
  name: Rin                    # short name used inside prompts
  full_name: Rin Asakura       # used by greetings, farewells, brief, diary
  system_prompt_path: characters/rin.txt   # or inline via system_prompt
```

The custom prompt replaces the built-in Kurisu identity, lore, and few-shot
examples. Everything generic stays wrapped around it automatically: the
companion behavior (listen first, no therapy-speak, match the user's
energy), healthy relationship boundaries, the safety block, teasing budget,
mood and trust dials, and the length policy.

Write the prompt like a character sheet, not a paragraph of adjectives. The
built-in Kurisu persona works because it carries concrete behaviors and
example exchanges, and the model imitates examples far more reliably than it
follows rules. A good pack prompt includes:

- who the character is, in second person ("You are ...")
- how they talk: register, verbal habits, what they would never say
- how they handle being compliments, sadness, and excitement
- three to five short example exchanges in their voice (good and bad)

The greeting, farewell, poke, idle-chatter, morning-brief, and diary prompts
all pick up `full_name` automatically. Memory summaries use `name`.

Those satellite prompts are much smaller than the chat prompt, so they can't
carry your whole character sheet. They include a short voice capsule instead:
the manifest's `voice_style` if you set one (recommended — one to three
sentences describing the register), otherwise the opening sentences of your
system prompt. Make sure one of those two reads as a voice spec.

## 2. Avatar

Drop a VRM file under `static/models/` and list it in
`static/models/models.json`:

```json
{
  "models": [
    { "id": "rin", "label": "Rin", "file": "/static/models/Rin.vrm" }
  ]
}
```

It appears in the model picker in the web UI and hot-swaps live. The
animation library and emotion-to-expression mapping use standard VRM
expression presets (happy, angry, sad, surprised, relaxed, neutral), so any
VRM model (0.x or 1.0) with those presets gets the full body-language system.

## 3. Voice

All voice assets are config fields. For the GPU cloned voice, point the
zero-shot reference at a clip of your character (10 to 20 seconds of clean
speech) and transcribe it:

```yaml
providers:
  tts: qwen3_tts
  qwen3_tts_ref_audio: characters/rin_ref.wav
  qwen3_tts_ref_text: "The exact transcript of the reference clip."
  qwen3_tts_language: Japanese   # or English
```

CPU alternatives: `melotts_onnx` with your own checkpoint paths, `openai`,
or `pyttsx3` for a keyless system voice.

## 4. Spoken language

The default pipeline asks the LLM for bilingual tagged output (English text
for display and memory, Japanese for the voice). For an English-speaking
character set:

```yaml
audio:
  voice_language: english
```

## Known limits

- The web UI branding (window title, loading screen, the onboarding wizard
  intro line) still says Kurisu/Amadeus. Cosmetic, on the roadmap.
- The conversation eval harness (`make eval`) scores Kurisu-specific
  behavior; a custom pack needs its own scenarios before that gate means
  anything for it.
- The VADER fallback emotion keywords (used only when tagged output is off
  or the model ignores the format) are tuned to Kurisu. With tagged output
  on (the default), emotions come from the LLM and this does not matter.
