# Voice expressiveness

The optional features below make her voice and presence feel more alive in
conversation. Most are driven by audio files you generate or record yourself,
and those stay silently off until the files exist.

## Emotion voice references (qwen3_tts)

The tagged output already labels every sentence with an emotion, and the face
and gestures use it. With emotion references, the voice uses it too.

Record (or cut from source material) a short clip of the voice in a given
emotion and place it next to the base reference clip, using the base name plus
the emotion label:

```
models/kurisu_voice_qwen/kurisu_ref.wav          # base (already there)
models/kurisu_voice_qwen/kurisu_ref_happy.wav    # cheerful delivery
models/kurisu_voice_qwen/kurisu_ref_happy.txt    # transcript of that clip
models/kurisu_voice_qwen/kurisu_ref_angry.wav
models/kurisu_voice_qwen/kurisu_ref_angry.txt
```

Each clip needs its own `.txt` transcript containing exactly what is said in
that clip. A clip without a transcript is skipped with a warning, because
cloning against the wrong text degrades the voice. Aim for the same quality
bar as the base reference: a clean 5 to 15 second line with the emotion
clearly audible.

Restart the server and check the log. It lists every emotion reference it
loaded. From then on, sentences tagged with a matching emotion are cloned from
that clip instead of the base one. The basic labels (happy, angry, sad,
relaxed, surprised) cover most sentences; rich labels like `excited` or
`embarrassed` also match when you provide clips for them, and anything without
a matching clip falls back to the base voice.

Notes:

- Full-reply synthesis (the default) voices each generation unit with the
  emotion that dominates it, so one grumpy aside inside a happy reply will not
  flip the whole clip.
- The TTS disk cache keys on the emotion label, not the clip contents. If you
  re-record an emotion clip, delete `cache/tts` so canned lines regenerate.
- The speaker drift guard keeps scoring against the base reference, which is
  fine because every clip is the same speaker.

## Thinking sounds (fillers)

Short pre-synthesized clips in her voice ("んー…", "そうね…") that play while
she is composing a reply, so the wait between your message and her answer
reads as thinking rather than loading.

Generate them once with your configured voice:

```
python scripts/generate_fillers.py            # writes static/fillers/
```

The web client checks for `static/fillers/manifest.json`. When it exists, a
random clip (never the same one twice in a row) plays if the reply's first
audio takes longer than about 0.6 seconds, and the real reply always cuts it
off cleanly. When it does not exist, nothing plays and nothing is fetched.

You can turn the behavior off in Settings under "Thinking sounds". The clips
are your configured voice, so re-run the script after switching TTS providers,
reference clips, or character packs. The output directory is gitignored.

## Non-verbal vocalizations (laughs, sighs)

Real reactions are sounds, not sentences. With these clips in place, the model
can put a standalone `[laugh]`, `[giggle]`, `[sigh]`, `[gasp]`, or `[hmm]`
line inside a streamed reply, and that exact spot in the audio plays a
pre-recorded vocalization in her voice instead of a synthesized sentence.

Generate the clips once with your configured voice:

```
python scripts/generate_nonverbals.py         # writes static/nonverbals/
```

Then restart the server. The backend loads `static/nonverbals/manifest.json`
at first use and only advertises the tags it actually has clips for, so the
model is never told about a sound it cannot make. Without the manifest the
feature is fully inert: tags are stripped from display, memory, and audio.

Listen to every generated wav before trusting it, because zero-shot TTS
renders laughter with mixed results. Delete a bad clip and re-run the script (it only
fills gaps unless you pass `--force`), or tweak the line texts at the top of
the script. Several variants per kind are generated so repeated laughs do not
sound identical; the picker never repeats the previous variant.

Behavior details:

- Clips splice into the ordered audio stream at their sequence slot, so a
  laugh right before a teasing line plays right before that line's audio.
- Display text, memory, and transcripts never contain the tags; a reply that
  is nothing but a `[laugh]` still commits whatever spoken lines it has.
- `web.streaming_nonverbals: false` turns the whole feature off regardless of
  clips. The POST (non-WebSocket) chat path strips the tags but does not
  splice audio; the default streaming path is where they play.
- The clips are voice-specific like the fillers: re-run the script after
  changing the TTS provider, reference clip, or character pack.

## Listening reactions (while you speak)

While your mic is capturing, in push-to-talk or hands-free mode, she behaves
like someone actually listening: she holds eye contact, nods along at natural
moments, and after a long stretch of speech may voice one soft acknowledgement
("うん。") without interrupting you.

The eye contact and nodding need no assets at all and are on by default; turn
the whole behavior off in Settings under "Listening reactions". The voiced
acknowledgements come from the same manifest as the thinking sounds, so
re-running the filler script adds them:

```
python scripts/generate_fillers.py            # now also writes backchannels
```

The vocal part is deliberately rare (sustained speech, a global cooldown, and
a coin flip), because backchannels read as human only when they are
occasional. The mic capture uses browser echo cancellation, so her "うん。"
does not leak into your recording or the transcript.
