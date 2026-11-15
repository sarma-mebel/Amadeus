# Voice reference clips — what each pack needs

Every pack below activates with the default voice until a reference clip is
added. The qwen3 zero-shot clone needs ONE clean clip per character:

- 5–15 seconds of the character speaking Japanese
- quiet dialogue scene — no BGM, no SFX, no other voices (this matters more
  than length; a shorter clean clip beats a longer dirty one)
- an exact transcript of what is said in the clip

Save the clip as `voice_ref.wav` inside the pack folder, fill in the
`voice:` block in `character.yaml` (ref_text = the transcript), and re-run
the installer. Character-song intros and drama-CD tracks are often the
cleanest sources. `scripts/split_kurisu_audio.py` and
`scripts/transcribe_kurisu_clips.py` show the prep pattern used for the
Kurisu voice; emotion variants (`voice_ref_happy.wav` + `.txt` etc.) work
the same way as Kurisu's once the base clip exists.

| Pack | Character | Voice actress to source |
|------|-----------|------------------------|
| rin_tohsaka | Rin Tohsaka | Kana Ueda |
| rem | Rem | Inori Minase |
| frieren | Frieren | Atsumi Tanezaki |
| marin | Marin Kitagawa | Hina Suzuki |
| holo | Holo | Ami Koshimizu |
| megumin | Megumin | Rie Takahashi |

Activate any pack with:

    python -m amadeus.characters characters/packs/<id> --apply

(Your previous config is kept as config.local.yaml.bak-character; pick the
matching outfit in Settings — the VRMs are already registered.) One pack is
active at a time; re-run with another pack to switch, or restore the backup
to return to Kurisu.

All clips are for private use on this machine only, same stance as the
models and the Kurisu voice itself.
