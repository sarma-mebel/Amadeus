# Animation asset provenance

`manifest.json` in this directory is runtime-required (fetched by
`frontend/src/components/VRMViewer.tsx`) and, along with the `.vrma` files it
references, should be committed. This file documents where everything came
from so licensing stays traceable.

## Built-in (pre-existing, unchanged)

- `idle/idle_animation01-07.vrma` — official VRoid Project sample idle pack.
- `action/VRMA_01-09.vrma` — official VRoid Project "VRMA animation set"
  sample pack (Show Full Body, Greeting, Peace Sign, Shoot, Spin, Model Pose,
  Squat, Look Around, Action 09), originally distributed free via BOOTH
  (https://vroid.booth.pm/items/5512385, pixiv Inc. terms: free, commercial
  use permitted with credit, no redistribution restrictions beyond that).
  Note: as of this writing that BOOTH listing shows as unpublished/private,
  but the files already vendored into this repo predate that and are kept.

## New: `sourced/` (added this pass)

Source: https://github.com/tk256ailab/vrm-viewer (commit
`0640427f053719b4c456d9c793c18a44e60ea87b`, 2025-11-21)
License: MIT (repository root `LICENSE`, Copyright (c) 2025 TK256) — covers
the whole repository tree, including the `VRMA/` directory; no separate or
conflicting license is attached to the animation files, and the README does
not attribute them to Mixamo or any other third party. Verified as
native VRM Animation (`VRMC_vrm_animation` glTF extension) binaries, not
Mixamo FBX conversions.

Initially 7 of the 11 available clips in that repo were copied into
`sourced/`; the remaining 4 (Angry/Sad/Surprised/Jump, originally skipped as
redundant with the blend-shape expression system) were added in a later pass
(July 2026, same pinned commit) once the emotion-gesture map in
`VRMViewer.tsx` grew real body-gesture slots for angry/sad/surprised. All
were visually vetted (Playwright, early/mid/late/after-return-to-idle
frames) before wiring.

| File | Status | Notes |
|---|---|---|
| `sourced/Thinking.vrma` | wired — gesture (`action`, role: `gesture`, name "Thinking Pose") | one-shot only; loaded but not currently triggered by any automatic timer, reachable via `window.vrmAnimations.play("Thinking Pose")` |
| `sourced/Goodbye.vrma` | wired — gesture (`action`, role: `gesture`, name "Goodbye Wave") | same as above: one-shot, manual-trigger only for now |
| `sourced/LookAround.vrma` | wired — idle-variety pool (`action`, role: `subtle`, name "Look Around") + emotion gesture ("surprised") | visually vetted (Playwright screenshots, start/mid/after-return-to-idle): clean head-turn motion, no breakage. One-shot only (`mode: 'actual'`), never a looping idle. Fires from the idle-variety scheduler (user idle >2min, no TTS playing, 3-6min cadence, no-repeat) and from `window.kurisuReact('surprised', …)`. |
| `sourced/Relax.vrma` | wired — idle-variety pool (`action`, role: `subtle`, name "Relax Pose") | visually vetted: no anatomical breakage (no T-pose/limb distortion), but the actual motion is a bigger arms-out stretch than the name suggests — still natural-looking, just more energetic than a typical "idle variety" beat. Kept wired since nothing is broken; revisit if it reads as too big in practice. |
| `sourced/Sleepy.vrma` | wired — idle-variety pool (`action`, role: `subtle`, name "Sleepy") | visually vetted: eyes-closed/rub-eye pose, clean and natural. Gated further than the other subtle clips per item 4: only eligible after >10 minutes of continuous user idle (matched by "sleepy" in the clip name in `VRMViewer.tsx`, not a separate manifest field). |
| `sourced/Clapping.vrma` | wired — emotion gesture (`action`, role: `gesture`, name "Clapping", `additive: true`) | visually vetted both as a normal one-shot (`window.vrmAnimations.play`) and as an additive layer over the running idle (`window.kurisuReact('happy', …)`) — both read clean, no double-transform artifacts. Fires on `emotion: "happy"` with intensity ≥0.55, at most once per reply. |
| `sourced/Angry.vrma` | wired — emotion gesture (`action`, role: `gesture`, name "Angry Pose") | visually vetted: hands-on-hips annoyed pose with a lean toward camera, clean, very in-character. Fires from `window.kurisuReact('angry', …)` at intensity ≥0.75 — the first body gesture the angry emotion has had. |
| `sourced/Sad.vrma` | wired — emotion gesture (`action`, role: `gesture`, name "Sad Pose") | visually vetted: head-down dejected pose, arms behind back, clean. Replaces the old Thinking Pose stand-in for `emotion: "sad"` (≥0.75). Thinking Pose stays wired for pokes/manual. |
| `sourced/Surprised.vrma` | wired — emotion gesture (`action`, role: `gesture`, name "Surprised Pose") | visually vetted: startled arms-out pose, clean. Replaces Action Look Around for `emotion: "surprised"` (≥0.78); Action Look Around remains available for manual/poke use. |
| `sourced/Jump.vrma` | wired — manual-only gesture (`action`, role: `gesture`, tags `manual`, name "Jump") | visually vetted: energetic full jump, spring-bone hair motion reads fine, clean return to idle. Deliberately NOT in the emotion map or idle pools (too big for an unattended beat, per the original skip rationale); reachable via `window.vrmAnimations.play("Jump")`. |
| `sourced/Blush.vrma` | still unwired | Item 1 required checking the VRM's `expressionManager.expressionMap` for a blush-like blend shape before wiring this clip or any blush reaction beat, and skipping it entirely if none exists. The bundled VRM's expression map was checked at runtime (`VRMViewer.tsx`, the `blushExpressionName` detection logged under `kurisu_debug`) and only wires an *expression* nudge (not this clip) when a `blush`/`embarrass`/`shy`-named expression is present. This specific `Blush.vrma` *clip* remains unwired — no manifest entry — since the reaction beat is expression-only, per the task's explicit "else leave Blush unwired" fallback. |

Note on unattended playback: the old `SUBTLE_ACTION_NAMES` allowlist (kept
deliberately empty) has been replaced (see the comment now in its place in
`VRMViewer.tsx`) by a manifest-driven, properly gated mechanism: clips tagged
`role: "subtle"` above only fire through the one-shot `mode: 'actual'` path,
never as a looping idle, and only when the user has been idle >2 minutes AND
no TTS audio is currently playing, with no-repeat (never the same clip twice
in a row) and a further >10 minute gate for Sleepy. Base cadence is a random
3-6 minutes. Any *additional* unwired clips beyond the ones listed above
should still get a fresh manual review before being added to any automatic
trigger list.

## New: `sourced/idle-maid.vrma` (added 2026-07-11)

Source: https://github.com/not-elm/desktop-homunculus (commit `30d11a5ad72f`,
2026-05-04), file `mods/assets/vrma/idle-maid.vrma`.
License: the assets package (`mods/assets/package.json`, "Official built-in
assets for Desktop Homunculus — includes the default VRM character, VRMA
animations") declares `MIT AND CC-BY-4.0`, author notelm; the project's
CREDITS.md attributes only the Bevy icon and the VRoid sample VRM to third
parties, so the VRMA files are first-party. **Attribution (CC-BY-4.0):
"idle-maid.vrma by notelm (Desktop Homunculus project), CC-BY-4.0".**

Wired as `role: "idle"`, name **"Calm Idle"** — the second clip in the
looping-idle rotation next to Standing Idle. Loop-vetted on Kurisu
(Playwright, early/mid/pre-seam/post-seam): hands-clasped natural standing
pose, near-identical pose across the 2.5s loop seam, no breakage.

Rejected from the same source (checked, do not add): `idle-sitting.vrma`
(seated pose reads broken on a standing scene) and `grabbed.vrma` (a
dragged-by-the-cursor flail that translates the whole model; the camera
follow chases it).

## New: `sourced/UAL_Idle_Loop.vrma` + `sourced/UAL_Idle_Talking_Loop.vrma` (added 2026-07-11)

Source: Quaternius **Universal Animation Library** (free Standard version),
**CC0** ("Free to use in personal, educational and commercial projects.
(CC0 License)" — https://quaternius.com/packs/universalanimationlibrary.html).
Obtained via the glTF mirror https://github.com/J-Ponzo/gltf-universal-animation-library
(the 2025-06-10 itch.io distribution, glTF flavor), animations `Idle_Loop`
and `Idle_Talking_Loop` out of the 46 in `AnimationLibrary_Godot_Standard.gltf`.

Converted to VRMA with `scripts/ual_gltf_to_vrma.py`: the pack's Blender
Rigify skeleton (`DEF-*` bones) is mapped to VRM humanoid bone names via a
`VRMC_vrm_animation` extension block, keeping only the skeleton subtree and
one animation per file. Godot-exported `scale` tracks (which
three-vrm-animation's parser rejects — "Invalid path 'scale'") and non-hips
`translation` tracks are stripped per the VRMA spec.

- **"Relaxed Idle"** (`UAL_Idle_Loop.vrma`, role `idle`) — third clip in the
  looping-idle rotation. Loop-vetted on Kurisu: relaxed arms-at-sides
  stance, smooth 2.5s seam, no breakage.
- **"Talking"** (`UAL_Idle_Talking_Loop.vrma`, role `talk`) — the first
  `role: "talk"` clip this app has shipped; the scheduler crossfades to it
  while TTS is streaming and back to the idle pool afterwards. Loop-vetted:
  natural conversational hand gesturing, clean 2.9s loop.
- Rejected from the same pack: `Walk_Formal_Loop` (walks in place — reads
  wrong in a stationary scene). The remaining 43 free-pack animations are
  combat/locomotion/sitting-oriented; re-run the converter script if any
  become relevant.

## New: `ual/UAL2_*.vrma` (added 2026-07-11)

Source: Quaternius **Universal Animation Library 2** (free Standard version),
**CC0** (License.txt in the pack; also stated on
https://quaternius.com/packs/universalanimationlibrary2.html). Obtained as a
direct zip from OpenGameArt
(https://opengameart.org/content/universal-animation-library-2,
`universal_animation_library_2standard.zip`); the pack ships a single
`UAL2_Standard.glb` (UE-style skeleton — `pelvis`/`spine_01`/`clavicle_l` —
handled by the `UE_BONE_MAP` in `scripts/ual_gltf_to_vrma.py`, which
auto-detects UAL1's Rigify rig vs UAL2's UE rig). All 42 animations
(minus `A_TPose`) converted and wired as lazy manual-only library clips.
Vetted + promoted: **"Arms Crossed Idle"** (`UAL2_Idle_FoldArms_Loop`,
role `idle`) — clean arm crossing, no jacket clipping, very in-character.
Vetted, kept manual: `Idle_TalkingPhone_Loop`, `Yes` (head-motion clips
lose most of their point since head/neck tracks are stripped at load).

## Mixamo and community clips (not included)

Mixamo FBX exports and Mixamo-derived community BVH packs are not redistributed
in this repository, because Mixamo's terms do not allow sharing the raw animation
files. The loaders for both formats are included (`frontend/src/mixamoRetarget.ts`,
`frontend/src/bvhRetarget.ts`). To use your own clips, download them from Mixamo and
follow `docs/animations.md`.

## Looping-idle vetting (2026-07-11)

All six non-looping clips from the VRoid idle sample pack (`idle_animation01/
02/04/05/06/07.vrma`) were loop-vetted on Kurisu (Playwright, early/mid/
pre-seam/post-seam frames) as candidates to widen the looping-idle rotation
beyond Standing Idle. **All six were rejected as loops** — despite the pack
name they are short motion beats, not standing loops: 01 holds a near-T-pose
arm position, 02 is an overhead stretch, 04 raises a fist, 07 is a
hand-inspection gesture, and 05/06 have zero-duration (static single-pose)
tracks. Their existing `role: "gesture"` + `tags: ["manual"]` wiring is
correct; do not promote them without new evidence. Additional looping idles
must come from external sourcing — use the gitignored local overlay
(`manifest.local.json` + `static/animations/mixamo|local/`, see
docs/animations.md) since Mixamo exports can't be committed to this public
repo, and vet in the `?view=animlab` workbench.

## Skipped candidates (do not add without re-review)

- `EmoteCollector_+192_1.2.0.unitypackage` / `AvatarFadeSystem.unitypackage`
  (both in this directory) — inspected July 2026. Despite the name, the
  EmoteCollector pack contains only 14 `.anim` files (8 unique static poses:
  AFK, 3 sit, 4 sleep — the "+192" is VRChat action-menu slots); they are
  Unity humanoid *muscle-space* clips with no bone transform curves, only
  convertible via a manual Unity+UniVRM export, and carry no license text
  (BOOTH-style VRChat asset, redistribution rights unverified). The
  AvatarFadeSystem package is a VRChat fade shader effect with no character
  animation at all. Neither is usable by this app's VRMA/FBX pipeline.
- `test157t/VRM-Assets-Pack-For-Silly-Tavern` (111 VRM Animation Pack) —
  README explicitly states the animations are Mixamo FBX exports converted
  to BVH/VRM rigs. Mixamo's terms do not permit redistributing exported
  animation data outside Adobe's own pipeline, and this repo carries no
  license file addressing that. Skipped per this task's explicit instruction
  not to take Mixamo re-uploads with unclear redistribution rights.
