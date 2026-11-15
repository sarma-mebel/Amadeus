# Avatar animations

`VRMViewer.tsx` drives Kurisu's idle sway, gestures, and (optionally) a
dedicated talking pose from two kinds of clip:

- **`.vrma`** files — VRM Animation format, exported for a VRM-rigged
  character directly (e.g. from VSeeFace or Blender VRM Animation exporter).
  These play back with no extra processing.
- **`.fbx`** files — Mixamo-style FBX animations, retargeted onto Kurisu's
  VRM humanoid rig at load time by `frontend/src/mixamoRetarget.ts`. Use
  these to pull in the huge Mixamo animation library without needing a
  VRM-specific re-export pipeline.

The built-in idle/action clip sets (`idle_animation01-07.vrma`,
`VRMA_01-09.vrma`) keep working with no manifest at all — everything below is
opt-in, additive sourcing for more variety.

## Where to get Mixamo clips

1. Go to [mixamo.com](https://www.mixamo.com) (free Adobe account required)
   and upload/select a humanoid character (any rig — Mixamo's own default
   character works fine; the retargeter only reads bone names and rotations,
   not the mesh).
2. Search for and pick an animation. Good starting points for a
   companion-app idle/gesture set:
   - **Breathing Idle** — a good default idle base
   - **Standing Idle Variations** (there are several numbered variants) —
     idle-rotation variety
   - **Thinking** — subtle gesture, reads well as an unattended "beat"
   - **Talking** — pairs with the manifest's `"role": "talk"` field (see
     below)
   - **Head Nod** — subtle gesture
3. On the download panel, choose:
   - Format: **FBX Binary (.fbx)**
   - Skin: **Without Skin** (Kurisu's own VRM mesh/skeleton is used; the
     Mixamo mesh isn't needed and download-without-skin is much smaller)
   - Frames per Second: **30**
   - Keyframe Reduction: **none** (uniform) is fine
4. Download the `.fbx` file.

## Where to drop them

Put downloaded files under:

```
static/animations/mixamo/<your-file>.fbx
```

(`static/animations/idle/` and `static/animations/action/` remain reserved
for the built-in `.vrma` sets; keep Mixamo `.fbx` files in their own
subdirectory so the two sourcing pipelines don't collide on naming.)

## Local clips: the gitignored overlay

`static/animations/mixamo/`, `static/animations/local/`, and
`static/animations/manifest.local.json` are all gitignored. This is the
intended home for clips that must not be committed — Mixamo's terms allow
using exported animations in your own running app but not redistributing
the exported data, and this repo is public.

`manifest.local.json` sits next to `manifest.json`, has the same shape, and
is merged over it at load time: an entry for a **new** file is appended; an
entry naming the **same** `directory` + `file` as a committed entry replaces
it (so you can re-role or `"disabled": true` a built-in clip without
touching the committed manifest). Missing files are warn-and-skipped, so a
checkout without your local clips still works.

Example — a Mixamo talking loop plus a demoted built-in:

```jsonc
{
  "action": [
    { "file": "Talking.fbx", "directory": "mixamo", "name": "Talking", "role": "talk" },
    { "file": "VRMA_03.vrma", "directory": "action", "name": "Peace Sign", "role": "gesture", "disabled": true }
  ]
}
```

## Vetting clips: the anim lab

Open the app with `?view=animlab` (e.g. `http://127.0.0.1:8000/?view=animlab`)
to get the full scene next to a clip-inventory panel: play any loaded clip
once or looping, see which clips are in the looping-idle pool, and copy a
ready-made `manifest.local.json` entry for a clip. Checklist for judging a
clip before giving it an automatic role (`idle`, `talk`, `subtle`, or an
emotion-gesture slot in `animationScheduler.ts`): watch a full loop for pose
snaps at the seam, limb distortion/clipping, foot slide, and how the
crossfade back to the idle rotation reads.

The `⧉ additive` button previews a clip as an additive layer over the
running idle instead of a full-body crossfade (`vrm/additiveGestures.ts`:
position tracks stripped, keyframes converted to deltas against the clip's
first frame, so it enters at zero offset). This is how the automatic
emotion-gesture pool plays when the `kurisu_additive_gestures` localStorage
toggle is `true` — the toggle ships off until every pool clip has been
re-vetted with this button, since additive playback changes how a clip
reads (clips whose motion rides mostly on hips/root translation lose it;
upper-body gestures generally read better, and the idle sway/breathing
stay alive underneath).

Related, also off by default: the `kurisu_inertialization` localStorage
toggle (`vrm/inertialization.ts`) replaces the linear crossfade between
body clips with a hard cut plus a spring-decayed pose offset — the old
motion's momentum carries through the transition instead of both clips
blending at half weight ("crossfade mush"). It changes how every
transition reads, so vet it the same way: set the toggle, then play
one-shots and let them return to idle here and judge the entries/exits.

## Manifest format

By default the app only loads the built-in `.vrma` presets. To add more
clips (VRMA or FBX, mixed freely), create `static/animations/manifest.json`
— see `docs/animations-manifest.example.json` in this repo for a full
example. Shape:

```jsonc
{
  "idle": [ /* AnimationManifestEntry[] */ ],
  "action": [ /* AnimationManifestEntry[] */ ]
}
```

Each entry is either a bare filename string, or an object:

| Field       | Type                          | Notes                                                                 |
|-------------|-------------------------------|------------------------------------------------------------------------|
| `file`      | string                        | Filename, or `subdir/filename`. `.fbx` (Mixamo rig) and `.bvh` (VRM-named rig) are retargeted at load; `.vrma` plays directly. |
| `directory` | string (optional)              | Base directory under `static/animations/` if not embedded in `file`.  |
| `name`      | string (optional)              | Friendly display/lookup name; defaults to a title-cased filename.     |
| `role`      | `"idle" \| "gesture" \| "talk"` (optional) | See below.                                                |
| `tags`      | string[] (optional)            | Free-form; the anim lab shows them as badges. |
| `lazy`      | boolean (optional)             | Library clip: listed in the anim lab but only fetched/parsed on first play. Use for bulk packs; auto-role clips (`idle`/`talk`) must stay eager. |

Entries in the `idle` array loop forever (`LoopRepeat`) and are the pool the
idle-variety rotation crossfades between. Entries in the `action` array play
once (`LoopOnce`) — manual triggers (`window.vrmAnimations.play(...)`, chat
reply flourishes, poke reactions) and the rare unattended "subtle gesture"
beat all pick from this pool. See `VRMViewer.tsx`'s `SUBTLE_ACTION_NAMES` /
`POKE_ACTION_NAMES` lists for which *built-in* clip names are considered
subtle enough to fire unattended or as a poke reaction — if you add your own
gesture clips and want them eligible for those unattended/poke pools too,
give them a matching display `name` or extend those lists.

### `role: "talk"`

If a manifest entry (idle or action — either works, though `action` with
`role: "talk"` is the more accurate shape) is tagged `"role": "talk"`, the
viewer crossfades to it whenever the avatar starts speaking (TTS playback —
`window.kurisuLipSync.beginStream()`/streaming chunks) and crossfades back
to a random idle once speech ends. Only the first `talk`-tagged clip found
is used. This is entirely optional — with no `talk`-tagged entry, speaking
has no effect on which body animation is playing (lip sync/visemes still
run independently, as always).

## Performance note

`FBXLoader` (and the retargeting math) are loaded with a dynamic
`import()`, only when the manifest actually references an `.fbx` file. A
manifest with only `.vrma` entries (or no manifest at all) never pulls
FBXLoader into the bundle, so the default install doesn't pay for Mixamo
support it isn't using.

## Discovery mode

Separately from the manifest, `window.__kurisuDiscoverAnimations = true`
(set before the app mounts, e.g. via a dev console) makes the viewer probe
`static/animations/{idle,action}/idle_animationNN.vrma` /
`VRMA_NN.vrma` sequentially past the built-in presets, for VRMA-only
experimentation without hand-writing a manifest. This does not currently
probe for `.fbx` files — use the manifest for those.
