/**
 * Canvas pointer interactions, moved verbatim out of VRMViewer's scene
 * effect: the poke raycast (click-not-drag on her model), the double-click
 * reframe escape hatch, and the B4 pointer-gaze tracking that feeds
 * gazeDirector's 'pointer' phase.
 *
 * Head-pat gesture: a pointerdown that lands on the 'head' region, is held
 * >= HEAD_PAT_HOLD_MS, and moves less than HEAD_PAT_MOVE_THRESHOLD_PX before
 * release is a PAT — reported as region 'head_pat' (a distinct region string
 * from the existing quick-click 'head', so ChatInterface/the backend can
 * give it a different, arc-aware line) instead of the normal poke. A quick
 * click on the head (released before the hold threshold, same as any other
 * region) behaves exactly as it always has. If the hold turns into an
 * OrbitControls drag (pointer moves past the poke drag threshold before the
 * hold timer fires), the pat is cancelled outright — it must never fire
 * alongside, or instead of, a camera orbit the user was actually going for.
 */

import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'
import { kurisuBridge } from './kurisuBridge'
import { isNameInList, POKE_ACTION_NAMES, POKE_REGION_CLIPS, resolvePokedRegion } from './pokeRegions'
import type { GazeRuntimeState } from './gazeDirector'
import type { CameraFollowState } from './cameraDirector'

export interface PointerInteractions {
  attach: () => void
  detach: () => void
}

export const createPointerInteractions = ({
  canvas,
  camera,
  getVrm,
  getMixer,
  gazeStateRef,
  followState,
  lastInteractionAtRef,
  onReframeRequest,
}: {
  canvas: HTMLCanvasElement
  camera: THREE.PerspectiveCamera
  getVrm: () => VRM | null
  getMixer: () => THREE.AnimationMixer | null
  gazeStateRef: { current: GazeRuntimeState }
  followState: CameraFollowState
  lastInteractionAtRef: { current: number }
  /** Double-click-to-reframe — VRMViewer wires this to the camera
   * director's startReframeTween. */
  onReframeRequest: () => void
}): PointerInteractions => {
  // Poke: clicking directly on Kurisu's model triggers a short reaction
  // (POST /api/poke, wired up by ChatInterface via window.kurisuPoke).
  // OrbitControls listens on the same canvas for drag-to-orbit, so a
  // plain 'click' handler would also fire at the end of every orbit
  // drag; instead track pointerdown/pointerup positions ourselves and
  // only treat it as a poke if the pointer barely moved between them.
  const raycaster = new THREE.Raycaster()
  const pointerNdc = new THREE.Vector2()
  const pokeRegionScratch = new THREE.Vector3()
  const pointerDownPos = { x: 0, y: 0 }
  const POKE_DRAG_THRESHOLD_PX = 6
  const POKE_CLIENT_DEBOUNCE_MS = 1000
  let lastPokeAt = 0

  // Head-pat: a hold-and-release on the 'head' region (see module doc).
  // HEAD_PAT_MOVE_THRESHOLD_PX is intentionally tighter than
  // POKE_DRAG_THRESHOLD_PX — a pat is a still hold, not just "didn't
  // technically cross the orbit threshold" — so a hold that wanders more
  // than a normal poke's tolerance (but not quite enough to trigger
  // OrbitControls) still correctly reads as "not a pat" rather than a
  // falsely-still one.
  const HEAD_PAT_HOLD_MS = 500
  const HEAD_PAT_MOVE_THRESHOLD_PX = 8
  let headPatTimeoutId: number | null = null
  let headPatArmed = false // true once the hold timer has actually fired
  let headPatCancelled = false // set once movement/drag disqualifies this hold

  const cancelHeadPatHold = () => {
    if (headPatTimeoutId !== null) {
      window.clearTimeout(headPatTimeoutId)
      headPatTimeoutId = null
    }
    headPatArmed = false
    headPatCancelled = true
  }

  const handlePointerDown = (event: PointerEvent) => {
    pointerDownPos.x = event.clientX
    pointerDownPos.y = event.clientY
    // Any pointer interaction on the canvas counts as "user is touching
    // the camera" for the auto-return idle timer, even before OrbitControls
    // decides whether this becomes a drag (its own 'start' event fires
    // slightly later/only on drag) — this way a click-and-release still
    // resets the 20s idle clock.
    followState.lastUserInputAt = performance.now()

    // Arm the head-pat hold timer only if this pointerdown actually landed
    // on the head region — everywhere else, a hold is just a hold (poke
    // regions other than head have no "held" variant per spec).
    headPatArmed = false
    headPatCancelled = false
    const vrm = getVrm()
    if (!vrm?.scene) return
    const rect = canvas.getBoundingClientRect()
    pointerNdc.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
    pointerNdc.y = -((event.clientY - rect.top) / rect.height) * 2 + 1
    raycaster.setFromCamera(pointerNdc, camera)
    const hits = raycaster.intersectObject(vrm.scene, true)
    if (hits.length === 0) return
    const pokedRegion = resolvePokedRegion(vrm, hits[0].point, pokeRegionScratch)
    if (pokedRegion !== 'head') return

    headPatTimeoutId = window.setTimeout(() => {
      headPatTimeoutId = null
      // Only actually "arm" if nothing cancelled it in the meantime (moved
      // too far / OrbitControls took over) — see handlePointerMove below.
      if (!headPatCancelled) {
        headPatArmed = true
      }
    }, HEAD_PAT_HOLD_MS)
  }

  const handlePointerUp = (event: PointerEvent) => {
    const dx = event.clientX - pointerDownPos.x
    const dy = event.clientY - pointerDownPos.y
    const moved = Math.hypot(dx, dy)

    if (headPatArmed && !headPatCancelled && moved <= HEAD_PAT_MOVE_THRESHOLD_PX) {
      // A completed pat: distinct region string ('head_pat', not 'head')
      // so the backend/ChatInterface can give it its own arc-aware line
      // instead of the ordinary quick-poke reaction — no gesture clip here
      // per spec ("locally just add a small soft expression on release —
      // nothing loud").
      cancelHeadPatHold()
      lastInteractionAtRef.current = performance.now()
      // Wake trigger (spec: "a poke click ... wakes her") — a pat counts
      // too, it's still a physical poke as far as doze is concerned.
      kurisuBridge.petDoze?.wake('head_pat')
      kurisuBridge.expressions?.set('relaxed', 0.45, 1400)
      kurisuBridge.poke?.('head_pat')
      return
    }
    cancelHeadPatHold()

    if (moved > POKE_DRAG_THRESHOLD_PX) return // was a drag/orbit, not a poke

    const vrm = getVrm()
    if (!vrm?.scene) return

    const rect = canvas.getBoundingClientRect()
    pointerNdc.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
    pointerNdc.y = -((event.clientY - rect.top) / rect.height) * 2 + 1

    raycaster.setFromCamera(pointerNdc, camera)
    const hits = raycaster.intersectObject(vrm.scene, true)
    if (hits.length === 0) return // clicked empty space / the environment, not the character

    const now = performance.now()
    if (now - lastPokeAt < POKE_CLIENT_DEBOUNCE_MS) return
    lastPokeAt = now
    lastInteractionAtRef.current = now
    // Wake trigger (spec: "a poke click ... wakes her") — see the
    // head-pat's identical call above for the same reasoning.
    kurisuBridge.petDoze?.wake('poke')

    // Optional flourish: play a reaction clip so the poke has some motion
    // behind it. Region-aware when the hitarea clips are available: the
    // hit point picks the nearest mapped bone (head/chest/hands) and its
    // vetted reaction plays; otherwise fall back to the curated generic
    // pool (POKE_ACTION_NAMES — not "any non-idle clip", so a poke can't
    // land on a non-sequitur like Shoot or Squat). Safe no-op if nothing
    // loaded/matched yet — expression/audio still fire.
    // Resolve the poked body region from the hit point regardless of
    // which clips are available — the dialogue side (/api/poke) wants it
    // even on checkouts without the hitarea clip pack. Pure nearest-bone
    // lookup imported from ./pokeRegions (unit-tested there).
    const hitPoint = hits[0].point
    const pokedRegion = resolvePokedRegion(vrm, hitPoint, pokeRegionScratch)

    if (getMixer() && kurisuBridge.vrmAnimations) {
      const listedNames = new Set(
        kurisuBridge.vrmAnimations.describe().map((clipInfo) => clipInfo.name)
      )
      const regionClip = POKE_REGION_CLIPS.find(
        (region) => region.region === pokedRegion && listedNames.has(region.clip)
      )?.clip
      if (regionClip) {
        kurisuBridge.vrmAnimations.play(regionClip)
      } else {
        const availableClipNames = Object.keys(kurisuBridge.vrmAnimations.clips)
        const reactionClips = availableClipNames.filter((name) =>
          isNameInList(name, POKE_ACTION_NAMES)
        )
        if (reactionClips.length > 0) {
          const pick = reactionClips[Math.floor(Math.random() * reactionClips.length)]
          kurisuBridge.vrmAnimations.play(pick)
        }
      }
    }

    kurisuBridge.poke?.(pokedRegion)
  }

  // Double-click-to-reframe (item 5): an explicit escape hatch back to the
  // computed default shot after the user has orbited somewhere odd.
  const handleDoubleClick = () => {
    followState.lastUserInputAt = performance.now()
    onReframeRequest()
  }

  // B4 (pointer gaze): reuses the raycaster/NDC scratch objects above.
  // Unprojects the pointer to a plane at the head bone's depth (rather
  // than projecting the head to screen space) — simpler, and correct
  // regardless of where in frame her head currently sits. "Near her face"
  // uses a generous NDC radius around the head's own screen-space
  // position so it's forgiving of exact aim.
  const POINTER_FACE_NDC_RADIUS = 0.35
  const pointerGazePlane = new THREE.Plane()
  const pointerGazeHit = new THREE.Vector3()
  const pointerHeadWorldScratch = new THREE.Vector3()
  const pointerHeadProjectedScratch = new THREE.Vector3()
  const pointerPlaneNormalScratch = new THREE.Vector3()
  const handlePointerMove = (event: PointerEvent) => {
    // Head-pat cancel: if a hold is in progress (timer armed or still
    // pending) and the pointer has now wandered past the pat's tighter
    // movement threshold, this is turning into an OrbitControls drag —
    // cancel the pat outright so it can never fire alongside, or instead
    // of, the orbit the user was actually going for. Checked unconditionally
    // (before the VRM/head-bone early-returns below) since it only needs
    // the down-position, not the avatar.
    if ((headPatTimeoutId !== null || headPatArmed) && !headPatCancelled) {
      const dx = event.clientX - pointerDownPos.x
      const dy = event.clientY - pointerDownPos.y
      if (Math.hypot(dx, dy) > HEAD_PAT_MOVE_THRESHOLD_PX) {
        cancelHeadPatHold()
      }
    }

    const vrm = getVrm()
    if (!vrm?.humanoid) return
    const headBone = vrm.humanoid.getNormalizedBoneNode('head')
    if (!headBone) return

    const rect = canvas.getBoundingClientRect()
    pointerNdc.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
    pointerNdc.y = -((event.clientY - rect.top) / rect.height) * 2 + 1

    const headWorld = headBone.getWorldPosition(pointerHeadWorldScratch)

    // Gate "near her face": project the head's world position to screen
    // space (NDC) and compare against the pointer's own NDC position with
    // a generous radius.
    const headProjected = pointerHeadProjectedScratch.copy(headWorld).project(camera)
    const ndcDist = Math.hypot(pointerNdc.x - headProjected.x, pointerNdc.y - headProjected.y)
    gazeStateRef.current.pointerNear = ndcDist <= POINTER_FACE_NDC_RADIUS

    // Unproject the pointer onto a plane through the head, facing the
    // camera, to get a world-space point the gaze proxy can ease toward.
    const planeNormal = camera.getWorldDirection(pointerPlaneNormalScratch).negate()
    pointerGazePlane.setFromNormalAndCoplanarPoint(planeNormal, headWorld)
    raycaster.setFromCamera(pointerNdc, camera)
    const hit = raycaster.ray.intersectPlane(pointerGazePlane, pointerGazeHit)
    if (hit) {
      gazeStateRef.current.pointerWorld.copy(hit)
    }
    gazeStateRef.current.lastPointerMoveAt = performance.now()
  }

  return {
    attach: () => {
      canvas.addEventListener('pointerdown', handlePointerDown)
      canvas.addEventListener('pointerup', handlePointerUp)
      canvas.addEventListener('dblclick', handleDoubleClick)
      canvas.addEventListener('pointermove', handlePointerMove)
    },
    detach: () => {
      cancelHeadPatHold()
      canvas.removeEventListener('pointerdown', handlePointerDown)
      canvas.removeEventListener('pointerup', handlePointerUp)
      canvas.removeEventListener('dblclick', handleDoubleClick)
      canvas.removeEventListener('pointermove', handlePointerMove)
    },
  }
}
