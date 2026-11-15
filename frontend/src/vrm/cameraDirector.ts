/**
 * Camera direction, moved verbatim out of VRMViewer's scene effect: the
 * follow-camera (anchor blends hips + head so crouches/leans keep her face
 * in frame), the 20s-idle auto-return to the default cinematography, the
 * aspect-driven reframe ease, and the double-click reframe tween. Tuning
 * constants are the exported ones in cameraFraming.ts (they were duplicated
 * inline in the component before this split).
 *
 * `followState` is exposed as a plain mutable state object — the pointer
 * handlers stamp `lastUserInputAt` directly, exactly as they did when it
 * was an effect-local.
 */

import * as THREE from 'three'
import type { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { VRM } from '@pixiv/three-vrm'
import {
  AUTO_RETURN_IDLE_MS,
  AUTO_RETURN_LERP,
  DEADZONE,
  FOLLOW_HEAD_WEIGHT,
  FOLLOW_HIPS_WEIGHT,
  FOLLOW_LERP,
  orbitClampsForHeight,
  PRE_LOAD_CAMERA_TARGET,
  REFRAME_DURATION_S,
  REFRAME_TWEEN_DURATION_S,
  RESUME_DELAY_MS,
  solveFramingDistance,
  TARGET_CLAMP_RADIUS,
  type DefaultFraming,
} from './cameraFraming'
import { easeOutQuad } from './proceduralMotion'

export interface CameraFollowState {
  /** Last blended chest position the camera settled on. */
  anchor: THREE.Vector3
  userDriving: boolean
  resumeAt: number
  /** Wall-clock of the last user camera input (orbit start OR pointerdown
   * on the canvas) — drives the 20s-idle auto-return-to-default-offset. */
  lastUserInputAt: number
}

export interface CameraDirector {
  followState: CameraFollowState
  /** Adopt the framing computed from the model's own bones (item 1): snap
   * camera/target/rig to it and derive the orbit clamps from the model
   * height. Called from the VRM load callback. */
  adoptDefaultFraming: (framing: DefaultFraming) => void
  /** Aspect-aware reframe (item 2): eases the rig's distance toward the
   * value required by the current aspect instead of snapping, so a window
   * resize doesn't visibly jump-cut the shot. */
  beginReframeToAspect: () => void
  /** Double-click-to-reframe (item 5) / initial snap-to-computed-framing:
   * smoothly tweens position+target back to the exact default shot. */
  startReframeTween: () => void
  /** Per-frame follow + auto-return step — the original animate() ran this
   * inside the `if (vrmRef.current)` branch. */
  updateFollow: (now: number) => void
  /** Per-frame reframe tween / aspect-ease step — runs after the lighting
   * lerp, before controls.update(). */
  updateReframe: (deltaTime: number) => void
  /** True while either camera tween (double-click reframe or aspect
   * reframe) is easing — an in-flight camera move disqualifies the render
   * loop's idle throttle. */
  isReframing: () => boolean
}

export const createCameraDirector = ({
  camera,
  controls,
  getCanvas,
  getVrm,
}: {
  camera: THREE.PerspectiveCamera
  controls: OrbitControls
  getCanvas: () => HTMLCanvasElement | null
  getVrm: () => VRM | null
}): CameraDirector => {
  // Computed once the VRM's bones are available (see computeDefaultFraming,
  // imported from ./cameraFraming, invoked from the VRM load callback).
  // Holds the "home" shot that orbit clamps, aspect reframing,
  // follow-offset auto-return, and double-click-to-reframe all reference.
  let defaultFraming: DefaultFraming | null = null

  // Orbit clamps derived from the model's own vertical extent (item 3):
  // prevents zooming to infinity, clipping through the model, tipping
  // under the floor, or looking straight down from directly overhead.
  const applyOrbitClampsForHeight = (modelHeight: number) => {
    const clamps = orbitClampsForHeight(modelHeight)
    controls.minDistance = clamps.minDistance
    controls.maxDistance = clamps.maxDistance
    controls.minPolarAngle = clamps.minPolarAngle
    controls.maxPolarAngle = clamps.maxPolarAngle
  }

  // Featured/idle animations can walk Kurisu around the set; without this,
  // the carefully framed default shot only holds until the first
  // animation plays. Track a hips/head blend in world space and nudge the
  // camera rig to follow, but back off completely while the user is
  // orbiting.
  const followState: CameraFollowState = {
    anchor: PRE_LOAD_CAMERA_TARGET.clone(),
    userDriving: false,
    resumeAt: 0,
    lastUserInputAt: performance.now(),
  }
  const cameraRig = { offset: camera.position.clone().sub(controls.target) }

  const reframeState = {
    active: false,
    elapsed: 0,
    fromDistance: cameraRig.offset.length(),
    toDistance: cameraRig.offset.length(),
  }

  const beginReframeToAspect = () => {
    const canvas = getCanvas()
    if (!defaultFraming || !canvas) return
    const aspect = canvas.clientWidth / canvas.clientHeight
    const requiredDistance = solveFramingDistance(
      defaultFraming.target,
      defaultFraming.direction,
      defaultFraming.mustFramePoints,
      aspect,
      camera.fov
    )
    reframeState.fromDistance = cameraRig.offset.length()
    reframeState.toDistance = requiredDistance
    reframeState.elapsed = 0
    reframeState.active = true
  }

  const reframeTween = {
    active: false,
    elapsed: 0,
    fromPosition: new THREE.Vector3(),
    fromTarget: new THREE.Vector3(),
  }

  const startReframeTween = () => {
    if (!defaultFraming) return
    reframeTween.fromPosition.copy(camera.position)
    reframeTween.fromTarget.copy(controls.target)
    reframeTween.elapsed = 0
    reframeTween.active = true
  }

  controls.addEventListener('start', () => {
    followState.userDriving = true
    followState.lastUserInputAt = performance.now()
    reframeTween.active = false
  })
  controls.addEventListener('end', () => {
    followState.userDriving = false
    followState.resumeAt = performance.now() + RESUME_DELAY_MS
    // Re-anchor to wherever the user left the camera so the follow logic
    // resumes from there instead of snapping back to the old target.
    followState.anchor.copy(controls.target)
    cameraRig.offset.copy(camera.position).sub(controls.target)
    reframeState.fromDistance = cameraRig.offset.length()
    reframeState.toDistance = reframeState.fromDistance
    reframeState.active = false
  })

  const adoptDefaultFraming = (framing: DefaultFraming) => {
    defaultFraming = framing
    controls.target.copy(framing.target)
    camera.position.copy(framing.target).add(framing.offset)
    cameraRig.offset.copy(framing.offset)
    followState.anchor.copy(framing.target)
    applyOrbitClampsForHeight(framing.modelHeight)
  }

  // Per-frame scratch (same convention as gazeDirector's gazeScratchVec):
  // camera follow, target-clamp, and reframe-tween math below all run every
  // rendered frame — allocating fresh Vector3s there churned the GC on the
  // always-on pet.
  const followHipsScratchVec = new THREE.Vector3()
  const followHeadScratchVec = new THREE.Vector3()
  const followDriftScratchVec = new THREE.Vector3()
  const reframeToPositionScratchVec = new THREE.Vector3()

  const updateFollow = (now: number) => {
    const vrm = getVrm()
    if (!vrm) return
    // Gentle camera follow: keep her in frame as idle/featured
    // animations move her around, without fighting the user's orbit.
    // Anchor is a hips/head blend (item 4) rather than hips alone, so a
    // crouch or lean (Squat, Spin, clips that translate her) keeps her
    // face in frame instead of just tracking her waist.
    const canFollow = !followState.userDriving && now >= followState.resumeAt
    const hips = vrm.humanoid?.getNormalizedBoneNode('hips')
    const head = vrm.humanoid?.getNormalizedBoneNode('head')
    if (canFollow && hips && head) {
      const hipsWorld = hips.getWorldPosition(followHipsScratchVec)
      const headWorld = head.getWorldPosition(followHeadScratchVec)
      const chestBlend = hipsWorld
        .multiplyScalar(FOLLOW_HIPS_WEIGHT)
        .addScaledVector(headWorld, FOLLOW_HEAD_WEIGHT)

      if (chestBlend.distanceTo(followState.anchor) > DEADZONE) {
        followState.anchor.lerp(chestBlend, FOLLOW_LERP)
      }

      // Clamp the anchor (and therefore controls.target) to stay within
      // TARGET_CLAMP_RADIUS of the default framing's chest point, so a
      // long walk/translate animation can't drag the pan target off
      // into the set.
      if (defaultFraming) {
        const drift = followDriftScratchVec.copy(followState.anchor).sub(defaultFraming.target)
        if (drift.length() > TARGET_CLAMP_RADIUS) {
          drift.setLength(TARGET_CLAMP_RADIUS)
          followState.anchor.copy(defaultFraming.target).add(drift)
        }
      }

      controls.target.lerp(followState.anchor, FOLLOW_LERP)
      camera.position.copy(controls.target).add(cameraRig.offset)
    }

    // Auto-return (item 4): after AUTO_RETURN_IDLE_MS of no user camera
    // input, gently ease the rig's offset (distance + direction) back
    // toward the original cinematography, so orbiting away to look at
    // something doesn't leave the shot stuck there indefinitely. Any
    // pointerdown/orbit-start resets lastUserInputAt and cancels this
    // instantly (see the controls 'start' listener and handlePointerDown).
    if (
      defaultFraming &&
      !followState.userDriving &&
      now - followState.lastUserInputAt >= AUTO_RETURN_IDLE_MS
    ) {
      cameraRig.offset.lerp(defaultFraming.offset, AUTO_RETURN_LERP)
      camera.position.copy(controls.target).add(cameraRig.offset)
    }
  }

  const updateReframe = (deltaTime: number) => {
    // Double-click reframe tween (item 5): smoothly carries camera
    // position + controls.target back to the exact computed default shot.
    // Takes priority over the aspect reframe below since it's a direct,
    // explicit user action.
    if (reframeTween.active && defaultFraming) {
      reframeTween.elapsed = Math.min(reframeTween.elapsed + deltaTime, REFRAME_TWEEN_DURATION_S)
      const t = easeOutQuad(reframeTween.elapsed / REFRAME_TWEEN_DURATION_S)
      const toPosition = reframeToPositionScratchVec.copy(defaultFraming.target).add(defaultFraming.offset)
      camera.position.lerpVectors(reframeTween.fromPosition, toPosition, t)
      controls.target.lerpVectors(reframeTween.fromTarget, defaultFraming.target, t)
      if (t >= 1) {
        reframeTween.active = false
        cameraRig.offset.copy(defaultFraming.offset)
        followState.anchor.copy(defaultFraming.target)
        reframeState.fromDistance = cameraRig.offset.length()
        reframeState.toDistance = reframeState.fromDistance
      }
    } else if (reframeState.active && !followState.userDriving) {
      // Aspect-aware reframe (item 2): eases the rig's distance toward the
      // value required by the current viewport aspect, only while the
      // user isn't actively orbiting.
      reframeState.elapsed = Math.min(reframeState.elapsed + deltaTime, REFRAME_DURATION_S)
      const t = easeOutQuad(reframeState.elapsed / REFRAME_DURATION_S)
      const nextDistance =
        reframeState.fromDistance + (reframeState.toDistance - reframeState.fromDistance) * t
      cameraRig.offset.setLength(nextDistance)
      camera.position.copy(controls.target).add(cameraRig.offset)
      if (t >= 1) {
        reframeState.active = false
      }
    }
  }

  return {
    followState,
    adoptDefaultFraming,
    beginReframeToAspect,
    startReframeTween,
    updateFollow,
    updateReframe,
    isReframing: () => reframeTween.active || reframeState.active,
  }
}
