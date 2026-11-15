/**
 * Additive gesture layering (lifelike pass 2026-07-30, research item 3):
 * play emotion-gesture one-shots as an ADDITIVE layer over the running
 * idle loop instead of crossfading the whole body to them. A crossfade
 * replaces the pose — the visible "pose teleport" on gesture entry/exit,
 * and the reason a one-shot freezes the body's idle sway for its whole
 * duration. An additive layer keeps the idle loop (and breathing/weight
 * shift) alive underneath and adds only the gesture's delta.
 *
 * Conversion MUST go through THREE.AnimationUtils.makeClipAdditive —
 * setting an action's blendMode alone leaves absolute keyframe values
 * being *summed* onto the base pose, which breaks every transform.
 * makeClipAdditive rewrites the keyframes as deltas against a reference
 * frame (frame 0 of the clip itself here — so a gesture always enters at
 * identity, killing the entry teleport by construction) and stamps
 * clip.blendMode = AdditiveAnimationBlendMode for clipAction() to pick up.
 *
 * Position tracks are stripped before conversion: gesture root/hips
 * translation layered on top of an idle that also moves the hips reads as
 * sliding, and the emotion pool's value is in the upper-body motion.
 *
 * VISUAL VETTING GATE: converted playback changes how every already-vetted
 * gesture clip looks, so the whole path ships OFF behind the
 * `kurisu_additive_gestures` localStorage toggle until the pool has been
 * re-vetted in the anim lab (?view=animlab now has a per-clip "additive"
 * preview button riding playByName's additive option).
 */
import * as THREE from 'three'

export const ADDITIVE_GESTURES_STORAGE_KEY = 'kurisu_additive_gestures'

export const loadAdditiveGesturesEnabled = (): boolean => {
  try {
    return window.localStorage.getItem(ADDITIVE_GESTURES_STORAGE_KEY) === 'true'
  } catch {
    return false
  }
}

export const saveAdditiveGesturesEnabled = (enabled: boolean): void => {
  try {
    window.localStorage.setItem(ADDITIVE_GESTURES_STORAGE_KEY, String(enabled))
  } catch {
    // Preference just doesn't persist.
  }
}

// Converted-clip cache: conversion clones tracks and runs once per clip
// per session; keyed on the source clip object so a character/manifest
// reload (fresh clip instances) can never serve stale conversions.
const additiveClipCache = new WeakMap<THREE.AnimationClip, THREE.AnimationClip>()

/**
 * Returns the additive-converted twin of `clip` (cached). The source clip
 * is never mutated — normal crossfade playback of the same clip elsewhere
 * (manual play_animation requests, the toggle turned back off) keeps
 * working from the original keyframes.
 */
export const getAdditiveGestureClip = (clip: THREE.AnimationClip): THREE.AnimationClip => {
  const cached = additiveClipCache.get(clip)
  if (cached) return cached
  const stripped = clip.tracks.filter((track) => !track.name.endsWith('.position'))
  const clone = new THREE.AnimationClip(
    `${clip.name}__additive`,
    clip.duration,
    stripped.map((track) => track.clone()),
  )
  THREE.AnimationUtils.makeClipAdditive(clone)
  additiveClipCache.set(clip, clone)
  return clone
}
