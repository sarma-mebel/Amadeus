/** Body-animation state machine, extracted verbatim from VRMViewer's VRM
 * load callback: idle looping + crossfades, the idle-variety/subtle-gesture
 * timer, talk-clip handling on speaking changes, and the emotion→one-shot
 * gesture map with its per-reply throttle.
 *
 * The five refs in the deps are VRMViewer's own component refs (teardown and
 * the poke-handler's mid-gesture check read them directly) — the scheduler
 * mutates them in place, exactly as the inline code did. Expression/gaze
 * side effects of strong emotions (blush + look-away) stay in VRMViewer and
 * are injected via `onFlusteredBeat`.
 */

import * as THREE from 'three'
import type { RefObject } from 'react'
import { getAdditiveGestureClip, loadAdditiveGesturesEnabled } from './additiveGestures'

export type AnimationRole = 'idle' | 'gesture' | 'talk' | 'subtle'

// Body language while she speaks (localStorage, instant like the other
// client preferences). Default OFF = mouth-only talking: speech drives
// lip-sync, gaze and the subtle idle sway, nothing else. The looping
// role:"talk" clip repeated its few-second motion for the entire reply,
// which read robotic ("she keeps doing the same action while talking") —
// so talk-loop clips and mid-speech emotion gesture clips are the opt-in
// now, not the default. Read live at each speaking change / gesture
// attempt, so flipping the toggle applies to the very next reply.
export const TALK_BODY_STORAGE_KEY = 'kurisu_talk_body'

export const loadTalkBodyEnabled = (): boolean => {
  try {
    return window.localStorage.getItem(TALK_BODY_STORAGE_KEY) === 'true'
  } catch {
    return false
  }
}

export const saveTalkBodyEnabled = (enabled: boolean): void => {
  try {
    window.localStorage.setItem(TALK_BODY_STORAGE_KEY, String(enabled))
  } catch {
    // Preference just doesn't persist.
  }
}

export interface ClipMeta {
  additive?: boolean
  disabled?: boolean
  role?: AnimationRole
}

// Emotion reactions: expression first, with a conservative one-shot body
// gesture for strong emotions only. Big theatrical clips stay manual-only.
//
// Each emotion maps to a small POOL of vetted clips (picked no-repeat)
// rather than one fixed clip, and the old hard once-per-reply throttle is
// now a cooldown: a long multi-sentence reply can land a second body beat
// once GESTURE_COOLDOWN_MS has passed, capped at MAX_GESTURES_PER_TURN so
// she never turns into a gesticulating mime. Every clip listed here was
// visually vetted (static/animations/SOURCES.md) — don't add unvetted ones.
//
// Module-level (not scheduler-local) so findUnresolvedGestureClipNames below
// — and its tests — can check every referenced name against the loaded clip
// library without constructing a scheduler.
export const EMOTION_GESTURE_MAP: Record<
  string,
  Array<{ name: string; threshold: number }> | undefined
> = {
  // "Peace Sign" is the subtler on-brand default for happy; bigger
  // celebratory clips need a clearly-elated sentence to fire. Thresholds
  // were lowered overall from 0.75-0.95 — at the old bar they almost never
  // cleared it. ("Clapping" was cut 2026-07-20: it read as out-of-context
  // applause on merely cheerful replies. Same pass dropped "Surprised
  // Pose" from surprised.)
  //
  // 2026-07-14 vetting pass (screenshot review of every pool clip): several
  // clips were dropped from these pools as broken-looking or off-character —
  // Joy (inert then root-drifts out of frame), Amusement 3 (whip-snap blows
  // the springbone hair across her face), Action Look Around (turns her back
  // and wanders), Sadness 2 (1.5s whiplash hunch), Excitement 2 (flaily
  // fist-pump), Disappointment 2 (root motion strides through the camera),
  // Fear 2 (cowers in profile, face hidden), Relief 1 (reads as a stretch
  // routine), Curiosity 2 (face-to-floor lurch) and Curiosity 3 (off-balance
  // theatrical lean). They all remain in the manifest for explicit
  // play_animation requests — just never as automatic body language. An
  // emotion with no surviving clip gets face expression only.
  //
  // 2026-07-19 audit (full-library screenshot sweep, all 230 clips): the
  // dormant emote library had several clean conversational clips that were
  // never wired up — added below (each flipped eager in the manifest).
  // Confusion 3 (deep sideways lurch) was swapped for Confusion 2 (a plain
  // head-scratch). Neutral 3 was demoted from the subtle pool (root-drifts
  // toward the camera); plain "Neutral" replaces it via its manifest role.
  happy: [
    { name: 'Peace Sign', threshold: 0.7 },
    { name: 'Amusement 2', threshold: 0.78 },
    { name: 'Cheering', threshold: 0.85 },
  ],
  surprised: [
    { name: 'Surprise 2', threshold: 0.72 },
    { name: 'Surprise', threshold: 0.75 },
    { name: 'Startled', threshold: 0.72 },
  ],
  sad: [
    { name: 'Sad Pose', threshold: 0.68 },
    { name: 'Remorse', threshold: 0.72 },
    { name: 'Sadness', threshold: 0.8 },
  ],
  angry: [
    { name: 'Angry Pose', threshold: 0.68 },
    { name: 'Annoyance', threshold: 0.7 },
  ],
  // Rich body-language tags (RICH_EMOTION_TO_EXPRESSION on the backend): the
  // face already got the mapped basic expression; these pick a more
  // specific body beat. Lower thresholds — a rich tag is already a
  // deliberate choice by the model.
  embarrassed: [
    { name: 'Embarrassment', threshold: 0.6 },
    { name: 'Blush Pose', threshold: 0.65 },
    { name: 'Nervousness 2', threshold: 0.72 },
    { name: 'Bashful', threshold: 0.7 },
  ],
  proud: [
    { name: 'Pride', threshold: 0.6 },
    { name: 'Pride 2', threshold: 0.68 },
  ],
  grateful: [
    { name: 'Gratitude', threshold: 0.6 },
    { name: 'Quick Informal Bow', threshold: 0.7 },
  ],
  curious: [
    { name: 'Curiosity', threshold: 0.6 },
    { name: 'Realization', threshold: 0.75 },
  ],
  confused: [
    { name: 'Confusion 2', threshold: 0.6 },
    { name: 'Shrugging', threshold: 0.65 },
  ],
  annoyed: [{ name: 'Annoyance', threshold: 0.6 }],
  // Rich tags that had no gesture pool at all until the 2026-07-19 Mixamo
  // batch (the backend has emitted them all along — they just fell through
  // to face-only).
  excited: [
    { name: 'Excited', threshold: 0.6 },
    { name: 'Fist Pump', threshold: 0.7 },
  ],
  amused: [{ name: 'Laughing', threshold: 0.6 }],
  disappointed: [
    { name: 'Disappointed', threshold: 0.6 },
    { name: 'Defeated', threshold: 0.75 },
  ],
  relieved: [{ name: 'Relieved Sigh', threshold: 0.6 }],
}

/** Every clip name EMOTION_GESTURE_MAP can reach, de-duplicated. */
export const gestureMapClipNames = (): string[] =>
  Array.from(
    new Set(
      Object.values(EMOTION_GESTURE_MAP).flatMap((pool) => pool?.map((gesture) => gesture.name) ?? [])
    )
  )

/**
 * Dev-mode safety net: a rename or a `lazy: true` flip on a manifest entry
 * currently makes an emotion (or a poke region/reaction) silently
 * gestureless — resolveClipName falls through to the raw input, which
 * never matches loadedClips, and kurisuReact/the poke handler just no-op.
 * Pass every clip name referenced by name from gesture pools (the emotion
 * map, poke regions, poke reaction lists, ...) plus a resolver that mirrors
 * production alias-resolution and the set of names that actually resolve
 * to a loaded EAGER clip; anything left over gets a loud console.warn
 * instead of failing silently. Pure — no THREE/DOM — so it's usable from
 * both the scheduler and plain unit tests.
 */
export const findUnresolvedGestureClipNames = (
  referencedNames: string[],
  resolveClipName: (name: string) => string,
  loadedEagerClipNames: Set<string>
): string[] =>
  Array.from(new Set(referencedNames)).filter(
    (name) => !loadedEagerClipNames.has(resolveClipName(name))
  )

/** DEV-only console.warn wrapper around findUnresolvedGestureClipNames. */
export const warnUnresolvedGestureClipNames = (
  referencedNames: string[],
  resolveClipName: (name: string) => string,
  loadedEagerClipNames: Set<string>
): void => {
  const unresolved = findUnresolvedGestureClipNames(
    referencedNames,
    resolveClipName,
    loadedEagerClipNames
  )
  if (unresolved.length === 0) return
  console.warn(
    '⚠️ These gesture clip names never resolve to a loaded eager clip (rename, typo, or a stray lazy: true?):',
    unresolved
  )
}

export interface AnimationSchedulerDeps {
  mixer: THREE.AnimationMixer
  clips: Record<string, THREE.AnimationClip>
  clipMeta: Record<string, ClipMeta>
  clipAliases: Map<string, string>
  idleNames: string[]
  actionNames: string[]
  talkClipNames: string[]
  /** Load-on-demand for lazy library clips (clipLoader.ensureClipLoaded). */
  ensureClipLoaded?: (name: string) => Promise<boolean>
  /** Inertialized transitions (vrm/inertialization, lifelike pass): return
   * true to take over an action switch — playAnimation then hard-cuts to
   * the next action at full weight and the caller's spring-decayed offset
   * absorbs the pose difference; false (or absent) keeps the classic
   * crossfade. The hook owns the kurisu_inertialization toggle read. */
  requestInertializedTransition?: (options: {
    fadeDuration: number
    hasEvaluated: () => boolean
  }) => boolean
  /** () => lipSyncState.streaming — gates gestures while she's speaking. */
  isSpeechStreaming: () => boolean
  getLastInteractionAt: () => number
  /** Blush + tsundere look-away for strong happy/surprised — expression/gaze
   * territory, owned by the caller. */
  onFlusteredBeat: (emotion: string, intensity: number) => void
  debugLog: (...args: unknown[]) => void
  // VRMViewer's refs, shared so its teardown/mid-gesture checks are unchanged.
  currentActionRef: RefObject<THREE.AnimationAction | null>
  currentClipNameRef: RefObject<string | null>
  currentModeRef: RefObject<'idle' | 'actual'>
  scheduledActualTimeoutRef: RefObject<number | null>
  mixerFinishedListenerRef: RefObject<
    ((event: THREE.Event & { action: THREE.AnimationAction }) => void) | null
  >
}

export const createAnimationScheduler = ({
  mixer,
  clips: loadedClips,
  clipMeta,
  clipAliases,
  idleNames,
  actionNames,
  talkClipNames,
  ensureClipLoaded,
  requestInertializedTransition,
  isSpeechStreaming,
  getLastInteractionAt,
  onFlusteredBeat,
  debugLog,
  currentActionRef,
  currentClipNameRef,
  currentModeRef,
  scheduledActualTimeoutRef,
  mixerFinishedListenerRef,
}: AnimationSchedulerDeps) => {
  const pickRandom = (names: string[], exclude?: string | null) => {
    const filtered = exclude ? names.filter((name) => name !== exclude) : names
    const pool = filtered.length > 0 ? filtered : names
    return pool[Math.floor(Math.random() * pool.length)]
  }

  const cancelScheduledActual = () => {
    if (scheduledActualTimeoutRef.current !== null) {
      window.clearTimeout(scheduledActualTimeoutRef.current)
      scheduledActualTimeoutRef.current = null
    }
  }

  const playAnimation = (name: string, mode: 'idle' | 'actual', fadeDuration = 0.6) => {
    if (!mixer || !loadedClips[name]) return

    const clip = loadedClips[name]
    const nextAction = mixer.clipAction(clip)
    nextAction.reset()
    nextAction.setEffectiveTimeScale(1)
    nextAction.setEffectiveWeight(1)

    if (mode === 'idle') {
      nextAction.setLoop(THREE.LoopRepeat, Infinity)
      nextAction.clampWhenFinished = false
    } else {
      nextAction.setLoop(THREE.LoopOnce, 0)
      nextAction.clampWhenFinished = true
    }

    const currentAction = currentActionRef.current
    if (currentAction && currentAction !== nextAction) {
      // Inertialized switch (opt-in): hard-cut to the new action at full
      // weight and let the offset spring carry the old pose's momentum out.
      // `hasEvaluated` defers the pose capture until the mixer has actually
      // run the new action once — a switch fired from inside mixer.update
      // (the 'finished' handler) lands mid-evaluation.
      if (
        requestInertializedTransition?.({
          fadeDuration,
          hasEvaluated: () => nextAction.time > 0,
        })
      ) {
        currentAction.stop()
      } else {
        currentAction.crossFadeTo(nextAction, fadeDuration, false)
      }
    } else {
      nextAction.fadeIn(fadeDuration)
    }

    nextAction.play()
    currentActionRef.current = nextAction
    currentClipNameRef.current = name
    currentModeRef.current = mode
    debugLog(`▶️  Playing: ${name} (${mode})`)
  }

  // Additive gesture layer (lifelike pass 2026-07-30, vrm/additiveGestures):
  // one-shot gestures play as a delta over the RUNNING idle loop instead of
  // crossfading the whole body away from it — the idle sway, breathing and
  // weight shift stay alive underneath, and entry/exit pose-teleports
  // disappear (the converted clip starts at identity by construction).
  // Gated by the kurisu_additive_gestures toggle until the pool is
  // re-vetted; one active layer at a time, latest wins.
  let activeAdditiveAction: THREE.AnimationAction | null = null
  let additiveFadeTimeoutId: number | null = null

  const stopActiveAdditive = (fade = 0.2) => {
    if (additiveFadeTimeoutId !== null) {
      window.clearTimeout(additiveFadeTimeoutId)
      additiveFadeTimeoutId = null
    }
    if (activeAdditiveAction) {
      activeAdditiveAction.fadeOut(fade)
      activeAdditiveAction = null
    }
  }

  const playAdditiveGesture = (name: string, fadeIn = 0.25): boolean => {
    const clip = loadedClips[name]
    if (!clip || !mixer) return false
    stopActiveAdditive()
    const additiveClip = getAdditiveGestureClip(clip)
    const action = mixer.clipAction(additiveClip)
    action.reset()
    action.setLoop(THREE.LoopOnce, 0)
    action.clampWhenFinished = false
    action.setEffectiveTimeScale(1)
    action.setEffectiveWeight(1)
    action.fadeIn(fadeIn)
    action.play()
    activeAdditiveAction = action
    // Fade the layer out just before the clip ends: a gesture that doesn't
    // return to its first-frame pose would otherwise release with a snap
    // when LoopOnce stops contributing.
    const fadeOutS = Math.min(0.4, Math.max(0.15, additiveClip.duration * 0.2))
    const delayMs = Math.max(0, (additiveClip.duration - fadeOutS) * 1000)
    additiveFadeTimeoutId = window.setTimeout(() => {
      additiveFadeTimeoutId = null
      if (activeAdditiveAction === action) {
        action.fadeOut(fadeOutS)
        activeAdditiveAction = null
      }
    }, delayMs)
    debugLog(`▶️  Playing additive layer: ${name}`)
    return true
  }

  const subtleActionNames = actionNames.filter((name) => {
    const meta = clipMeta[name]
    return meta?.role === 'subtle' && !meta.disabled
  })
  let lastSubtleClipName: string | null = null
  const playRandomSubtleGesture = (fadeDuration = 1.2) => {
    if (subtleActionNames.length === 0) return false
    const idleSinceMs = performance.now() - getLastInteractionAt()
    const eligible =
      idleSinceMs > 600000
        ? subtleActionNames
        : subtleActionNames.filter((name) => !name.toLowerCase().includes('sleepy'))
    if (eligible.length === 0) return false
    const nextName = pickRandom(eligible, lastSubtleClipName)
    lastSubtleClipName = nextName
    playAnimation(nextName, 'actual', fadeDuration)
    return true
  }

  // Idle variety: true idle clips may loop; manifest role "subtle"
  // clips may fire as rare one-shot actions after the user has been
  // idle. Greeting/wave/action clips never become looping idle.
  const scheduleActual = () => {
    cancelScheduledActual()
    // Was 3-6 min, which left Kurisu near-static (breathing/blink/gaze
    // only) for minutes at a time. Tightened to ~45-90s so a vetted
    // subtle beat (look-around / relax) fires roughly once a minute while
    // idle — the clips used are one-shot and already visually vetted, so
    // this is a cadence change, not new (unvetted) looping-idle motion.
    const delayMs = 45000 + Math.random() * 45000 // 45-90s
    scheduledActualTimeoutRef.current = window.setTimeout(() => {
      const idleSinceMs = performance.now() - getLastInteractionAt()
      const playedSubtle =
        idleSinceMs > 40000 && !isSpeechStreaming() && playRandomSubtleGesture(1.2)
      if (!playedSubtle && idleNames.length > 1) {
        const nextName = pickRandom(idleNames, currentClipNameRef.current)
        // Item 5a: vary the idle-swap crossfade duration (~0.9-1.6s)
        // instead of a fixed 1.2s, so consecutive swaps don't all feel
        // identically paced.
        playAnimation(nextName, 'idle', 0.9 + Math.random() * 0.7)
      }
      scheduleActual()
    }, delayMs)
    debugLog(`🕒 Next idle variety/gesture beat in ${(delayMs / 1000).toFixed(1)}s`)
  }

  const playRandomIdle = (fadeDuration = 0.6) => {
    if (idleNames.length === 0) return
    const excludeCurrent = idleNames.includes(currentClipNameRef.current ?? '')
      ? currentClipNameRef.current
      : undefined
    const nextName = pickRandom(idleNames, excludeCurrent)
    playAnimation(nextName, 'idle', fadeDuration)
    scheduleActual()
  }

  const onFinished = (event: THREE.Event & { action: THREE.AnimationAction }) => {
    if (currentModeRef.current === 'actual' && event.action === currentActionRef.current) {
      // A one-shot (e.g. an emotion gesture) that finishes while she's
      // still speaking returns to the current talking loop, not the idle
      // pool. (Not in mouth-only mode — there the talk loop never runs,
      // and a mid-toggle flip must not revive a stale currentTalkName.)
      if (
        isSpeechStreaming() &&
        currentTalkName &&
        loadedClips[currentTalkName] &&
        loadTalkBodyEnabled()
      ) {
        playAnimation(currentTalkName, 'idle', 0.5)
        return
      }
      playRandomIdle()
    }
  }

  mixer.addEventListener('finished', onFinished)
  mixerFinishedListenerRef.current = onFinished

  // Role: "talk" clip support (docs/animations.md / manifest role
  // field) — crossfade to it while a TTS reply is streaming and back
  // to idle once it ends, for manifests that define one.
  //
  // B6 (historical): early manifests shipped no role: "talk" clip, which
  // used to leave onSpeakingChange null and mean nothing managed body
  // animation when speech started. The committed manifest now ships ~10
  // eager talk clips, so the talk-loop branch is the live path.
  // onSpeakingChange is ALWAYS set regardless: when a manifest talk clip
  // exists that path is unchanged; otherwise,
  // if speech starts while a one-shot ('actual' mode) gesture is
  // currently playing, crossfade back to a random idle so the body
  // doesn't stay frozen mid-gesture through the whole reply.
  // With several talk clips, each speaking stretch picks one at random
  // (no immediate repeat) so long conversations don't loop a single 3s
  // gesture pattern forever.
  const loadedTalkNames = () => talkClipNames.filter((name) => loadedClips[name])
  let currentTalkName: string | null = null
  const pickTalkClip = () => {
    const pool = loadedTalkNames()
    if (pool.length === 0) return null
    currentTalkName = pickRandom(pool, pool.length > 1 ? currentTalkName : undefined)
    return currentTalkName
  }

  const onSpeakingChange = (speaking: boolean) => {
    if (!loadTalkBodyEnabled()) {
      // Mouth-only talking (the default): no talk-loop clip. Keep whatever
      // idle loop is playing (continuous, subtle — never a repeating
      // gesture), pause the idle-variety/gesture beat so a random one-shot
      // can't fire mid-reply, and resume it when she stops. A one-shot
      // already playing when speech starts crossfades back to idle so she
      // doesn't hold a frozen gesture through the whole reply.
      if (speaking) {
        cancelScheduledActual()
        if (currentModeRef.current === 'actual') playRandomIdle(0.5)
      } else {
        scheduleActual()
      }
      return
    }
    if (loadedTalkNames().length > 0) {
      if (speaking) {
        const talkName = pickTalkClip()
        if (talkName) {
          cancelScheduledActual()
          playAnimation(talkName, 'idle', 0.5)
        }
      } else {
        playRandomIdle(0.6)
      }
      return
    }

    if (speaking && currentModeRef.current === 'actual') {
      cancelScheduledActual()
      playRandomIdle(0.5)
    }
  }

  const resolveClipName = (input: string): string => {
    if (loadedClips[input]) return input
    const trimmed = input.trim()
    const withoutExt = trimmed.replace(/\.vrma$/i, '')
    return (
      clipAliases.get(trimmed) ||
      clipAliases.get(trimmed.toLowerCase()) ||
      clipAliases.get(withoutExt) ||
      clipAliases.get(withoutExt.toLowerCase()) ||
      trimmed
    )
  }

  /** kurisuBridge.vrmAnimations.play — resolve aliases, loop idles, one-shot rest. */
  const playByName = (
    name: string,
    options?: { fadeDuration?: number; loop?: boolean; additive?: boolean },
  ) => {
    const resolved = resolveClipName(name)
    if (!loadedClips[resolved] && ensureClipLoaded) {
      // Lazy library clip: fetch it, then re-enter (the load is async but
      // play() callers are fire-and-forget anyway).
      void ensureClipLoaded(resolved).then((ok) => {
        if (ok) playByName(name, options)
        else console.error(`Animation "${name}" not found or failed to load`)
      })
      return
    }
    if (!loadedClips[resolved] || !mixer) {
      console.error(`Animation "${name}" not found or mixer not ready`)
      return
    }
    // Explicit additive preview (anim-lab vetting / console): plays the
    // converted layer regardless of the global toggle — an explicit
    // request IS the vetting workflow.
    if (options?.additive) {
      playAdditiveGesture(resolved, options.fadeDuration ?? 0.25)
      return
    }
    const fadeDuration = options?.fadeDuration ?? 0.6
    const loop = options?.loop ?? false
    const isIdle = idleNames.includes(resolved)
    const mode: 'idle' | 'actual' = loop || isIdle ? 'idle' : 'actual'
    if (mode === 'idle') {
      playAnimation(resolved, 'idle', fadeDuration)
      if (actionNames.length > 0) {
        scheduleActual()
      }
    } else {
      cancelScheduledActual()
      playAnimation(resolved, 'actual', fadeDuration)
    }
  }

  // Emotion reactions: expression first, with a conservative one-shot
  // body gesture for strong emotions only. Big theatrical clips stay
  // manual-only. Pool/cooldown/throttle behavior is documented on
  // EMOTION_GESTURE_MAP itself (module scope, above).
  const GESTURE_COOLDOWN_MS = 14000
  const MAX_GESTURES_PER_TURN = 2
  let gesturesThisTurn = 0
  let lastGestureAt = 0
  let lastGestureName: string | null = null
  const kurisuReact = (emotion: string, intensity = 0.8) => {
    const normalized = emotion.trim().toLowerCase()
    if (normalized === 'neutral' || normalized === 'relaxed') return

    // Embarrassed-adjacent beat (blush + tsundere look-away) is expression/
    // gaze territory — delegated to the caller.
    if ((normalized === 'happy' || normalized === 'surprised') && intensity > 0.85) {
      onFlusteredBeat(normalized, intensity)
    }
    // An explicit "embarrassed" tag IS the flustered beat — lower bar.
    if (normalized === 'embarrassed' && intensity > 0.55) {
      onFlusteredBeat(normalized, intensity)
    }

    // Mouth-only talking: no body gesture clips while she's speaking. The
    // flustered beat above still runs — blush/gaze is face territory, not
    // a body clip. Emotions that arrive outside a speech stretch (the POST
    // path delivers them at done) still gesture normally.
    if (isSpeechStreaming() && !loadTalkBodyEnabled()) return

    if (gesturesThisTurn >= MAX_GESTURES_PER_TURN) return
    if (gesturesThisTurn > 0 && performance.now() - lastGestureAt < GESTURE_COOLDOWN_MS) return
    const pool = EMOTION_GESTURE_MAP[normalized]
    if (!pool) return
    const eligible = pool.filter(
      (gesture) => intensity >= gesture.threshold && loadedClips[resolveClipName(gesture.name)]
    )
    if (eligible.length === 0) return
    // Prefer not repeating the previous gesture, but a single-clip pool may
    // still repeat (better than that emotion going permanently gestureless).
    const varied = eligible.filter((gesture) => gesture.name !== lastGestureName)
    const candidates = varied.length > 0 ? varied : eligible
    const gesture = candidates[Math.floor(Math.random() * candidates.length)]
    const resolved = resolveClipName(gesture.name)
    gesturesThisTurn += 1
    lastGestureAt = performance.now()
    lastGestureName = gesture.name
    // Additive layer (opt-in until re-vetted): the idle loop keeps running
    // underneath, so the idle-variety timer stays armed too. Falls back to
    // the classic full-body crossfade when off or the layer can't start.
    if (loadAdditiveGesturesEnabled() && playAdditiveGesture(resolved)) return
    cancelScheduledActual()
    playAnimation(resolved, 'actual', 0.5)
  }
  const kurisuReactReset = () => {
    gesturesThisTurn = 0
  }

  // Dev-mode safety net (item 3a, docs/animations.md): a manifest rename or
  // a stray `lazy: true` on a clip EMOTION_GESTURE_MAP references currently
  // makes that emotion silently gestureless — nothing crashes, she just
  // never gestures. Checked once per scene load against the clips that
  // actually finished eager-loading (loadedClips; lazy clips only join it
  // once first played, so a lazy-flagged gesture clip correctly warns too).
  if (import.meta.env.DEV) {
    warnUnresolvedGestureClipNames(gestureMapClipNames(), resolveClipName, new Set(Object.keys(loadedClips)))
  }

  return {
    playAnimation,
    playRandomIdle,
    cancelScheduledActual,
    scheduleActual,
    onSpeakingChange,
    resolveClipName,
    playByName,
    kurisuReact,
    kurisuReactReset,
    // Exposed for petIdleBehaviors.ts (pet-window ambient idle beat): plays
    // one clip from the SAME vetted `subtle` pool the unattended-idle timer
    // above already draws from automatically — never a new pool, never a
    // lazy/unvetted clip. `preferCalm` narrows to name-matched candidates
    // (e.g. "Relax"/"Sleepy" at night) when any exist in the pool, exactly
    // the same fallback-to-full-pool shape playRandomSubtleGesture already
    // uses for its own "sleepy after 10min idle" gate.
    playRandomSubtleGesture: (fadeDuration?: number, preferCalm?: (name: string) => boolean) => {
      if (!preferCalm) return playRandomSubtleGesture(fadeDuration)
      const calmCandidates = subtleActionNames.filter(preferCalm)
      if (calmCandidates.length === 0) return playRandomSubtleGesture(fadeDuration)
      const nextName = pickRandom(calmCandidates, lastSubtleClipName)
      lastSubtleClipName = nextName
      playAnimation(nextName, 'actual', fadeDuration ?? 1.2)
      return true
    },
    /** True if the vetted `subtle` pool has at least one clip — the pet
     * idle beat has nothing to do (and shouldn't fall back to any other
     * pool) when it's empty. */
    hasSubtlePool: () => subtleActionNames.length > 0,
  }
}
