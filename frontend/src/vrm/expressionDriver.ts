/**
 * Facial expression driver, moved verbatim out of VRMViewer's scene effect:
 * target-based expression easing (in -> hold -> out to a relaxed baseline),
 * the mood-EMA resting face, and the eased blink state machine.
 *
 * State lives in refs the COMPONENT owns (not this factory) because the
 * refs deliberately survive scene-effect re-runs — e.g. the mood baseline
 * persists across an outfit swap. The factory just closes over them.
 */

import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'
import { easeInQuad, easeOutQuad } from './proceduralMotion'

export const VRM_EXPRESSION_NAMES = ['happy', 'angry', 'sad', 'relaxed', 'surprised', 'neutral']

/** Case-insensitive expression-name resolution. VRM 0.x models expose
 * non-preset expressions as *custom* names with their authored casing —
 * every VRM0 pack in the bundle (Holo, the quintuplets, Miku, Rem, Rin…)
 * ships surprised as custom "Surprised", which a blind
 * setValue('surprised') silently misses. Resolves the canonical
 * lowercase name to whatever key the manager actually has, cached per
 * manager instance (managers are swapped wholesale on character change,
 * so a WeakMap keyed on the manager can never serve stale aliases). */
const expressionAliasCache = new WeakMap<object, Map<string, string>>()

export const resolveExpressionName = (
  manager: { expressionMap?: Record<string, unknown> },
  name: string,
): string => {
  const map = manager.expressionMap
  if (!map || name in map) return name
  let aliases = expressionAliasCache.get(manager)
  if (!aliases) {
    aliases = new Map()
    for (const key of Object.keys(map)) {
      aliases.set(key.toLowerCase(), key)
    }
    expressionAliasCache.set(manager, aliases)
  }
  return aliases.get(name.toLowerCase()) ?? name
}

/** Target-based expression easing state, driven every frame from animate().
 * Replaces the old hard-set + setTimeout-hard-reset approach so
 * window.kurisuExpressions.set(...) reads as an eased hold instead of a
 * snap-and-pop. */
export interface ExpressionEasingState {
  name: string
  fromIntensity: number
  targetIntensity: number
  phase: 'in' | 'hold' | 'out'
  elapsed: number
  inDuration: number
  holdDuration: number
  outDuration: number
  relaxedBaseline: number
}

export const createExpressionEasingState = (): ExpressionEasingState => ({
  name: 'neutral',
  fromIntensity: 0,
  targetIntensity: 0,
  phase: 'hold',
  elapsed: 0,
  inDuration: 0.2,
  holdDuration: 0,
  outDuration: 0.6,
  relaxedBaseline: 0.4,
})

/** Item 5b: mood-biased resting face. The backend's ChatResponse doesn't
 * carry a mood/KurisuState field to the frontend (checked amadeus/web/
 * app.py — only emotion/emotion_intensity per reply), so this derives a
 * cheap mood proxy client-side: an exponential moving average of recent
 * reply emotions, where happy/relaxed nudge the baseline up and sad/angry
 * nudge it down. `current` is what advanceExpression eases the relaxed
 * baseline toward each frame (see relaxedBaseline assignment there);
 * `target` is what setKurisuExpression's EMA update aims for. */
export interface MoodBaselineState {
  current: number
  target: number
}

export const createMoodBaselineState = (): MoodBaselineState => ({
  current: 0.4,
  target: 0.4,
})

/** Eased blink state machine, advanced by deltaTime in animate(). */
export interface BlinkState {
  phase: 'idle' | 'closing' | 'closed' | 'opening'
  elapsed: number
  nextBlinkAt: number
  pendingDoubleBlink: boolean
}

export const createBlinkState = (): BlinkState => ({
  phase: 'idle',
  elapsed: 0,
  nextBlinkAt: 2.5 + Math.random() * 3.5,
  pendingDoubleBlink: false,
})

export interface ExpressionDriver {
  /** Drives the VRM's discrete facial expressions (as opposed to the
   * continuous viseme blending in the lip-sync loop). Ported from the
   * legacy inline client (amadeus/web/app.py setKurisuExpression) so
   * window.kurisuExpressions.set(...) — already called by ChatInterface on
   * every reply — actually does something in the React app. Signature is
   * unchanged so ChatInterface needs no changes. */
  set: (expressionName: string, intensity?: number, duration?: number) => void
  advanceExpression: (deltaTime: number) => void
  advanceBlink: (deltaTime: number) => void
  /** Event-coupled blink (lifelike pass 2026-07-30): humans blink on most
   * large gaze shifts and at speech boundaries, not just on a timer.
   * Starts a blink immediately if the machine is idle, at least ~0.6s has
   * passed since the last one, and the surprised-suppression doesn't
   * apply. `probability` (0-1, default 1) lets callers pass the
   * VMagicMirror-style "97% for big shifts, less for small ones" odds
   * without owning the dice roll themselves. */
  requestBlink: (probability?: number) => void
  /** Doze support (pet variant, see ../vrm/petDoze): sustains the `blink`
   * VRM expression at `amount` (0 = open, 1 = fully closed) instead of the
   * quick open/close pulse advanceBlink produces on its own. Call AFTER
   * advanceBlink every frame — takes the max of the two, so a genuine blink
   * mid-doze (the eyes fluttering shut a little further, or a real blink
   * layering on top of a partial drowsy lid) still reads correctly instead
   * of the doze override stomping it back down. amount 0 is a no-op (never
   * fights a fully-open blink state), so calling this every frame with 0
   * while fully awake is always safe/inert. */
  setEyeClose: (amount: number) => void
}

export const createExpressionDriver = ({
  getVrm,
  expressionStateRef,
  moodBaselineStateRef,
  blinkStateRef,
  onEmotionCue,
  isLipSyncActive,
}: {
  getVrm: () => VRM | null
  expressionStateRef: { current: ExpressionEasingState }
  moodBaselineStateRef: { current: MoodBaselineState }
  blinkStateRef: { current: BlinkState }
  /** Gaze coupling: called with the normalized emotion + intensity on every
   * set() so the gaze director can apply its emotion-driven aversion/
   * lock-on semantics (VRMViewer wires this to applyEmotionGaze). */
  onEmotionCue: (emotion: string, intensity: number) => void
  /** Lip-sync ownership handshake: while this returns true, the driver
   * leaves the 'relaxed' expression entirely alone. The lip-sync analysis
   * tick writes a per-frame jaw-slack 'relaxed' duck while audio plays,
   * and the driver's exclusive-zeroing pass used to run later in the same
   * frame and stomp it — the duck had been dead whenever any non-relaxed
   * emotion was showing. Optional so tests (and any future headless
   * caller) don't have to care. */
  isLipSyncActive?: () => boolean
}): ExpressionDriver => {
  const setKurisuExpression = (expressionName: string, intensity = 0.8, duration = 2000) => {
    const vrm = getVrm()
    if (!vrm?.expressionManager) {
      console.warn('VRM or expressionManager not available yet')
      return
    }
    const state = expressionStateRef.current
    onEmotionCue(expressionName.trim().toLowerCase(), intensity)
    // Read back whatever's currently showing so the ease-in starts from
    // the true current value, not a snap.
    const currentValue =
      vrm.expressionManager.getValue(resolveExpressionName(vrm.expressionManager, state.name)) ?? 0
    state.name = expressionName
    state.fromIntensity = currentValue
    state.targetIntensity = intensity
    state.phase = 'in'
    state.elapsed = 0
    state.inDuration = 0.2
    state.holdDuration = Math.max(0, duration / 1000)
    state.outDuration = 0.6

    // Item 5b: fold this reply's emotion into the mood EMA proxy. Kept
    // subtle — a modest 0.25-0.55 range around the original fixed 0.4
    // baseline, nudged a small step per reply rather than jumping. Sample
    // for this reply: raise baseline toward ~0.55 for happy/relaxed, lower
    // toward ~0.25 for sad/angry, hold steady otherwise (surprised/
    // neutral don't carry valence for this proxy).
    const moodState = moodBaselineStateRef.current
    const normalized = expressionName.trim().toLowerCase()
    let moodSample: number | null = null
    if (normalized === 'happy' || normalized === 'relaxed') {
      moodSample = 0.4 + 0.15 * Math.min(1, intensity)
    } else if (normalized === 'sad' || normalized === 'angry') {
      moodSample = 0.4 - 0.15 * Math.min(1, intensity)
    }
    if (moodSample !== null) {
      const emaAlpha = 0.25
      moodState.target = THREE.MathUtils.clamp(
        moodState.target + (moodSample - moodState.target) * emaAlpha,
        0.25,
        0.55,
      )
    }
  }

  // Target-based expression easing (item 5): advances expressionStateRef
  // (populated by setKurisuExpression above) through in -> hold -> out,
  // settling on the 'relaxed' baseline. Runs every frame regardless of
  // whether setKurisuExpression has ever been called (state defaults to
  // an already-settled 'neutral'/hold, so this is a no-op until the first
  // real call).
  const advanceExpression = (deltaTime: number) => {
    const vrm = getVrm()
    if (!vrm?.expressionManager) return
    const state = expressionStateRef.current
    state.elapsed += deltaTime

    // Item 5b: slowly ease the relaxed baseline toward the mood EMA's
    // current target (very slow — full range 0.25-0.55, ~30s to fully
    // settle to a new target — this must read as a mood drift, not a
    // visible pop). While currently resting at the settled 'relaxed'
    // hold, also live-update the target/from so the drift is visible
    // instead of only taking effect after the next expression event.
    const moodState = moodBaselineStateRef.current
    state.relaxedBaseline += (moodState.target - state.relaxedBaseline) * Math.min(1, deltaTime * 0.05)
    if (state.name === 'relaxed' && state.phase === 'hold' && state.holdDuration === Number.POSITIVE_INFINITY) {
      state.fromIntensity = state.relaxedBaseline
      state.targetIntensity = state.relaxedBaseline
    }

    let value: number
    if (state.phase === 'in') {
      const t = state.inDuration > 0 ? Math.min(1, state.elapsed / state.inDuration) : 1
      value = state.fromIntensity + (state.targetIntensity - state.fromIntensity) * easeOutQuad(t)
      if (t >= 1) {
        state.phase = 'hold'
        state.elapsed = 0
      }
    } else if (state.phase === 'hold') {
      value = state.targetIntensity
      // 'neutral' used to be excluded here, which pinned the face forever
      // after any neutral set() (holdDuration never checked -> never
      // released, mood-baseline recovery skipped). It now holds and
      // releases like every other expression; the settled resting state
      // below is what's deliberately permanent (holdDuration Infinity).
      if (state.elapsed >= state.holdDuration) {
        state.phase = 'out'
        state.elapsed = 0
      }
    } else {
      const t = state.outDuration > 0 ? Math.min(1, state.elapsed / state.outDuration) : 1
      value = state.targetIntensity + (0 - state.targetIntensity) * easeOutQuad(t)
      if (t >= 1) {
        // Settled back to baseline: fold into the 'relaxed' expression so
        // the face doesn't just go slack, matching the old behavior's
        // "hard reset to relaxed 0.4".
        state.phase = 'hold'
        state.elapsed = 0
        state.holdDuration = Number.POSITIVE_INFINITY
        state.name = 'relaxed'
        state.fromIntensity = state.relaxedBaseline
        state.targetIntensity = state.relaxedBaseline
        value = state.relaxedBaseline
      }
    }

    const lipSyncOwnsRelaxed = isLipSyncActive?.() ?? false
    VRM_EXPRESSION_NAMES.forEach((name) => {
      // While speech audio is live, 'relaxed' belongs to the lip-sync
      // tick's jaw-slack duck — writing it here (whether the active value
      // or the exclusive zero) would stomp that duck every frame. The
      // driver resumes ownership the moment audio stops.
      if (name === 'relaxed' && lipSyncOwnsRelaxed) return
      const manager = vrm.expressionManager
      manager?.setValue(resolveExpressionName(manager, name), name === state.name ? value : 0)
    })
  }

  // Eased blink state machine (item 3), replacing the old hard-set +
  // setTimeout(150ms) hard-reset. Suppressed while a strong non-neutral
  // expression (e.g. mid-surprise) is showing, since a blink stomping
  // through a wide-eyed surprised pose reads as an animation glitch.
  const advanceBlink = (deltaTime: number) => {
    const vrm = getVrm()
    if (!vrm?.expressionManager) return
    const state = blinkStateRef.current
    const expr = expressionStateRef.current
    const suppressed =
      expr.name === 'surprised' && expr.targetIntensity > 0.7 && expr.phase !== 'out'

    if (state.phase === 'idle') {
      state.elapsed += deltaTime
      if (!suppressed && state.elapsed >= state.nextBlinkAt) {
        state.phase = 'closing'
        state.elapsed = 0
      }
      return
    }

    state.elapsed += deltaTime
    if (state.phase === 'closing') {
      const t = Math.min(1, state.elapsed / 0.07)
      vrm.expressionManager.setValue('blink', easeInQuad(t))
      if (t >= 1) {
        state.phase = 'closed'
        state.elapsed = 0
      }
    } else if (state.phase === 'closed') {
      if (state.elapsed >= 0.05) {
        state.phase = 'opening'
        state.elapsed = 0
      }
    } else if (state.phase === 'opening') {
      const t = Math.min(1, state.elapsed / 0.12)
      vrm.expressionManager.setValue('blink', 1 - easeOutQuad(t))
      if (t >= 1) {
        // 15% chance of an immediate second blink (a natural-feeling
        // double-blink), otherwise schedule the next single blink 2.5-6s
        // out.
        if (state.pendingDoubleBlink) {
          state.pendingDoubleBlink = false
          state.phase = 'closing'
          state.elapsed = 0
        } else {
          state.phase = 'idle'
          state.elapsed = 0
          state.nextBlinkAt = 2.5 + Math.random() * 3.5
          state.pendingDoubleBlink = Math.random() < 0.15
        }
      }
    }
  }

  const setEyeClose = (amount: number) => {
    if (amount <= 0) return
    const vrm = getVrm()
    if (!vrm?.expressionManager) return
    const current = vrm.expressionManager.getValue('blink') ?? 0
    vrm.expressionManager.setValue('blink', Math.max(current, amount))
  }

  // Event-coupled blink: rides the exact same eased state machine as the
  // ambient timer blink (advanceBlink drives the closing/opening phases
  // either way), so an event blink can never fight or double-layer one
  // already in flight — a non-idle phase simply ignores the request. The
  // idle `elapsed` counter doubles as time-since-last-blink, giving the
  // ~0.6s spacing guard for free.
  const requestBlink = (probability = 1) => {
    const state = blinkStateRef.current
    const expr = expressionStateRef.current
    const suppressed =
      expr.name === 'surprised' && expr.targetIntensity > 0.7 && expr.phase !== 'out'
    if (suppressed || state.phase !== 'idle' || state.elapsed < 0.6) return
    if (probability < 1 && Math.random() >= probability) return
    state.phase = 'closing'
    state.elapsed = 0
  }

  return { set: setKurisuExpression, advanceExpression, advanceBlink, setEyeClose, requestBlink }
}
