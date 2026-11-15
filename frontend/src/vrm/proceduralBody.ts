/**
 * Procedural body motion, moved verbatim out of VRMViewer's render loop:
 * the record-and-undo additive stack, breathing + noise drift, head aim
 * toward the gaze target, and the onset-driven speech/listening nod
 * machinery. One instance per scene mount; the component's animate() calls
 * `undoAdditives()` before mixer.update() and `update()` after it, in the
 * same spots the inline code ran.
 */

import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'
import type { FaceChannel } from './faceMorphs'
import {
  advanceWeightShift,
  createDriftState,
  createWeightShiftState,
  driftNoise,
  nodImpulseShape,
  NOD_DURATION_S,
} from './proceduralMotion'
import {
  computeHeadAimDelta,
  readHeadTrackPref,
  smoothingAlpha,
  worldDeltaToLocalAdditive,
} from './headAim'

/**
 * Procedural additive rotations (breathing, speech nods) are layered on
 * top of the mixer's pose, but the mixer is NOT guaranteed to re-pose
 * every bone every frame: head/neck tracks are stripped from all clips
 * (withoutBoneMotionTracks), and a clip may simply not animate chest/
 * spine. A bare read-base-then-multiply against such a bone reads back
 * its own previous output and compounds forever — a ~2° nod integrates
 * into tens of degrees over its lifetime and never returns to neutral.
 * Instead, every additive applied is recorded here and explicitly undone
 * at the top of the next frame BEFORE mixer.update(): bones the mixer
 * drives get overwritten anyway (undo is a harmless no-op), and bones it
 * doesn't drive are restored to their true base, so additives stay
 * additive either way.
 */
export interface AdditiveStack {
  undo: () => void
  apply: (bone: THREE.Object3D, additive: THREE.Quaternion) => void
}

export const createAdditiveStack = (): AdditiveStack => {
  const proceduralAdditives = new Map<THREE.Object3D, THREE.Quaternion>()
  const additiveUndoScratchQuat = new THREE.Quaternion()
  // Recycled quaternions for the recorded additives: the bone set is tiny
  // and stable (chest/upperChest/spine/head), so after the first few
  // frames the pool covers every record and the per-frame .clone() this
  // replaced allocates nothing. Callers may therefore pass a reused
  // scratch quaternion as `additive` — it's copied, never retained.
  const additiveQuatPool: THREE.Quaternion[] = []

  const undo = () => {
    proceduralAdditives.forEach((applied, bone) => {
      // bone.quaternion === base * applied, so right-multiplying by
      // applied⁻¹ restores the base pose exactly.
      bone.quaternion.multiply(additiveUndoScratchQuat.copy(applied).invert())
      additiveQuatPool.push(applied)
    })
    proceduralAdditives.clear()
  }

  const apply = (bone: THREE.Object3D, additive: THREE.Quaternion) => {
    bone.quaternion.multiply(additive)
    const applied = proceduralAdditives.get(bone)
    if (applied) {
      // Two systems touched the same bone this frame — fold both into one
      // recorded additive so the undo removes the full stack.
      applied.multiply(additive)
    } else {
      const stored = additiveQuatPool.pop() ?? new THREE.Quaternion()
      proceduralAdditives.set(bone, stored.copy(additive))
    }
  }

  return { undo, apply }
}

export interface ProceduralBody {
  /** Strip last frame's procedural additives (speech nods, breathing)
   * BEFORE the mixer runs — see the AdditiveStack doc above. */
  undoAdditives: () => void
  /** Per-frame procedural layers, in the original animate() order: speech
   * micro-motion first (it updates the smoothed speechEnvelope this frame),
   * then head aim, then breathing (which reads that envelope to amplify
   * itself while speaking — item 2). Call after mixer.update() and before
   * vrm.update(). */
  update: (deltaTime: number) => void
  /** Queue an active-listening nod (the 'kurisu:listening-nod' event, fired
   * while the USER is talking into the mic). Capped so a burst of events
   * can't turn her into a bobblehead. */
  queueListeningNod: () => void
  /** Speech-beat gesture (lifelike pass 2026-07-30, vrm/speechBeats):
   * fired from the 'kurisu:sentence-audio-start' event when the starting
   * sentence classified as question (head tilt + brow raise), negation
   * (small headshake), or emphasis (a crisper nod through the activeNod
   * machinery). Rate-limited internally (min gap + never over a beat/nod
   * already in flight) so dense replies can't turn her into a metronome. */
  queueBeatGesture: (kind: 'question' | 'negation' | 'emphasis') => void
  /** Standalone-legible inquisitive tilt for the cursor-linger reaction. */
  queueCuriousTilt: () => void
}

export const createProceduralBody = ({
  getVrm,
  getRawSpeechEnvelope,
  getBrowChannel,
  getGazeProxy,
  isGazeAverted,
  getBreathingScale,
  getDroopAmount,
}: {
  getVrm: () => VRM | null
  /** The live lip-sync analyser envelope (lipSyncStateRef.current.speechEnvelope). */
  getRawSpeechEnvelope: () => number
  /** Brow-raise face channel detected on the loaded VRM (expression or raw
   * morph — see ../vrm/faceMorphs), or null (item 2: eyebrow-raise-on-peak
   * only if this VRM exposes one). */
  getBrowChannel: () => FaceChannel | null
  /** The gaze proxy Object3D the head aims toward (null before VRM load). */
  getGazeProxy: () => THREE.Object3D | null
  /** True while an emotional gaze aversion is active — the head then mostly
   * stays with the animation (a sad look-down loses its meaning if the head
   * keeps squaring up to the camera). */
  isGazeAverted: () => boolean
  /** Doze support (pet variant, see ../vrm/petDoze): { rate, amplitude }
   * multipliers on the breathing sine's frequency/swing, sampled once per
   * frame. Defaults to {1, 1} (unchanged behavior) when omitted — every
   * existing caller (the main window, and the pet variant before this
   * feature) never passes it. A dozing avatar breathes slower and deeper
   * (rate < 1, amplitude > 1); the caller (VRMViewer) owns deciding those
   * numbers from the doze phase, this module just applies whatever it's
   * given. */
  getBreathingScale?: () => { rate: number; amplitude: number }
  /** Doze support: 0 (fully alert) -> 1 (fully drooped) head-droop amount,
   * sampled once per frame. Applied as a small extra downward-pitch
   * additive on the head bone, independent of (and layered after) the
   * normal gaze-target head aim above — a nodding-off droop is a fixed
   * "chin drops" pose, not a gaze target, so it doesn't belong in the
   * gaze-proxy math. Omitted/0 is a byte-identical no-op. */
  getDroopAmount?: () => number
}): ProceduralBody => {
  const additives = createAdditiveStack()

  let breathPhase = 0

  // Breathing/nod additive scratch: fixed axes plus quaternion/euler slots
  // reused per bone per frame (safe because the additive stack's apply()
  // never retains the passed quaternion — it copies into a pooled one).
  const breathAxisX = new THREE.Vector3(1, 0, 0)
  const breathAxisY = new THREE.Vector3(0, 1, 0)
  const breathQuatScratch = new THREE.Quaternion()
  const breathYawQuatScratch = new THREE.Quaternion()
  const nodEulerScratch = new THREE.Euler()
  const nodQuatScratch = new THREE.Quaternion()

  // easeInQuad/easeOutQuad/driftNoise/nodImpulseShape are pure functions
  // imported from ./proceduralMotion — see that module for their
  // implementation (unit-tested there). driftState is this mount's
  // instance of the smoothed-random-walk state driftNoise mutates (item 2:
  // slow noise-based torso/weight drift, NOT sinusoidal — see the module
  // doc for why).
  const driftState = createDriftState()

  // Additive breathing rotation (item 4).
  //
  // Item 2 (body presence while talking): the manifest now ships several
  // role: "talk" clips (the scheduler crossfades to one while speech
  // streams), but this layer still matters — breathing amplitude is
  // boosted while speaking (arg `speakingAmount`, 0-1 eased envelope from
  // applySpeechMicroMotion) and a slow noise-based torso drift is layered
  // in (see driftNoise()), which is what keeps a single looping talk clip
  // from reading as canned.
  const applyBreathing = (deltaTime: number, speakingAmount: number) => {
    const vrm = getVrm()
    if (!vrm?.humanoid) return
    // Doze support: {rate, amplitude} defaults to {1, 1} (byte-identical to
    // the pre-doze behavior) whenever the caller doesn't pass a scale getter
    // at all, or while awake (VRMViewer's getter returns {1,1} outside
    // drowsy/dozing).
    const { rate: rateScale, amplitude: ampScale } = getBreathingScale?.() ?? { rate: 1, amplitude: 1 }
    breathPhase += deltaTime * (2 * Math.PI * 0.25) * rateScale // ~0.25Hz fundamental, slower while dozing
    // Slight secondary harmonic keeps the motion from feeling like a
    // pure metronome sine.
    const primary = Math.sin(breathPhase)
    const secondary = Math.sin(breathPhase * 2.0 + 0.6) * 0.25
    const breathSignal = primary + secondary
    // Up to ~70% larger breathing swing at full speech envelope — reads as
    // "more animated while talking" without a fixed-rate oscillation.
    // Deeper while dozing (ampScale > 1) composes multiplicatively with the
    // speech boost, though in practice the two never overlap (she doesn't
    // speak while dozing).
    const breathAmpScale = (1 + speakingAmount * 0.7) * ampScale

    const driftAmount = driftNoise(driftState, deltaTime, speakingAmount)

    const applyTo = (boneName: 'chest' | 'upperChest' | 'spine', amplitude: number, driftScale: number) => {
      const bone = vrm.humanoid?.getNormalizedBoneNode(boneName)
      if (!bone) return
      const totalX = breathSignal * amplitude * breathAmpScale + driftAmount.pitch * driftScale
      // Scratch quaternions/axes hoisted to factory scope — this runs for
      // three bones every frame, and additives.apply copies the value
      // rather than retaining the reference, so reuse is safe.
      const additive = breathQuatScratch
        .setFromAxisAngle(breathAxisX, totalX)
        .multiply(breathYawQuatScratch.setFromAxisAngle(breathAxisY, driftAmount.yaw * driftScale))
      additives.apply(bone, additive)
    }

    applyTo('chest', 0.015, 1)
    applyTo('upperChest', 0.0075, 0.6)
    applyTo('spine', 0.0075, 0.8)
  }

  // Procedural weight shift (lifelike pass 2026-07-30): a standing body
  // settles onto one hip every 10-30s. Hip roll with a counter-rotated
  // spine/chest keeps the shoulders near level (contrapposto) so it reads
  // as weight transfer, not a lean. Written on normalized spine bones
  // through the same additive stack as breathing, so springbones (hair/
  // clothes) inherit the settle and the undo pass keeps it from
  // compounding on bones a clip doesn't animate.
  const weightShiftState = createWeightShiftState()
  const weightAxisZ = new THREE.Vector3(0, 0, 1)
  const weightQuatScratch = new THREE.Quaternion()

  const applyWeightShift = (deltaTime: number) => {
    const vrm = getVrm()
    if (!vrm?.humanoid) return
    const roll = advanceWeightShift(weightShiftState, deltaTime)
    if (Math.abs(roll) < 1e-5) return
    const applyRoll = (boneName: 'hips' | 'spine' | 'chest', amount: number) => {
      const bone = vrm.humanoid?.getNormalizedBoneNode(boneName)
      if (!bone) return
      additives.apply(bone, weightQuatScratch.setFromAxisAngle(weightAxisZ, amount))
    }
    applyRoll('hips', roll)
    // ~70% counter-rotation split across the spine chain: pelvis tilts,
    // shoulders stay close to level, silhouette gets the S-curve.
    applyRoll('spine', -roll * 0.4)
    applyRoll('chest', -roll * 0.3)
  }

  // Head aim: the head itself turns toward the gaze target (weighted),
  // so she faces you instead of only cutting her eyes sideways while the
  // idle clip leaves her head wherever it wants. Composes with the nod
  // additive below through the same procedural stack; the eyes (lookAt)
  // cover whatever angle the head doesn't take. Convention-safe forward
  // axis comes from three-vrm's own lookAt.faceFront.
  const headTrackStrength = readHeadTrackPref()
  const headAimState = {
    current: new THREE.Quaternion(), // smoothed world-space delta
    scratchTarget: new THREE.Quaternion(),
    scratchWorldQuat: new THREE.Quaternion(),
    scratchWorldPos: new THREE.Vector3(),
  }
  // Doze droop scratch (reused every frame; additives.apply copies, never
  // retains — same pooling convention as the breathing/nod scratches above).
  const droopQuatScratch = new THREE.Quaternion()

  const applyHeadAim = (deltaTime: number) => {
    if (headTrackStrength <= 0) return
    const vrm = getVrm()
    const proxy = getGazeProxy()
    if (!vrm?.humanoid || !proxy) return
    const head = vrm.humanoid.getNormalizedBoneNode('head')
    if (!head) return

    head.getWorldQuaternion(headAimState.scratchWorldQuat)
    head.getWorldPosition(headAimState.scratchWorldPos)
    const faceFront = vrm.lookAt?.faceFront ?? new THREE.Vector3(0, 0, 1)

    // During an emotional gaze aversion the head should mostly stay with
    // the animation — a sad look-down loses its meaning if the head keeps
    // squaring up to the camera.
    const averted = isGazeAverted()
    const weight = 0.5 * headTrackStrength * (averted ? 0.25 : 1)

    const target = computeHeadAimDelta(
      headAimState.scratchWorldQuat,
      headAimState.scratchWorldPos,
      proxy.position,
      faceFront,
      { maxAngleRad: 0.5, weight },
    )
    // Ease toward the target so head turns read as deliberate (~0.25s
    // settle), never snappy; saccade jitter in the proxy filters out.
    headAimState.current.slerp(target, smoothingAlpha(deltaTime, 8))
    const local = worldDeltaToLocalAdditive(
      headAimState.current,
      headAimState.scratchWorldQuat,
    )
    additives.apply(head, local)

    // Doze droop: a small extra forward/downward pitch layered on top,
    // scaled by how far into dozing she is. Kept gentle (max ~18°) — this
    // is "nodding off", not "head flopped onto the desk". Uses the same
    // additive stack (undone/reapplied every frame like everything else
    // here), so it composes cleanly with head-aim and never compounds.
    const droop = getDroopAmount?.() ?? 0
    if (droop > 0) {
      const droopAngle = THREE.MathUtils.degToRad(18) * droop
      additives.apply(head, droopQuatScratch.setFromAxisAngle(breathAxisX, droopAngle))
    }
  }

  // Item 1 rework: speech-synchronized micro-motion, now ONSET-DRIVEN
  // instead of a continuous fixed-frequency oscillator. A continuous
  // ~1.6Hz sine scaled by loudness reads as metronomic no matter how small
  // the amplitude — real head nods while talking are irregular, tied to
  // emphasis/onsets in speech, not a running clock. Approach: track a
  // slow moving average of the envelope; whenever the raw envelope rises
  // above that average by a margin (and we're not in the refractory
  // window), fire a single one-shot damped "nod impulse" — an eased
  // down-and-back motion with randomized amplitude/tilt — and then let the
  // head rest on the underlying animation until the next onset. Nothing
  // here is periodic.
  let speechEnvelope = 0
  let envelopeSlowAvg = 0
  let refractoryRemaining = 0
  // Active nod impulse, or null when at rest (resting on the underlying
  // animation exactly as the user asked).
  let activeNod: {
    elapsed: number
    duration: number
    nodAmpRad: number
    tiltAmpRad: number
  } | null = null
  // Active-listening nods requested from outside the speech path (the
  // 'kurisu:listening-nod' event, fired while the USER is talking into
  // the mic). Consumed by the same activeNod machinery below, so a
  // listening nod and a speech-onset nod can never fight over the head.
  let pendingListeningNods = 0
  // NOD_DURATION_S/nodImpulseShape (the eased damped impulse shape: rises
  // fast to the peak at 40% of the duration, then relaxes back to zero —
  // reads as a single critically-damped nod rather than a spring that
  // overshoots) are imported from ./proceduralMotion.

  // Speech-beat gestures (lifelike pass 2026-07-30): text-rule-driven
  // head beats synchronized to sentence audio starts. Question/negation
  // run as their own eased head envelope here; emphasis rides the
  // existing activeNod machinery (below) so it can never fight an
  // onset-driven nod over the head bone.
  const BEAT_MIN_GAP_MS = 2500
  let lastBeatAt = -Infinity
  let activeHeadBeat: {
    kind: 'question' | 'negation'
    elapsed: number
    duration: number
    ampRad: number
  } | null = null
  let pendingEmphasisNod = false
  // Brow-raise pulse for the question beat, read by the brow section of
  // applySpeechMicroMotion (max'd with the envelope-driven raise).
  let beatBrowLevel = 0
  const beatEulerScratch = new THREE.Euler()
  const beatQuatScratch = new THREE.Quaternion()

  /** Cursor-linger reaction (vrm/cursorGaze): the same inquisitive-tilt
   * shape as the question beat, but at a standalone-legible amplitude —
   * the 2-2.8° speech-beat tilt reads as an accent DURING speech and is
   * invisible as an idle reaction under the ambient sway (live finding,
   * 2026-08-03 shakedown: "cursor play doesn't work" — it fired, at 2°).
   * Skips the beat gap gate (the linger has its own 150s cooldown) but
   * still stamps lastBeatAt so a speech beat can't stack right on top. */
  const queueCuriousTilt = () => {
    if (activeHeadBeat) return
    lastBeatAt = performance.now()
    activeHeadBeat = {
      kind: 'question',
      elapsed: 0,
      duration: 1.6,
      ampRad: THREE.MathUtils.degToRad(7.0 + Math.random() * 2.0),
    }
  }

  const queueBeatGesture = (kind: 'question' | 'negation' | 'emphasis') => {
    const now = performance.now()
    if (now - lastBeatAt < BEAT_MIN_GAP_MS) return
    if (kind === 'emphasis') {
      if (activeNod || pendingEmphasisNod) return
      pendingEmphasisNod = true
      lastBeatAt = now
      return
    }
    if (activeHeadBeat) return
    lastBeatAt = now
    activeHeadBeat = {
      kind,
      elapsed: 0,
      duration: kind === 'question' ? 0.9 : 0.8,
      ampRad:
        kind === 'question'
          ? THREE.MathUtils.degToRad(2.0 + Math.random() * 0.8)
          : THREE.MathUtils.degToRad(1.5 + Math.random() * 0.6),
    }
  }

  const applyBeatGesture = (deltaTime: number) => {
    beatBrowLevel = 0
    if (!activeHeadBeat) return
    const vrm = getVrm()
    const head = vrm?.humanoid?.getNormalizedBoneNode('head')
    if (!vrm?.humanoid || !head) {
      activeHeadBeat = null
      return
    }
    activeHeadBeat.elapsed += deltaTime
    const t = Math.min(1, activeHeadBeat.elapsed / activeHeadBeat.duration)
    // Smooth rise-and-fall base envelope for both beat kinds.
    const envelope = Math.sin(Math.PI * t)
    if (activeHeadBeat.kind === 'question') {
      // Inquisitive tilt: head rolls to one side with a whisper of chin
      // lift, brows up — the classic "oh?" posture.
      const roll = activeHeadBeat.ampRad * envelope
      const pitch = -activeHeadBeat.ampRad * 0.25 * envelope
      additives.apply(
        head,
        beatQuatScratch.setFromEuler(beatEulerScratch.set(pitch, 0, roll)),
      )
      beatBrowLevel = 0.28 * envelope
    } else {
      // Negation: two small shakes under the smooth envelope — visible
      // "no", nowhere near a theatrical whip.
      const yaw = activeHeadBeat.ampRad * Math.sin(2 * Math.PI * 2 * t) * envelope
      additives.apply(head, beatQuatScratch.setFromEuler(beatEulerScratch.set(0, yaw, 0)))
    }
    if (t >= 1) activeHeadBeat = null
  }

  const applySpeechMicroMotion = (deltaTime: number) => {
    const vrm = getVrm()
    if (!vrm?.humanoid) return

    const rawEnvelope = getRawSpeechEnvelope()
    // Fast-ish attack, slower release so the envelope doesn't chatter on
    // every syllable but still decays cleanly to zero shortly after
    // speech ends. Used both for onset detection and to feed
    // applyBreathing's speaking-amount amplification (item 2).
    const smoothing = rawEnvelope > speechEnvelope ? 0.35 : 0.08
    speechEnvelope += (rawEnvelope - speechEnvelope) * smoothing
    if (speechEnvelope < 0.005) speechEnvelope = 0

    // Slow moving average the raw envelope rides on top of — an onset is
    // a rise *above this trailing baseline*, not just "any loud frame",
    // so a sustained loud passage doesn't keep re-triggering every frame.
    envelopeSlowAvg += (speechEnvelope - envelopeSlowAvg) * Math.min(1, deltaTime * 1.2)

    if (refractoryRemaining > 0) refractoryRemaining -= deltaTime

    // Listening nods: deeper and slower than speech-onset micro-nods (an
    // acknowledgement, not an emphasis twitch), and only when the head is
    // free. They ignore the speech envelope — she isn't the one talking.
    if (!activeNod && pendingListeningNods > 0) {
      pendingListeningNods -= 1
      activeNod = {
        elapsed: 0,
        duration: NOD_DURATION_S * (1.3 + Math.random() * 0.4),
        nodAmpRad: THREE.MathUtils.degToRad(2.4 + Math.random() * 1.2),
        tiltAmpRad: THREE.MathUtils.degToRad((Math.random() * 2 - 1) * 0.8),
      }
      refractoryRemaining = 0.5 + Math.random() * 0.3
    }

    // Emphasis beat (queueBeatGesture): a crisper, slightly deeper nod
    // than an envelope-onset one, consumed through the same single
    // activeNod slot so beats and onsets can never stack on the head.
    if (!activeNod && pendingEmphasisNod) {
      pendingEmphasisNod = false
      activeNod = {
        elapsed: 0,
        duration: NOD_DURATION_S * (0.9 + Math.random() * 0.2),
        nodAmpRad: THREE.MathUtils.degToRad(1.9 + Math.random() * 0.6),
        tiltAmpRad: THREE.MathUtils.degToRad((Math.random() * 2 - 1) * 0.8),
      }
      refractoryRemaining = 0.5 + Math.random() * 0.3
    }

    const onsetMargin = 0.16
    if (
      !activeNod &&
      refractoryRemaining <= 0 &&
      speechEnvelope > 0.05 &&
      speechEnvelope > envelopeSlowAvg + onsetMargin
    ) {
      // Fire one nod impulse. Amplitude randomized +/-40% per the spec,
      // plus a small random tilt component so no two nods are identical.
      const amplitudeJitter = 0.6 + Math.random() * 0.8 // 0.6x - 1.4x (+/-40%)
      const baseNodDeg = 1.5 + Math.random() * 0.5 // 1.5-2deg, capped ~2deg by spec
      activeNod = {
        elapsed: 0,
        duration: NOD_DURATION_S * (0.85 + Math.random() * 0.3),
        nodAmpRad: THREE.MathUtils.degToRad(baseNodDeg) * amplitudeJitter,
        tiltAmpRad: THREE.MathUtils.degToRad((Math.random() * 2 - 1) * 1.2),
      }
      // 400-700ms refractory period before the next onset can fire.
      refractoryRemaining = 0.4 + Math.random() * 0.3
    }

    let nodRad = 0
    let tiltRad = 0
    if (activeNod) {
      activeNod.elapsed += deltaTime
      const t = Math.min(1, activeNod.elapsed / activeNod.duration)
      const shape = nodImpulseShape(t)
      nodRad = activeNod.nodAmpRad * shape
      tiltRad = activeNod.tiltAmpRad * shape
      if (t >= 1) activeNod = null
    }

    // Head: recorded additive on top of the base pose (see
    // AdditiveStack — head tracks are stripped from every clip, so
    // the undo pass is what returns the head to base between frames; the
    // mixer never will). Only touched while a nod impulse is active or
    // mid-decay; at rest the head genuinely rests on the underlying
    // animation between onsets.
    const headBone = vrm.humanoid.getNormalizedBoneNode('head')
    if (headBone && (nodRad !== 0 || tiltRad !== 0)) {
      // Hoisted scratch (nod runs every frame while nodding); safe to
      // reuse — additives.apply copies, never retains.
      const additive = nodQuatScratch.setFromEuler(nodEulerScratch.set(nodRad, 0, tiltRad))
      additives.apply(headBone, additive)
    }

    // Slight eyebrow raise on amplitude, eased toward the envelope rather
    // than snapped — kept envelope-driven (no sine terms). Goes through
    // the FaceChannel abstraction so morph-only rigs (VRoid's unmapped
    // Fcl_BRW_* morphs — Kurisu included) get the raise too, not just
    // models with a brow-named VRM expression.
    const browChannel = getBrowChannel()
    if (browChannel) {
      // beatBrowLevel: the question-beat brow pulse (applyBeatGesture) —
      // max'd with the envelope raise, eased through the same smoothing.
      const browTarget = Math.min(0.35, Math.max(speechEnvelope * 0.4, beatBrowLevel))
      const current = browChannel.get()
      browChannel.set(current + (browTarget - current) * 0.3)
    }
  }

  return {
    undoAdditives: additives.undo,
    update: (deltaTime: number) => {
      // Speech micro-motion (head nod onsets) runs first so it updates the
      // smoothed speechEnvelope this frame; breathing then reads that
      // envelope to amplify itself while speaking (item 2). Both run after
      // mixer.update() so they layer on top of the freshly-written pose,
      // and before vrm.update() so the normalized-bone changes reach the
      // raw skeleton (and spring bones) this same frame.
      // Beat gestures advance first so the question-beat brow pulse is
      // fresh when applySpeechMicroMotion's brow section reads it.
      applyBeatGesture(deltaTime)
      applySpeechMicroMotion(deltaTime)
      applyHeadAim(deltaTime)
      applyBreathing(deltaTime, speechEnvelope)
      applyWeightShift(deltaTime)
    },
    queueListeningNod: () => {
      pendingListeningNods = Math.min(pendingListeningNods + 1, 2)
    },
    queueBeatGesture,
    queueCuriousTilt,
  }
}
