/**
 * Attention/presence reflex window-event handlers, moved verbatim out of
 * VRMViewer's scene effect. Each reacts to a global CustomEvent a sibling
 * pane (ChatInterface, the audio listening reactor, the acknowledgment
 * controller, or the karaoke timing emitter) dispatches, and mutates the
 * gaze/blink/procedural-body state the render loop reads every frame:
 *
 *   - kurisu:activity (typing) — holds attentive eye contact while firing,
 *     snaps back to camera + primes an early blink on a >8s-idle return.
 *   - kurisu:user-speaking (mic) — same idle-return snap, keyed off the
 *     FIRST event after a silence gap; holds gaze lock-on while she's being
 *     talked to.
 *   - kurisu:listening-nod / kurisu:cursor-linger / kurisu:sentence-audio-start
 *     — queue procedural-body micro-gestures (nod / curious tilt / speech
 *     beat) on their respective triggers.
 *   - kurisu:thinking-start / kurisu:thinking-end — bracket the LLM wait
 *     with the gaze director's thinking-aversion beat.
 *
 * Exposes attach()/detach() (same convention as pointerInteractions.ts) so
 * the component only needs to call both at the right points in its effect
 * and cleanup.
 *
 * NEW discrete one-shot reflexes (barge-in, focus return, ambient events)
 * belong in reflexes.ts's declarative table — this module stays the
 * high-rate continuous machinery that needs direct frame-state access.
 */

import type { GazeRuntimeState, GazeSocialState, GazeDirector } from './gazeDirector'
import type { BlinkState } from './expressionDriver'
import type { ProceduralBody } from './proceduralBody'

export interface AttentionEventsDeps {
  gazeSocial: GazeSocialState
  gazeStateRef: { current: GazeRuntimeState }
  blinkStateRef: { current: BlinkState }
  gazeDirector: Pick<GazeDirector, 'beginThinking' | 'endThinking'>
  proceduralBody: Pick<ProceduralBody, 'queueListeningNod' | 'queueCuriousTilt' | 'queueBeatGesture'>
  now?: () => number
  random?: () => number
}

export interface AttentionEventsController {
  attach: () => void
  detach: () => void
}

const IDLE_RETURN_THRESHOLD_MS = 8000

export const createAttentionEvents = (deps: AttentionEventsDeps): AttentionEventsController => {
  const now = deps.now ?? (() => performance.now())
  const random = deps.random ?? Math.random

  // Typing/presence awareness: ChatInterface dispatches 'kurisu:activity' on
  // every keystroke (and other engagement). While it's firing she holds
  // attentive eye contact (advanceGaze suppresses the random glance-away),
  // and re-engaging after >8s idle earns an immediate acknowledging
  // glance-plus-blink — she noticed you came back to the keyboard.
  const handleUserActivity = () => {
    const nowMs = now()
    const idleFor = nowMs - deps.gazeSocial.lastTypingAt
    deps.gazeSocial.lastTypingAt = nowMs
    deps.gazeSocial.typingUntil = nowMs + 2500
    if (idleFor > IDLE_RETURN_THRESHOLD_MS) {
      const gaze = deps.gazeStateRef.current
      if (gaze.phase === 'offset') {
        gaze.phase = 'onCamera'
        gaze.nextChangeAt = 2 + random() * 4
      }
      const blink = deps.blinkStateRef.current
      if (blink.phase === 'idle') {
        blink.nextBlinkAt = Math.min(blink.nextBlinkAt, blink.elapsed + 0.15)
      }
    }
  }

  // Active listening: while the user speaks into the mic, the listening
  // reactor (audio/listeningReactions.ts) dispatches these. Eye contact
  // holds via the same lockOn the emotion gaze uses; nods queue into the
  // speech micro-motion machinery (capped so a burst of events can't turn
  // her into a bobblehead).
  let lastSpeakingAt = 0
  const handleUserSpeaking = () => {
    const nowMs = now()
    // Speaking events fire repeatedly while the mic captures; only the
    // FIRST one after a silence gap is "she started talking after a lull".
    const silentFor = nowMs - lastSpeakingAt
    lastSpeakingAt = nowMs
    deps.gazeSocial.lockOnUntil = Math.max(deps.gazeSocial.lockOnUntil, nowMs + 1600)
    // lockOnUntil only suppresses NEW glance-aways; an in-progress glance
    // still finishes wandering. Mirror the typing-return snap so she pulls
    // attention straight back to the user the moment they start speaking
    // (same idle threshold as the keyboard-return acknowledging glance).
    if (silentFor > IDLE_RETURN_THRESHOLD_MS) {
      const gaze = deps.gazeStateRef.current
      if (gaze.phase === 'offset') {
        gaze.phase = 'onCamera'
        gaze.nextChangeAt = 2 + random() * 4
      }
    }
  }

  const handleListeningNod = () => {
    deps.proceduralBody.queueListeningNod()
  }

  // Cursor-linger play (pet, vrm/cursorGaze): a still cursor parked on her
  // window for a beat gets a curious head-tilt. Uses the dedicated
  // standalone-amplitude tilt, NOT the speech question beat — 2° speech
  // accents are invisible as idle reactions (2026-08-03 shakedown). Client-
  // side reflex; the gaze half (tracking the cursor) is already running, so
  // the tilt lands while she's visibly looking at it.
  const handleCursorLinger = () => {
    deps.proceduralBody.queueCuriousTilt()
  }

  // Thinking-aversion beat (lifelike pass 2026-07-30): the acknowledgment
  // controller (vrm/acknowledgment.ts) brackets every LLM wait with these
  // two events — after the acknowledging nod she visibly looks up-and-away
  // to "consider", and re-engages eye contact the moment the first token
  // proves she's answering.
  const handleThinkingStart = () => deps.gazeDirector.beginThinking()
  const handleThinkingEnd = () => deps.gazeDirector.endThinking()

  // Speech-beat gestures (lifelike pass, vrm/speechBeats): the karaoke
  // timing event carries the starting sentence's beat classification, so
  // the tilt/headshake/nod lands exactly when that sentence's audio does.
  const handleSentenceAudioStart = (event: Event) => {
    const beat = (event as CustomEvent<{ beat?: string }>).detail?.beat
    if (beat === 'question' || beat === 'negation' || beat === 'emphasis') {
      deps.proceduralBody.queueBeatGesture(beat)
    }
  }

  const attach = () => {
    window.addEventListener('kurisu:activity', handleUserActivity)
    window.addEventListener('kurisu:user-speaking', handleUserSpeaking)
    window.addEventListener('kurisu:listening-nod', handleListeningNod)
    window.addEventListener('kurisu:cursor-linger', handleCursorLinger)
    window.addEventListener('kurisu:thinking-start', handleThinkingStart)
    window.addEventListener('kurisu:thinking-end', handleThinkingEnd)
    window.addEventListener('kurisu:sentence-audio-start', handleSentenceAudioStart)
  }

  const detach = () => {
    window.removeEventListener('kurisu:activity', handleUserActivity)
    window.removeEventListener('kurisu:user-speaking', handleUserSpeaking)
    window.removeEventListener('kurisu:listening-nod', handleListeningNod)
    window.removeEventListener('kurisu:cursor-linger', handleCursorLinger)
    window.removeEventListener('kurisu:thinking-start', handleThinkingStart)
    window.removeEventListener('kurisu:thinking-end', handleThinkingEnd)
    window.removeEventListener('kurisu:sentence-audio-start', handleSentenceAudioStart)
  }

  return { attach, detach }
}
