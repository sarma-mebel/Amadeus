/**
 * Hands-free conversation mode: the pure endpointing state machine plus the
 * localStorage preference plumbing shared by useHandsFree (main chat + pet)
 * and SettingsPanel.
 *
 * The endpointer is fed (level, timestampMs) samples from the live capture
 * stream and decides when the user has finished an utterance. It is pure and
 * clock-free — the caller supplies timestamps (useHandsFree feeds an
 * audio-derived clock accumulated from buffer durations, which makes the
 * whole pipeline deterministic under test) — and self-resets after every
 * endpoint/discard so one instance can endpoint a whole conversation.
 *
 *   idle ──(level ≥ voiceThreshold)──▶ pending ──(≥ minVoicedMs sustained)──▶ voiced
 *     ▲        │ dip below releaseThreshold                                     │
 *     └────────┘ (blip: cough/click, discarded silently)                        │
 *     ▲                                                                         │
 *     ├──(trailing silence ≥ trailingSilenceMs, enough voiced audio)── endpoint ┤
 *     ├──(trailing silence, but voiced total < minUtteranceMs)──────── discard ─┤
 *     └──(utterance running ≥ maxUtteranceMs)──────── endpoint (max-utterance) ─┘
 */

// --- Preference (shared by the hook and SettingsPanel) ----------------------

export const HANDS_FREE_STORAGE_KEY = 'kurisu_hands_free'

/** Window event dispatched whenever the preference changes, so every surface
 * (the useHandsFree instance driving capture, SettingsPanel's checkbox, the
 * composer/pet toggle buttons) stays in sync without prop-drilling across
 * component trees. detail: { enabled: boolean }. */
export const HANDS_FREE_EVENT = 'kurisu:hands-free'

/** Default OFF: an always-open microphone must be an explicit choice. */
export const loadHandsFreeEnabled = (): boolean => {
  try {
    return window.localStorage.getItem(HANDS_FREE_STORAGE_KEY) === 'true'
  } catch {
    return false
  }
}

export const saveHandsFreeEnabled = (enabled: boolean): void => {
  try {
    window.localStorage.setItem(HANDS_FREE_STORAGE_KEY, String(enabled))
  } catch {
    // Preference just won't persist; the event still updates this session.
  }
  window.dispatchEvent(new CustomEvent(HANDS_FREE_EVENT, { detail: { enabled } }))
}

// --- Endpointer thresholds --------------------------------------------------

/** RMS at/above this acquires a voice onset. Same floor the listening
 * reactor uses: AGC'd speech on the echo-cancelled capture path typically
 * sits at 0.05–0.2 RMS while room noise stays under 0.01. */
export const HANDS_FREE_VOICE_THRESHOLD = 0.015

/** Hysteresis release: once an onset/utterance is running, only a drop BELOW
 * this counts as silence. A level hovering right at the acquire threshold
 * would otherwise flap between voiced/silent and endpoint mid-word. */
export const HANDS_FREE_RELEASE_THRESHOLD = 0.01

/** Sustained voice needed before an onset is believed. Keyboard clicks and
 * short coughs rarely hold 250ms continuously above threshold; real speech
 * always does. (Matches the barge-in detector's min-voiced rationale.) */
export const HANDS_FREE_MIN_VOICED_MS = 250

/** Trailing silence that ends the utterance. Middle of the 700–900ms band:
 * long enough to survive natural inter-word pauses, short enough that the
 * reply still feels immediate. (The old auto-listen mode used 1400ms, tuned
 * for a single one-shot window; a continuously open mic can afford to be
 * snappier because a cut-off user can just keep talking for the next turn.) */
export const HANDS_FREE_TRAILING_SILENCE_MS = 800

/** Hard cap on one utterance. Steady background noise (fan, music) that
 * holds the level up must not pin the machine in 'voiced' forever — the
 * capture buffer would grow unboundedly and nothing would ever send. At the
 * cap the utterance endpoints as-is; STT gets whatever was heard. */
export const HANDS_FREE_MAX_UTTERANCE_MS = 30_000

/** Cumulative voiced audio an utterance must contain by endpoint time or it
 * is discarded, never sent. A cough or door slam can sustain >250ms (passing
 * the onset gate) but almost never totals 600ms of voiced audio; the
 * shortest real requests ("hey", "stop") ride the utterance's own trailing
 * energy past this comfortably. */
export const HANDS_FREE_MIN_UTTERANCE_MS = 600

/** Clamp on the per-sample dt derived from caller timestamps. If feeding
 * pauses (main thread hitch, caller suspends without reset) the gap must not
 * be booked as one giant slab of silence/voice on the next sample. */
export const HANDS_FREE_MAX_SAMPLE_GAP_MS = 250

// --- State machine ----------------------------------------------------------

export type HandsFreePhase = 'idle' | 'pending' | 'voiced'

export type HandsFreeFeedResult =
  /** Keep feeding. While phase is 'idle' the caller may trim its capture
   * buffer to a short pre-roll — nothing being buffered is speech yet. */
  | { kind: 'listening'; phase: HandsFreePhase }
  /** Utterance complete — slice the recording and send it to STT. */
  | { kind: 'endpoint'; reason: 'silence' | 'max-utterance'; utteranceMs: number }
  /** What looked like an utterance was too short to be speech — drop the
   * buffered audio silently and keep listening. */
  | { kind: 'discard'; reason: 'too-short' }

export interface HandsFreeEndpointerOptions {
  voiceThreshold?: number
  releaseThreshold?: number
  minVoicedMs?: number
  trailingSilenceMs?: number
  maxUtteranceMs?: number
  minUtteranceMs?: number
}

export interface HandsFreeEndpointer {
  /** Feed one RMS sample observed at timestampMs (monotonic, caller-owned).
   * Endpoint/discard results self-reset the machine to 'idle', so the caller
   * just keeps feeding. */
  feed: (level: number, timestampMs: number) => HandsFreeFeedResult
  /** Hard reset (suspend boundaries): forget the utterance AND the timestamp
   * baseline, so the first sample after a gap carries no phantom dt. */
  reset: () => void
}

export const createHandsFreeEndpointer = (
  options: HandsFreeEndpointerOptions = {},
): HandsFreeEndpointer => {
  const voiceThreshold = options.voiceThreshold ?? HANDS_FREE_VOICE_THRESHOLD
  // Release can never exceed acquire, or the hysteresis would invert.
  const releaseThreshold = Math.min(
    options.releaseThreshold ?? HANDS_FREE_RELEASE_THRESHOLD,
    voiceThreshold,
  )
  const minVoicedMs = options.minVoicedMs ?? HANDS_FREE_MIN_VOICED_MS
  const trailingSilenceMs = options.trailingSilenceMs ?? HANDS_FREE_TRAILING_SILENCE_MS
  const maxUtteranceMs = options.maxUtteranceMs ?? HANDS_FREE_MAX_UTTERANCE_MS
  const minUtteranceMs = options.minUtteranceMs ?? HANDS_FREE_MIN_UTTERANCE_MS

  let phase: HandsFreePhase = 'idle'
  let lastTs: number | null = null
  let utteranceStartTs = 0
  let pendingRunMs = 0 // continuous voiced run while confirming an onset
  let voicedTotalMs = 0 // cumulative voiced audio in the utterance
  let silenceMs = 0 // current trailing-silence run while voiced

  /** Back to 'idle' after an endpoint/discard — keeps the timestamp baseline
   * so dt stays continuous across utterances. */
  const settle = () => {
    phase = 'idle'
    pendingRunMs = 0
    voicedTotalMs = 0
    silenceMs = 0
  }

  const reset = () => {
    settle()
    lastTs = null
  }

  const feed = (level: number, timestampMs: number): HandsFreeFeedResult => {
    const dt =
      lastTs === null
        ? 0
        : Math.max(0, Math.min(timestampMs - lastTs, HANDS_FREE_MAX_SAMPLE_GAP_MS))
    lastTs = timestampMs

    if (phase === 'idle') {
      if (level >= voiceThreshold) {
        phase = 'pending'
        utteranceStartTs = timestampMs
        pendingRunMs = 0
        voicedTotalMs = 0
        silenceMs = 0
      }
      return { kind: 'listening', phase }
    }

    if (phase === 'pending') {
      // Hysteresis: the run survives while the level stays above the release
      // threshold, even if it dips under the (stricter) acquire threshold.
      if (level >= releaseThreshold) {
        pendingRunMs += dt
        voicedTotalMs += dt
        if (pendingRunMs >= minVoicedMs) phase = 'voiced'
      } else {
        // Blip — too short to be a speech onset. Forget it entirely.
        settle()
      }
      return { kind: 'listening', phase }
    }

    // phase === 'voiced'
    if (timestampMs - utteranceStartTs >= maxUtteranceMs) {
      const utteranceMs = timestampMs - utteranceStartTs
      settle()
      return { kind: 'endpoint', reason: 'max-utterance', utteranceMs }
    }
    if (level >= releaseThreshold) {
      silenceMs = 0
      voicedTotalMs += dt
      return { kind: 'listening', phase }
    }
    silenceMs += dt
    if (silenceMs < trailingSilenceMs) {
      // Inside the trailing-silence window: still part of the utterance.
      return { kind: 'listening', phase }
    }
    const utteranceMs = timestampMs - utteranceStartTs
    const longEnough = voicedTotalMs >= minUtteranceMs
    settle()
    return longEnough
      ? { kind: 'endpoint', reason: 'silence', utteranceMs }
      : { kind: 'discard', reason: 'too-short' }
  }

  return { feed, reset }
}
