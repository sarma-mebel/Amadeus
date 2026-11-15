/** Active-listening reactions: while the USER speaks into the mic, she holds
 * eye contact, nods along, and occasionally voices a soft acknowledgement
 * ("うん。") — instead of standing frozen until the transcript arrives.
 *
 * Fed per-audio-buffer RMS from the level bus (both capture paths — manual
 * push-to-talk and hands-free — already publish there; the barge-in monitor
 * does not, so this only ever reacts to genuine user-capture windows).
 *
 * Reactions fan out three ways:
 *  - 'kurisu:user-speaking' window event → VRMViewer extends gaze lock-on
 *  - 'kurisu:listening-nod' window event → VRMViewer queues a slow nod
 *  - playBackchannel() → one quiet clip via the presence one-shot path
 *
 * The vocal channel is deliberately rare (long sustained speech, cooldown,
 * coin flip) — backchannels register when they are occasional, and every
 * clip played is echo-cancelled out of the ongoing recording, not mixed in.
 */
import { playBackchannel } from './fillerPlayer'

export const LISTENING_REACTIONS_STORAGE_KEY = 'kurisu_listening_reactions'

/** Default ON: purely additive presence behavior, off with one toggle. */
export const loadListeningReactionsEnabled = (): boolean => {
  try {
    return window.localStorage.getItem(LISTENING_REACTIONS_STORAGE_KEY) !== 'false'
  } catch {
    return true
  }
}

export const saveListeningReactionsEnabled = (enabled: boolean): void => {
  try {
    window.localStorage.setItem(LISTENING_REACTIONS_STORAGE_KEY, String(enabled))
  } catch {
    // Preference just doesn't persist.
  }
}

export interface ListeningReactorHooks {
  /** Fired periodically while speech is ongoing — "keep looking at them". */
  onAttention: () => void
  /** Fired at nod moments (utterance onset, then every few seconds). */
  onNod: () => void
  /** Fired at most once per utterance after sustained speech. */
  onVocal: () => void
}

export interface ListeningReactorOptions {
  /** RMS at/above this counts as voice (endpointer default is 0.015). */
  voiceThreshold?: number
  /** Silence longer than this ends the current utterance. */
  utteranceResetSilenceMs?: number
  /** Re-fire onAttention at this interval while voiced. */
  attentionIntervalMs?: number
  /** Continuous-ish speech needed before the first nod. */
  firstNodMs?: number
  /** Random gap bounds between subsequent nods. */
  nodMinGapMs?: number
  nodMaxGapMs?: number
  /** Sustained speech needed before the vocal backchannel can fire. */
  vocalMinSpeechMs?: number
  /** Global floor between vocal backchannels, across utterances. */
  vocalCooldownMs?: number
  /** Chance the eligible vocal moment actually voices (anti-metronome). */
  vocalChance?: number
  random?: () => number
}

export interface ListeningReactor {
  /** Feed one capture buffer's RMS + duration. */
  feed: (rms: number, dtMs: number) => void
  /** Capture ended — clear utterance state (cooldowns survive). */
  reset: () => void
}

export const createListeningReactor = (
  hooks: ListeningReactorHooks,
  options: ListeningReactorOptions = {},
): ListeningReactor => {
  const voiceThreshold = options.voiceThreshold ?? 0.015
  const resetSilenceMs = options.utteranceResetSilenceMs ?? 700
  const attentionIntervalMs = options.attentionIntervalMs ?? 900
  const firstNodMs = options.firstNodMs ?? 600
  const nodMinGapMs = options.nodMinGapMs ?? 2400
  const nodMaxGapMs = options.nodMaxGapMs ?? 4200
  const vocalMinSpeechMs = options.vocalMinSpeechMs ?? 3500
  const vocalCooldownMs = options.vocalCooldownMs ?? 10000
  const vocalChance = options.vocalChance ?? 0.6
  const random = options.random ?? Math.random

  let clock = 0 // accumulated feed time, ms
  let voicedMs = 0 // speech time in the current utterance (gaps tolerated)
  let silenceMs = 0
  let lastAttentionAt = -Infinity
  let nextNodAtVoicedMs: number | null = firstNodMs
  let vocalFiredThisUtterance = false
  let lastVocalAt = -Infinity

  const endUtterance = () => {
    voicedMs = 0
    silenceMs = 0
    nextNodAtVoicedMs = firstNodMs
    vocalFiredThisUtterance = false
  }

  return {
    feed: (rms: number, dtMs: number) => {
      clock += dtMs
      if (rms >= voiceThreshold) {
        silenceMs = 0
        voicedMs += dtMs
      } else {
        silenceMs += dtMs
        if (silenceMs >= resetSilenceMs) endUtterance()
        return
      }

      // Attention: cheap and idempotent, just keeps the gaze window open.
      if (voicedMs >= 150 && clock - lastAttentionAt >= attentionIntervalMs) {
        lastAttentionAt = clock
        hooks.onAttention()
      }

      // Nods: one early acknowledgement, then loosely-spaced follow-ups.
      if (nextNodAtVoicedMs !== null && voicedMs >= nextNodAtVoicedMs) {
        hooks.onNod()
        nextNodAtVoicedMs =
          voicedMs + nodMinGapMs + random() * (nodMaxGapMs - nodMinGapMs)
      }

      // Vocal: only for genuinely sustained speech, once per utterance,
      // globally cooled down, and still only a coin flip.
      if (
        !vocalFiredThisUtterance &&
        voicedMs >= vocalMinSpeechMs &&
        clock - lastVocalAt >= vocalCooldownMs
      ) {
        vocalFiredThisUtterance = true
        lastVocalAt = clock
        if (random() < vocalChance) hooks.onVocal()
      }
    },
    reset: endUtterance,
  }
}

// --- default wiring (module singleton, fed by the level bus) ---------------

const defaultHooks: ListeningReactorHooks = {
  onAttention: () => window.dispatchEvent(new Event('kurisu:user-speaking')),
  onNod: () => window.dispatchEvent(new Event('kurisu:listening-nod')),
  onVocal: () => {
    void playBackchannel()
  },
}

let reactor: ListeningReactor | null = null
let lastFeedAt = 0
let enabledThisCapture = true

/** Called by the level bus on every published capture buffer. dt comes from
 * wall-clock deltas (the PTT path doesn't thread buffer durations through),
 * clamped so a hiccup can't register as seconds of speech. */
export const feedListeningLevel = (rms: number): void => {
  const now = performance.now()
  const isCaptureStart = lastFeedAt === 0 || now - lastFeedAt > 1000
  const dt = isCaptureStart ? 50 : Math.min(now - lastFeedAt, 250)
  lastFeedAt = now
  if (isCaptureStart) {
    // Re-read the toggle at each capture boundary, not per buffer.
    enabledThisCapture = loadListeningReactionsEnabled()
  }
  if (!enabledThisCapture) return
  if (!reactor) reactor = createListeningReactor(defaultHooks)
  reactor.feed(rms, dt)
}

/** Called by the level bus reset (capture stopped/cancelled). */
export const resetListeningReactions = (): void => {
  lastFeedAt = 0
  reactor?.reset()
}

/** Test seam. */
export const resetListeningReactionsForTest = (): void => {
  reactor = null
  lastFeedAt = 0
  enabledThisCapture = true
}
