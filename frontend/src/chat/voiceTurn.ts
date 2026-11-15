/** Shared WebSocket turn-loop core for a single chat send, used by both
 * ChatInterface's full transcript UI and the desktop-pet's voice-first
 * loop (usePetChat). Owns everything that must stay byte-for-byte
 * identical between the two surfaces: building the streamChat call,
 * demuxing sentence/audio frames onto kurisuBridge.lipSync's
 * beginStream/enqueue/endStream, the seq->emotion map that lets an audio
 * chunk carry its sentence's emotion, the "has audio actually started"
 * latch, cancel-vs-endStream at done, and classifying a stream failure as
 * either "never got a live stream" (caller may fall back to POST, or for
 * the pet, just report an error) vs "died after frames already flowed"
 * (never silently re-send — the caller must surface the loss in place).
 *
 * Callers plug in their own per-surface concerns (transcript token
 * rendering, thought ghost lines, tool activity strips) via the
 * handlers below; the pet passes only what it needs.
 */
import { buildChatPayload, type ChatImageAttachment } from './payload'
import {
  streamChat,
  type DoneFrame,
  type SentenceFrame,
  type StreamChatHandle,
  type ThoughtFrame,
  type ToolProgressFrame,
} from '../streaming'
import { armFillerForTurn, type FillerHandle } from '../audio/fillerPlayer'
import { classifyStreamFailureCause, type StreamFailureCause } from './errors'
import { createFrameDemuxer } from './frameDemux'
import { diagnostics } from '../diagnostics'
import { kurisuBridge } from '../vrm/kurisuBridge'

export interface VoiceTurnInput {
  message: string
  sessionId: string | null
  userId: string
  audioEnabled: boolean
  images?: ChatImageAttachment[]
}

/** Why the stream ended without a normal 'done' frame. Mirrors streamChat's
 * two failure entry points (onError / onConnectError) but folds in whether
 * any frame had already flowed, since that's what decides recoverability. */
export type VoiceTurnFailure =
  /** No frame ever arrived (WS never opened, or the server reported
   * streaming is off before a 'session' frame) — nothing was shown yet, so
   * the caller may safely retry via POST without duplicating content. */
  | { kind: 'connect' }
  /** At least one frame (session/token/sentence/audio) had already flowed
   * — content may be partially rendered/spoken. Re-sending would duplicate
   * it, so the caller must surface the loss in place instead. `cause` is
   * set when the server's error frame (or its message text) identified the
   * failure as the PROVIDER's — rate limit or model error — rather than a
   * dropped connection; callers should caption those honestly instead of
   * claiming "connection lost". Absent cause = genuine transport loss. */
  | { kind: 'mid-stream'; cause?: StreamFailureCause }

export interface VoiceTurnHandlers {
  onSession?: (sessionId: string) => void
  /** Raw token text as it streams in (transcript rendering, etc). */
  onToken?: (text: string) => void
  /** Fires for every sentence frame, after this module has already applied
   * the emotion dispatch (expression + gesture) and recorded emotionBySeq
   * for the matching audio chunk. Use this only for surface-specific extras
   * (e.g. accumulating the Japanese subtitle). */
  onSentence?: (frame: SentenceFrame) => void
  onThought?: (frame: ThoughtFrame) => void
  onToolProgress?: (frame: ToolProgressFrame) => void
  /** Fires exactly once, right before the first audio chunk of the turn is
   * enqueued (i.e. lipSync.beginStream() is about to be/was just called). */
  onAudioStart?: () => void
  /** Terminal success — 'done' frame arrived normally (cancelled or not).
   * This module has already handled endStream()/stop() and animations are
   * NOT played here (ChatInterface gates them on tool allowlisting via
   * playRequestedAnimations; the pet plays them unconditionally) — callers
   * that want the requested-animation flourish read frame.response.animations
   * themselves. `ttsFailed` mirrors frame.tts_failed (count of sentences
   * whose server-side synthesis failed this turn, 0 when absent/zero) so
   * callers don't each have to re-read the raw frame field. */
  onDone?: (frame: DoneFrame, ttsFailed: number) => void
  /** Stream ended without a 'done' frame. See VoiceTurnFailure for the
   * connect-vs-mid-stream distinction the caller must respect. */
  onFailure?: (failure: VoiceTurnFailure) => void
}

/** Starts one streamed turn. Returns the same cancel-only handle shape as
 * streamChat — call .cancel() for a barge-in; this module takes care of
 * stopping lip sync immediately on that path via the 'done' frame's
 * cancelled flag (the backend always sends 'done' after a cancel). */
export const runVoiceTurn = (input: VoiceTurnInput, handlers: VoiceTurnHandlers): StreamChatHandle => {
  // Did any frame at all reach us? Distinguishes "WS never produced a live
  // stream" (safe to fall back to POST) from "died after content already
  // rendered" (must not silently re-send — would duplicate the turn).
  let streamStarted = false
  // Once-per-turn guard: onAudio can fire many times, but a missing
  // lipSync.enqueue is the same underlying condition every time — count it
  // once so a whole reply's worth of dropped chunks doesn't spam the
  // counter (fillerPlayer/other counters in this file follow the same
  // once-per-occurrence spirit).
  let noLipSyncCounted = false
  // Thinking-sound backchannel: one short pre-synthesized clip fires if the
  // first real audio frame takes longer than its delay. Cancelled on first
  // audio / done / failure; a clip already playing is superseded by
  // beginStream()'s teardown, never layered under the reply.
  const filler: FillerHandle | null = input.audioEnabled ? armFillerForTurn() : null
  // Sentence emotion dispatch + audio demux + done stop-vs-drain: shared
  // with duplexTurn.ts's per-turn handling via frameDemux.ts.
  const demux = createFrameDemuxer({
    onAudioStart: () => {
      // Real audio is here: a pending filler must not fire after it
      // (beginStream tears down one that already started).
      filler?.cancel()
      handlers.onAudioStart?.()
    },
    onAudioDropped: () => {
      if (!noLipSyncCounted) {
        noLipSyncCounted = true
        diagnostics.increment('voice.audio_dropped_no_lipsync')
      }
    },
  })

  const handle = streamChat(
    buildChatPayload({
      message: input.message,
      sessionId: input.sessionId,
      userId: input.userId,
      audioEnabled: input.audioEnabled,
      images: input.images,
    }),
    {
      onSession: (sessionId) => {
        streamStarted = true
        handlers.onSession?.(sessionId)
      },
      onToken: (text) => {
        handlers.onToken?.(text)
      },
      onSentence: (frame) => {
        demux.handleSentence(frame)
        handlers.onSentence?.(frame)
      },
      onThought: (frame) => {
        handlers.onThought?.(frame)
      },
      onToolProgress: (frame) => {
        handlers.onToolProgress?.(frame)
      },
      onAudio: (frame) => {
        demux.handleAudio(frame, input.audioEnabled)
      },
      onDone: (frame) => {
        // Voice-off or TTS-failed turns end without any audio frame — the
        // filler must not hum after the reply has fully landed.
        filler?.cancel()
        // endStream() drains remaining scheduled audio — wrong after a
        // cancel, where playback must stop immediately.
        demux.handleDone(frame)
        // Backend-reported per-sentence TTS synthesis failures: previously
        // read by nobody, so a partially- or fully-silent reply looked
        // identical to a healthy voice-off turn. Surfaced to callers so
        // they can drive an honest "her voice glitched" hint instead of
        // staying quiet.
        const ttsFailed = typeof frame.tts_failed === 'number' ? frame.tts_failed : 0
        if (ttsFailed > 0) {
          diagnostics.increment('voice.tts_failed_sentences', String(ttsFailed))
        }
        handlers.onDone?.(frame, ttsFailed)
      },
      onError: (message, fallbackToPost, errorKind) => {
        console.warn('Streaming error:', message)
        filler?.cancel()
        if (demux.hasAudioStarted()) {
          kurisuBridge.lipSync?.stop?.()
        }
        // Only "never started + server says it's fallback-safe" is a real
        // POST-fallback candidate. A pre-stream error the server didn't
        // flag as fallback-safe gets the same 'mid-stream' (surface-in-
        // place) treatment as a true mid-stream death — this matches the
        // original ChatInterface, which routed both through one else branch.
        // The provider-error cause deliberately does NOT change this
        // classification: a 429 must never trigger a POST retry (it would
        // hit the same rate limit) — it only changes the caption.
        if (!streamStarted && fallbackToPost) {
          handlers.onFailure?.({ kind: 'connect' })
        } else {
          const cause = classifyStreamFailureCause(errorKind, message)
          handlers.onFailure?.(cause ? { kind: 'mid-stream', cause } : { kind: 'mid-stream' })
        }
      },
      onConnectError: () => {
        filler?.cancel()
        // Deliberately does NOT stop() lip sync here even mid-stream (unlike
        // onError above) — matches the original ChatInterface behavior: a
        // close-before-done with no explicit error message lets whatever
        // audio is already scheduled finish draining rather than cutting it
        // off. Only the explicit onError path cuts audio short.
        if (!streamStarted) {
          handlers.onFailure?.({ kind: 'connect' })
        } else {
          handlers.onFailure?.({ kind: 'mid-stream' })
        }
      },
    },
  )

  return handle
}
