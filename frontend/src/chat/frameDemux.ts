/** Shared sentence/audio/done demuxing core for the streaming turn frame
 * vocabulary, factored out of voiceTurn.ts and duplexTurn.ts (both drive
 * kurisuBridge identically for these three frame kinds; only their
 * transport and surrounding frame set differ — see each module's own
 * header comment for what stays separate).
 *
 * Owns the per-turn state that must be threaded between frames: the
 * seq->emotion map (sentence frames arrive before the audio frame that
 * needs their emotion) and the "has audio actually started" latch that
 * decides stop() (cancel) vs endStream() (drain) at done. Callers own
 * everything transport-specific (WS demux, filler timing, no-lipsync
 * counters) via the optional hooks below.
 */
import { kurisuBridge } from '../vrm/kurisuBridge'
import { classifySpeechBeat, type SpeechBeatKind } from '../vrm/speechBeats'
import type { AudioFrame, DoneFrame, SentenceFrame } from '../streaming'

export interface FrameDemuxHooks {
  /** Fires exactly once, right before the first audio chunk of the turn is
   * enqueued (i.e. lipSync.beginStream() is about to be/was just called).
   * voiceTurn uses this to cancel a pending filler and notify its caller;
   * duplex has no filler and doesn't need the notification. */
  onAudioStart?: () => void
  /** Fires when an audio frame arrives but lipSync.enqueue isn't
   * available. voiceTurn counts this (once per turn) via diagnostics;
   * duplex has never counted it, so this stays opt-in rather than baked
   * into the shared core. */
  onAudioDropped?: () => void
}

/** One instance per turn. Not reused across turns — construct a fresh one
 * per runVoiceTurn call / per duplex turn boundary. */
export const createFrameDemuxer = (hooks: FrameDemuxHooks = {}) => {
  // seq -> emotion, so the audio chunk for a sentence can be tinted with
  // that sentence's emotion (sentence frames arrive before audio frames).
  const emotionBySeq = new Map<number, string>()
  // seq -> speech-beat kind (question/negation/emphasis, vrm/speechBeats):
  // classified from the sentence TEXT here, fired when that sentence's
  // AUDIO starts (lipSyncController's sentence-audio-start event), so the
  // gesture lands with her voice instead of seconds early. Per-turn like
  // emotionBySeq — a cancelled turn's beats die with the demuxer.
  const beatBySeq = new Map<number, SpeechBeatKind>()
  let audioStreamBegun = false

  return {
    /** Emotion -> avatar: expression blend + (throttled) one-shot body
     * gesture. Identical dance for every surface, so it lives here rather
     * than being duplicated per caller. Returns the frame's emotion (or
     * undefined) so callers can fold it into their own onSentence extras
     * without re-deriving it. */
    handleSentence(frame: SentenceFrame): void {
      if (frame.emotion) {
        const intensity = typeof frame.emotion_intensity === 'number' ? frame.emotion_intensity : 0.8
        kurisuBridge.expressions?.set(frame.emotion, intensity, 2500)
        kurisuBridge.react?.(frame.emotion_raw ?? frame.emotion, intensity)
        emotionBySeq.set(frame.seq, frame.emotion)
      }
      const beat = typeof frame.en === 'string' ? classifySpeechBeat(frame.en) : null
      if (beat) beatBySeq.set(frame.seq, beat)
    },

    /** Enqueues one audio chunk onto lipSync, tinted with its sentence's
     * emotion. No-ops (after the dropped hook) when lipSync isn't
     * available. Pass `audioEnabled: false` to skip entirely (voice-off
     * turns never touch lipSync). */
    handleAudio(frame: AudioFrame, audioEnabled: boolean): void {
      if (!audioEnabled) return
      const lipSync = kurisuBridge.lipSync
      if (!lipSync?.enqueue) {
        hooks.onAudioDropped?.()
        return
      }
      if (!audioStreamBegun) {
        audioStreamBegun = true
        lipSync.beginStream?.()
        hooks.onAudioStart?.()
      }
      // Pass seq + the sentence's emotion so VRMViewer can apply
      // per-emotion pitch detune and ordered karaoke-timing events.
      void lipSync
        .enqueue({
          base64Audio: frame.data,
          mimeType: frame.mime,
          seq: frame.seq,
          emotion: emotionBySeq.get(frame.seq),
          beat: beatBySeq.get(frame.seq),
        })
        .catch((err) => {
          console.error('Streamed audio chunk failed:', err)
        })
    },

    /** True once the first audio chunk of the turn has been enqueued
     * (i.e. beginStream() has fired). Callers need this to decide whether
     * endStream()/stop() at done is a no-op. */
    hasAudioStarted(): boolean {
      return audioStreamBegun
    },

    /** stop() (cancel — cuts already-scheduled audio short) or endStream()
     * (drains it) depending on the done frame's cancelled flag. No-ops if
     * no audio chunk ever started. Resets the started-latch so a demuxer
     * instance reused across multiple turns (duplex's long-lived socket)
     * re-arms for the next turn. */
    handleDone(frame: Pick<DoneFrame, 'cancelled'>): void {
      if (!audioStreamBegun) return
      if (frame.cancelled) {
        kurisuBridge.lipSync?.stop?.()
      } else {
        kurisuBridge.lipSync?.endStream?.()
      }
      audioStreamBegun = false
    },

    /** Unconditional stop for an out-of-band barge-in signal (duplex's
     * 'state' frame going to listening/user_speaking) — unlike handleDone,
     * this always calls stop() even if this instance never saw an audio
     * frame start, since the state frame is the server's own authority
     * that playback should be cut *now*. Resets the started-latch so the
     * demuxer re-arms for the next turn on a long-lived socket. */
    interrupt(): void {
      kurisuBridge.lipSync?.stop?.()
      audioStreamBegun = false
    },
  }
}

export type FrameDemuxer = ReturnType<typeof createFrameDemuxer>
