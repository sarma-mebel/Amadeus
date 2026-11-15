/**
 * Hands-free conversation mode (opt-in, localStorage `kurisu_hands_free`,
 * default OFF): the mic is acquired ONCE when the mode turns on (echo-
 * cancelled, same constraints as push-to-talk) and stays open until the mode
 * turns off, the view unmounts, or the tab/window is hidden. A pure RMS
 * endpointer (src/audio/handsFree.ts) watches the live level stream: when
 * the user starts speaking and then pauses (~800ms trailing silence) the
 * buffered utterance is sliced out of the running capture, transcribed via
 * the existing one-shot POST /api/stt, and sent through the same
 * onTranscript path push-to-talk uses. Shared verbatim by ChatInterface and
 * the pet's usePetChat — both mount this one hook.
 *
 * Interplay rules (the important invariants):
 *
 * - HER AUDIO NEVER ENDPOINTS. While `kurisu:speaking` is true the level
 *   loop is suspended: the capture buffer is discarded per-callback, the
 *   endpointer is dropped, and nothing is published to the level bus. Even
 *   though the stream is echo-cancelled, residual leakage sits right around
 *   the endpointer threshold on laptop speakers — so we don't endpoint
 *   "carefully", we don't endpoint at all. Interrupting her is exclusively
 *   the barge-in paths' job: the mic button / hold-Space (manual), or the
 *   opt-in auto barge-in detector, which now rides this same capture stream
 *   with its much stricter threshold + min-voiced run + per-speech warm-up
 *   (no separate MicMonitor stream needed while hands-free owns the mic).
 *   After a barge-in her audio stops, `kurisu:speaking` flips false, and
 *   normal listening resumes after a short grace.
 *
 * - NO OVERLAPPING SENDS. From the moment an utterance endpoints until its
 *   STT round-trip settles (`sttInFlight`), and then while the resulting
 *   turn is in flight (`isBusy` from the caller), the loop stays suspended —
 *   audio arriving meanwhile is discarded, not queued. A short post-
 *   transcript grace covers the render gap between onTranscript() firing
 *   and the caller's isBusy becoming visible here. Listening resumes after
 *   the turn ends (and, if she spoke, after her audio finishes or is barged
 *   in — the speaking:false grace).
 *
 * - PTT WINS. While the manual push-to-talk recorder is active
 *   (`manualMicActive`), hands-free suspends but keeps its stream. Chosen
 *   over hiding the PTT button because PTT is the reliable escape hatch
 *   when VAD misbehaves (noisy rooms), and "suspend, don't remove" keeps
 *   every existing PTT code path byte-identical whether the toggle is on or
 *   off. Two concurrent echo-cancelled captures of the same device are fine
 *   in every supported browser.
 *
 * - LEVEL BUS: hands-free publishes mic levels ONLY while actively
 *   listening (not while suspended), so the waveform indicator and her
 *   listening reactions (gaze/nods/backchannels) behave exactly as they do
 *   during a PTT capture — and never react to her own speech.
 */
import { useEffect, useRef, useState } from 'react'
import { apiFetch } from '../api'
import {
  HANDS_FREE_EVENT,
  HANDS_FREE_STORAGE_KEY,
  createHandsFreeEndpointer,
  loadHandsFreeEnabled,
  saveHandsFreeEnabled,
  type HandsFreeEndpointer,
} from '../audio/handsFree'
import { publishMicLevel, resetMicLevel } from '../audio/levelBus'
import { MicRecorder } from '../audio/micRecorder'
import { createBargeInDetector, type BargeInDetector } from '../audio/vad'

export type HandsFreeStatus = 'idle' | 'listening' | 'transcribing'

export { HANDS_FREE_STORAGE_KEY }
export const AUTO_BARGE_IN_STORAGE_KEY = 'kurisu_auto_barge_in'

/** Grace between her audio ending and endpointing arming, so the tail of her
 * speech (and the OS audio-device switch click) is never counted as user
 * voice. */
const RESUME_GRACE_MS = 350

/** Pre-roll kept in the capture buffer while nobody is speaking. Must cover
 * the endpointer's min-voiced onset gate (250ms) plus a capture buffer or
 * two, so the first syllable — heard before the onset is confirmed — makes
 * it into the sliced utterance. */
const PREROLL_MS = 400

/** Suspension window after a transcript is handed to the caller: the
 * caller's isBusy flips on its NEXT render, so without this gap the very
 * next audio buffer could start a second utterance while the first turn is
 * not yet visible here — the no-overlapping-sends invariant would leak. */
const POST_TRANSCRIPT_GRACE_MS = 600

export interface UseHandsFreeOptions {
  /** STT probe result from useMic — null while still probing. */
  micAvailable: boolean | null
  micPermissionDenied: boolean
  /** True while the push-to-talk mic is recording/transcribing. */
  manualMicActive: boolean
  /** True while a reply is loading or streaming — never endpoint then. */
  isBusy: boolean
  /** Called with the trimmed transcript of a completed utterance. */
  onTranscript: (text: string) => void
  /** Transient hint line (useMic's showMicHint) for auto-disable notices. */
  showHint: (hint: string) => void
  /** Interrupt the current reply (auto barge-in trigger) — same effect as
   * the mic button's barge-in: stop the stream + flush her audio. */
  onBargeIn?: () => void
}

export const useHandsFree = (options: UseHandsFreeOptions) => {
  const [handsFree, setHandsFree] = useState(
    () => typeof window !== 'undefined' && loadHandsFreeEnabled(),
  )
  const [handsFreeStatus, setHandsFreeStatus] = useState<HandsFreeStatus>('idle')

  // Latest props/state for the audio callbacks (they outlive the render
  // closure that created them).
  const optionsRef = useRef(options)
  useEffect(() => {
    optionsRef.current = options
  }, [options])
  const handsFreeRef = useRef(handsFree)

  const [autoBargeIn, setAutoBargeIn] = useState(
    () =>
      typeof window !== 'undefined' &&
      window.localStorage.getItem(AUTO_BARGE_IN_STORAGE_KEY) === 'true',
  )
  const autoBargeInRef = useRef(autoBargeIn)
  useEffect(() => {
    autoBargeInRef.current = autoBargeIn
    window.localStorage.setItem(AUTO_BARGE_IN_STORAGE_KEY, String(autoBargeIn))
  }, [autoBargeIn])

  // Tab/window visibility: a hidden surface releases the mic entirely (no
  // open microphone in a window the user can't see) and re-acquires on
  // return. Applies to both the browser tab and the pet window.
  const [docVisible, setDocVisible] = useState(
    () => typeof document === 'undefined' || document.visibilityState !== 'hidden',
  )
  useEffect(() => {
    const handleVisibility = () => setDocVisible(document.visibilityState !== 'hidden')
    document.addEventListener('visibilitychange', handleVisibility)
    return () => document.removeEventListener('visibilitychange', handleVisibility)
  }, [])

  const speakingRef = useRef(false)
  const recorderRef = useRef<MicRecorder | null>(null)
  const acquiringRef = useRef(false)
  const endpointerRef = useRef<HandsFreeEndpointer | null>(null)
  const detectorRef = useRef<BargeInDetector | null>(null)
  const sttInFlightRef = useRef(false)
  // Audio-derived clock: advances by each capture buffer's duration, so all
  // timing (endpointer, graces) is deterministic and immune to wall-clock
  // hiccups. Reset per acquisition.
  const clockRef = useRef(0)
  const resumeAtRef = useRef(0)
  const statusRef = useRef<HandsFreeStatus>('idle')

  const setStatus = (next: HandsFreeStatus) => {
    if (statusRef.current === next) return
    statusRef.current = next
    setHandsFreeStatus(next)
  }

  const releaseRecorder = () => {
    const recorder = recorderRef.current
    recorderRef.current = null
    if (recorder) {
      recorder.onLevel = null
      recorder.cancel()
    }
    endpointerRef.current = null
    detectorRef.current = null
    // An in-flight transcription is past the mic — leave 'transcribing' to
    // settle in its own finally.
    if (statusRef.current === 'listening') setStatus('idle')
    resetMicLevel()
  }

  const disableHandsFree = (hint: string) => {
    if (!handsFreeRef.current) return
    // Route through the shared preference so SettingsPanel/other toggles
    // stay in sync; the event listener below performs the actual teardown.
    saveHandsFreeEnabled(false)
    optionsRef.current.showHint(hint)
  }

  const transcribe = async (wav: Blob) => {
    sttInFlightRef.current = true
    setStatus('transcribing')
    resetMicLevel()
    try {
      const formData = new FormData()
      formData.append('audio', wav, 'clip.wav')
      const response = await apiFetch('/api/stt', { method: 'POST', body: formData })
      if (!response.ok) {
        if (response.status === 501) {
          disableHandsFree('Hands-free off — speech-to-text is unavailable on this server')
        }
        return
      }
      const data = await response.json()
      const text = typeof data.text === 'string' ? data.text.trim() : ''
      // Empty/whitespace transcript = the VAD heard something STT couldn't —
      // discard silently and keep listening.
      if (text) {
        resumeAtRef.current = clockRef.current + POST_TRANSCRIPT_GRACE_MS
        optionsRef.current.onTranscript(text)
      }
    } catch (error) {
      console.warn('Hands-free transcription failed:', error)
    } finally {
      sttInFlightRef.current = false
      // The level loop flips back to 'listening' on the next unsuspended
      // sample; settle to idle here so a released/suspended mic doesn't
      // stick on the spinner.
      if (statusRef.current === 'transcribing') setStatus('idle')
    }
  }

  /** The per-capture-buffer heart of the mode. Bound to one recorder; the
   * identity check makes callbacks from a superseded recorder inert. */
  const handleLevel = (recorder: MicRecorder, rms: number, dtMs: number) => {
    if (recorderRef.current !== recorder) return
    const clock = (clockRef.current += dtMs)
    const opts = optionsRef.current

    if (speakingRef.current) {
      // She's speaking: never endpoint (see module doc). Keep memory flat
      // and drop any half-built utterance.
      recorder.trimBuffer(0)
      endpointerRef.current = null
      if (statusRef.current === 'listening') setStatus('idle')
      // Auto barge-in rides this same stream: the detector's stricter
      // threshold + sustained-voice run + warm-up window keep her own
      // residual echo from self-triggering.
      const detector = detectorRef.current
      if (detector && autoBargeInRef.current) {
        if (detector.feed(rms, dtMs) === 'triggered') {
          detectorRef.current = null
          opts.onBargeIn?.()
        }
      }
      return
    }

    const suspended =
      !handsFreeRef.current ||
      sttInFlightRef.current ||
      opts.isBusy ||
      opts.manualMicActive ||
      clock < resumeAtRef.current
    if (suspended) {
      recorder.trimBuffer(0)
      endpointerRef.current = null
      if (statusRef.current === 'listening') setStatus('idle')
      return
    }

    // Actively listening: this is the only branch that publishes to the
    // level bus — waveform + listening reactions run exactly as during a
    // PTT capture, and only for genuine user-capture windows.
    if (statusRef.current === 'idle') setStatus('listening')
    publishMicLevel(rms)

    if (!endpointerRef.current) endpointerRef.current = createHandsFreeEndpointer()
    const result = endpointerRef.current.feed(rms, clock)
    if (result.kind === 'listening') {
      // Nothing being buffered is speech yet — keep only the pre-roll.
      if (result.phase === 'idle') recorder.trimBuffer(PREROLL_MS)
      return
    }
    if (result.kind === 'discard') {
      // Cough/blip: drop the audio, keep listening.
      recorder.trimBuffer(PREROLL_MS)
      return
    }
    // Endpoint: slice the utterance out of the running capture and run the
    // one-shot STT + send path. sttInFlight suspends the loop until settled.
    const wav = recorder.takeWav()
    if (!wav) return
    void transcribe(wav)
  }

  const desired =
    handsFree && options.micAvailable === true && !options.micPermissionDenied && docVisible
  const desiredRef = useRef(desired)
  desiredRef.current = desired

  const acquireRecorder = async () => {
    if (recorderRef.current || acquiringRef.current) return
    acquiringRef.current = true
    const recorder = new MicRecorder()
    recorder.onLevel = (rms, dtMs) => handleLevel(recorder, rms, dtMs)
    try {
      await recorder.start()
    } catch (error) {
      acquiringRef.current = false
      const domError = error as DOMException
      if (domError?.name === 'NotAllowedError' || domError?.name === 'SecurityError') {
        disableHandsFree('Hands-free off — microphone permission denied')
      } else {
        console.warn('Hands-free capture failed to start:', error)
        // A continuous mode with a dead mic would just look broken —
        // switch off rather than pretend to listen.
        disableHandsFree('Hands-free off — microphone unavailable')
      }
      return
    }
    acquiringRef.current = false
    // Superseded while getUserMedia was pending (toggled off, tab hidden):
    // release immediately, never listen.
    if (!desiredRef.current || recorderRef.current) {
      recorder.onLevel = null
      recorder.cancel()
      return
    }
    clockRef.current = 0
    resumeAtRef.current = 0
    endpointerRef.current = null
    recorderRef.current = recorder
    // Status settles via the level loop ('listening' on the first
    // unsuspended sample).
  }

  // Mic lifecycle: acquire once when the mode becomes usable, release when
  // it stops being (toggle off, STT unavailable, permission denied, hidden
  // tab, unmount).
  useEffect(() => {
    if (desired) {
      void acquireRecorder()
    } else {
      releaseRecorder()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [desired])

  // Her speaking state drives suspension + the auto barge-in detector. A
  // fresh detector per speech segment: its warm-up must start at HER speech
  // onset (residual echo burst before the AEC converges).
  useEffect(() => {
    const handleSpeakingChange = (event: Event) => {
      const detail = (event as CustomEvent<{ speaking: boolean }>).detail
      const speaking = Boolean(detail?.speaking)
      speakingRef.current = speaking
      if (speaking) {
        detectorRef.current =
          handsFreeRef.current && autoBargeInRef.current ? createBargeInDetector() : null
      } else {
        detectorRef.current = null
        resumeAtRef.current = clockRef.current + RESUME_GRACE_MS
      }
    }
    window.addEventListener('kurisu:speaking', handleSpeakingChange)
    return () => window.removeEventListener('kurisu:speaking', handleSpeakingChange)
  }, [])

  // Preference sync: every toggle surface (composer button, pet button,
  // SettingsPanel checkbox, auto-disable) routes through
  // saveHandsFreeEnabled, which dispatches this event; the mic lifecycle
  // effect above reacts to the state change.
  useEffect(() => {
    const handlePrefChange = (event: Event) => {
      const detail = (event as CustomEvent<{ enabled: boolean }>).detail
      const next = Boolean(detail?.enabled)
      if (handsFreeRef.current === next) return
      handsFreeRef.current = next
      setHandsFree(next)
    }
    window.addEventListener(HANDS_FREE_EVENT, handlePrefChange)
    return () => window.removeEventListener(HANDS_FREE_EVENT, handlePrefChange)
  }, [])

  // Guard rail: the mode can't stay on when STT is unavailable or mic
  // permission was denied — auto-disable with the standard hint.
  const { micAvailable, micPermissionDenied } = options
  useEffect(() => {
    if (!handsFree) return
    if (micAvailable === false) {
      disableHandsFree('Hands-free off — speech-to-text is unavailable on this server')
    } else if (micPermissionDenied) {
      disableHandsFree('Hands-free off — microphone permission denied')
    }
  }, [handsFree, micAvailable, micPermissionDenied])

  // Release the mic if the component unmounts.
  useEffect(() => {
    return () => {
      // Unmount can land while acquireRecorder is still awaiting
      // getUserMedia (recorderRef null, nothing to cancel yet). Every other
      // release is driven by a `desired` flip, which the post-await guard
      // reads via desiredRef — but unmount doesn't re-render, so desiredRef
      // would stay true and the just-resolved recorder would install itself
      // into the unmounted hook: a leaked open mic with live callbacks.
      // Flipping the ref here makes that guard release it immediately.
      desiredRef.current = false
      releaseRecorder()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const toggleHandsFree = () => {
    saveHandsFreeEnabled(!handsFreeRef.current)
  }

  const toggleAutoBargeIn = () => {
    const next = !autoBargeInRef.current
    autoBargeInRef.current = next
    setAutoBargeIn(next)
    if (!next) {
      detectorRef.current = null
    } else if (speakingRef.current && handsFreeRef.current) {
      // Turned on mid-speech: arm a detector now (warm-up from here).
      detectorRef.current = createBargeInDetector()
    }
  }

  return { handsFree, toggleHandsFree, handsFreeStatus, autoBargeIn, toggleAutoBargeIn }
}
