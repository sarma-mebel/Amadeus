/** Voice playback + lip sync, extracted verbatim from VRMViewer's scene
 * effect. One controller per scene mount owns the WebAudio graph (analyser →
 * dry/wet reverb → master gain), the viseme analysis tick (driven by
 * VRMViewer's render loop via `update()` — no rAF of its own), single-clip
 * playback, and the chunked-stream scheduler.
 *
 * CRITICAL INVARIANT: `stateRef` is the SAME ref object VRMViewer holds —
 * the render loop reads `state.speechEnvelope` (speech micro-motion) and the
 * animation scheduler reads `state.streaming` directly off it. Do not copy
 * the state object.
 *
 * Side effects on speaking-state changes (the 'kurisu:speaking' DOM event,
 * the talk-animation hook, the interaction timestamp) are injected via
 * `onSpeakingChange` — they belong to the scene effect, not the audio layer.
 */

import type { RefObject } from 'react'
import type { VRM } from '@pixiv/three-vrm'
import { diagnostics } from '../diagnostics'
import { loadVolumePrefs } from '../soundscape'

// App-wide volume control (item 4, distinct from the settings-panel voice
// slider soundscape.ts owns): a sibling-owned slider writes this key and
// fires this event after every change. "0".."1" as a string; absent =
// default 1 (unmuted) so a user who never opens the control hears exactly
// what they always heard.
export const APP_VOLUME_STORAGE_KEY = 'kurisu_volume'
export const APP_VOLUME_CHANGE_EVENT = 'kurisu-volume-changed'

export const loadAppVolume = (): number => {
  try {
    const raw = window.localStorage.getItem(APP_VOLUME_STORAGE_KEY)
    if (raw === null) return 1
    const parsed = Number(raw)
    return Number.isFinite(parsed) ? Math.min(1, Math.max(0, parsed)) : 1
  } catch {
    return 1
  }
}

const allocateFloat32Array = (length: number): Float32Array<ArrayBuffer> => {
  const buffer = new ArrayBuffer(length * Float32Array.BYTES_PER_ELEMENT)
  return new Float32Array(buffer) as Float32Array<ArrayBuffer>
}

// ONE AudioContext for the lifetime of the page, shared across controller
// mounts. The scene effect remounts on outfit swaps (and the component can
// unmount/remount on view switches), and Chrome caps ~6 live AudioContexts
// per page — creating one per mount exhausted the cap after a few swaps and
// silently killed all voice. Closing the old context on teardown is NOT the
// answer: close() is async and races the next mount's ensureAudioContext.
// Per-mount nodes (analyser/gains/convolver) are still built fresh on this
// shared context by ensureLipSyncGraph, and the autoplay unlock (resume()
// inside a user-gesture-initiated call) works the same on a reused context.
let sharedAudioContext: AudioContext | null = null

/** Test seam: forget the shared AudioContext singleton so a fresh mock
 * AudioContext stubbed in a later test actually gets picked up by
 * ensureAudioContext, instead of every test in the same file/worker
 * silently reusing whichever mock happened to be live first. Not used by
 * production code (there is exactly one real page, so the singleton is
 * intentional there). */
export const resetSharedAudioContextForTest = (): void => {
  sharedAudioContext = null
}

export type LipSyncState = {
  audioContext: AudioContext | null
  analyser: AnalyserNode | null
  // Voice reverb (B1): parallel dry/wet paths merged into a master gain that
  // scales TOTAL output — the analyser stays on the dry tap (pre-reverb) so
  // viseme detection reads the clean signal regardless of wetMix. The
  // app-wide volume slider (localStorage `kurisu_volume`) sits AFTER
  // masterGain, not before the analyser — see appVolumeGain below.
  //   source -> analyser -> dryGain \
  //                                  -> masterGain -> appVolumeGain -> destination
  //          -> convolver -> wetGain /
  dryGain: GainNode | null
  wetGain: GainNode | null
  convolver: ConvolverNode | null
  gainNode: GainNode | null // master (dry+wet) gain — this is what the settings-panel voice-volume slider drives
  // App-wide volume slider (kurisu_volume / 'kurisu-volume-changed', a
  // separate control from `gainNode`'s settings-panel voice slider — see
  // ensureLipSyncGraph). Deliberately the LAST node before destination:
  // the analyser (viseme/mouth-amplitude detection) taps the signal before
  // either gain stage, so muting/lowering this slider can never break lip
  // sync — the mouth keeps moving correctly even at volume 0.
  appVolumeGain: GainNode | null
  sourceNode: AudioBufferSourceNode | null
  freqData: Float32Array<ArrayBuffer> | null
  timeData: Float32Array<ArrayBuffer> | null
  expressionValues: Record<string, number>
  previousViseme: string | null
  // Chunked-stream playback: sources scheduled back-to-back on the shared
  // analyser graph, so the viseme loop runs continuously across chunks
  // instead of restarting (and cutting lip sync) per sentence.
  scheduledSources: AudioBufferSourceNode[]
  nextChunkStartTime: number
  streaming: boolean
  streamDraining: boolean
  // Item 2: raw (unsmoothed) speech amplitude for the current frame, written
  // by the lip-sync analysis tick (analyseAndAnimate, driven per-frame by
  // the render loop) and read by applySpeechMicroMotion, which does its own
  // smoothing/decay on top of this. 0 when not speaking.
  speechEnvelope: number
  // B3 (karaoke timing): pending setTimeout ids for 'kurisu:sentence-audio-start'
  // dispatches, one per scheduled chunk — cleared in stopLipSync so a
  // cancelled/replaced stream doesn't fire stale events.
  sentenceStartTimeoutIds: number[]
  // Last-resort bare <audio> playback (no VRM / no AudioContext). Tracked so
  // stopLipSync can silence it: an untracked element was unstoppable, and a
  // stream starting while one played put two voices on top of each other.
  fallbackAudio?: HTMLAudioElement | null
}

export interface ReverbSettings {
  irDurationSec: number
  decay: number
  wetMix: number
}

export interface LipSyncControllerDeps {
  /** VRMViewer's lipSyncStateRef — shared, never copied (see module doc). */
  stateRef: RefObject<LipSyncState>
  getVrm: () => VRM | null
  /** Active environment's reverb tuning; undefined falls back to defaults. */
  getReverb: () => ReverbSettings | undefined
  onSpeakingChange: (speaking: boolean) => void
}

// All five VRM visemes. 'oh' was missing for a long time — the centroid
// mapping below was authored with four bands and every /o/ landed on 'ou'
// (too closed) or 'aa' (too open).
const lipSyncVisemes = ['aa', 'ih', 'ee', 'ou', 'oh'] as const

export const createLipSyncController = ({
  stateRef,
  getVrm,
  getReverb,
  onSpeakingChange,
}: LipSyncControllerDeps) => {
  const dispatchSpeakingChange = (speaking: boolean) => {
    onSpeakingChange(speaking)
  }

const ensureAudioContext = () => {
  const state = stateRef.current
  if (!sharedAudioContext) {
    const AudioCtx = window.AudioContext ?? window.webkitAudioContext
    if (!AudioCtx) {
      console.warn('AudioContext not supported; lip sync disabled')
      return null
    }
    sharedAudioContext = new AudioCtx()
  }
  // state.audioContext stays a mirror of the shared context so existing
  // readers (enqueue, the analysis tick's sampleRate read) keep working.
  state.audioContext = sharedAudioContext
  return sharedAudioContext
}

// Speaking predicate covering BOTH playback paths. `streaming` only covers
// the chunked-stream path — the one-shot play() path (greetings, presence
// lines, reminder/reach-out announcements) never sets it, so guards that
// read only `streaming` (fillerPlayer, presence polls via the bridge's
// isSpeaking) used to pass while a presence clip was audibly playing and
// overlap two voices. `sourceNode` is the one-shot clip currently playing
// (cleared by stopLipSync via its onended/error paths); `scheduledSources`
// are stream chunks still scheduled or audible — they can outlive
// `streaming` (drain after endStream, and the barge-in fade window).
const isAudible = () => {
  const state = stateRef.current
  return (
    state.streaming ||
    state.sourceNode !== null ||
    state.scheduledSources.length > 0 ||
    Boolean(state.fallbackAudio && !state.fallbackAudio.ended && !state.fallbackAudio.paused)
  )
}

const resetLipSyncExpressions = (relaxedLevel = 0.35) => {
  const vrm = getVrm()
  if (!vrm?.expressionManager) return
  const state = stateRef.current
  lipSyncVisemes.forEach((name) => {
    state.expressionValues[name] = 0
    vrm.expressionManager?.setValue(name, 0)
  })
  state.expressionValues.relaxed = relaxedLevel
  vrm.expressionManager?.setValue('relaxed', relaxedLevel)
}


// Bumped on every teardown/begin so a pending delayed stop (the fade in
// fadeOutAndStopLipSync) can tell whether it's still tearing down the
// stream it faded, or would clobber a newer one.
let lipSyncGeneration = 0
// True while a fade's deferred teardown is pending — repeat stop() calls
// during the window must not hard-cut the already-fading audio (the click
// the fade exists to avoid). Cleared by stopLipSync itself, so any real
// teardown (a new stream/clip preempting the fade) re-arms cleanly.
let fadeStopPending = false

const stopLipSync = () => {
  lipSyncGeneration += 1
  fadeStopPending = false
  const state = stateRef.current
  const wasStreaming = state.streaming
  if (state.fallbackAudio) {
    try {
      state.fallbackAudio.pause()
    } catch {
      // already ended/never started; nothing to do
    }
    state.fallbackAudio = null
  }
  if (state.sourceNode) {
    try {
      state.sourceNode.stop()
    } catch (err) {
      console.warn('Failed to stop lip sync source:', err)
    }
    state.sourceNode.disconnect()
    state.sourceNode = null
  }
  state.scheduledSources.forEach((source) => {
    try {
      source.stop()
    } catch {
      // already finished playing; nothing to do
    }
    source.disconnect()
  })
  state.scheduledSources = []
  // B3: cancel any pending karaoke-timing dispatches from the stream
  // being torn down — a cancelled/replaced turn must not fire stale
  // 'kurisu:sentence-audio-start' events for audio that never plays.
  state.sentenceStartTimeoutIds.forEach((id) => window.clearTimeout(id))
  state.sentenceStartTimeoutIds = []
  state.streaming = false
  state.streamDraining = false
  state.nextChunkStartTime = 0
  if (wasStreaming) {
    dispatchSpeakingChange(false)
  }
  if (state.analyser) {
    state.analyser.disconnect()
    state.analyser = null
  }
  if (state.dryGain) {
    state.dryGain.disconnect()
    state.dryGain = null
  }
  if (state.convolver) {
    state.convolver.disconnect()
    state.convolver = null
  }
  if (state.wetGain) {
    state.wetGain.disconnect()
    state.wetGain = null
  }
  if (state.gainNode) {
    state.gainNode.disconnect()
    state.gainNode = null
  }
  if (state.appVolumeGain) {
    state.appVolumeGain.disconnect()
    state.appVolumeGain = null
  }
  state.freqData = null
  state.timeData = null
  state.previousViseme = null
  // Item 2: zero the raw envelope so applySpeechMicroMotion's own
  // smoothing eases the head/torso/brow motion back to neutral instead
  // of freezing at whatever the last frame of speech left it at (the
  // analysis tick that would otherwise decay this no-ops from here on —
  // its analyser was just nulled above).
  state.speechEnvelope = 0
  resetLipSyncExpressions(0.4)
}

// Barge-in stop: a hard source.stop() mid-word is an audible click and
// reads as broken; games always fade dialogue out over ~100ms. Ramps the
// master gain down, then does the real teardown — unless a newer stream
// has started in the meantime (generation guard), in which case the
// teardown already happened in beginStream and this must not touch it.
//
// The stream is declared dead SYNCHRONOUSLY at fade start (streaming =
// false): only the gain ramp + node teardown are deferred. Leaving
// `streaming` true for the fadeMs+40 window let enqueue's dead-stream
// guards pass, so a late chunk arriving mid-fade got decoded and scheduled
// onto the fading graph — a quiet blip, with its bookkeeping racing the
// deferred stopLipSync. The speaking-change side effects fire here too,
// since the deferred stopLipSync will see wasStreaming === false.
const fadeOutAndStopLipSync = (fadeMs = 140) => {
  const state = stateRef.current
  const context = state.audioContext
  // A fade is already in flight — its deferred stopLipSync finishes the
  // job. Falling through to the early hard-stop branch below (streaming is
  // already false mid-fade) would cut the still-audible ramp with a click.
  if (fadeStopPending) return
  if (!state.streaming || !context || !state.gainNode) {
    stopLipSync()
    return
  }
  state.streaming = false
  state.streamDraining = false
  dispatchSpeakingChange(false)
  const generation = lipSyncGeneration
  const gain = state.gainNode.gain
  const now = context.currentTime
  gain.cancelScheduledValues(now)
  gain.setTargetAtTime(0.0001, now, fadeMs / 1000 / 3)
  fadeStopPending = true
  window.setTimeout(() => {
    if (lipSyncGeneration === generation) stopLipSync()
  }, fadeMs + 40)
}

const decodeBase64Audio = (base64: string) => {
  const binary = atob(base64)
  const len = binary.length
  const bytes = new Uint8Array(len)
  for (let i = 0; i < len; i += 1) {
    bytes[i] = binary.charCodeAt(i)
  }
  return bytes.buffer
}

// Procedurally generates a mono impulse response: a short burst of noise
// with an exponential decay envelope, per B1's spec — no shipped IR
// asset, consistent with soundscape.ts's "everything is synthesized"
// approach. `durationSec`/`decay` come from the active environment's
// `audio.reverb` tuning (see environments.ts); decay is the exponent
// applied to the normalized (1 -> 0) envelope, so a bigger number decays
// faster/tighter and a smaller one leaves a longer tail.
const createImpulseResponse = (
  context: AudioContext,
  durationSec: number,
  decay: number
): AudioBuffer => {
  const length = Math.max(1, Math.floor(context.sampleRate * durationSec))
  const impulse = context.createBuffer(2, length, context.sampleRate)
  for (let channel = 0; channel < impulse.numberOfChannels; channel += 1) {
    const data = impulse.getChannelData(channel)
    for (let i = 0; i < length; i += 1) {
      const envelope = Math.pow(1 - i / length, decay)
      data[i] = (Math.random() * 2 - 1) * envelope
    }
  }
  return impulse
}

// Builds the analyser -> {dry, wet(reverb)} -> master gain -> destination
// graph once per playback session (single clip or chunked stream) and
// keeps it connected across however many AudioBufferSourceNodes get
// scheduled through it, so the viseme loop below never has to restart
// mid-utterance.
//
// B1 (voice reverb): the analyser stays on the DRY tap — it must see the
// clean pre-reverb signal or viseme/centroid detection degrades — while
// a parallel wet path (convolver -> wetGain) is summed into a master
// gain alongside the dry path (dryGain). The voice-volume slider
// (handleVolumeChange below / loadVolumePrefs().voice) drives that
// master gain so it still scales TOTAL output regardless of wetMix.
const ensureLipSyncGraph = (context: AudioContext) => {
  const state = stateRef.current
  if (
    state.analyser &&
    state.gainNode &&
    state.dryGain &&
    state.wetGain &&
    state.convolver &&
    state.appVolumeGain
  ) {
    return { analyser: state.analyser, gainNode: state.gainNode }
  }
  const analyser = context.createAnalyser()
  // 1024, not 2048: the viseme mapping only buckets the spectral centroid
  // into 5 coarse bands (see analyseAndAnimate), which doesn't need
  // 2048-bin resolution — halving the FFT halves the per-tick analysis
  // cost now that the tick runs inside the main render loop.
  analyser.fftSize = 1024
  analyser.smoothingTimeConstant = 0.6

  const reverb = getReverb() ?? { irDurationSec: 0.6, decay: 2.5, wetMix: 0.22 }

  const dryGain = context.createGain()
  dryGain.gain.value = 1

  const convolver = context.createConvolver()
  convolver.normalize = true
  convolver.buffer = createImpulseResponse(context, reverb.irDurationSec, reverb.decay)

  const wetGain = context.createGain()
  wetGain.gain.value = reverb.wetMix

  const masterGain = context.createGain()
  masterGain.gain.value = loadVolumePrefs().voice

  // App-wide volume slider (item 4): a second, independent gain stage
  // AFTER masterGain — both this and the settings-panel voice slider
  // above scale final loudness multiplicatively, but they're separate
  // controls (separate localStorage keys/events, separate owners) so
  // neither has to know about the other. The analyser tap that feeds
  // viseme/mouth-amplitude detection is upstream of BOTH gain stages, so
  // dragging this slider to 0 mutes audibly without ever breaking lip
  // sync — see LipSyncState's doc comment for the full chain diagram.
  const appVolumeGain = context.createGain()
  appVolumeGain.gain.value = loadAppVolume()

  analyser.connect(dryGain)
  analyser.connect(convolver)
  convolver.connect(wetGain)
  dryGain.connect(masterGain)
  wetGain.connect(masterGain)
  masterGain.connect(appVolumeGain)
  appVolumeGain.connect(context.destination)

  state.analyser = analyser
  state.dryGain = dryGain
  state.convolver = convolver
  state.wetGain = wetGain
  state.gainNode = masterGain
  state.appVolumeGain = appVolumeGain
  state.freqData = allocateFloat32Array(analyser.frequencyBinCount)
  state.timeData = allocateFloat32Array(analyser.fftSize)
  state.previousViseme = null
  return { analyser, gainNode: masterGain }
}

// Viseme analysis tick. Called once per rendered frame from VRMViewer's
// animate() loop (exposed as `update` on the controller) instead of
// self-scheduling its own 60fps rAF — the old concurrent rAF ran two
// full-spectrum passes + expression writes alongside the render loop and
// kept burning CPU with the tab hidden; riding the main loop inherits its
// document.hidden pause for free. Cheap no-op whenever no lip-sync graph
// is live (analyser is null outside a playback session). No dt parameter:
// the smoothing constants are per-call, and audible playback blocks the
// render loop's idle throttle, so this always runs at full frame rate
// while it matters.
const analyseAndAnimate = () => {
  const state = stateRef.current
  const activeVrm = getVrm()
  const analyserNode = state.analyser
  if (!analyserNode || !activeVrm?.expressionManager) {
    return
  }

  let timeData = state.timeData
  if (!timeData) {
    timeData = allocateFloat32Array(analyserNode.fftSize)
    state.timeData = timeData
  }
  let freqData = state.freqData
  if (!freqData) {
    freqData = allocateFloat32Array(analyserNode.frequencyBinCount)
    state.freqData = freqData
  }

  const timeDomainArray = timeData
  const frequencyArray = freqData

  analyserNode.getFloatTimeDomainData(timeDomainArray)
  analyserNode.getFloatFrequencyData(frequencyArray)

  let sumSquares = 0
  for (let i = 0; i < timeDomainArray.length; i += 1) {
    const sample = timeDomainArray[i]
    sumSquares += sample * sample
  }
  const rms = Math.sqrt(sumSquares / timeDomainArray.length)
  const silenceThreshold = 0.012

  let targetViseme: (typeof lipSyncVisemes)[number] | null = null
  let intensity = 0

  if (Number.isFinite(rms) && rms > silenceThreshold) {
    const scaled = (rms - silenceThreshold) * 10
    intensity = Math.min(1, Math.pow(Math.max(0, scaled), 0.85))

    const sampleRate = state.audioContext?.sampleRate ?? analyserNode.context.sampleRate
    let centroidNumerator = 0
    let magnitudeSum = 0
    for (let i = 0; i < frequencyArray.length; i += 1) {
      const db = frequencyArray[i]
      const magnitude = Number.isFinite(db) ? Math.max(0, 10 ** (db / 20)) : 0
      if (magnitude <= 0) continue
      const freq = (i * sampleRate) / analyserNode.fftSize
      centroidNumerator += freq * magnitude
      magnitudeSum += magnitude
    }
    const centroid = magnitudeSum > 0 ? centroidNumerator / magnitudeSum : 0

    // Five coarse spectral-centroid bands, lowest to highest: the rounded
    // back vowels sit lowest (/u/ then /o/), /a/ in the middle, /i/ and
    // /e/ up top. Boundaries are heuristic (a centroid is not a formant);
    // they were re-split from the original four bands when 'oh' was added.
    if (centroid < 380) {
      targetViseme = 'ou'
    } else if (centroid < 700) {
      targetViseme = 'oh'
    } else if (centroid < 1100) {
      targetViseme = 'aa'
    } else if (centroid < 2200) {
      targetViseme = 'ih'
    } else {
      targetViseme = 'ee'
    }

    if (state.previousViseme && targetViseme !== state.previousViseme && intensity < 0.25) {
      targetViseme = state.previousViseme as (typeof lipSyncVisemes)[number]
    }
  }

  const smoothing = 0.35
  lipSyncVisemes.forEach((name) => {
    const target = targetViseme === name ? intensity : 0
    const previous = state.expressionValues[name] ?? 0
    const next = previous + (target - previous) * smoothing
    state.expressionValues[name] = next
    activeVrm.expressionManager?.setValue(name, next)
  })

  // Jaw-slack duck on 'relaxed' while speaking. The expression driver
  // yields ownership of 'relaxed' whenever isAudible() (see its
  // isLipSyncActive dep), so this write actually reaches the mesh now.
  // Ease from the manager's LIVE value, not our own shadow — between
  // playback sessions the driver moves 'relaxed' (mood baseline drift),
  // and easing from a stale shadow caused a visible pop on stream start.
  const relaxedTarget = targetViseme ? Math.max(0.1, 0.25 - intensity * 0.2) : 0.4
  const relaxedPrev =
    activeVrm.expressionManager?.getValue('relaxed') ?? state.expressionValues.relaxed ?? 0
  const relaxedNext = relaxedPrev + (relaxedTarget - relaxedPrev) * 0.2
  state.expressionValues.relaxed = relaxedNext
  activeVrm.expressionManager?.setValue('relaxed', relaxedNext)

  state.previousViseme = targetViseme
  // Item 2: raw amplitude for this frame, read (and independently
  // smoothed/decayed) by applySpeechMicroMotion in the main render loop.
  state.speechEnvelope = intensity
}

// Serializing the lip-sync graph (the core race): startLipSync (one-shot
// play) and beginStream both call stopLipSync() unconditionally and both
// have async gaps (context.resume, decodeAudioData) after which they touch
// shared state (stateRef.current, sourceNode/scheduledSources, streaming).
// A stale in-flight startLipSync() whose decode completes AFTER a newer
// beginStream() has already started its own stream must not tear that
// newer stream down — that used to make the whole reply silent (its chunks
// then hit enqueueLipSyncChunk's `!state.streaming` guard and drop).
//
// lipSyncGeneration (declared above, already used by the fade-stop logic)
// is the same mechanism extended here: every operation that takes
// ownership of the graph — stopLipSync's own teardown, plus startLipSync
// and beginStream once they've called stopLipSync and are about to start
// something new — captures/bumps the generation, and re-checks it after
// every await. If the generation has moved on, this operation lost the
// race and abandons SILENTLY: no stop(), no start(), no touching state at
// all (a newer operation already owns the graph and may already be
// mid-playback).
// Last-resort playback when the graph path is unavailable. MUST stop the
// current graph audio first and register the element on state: a bare,
// untracked Audio used to play on top of an in-flight stream (nothing could
// stop it), which was one source of "two voices at once".
const playFallbackAudio = async (base64Audio: string, mimeType: string) => {
  stopLipSync()
  const state = stateRef.current
  const audio = new Audio(`data:${mimeType};base64,${base64Audio}`)
  state.fallbackAudio = audio
  audio.onended = () => {
    if (state.fallbackAudio === audio) state.fallbackAudio = null
  }
  await audio.play()
}

const startLipSync = async (base64Audio: string, mimeType = 'audio/wav') => {
  const vrm = getVrm()
  if (!vrm?.expressionManager) {
    console.warn('VRM not ready for lip sync, using basic audio playback')
    await playFallbackAudio(base64Audio, mimeType)
    return
  }

  const context = ensureAudioContext()
  if (!context) {
    await playFallbackAudio(base64Audio, mimeType)
    return
  }

  if (context.state === 'suspended') {
    try {
      await context.resume()
    } catch (err) {
      console.warn('Failed to resume AudioContext:', err)
    }
  }

  const state = stateRef.current
  state.audioContext = context

  stopLipSync()
  // stopLipSync() itself bumps lipSyncGeneration on every call (teardown or
  // fresh start alike) — reading it right after is this call's claim on the
  // graph. Any later operation that also calls stopLipSync (or wins its own
  // race) bumps it further, which is exactly what the check below detects.
  const generation = lipSyncGeneration

  // decodeAudioData detaches the buffer it's handed, but this one is
  // freshly built per call and never reused — no defensive copy needed.
  const arrayBuffer = decodeBase64Audio(base64Audio)
  const audioBuffer = await context.decodeAudioData(arrayBuffer)

  // A newer operation (another startLipSync/beginStream call) already
  // claimed the graph while this decode was in flight — abandon without
  // touching state; it may already be mid-playback.
  if (lipSyncGeneration !== generation) {
    diagnostics.increment('voice.lipsync_superseded')
    return
  }

  const { analyser } = ensureLipSyncGraph(context)

  const source = context.createBufferSource()
  source.buffer = audioBuffer
  source.connect(analyser)
  // sourceNode is what isAudible() reads for the one-shot path — it MUST
  // be cleared whenever this clip stops being audible, or filler/presence
  // guards see a phantom voice forever. onended → stopLipSync covers the
  // normal end; the catch below covers a start() failure.
  state.sourceNode = source

  try {
    source.start()
  } catch (err) {
    console.warn('Failed to start lip sync source:', err)
    // Only tear down if this operation still owns the graph — a start()
    // failure after being superseded must not stop a newer stream either.
    if (lipSyncGeneration === generation) stopLipSync()
    return
  }
  source.onended = () => {
    if (lipSyncGeneration === generation) stopLipSync()
  }
}

// --- Chunked-stream playback API (window.kurisuLipSync.beginStream /
// enqueue / endStream) -----------------------------------------------
//
// Sentences arrive one WAV clip at a time as the backend synthesizes
// them; each is scheduled to start exactly when the previous one ends
// (or immediately, if TTS has kept up and there's no gap), all feeding
// the same analyser graph so the viseme loop runs continuously across
// the whole reply instead of restarting — and cutting out — every
// sentence boundary.
// Ordering gate for enqueueLipSyncChunk: chunks DECODE concurrently, but
// each one must SCHEDULE only after every earlier-enqueued chunk has
// scheduled. Without this, two chunks decoding at once (a long sentence-1
// WAV + a short sentence-2 WAV arriving in the same WS burst) could resolve
// inverted and play swapped — enqueue order is the ordering authority here,
// not `seq`: the server emits audio frames in strictly ascending seq and the
// WS delivers them in order, so enqueue-call order IS seq order, and keying
// on arrival (instead of buffering on seq numbers) means a gap — a sentence
// whose synthesis failed server-side never produces a frame — simply never
// enters the chain and cannot deadlock it. Reset per beginStream so a new
// stream never waits on a dead stream's in-flight decodes; each link is
// .catch-wrapped so one failed decode can't wedge every later chunk.
let streamScheduleChain: Promise<void> = Promise.resolve()

const beginStream = async () => {
  const context = ensureAudioContext()
  if (!context) return
  const state = stateRef.current
  state.audioContext = context
  stopLipSync()
  streamScheduleChain = Promise.resolve()
  // stopLipSync() bumps lipSyncGeneration itself — reading it right after
  // is this call's claim on the graph (same mechanism as startLipSync).
  const generation = lipSyncGeneration
  ensureLipSyncGraph(context)
  // Mark the stream live synchronously — the caller enqueues the first
  // chunk on the same tick as beginStream(), and enqueue drops chunks
  // whenever `streaming` is false (dead-stream guard). Awaiting resume()
  // first left a window where the turn's opening audio was silently
  // eaten. Scheduling onto a still-suspended context is safe: sources
  // start once resume() completes.
  state.streaming = true
  state.streamDraining = false
  state.nextChunkStartTime = context.currentTime
  // No analyser loop to start: the graph ensureLipSyncGraph just built is
  // picked up by the next render-loop tick (controller.update()).
  dispatchSpeakingChange(true)
  if (context.state === 'suspended') {
    try {
      await context.resume()
    } catch (err) {
      console.warn('Failed to resume AudioContext:', err)
    }
  }
  // A newer operation preempted this stream while resume() was in flight
  // (e.g. a fast barge-in immediately followed by another beginStream) —
  // its own stopLipSync/setup already ran; this call must not touch state
  // any further (nothing to undo — the state writes above already
  // happened, but this generation no longer owns them going forward).
  if (lipSyncGeneration !== generation) {
    diagnostics.increment('voice.lipsync_superseded')
  }
}

// B3: per-emotion detune (cents) applied to each chunk's playback pitch
// — a small, cheap way to differentiate happy/surprised (brighter, +15c)
// from sad (flatter, -15c) without a second TTS pass.
const detuneForEmotion = (emotion?: string): number => {
  const normalized = emotion?.trim().toLowerCase()
  if (normalized === 'happy' || normalized === 'surprised') return 15
  if (normalized === 'sad') return -15
  return 0
}

// No mimeType param: decodeAudioData sniffs the container itself, and the
// standalone-Audio fallback that needed it is gone (late chunks from a
// dead stream are dropped instead of played over the superseding audio).
const enqueueLipSyncChunk = async (
  base64Audio: string,
  seq?: number,
  emotion?: string,
  beat?: string
) => {
  const state = stateRef.current
  if (!state.streaming) {
    // The stream this chunk belongs to was preempted (barge-in, a new
    // turn, or a presence line's one-shot play() tearing the graph down)
    // after the chunk left the server. A late chunk from a dead stream
    // must be dropped — scheduling it (or playing it via a standalone
    // Audio element, as this fallback once did) plays it over whatever
    // superseded it: two voices at once.
    return
  }
  const context = state.audioContext
  if (!context || !state.analyser) {
    return
  }
  // Claim: this chunk belongs to the stream that is live RIGHT NOW. The
  // `state.streaming` re-check below is not enough by itself — a barge-in
  // followed quickly by a new turn flips streaming false→true while this
  // chunk's decode is in flight, and the old turn's chunk then scheduled
  // itself into the NEW turn's timeline (a sentence from the cancelled
  // reply playing mid-way through the next one).
  const generation = lipSyncGeneration

  // decodeAudioData detaches the buffer it's handed, but this one is
  // freshly built per chunk and never reused — the defensive .slice(0)
  // copy this hot path used to make was pure overhead. The decode starts
  // NOW (concurrent with earlier chunks' decodes); only scheduling waits.
  const arrayBuffer = decodeBase64Audio(base64Audio)
  const decodePromise = context.decodeAudioData(arrayBuffer)

  const previous = streamScheduleChain
  const link = scheduleAfter(previous, decodePromise, state, generation, seq, emotion, beat)
  // The chain link never rejects (a failed decode must not wedge every
  // later chunk); the caller-visible promise still does.
  streamScheduleChain = link.catch(() => undefined)
  return link
}

// Body of one ordered chunk-schedule (see streamScheduleChain): awaits its
// own decode plus every earlier chunk's scheduling, re-validates the stream
// claim, then schedules — byte-identical scheduling math to the pre-ordering
// code, just moved behind the chain.
const scheduleAfter = async (
  previous: Promise<void>,
  decodePromise: Promise<AudioBuffer>,
  state: LipSyncState,
  generation: number,
  seq?: number,
  emotion?: string,
  beat?: string
) => {
  const audioBuffer = await decodePromise
  // Earlier-enqueued chunks must finish scheduling first (they may still be
  // decoding — the whole point of the chain). `previous` is already
  // .catch-wrapped, so an earlier failed decode falls through here.
  await previous

  const context = state.audioContext
  // The graph may have been torn down (stop()/new turn) while this
  // chunk was decoding — drop it rather than scheduling onto a stale
  // or foreign graph.
  if (
    !context ||
    stateRef.current !== state ||
    !state.analyser ||
    !state.streaming ||
    lipSyncGeneration !== generation
  ) {
    return
  }

  const source = context.createBufferSource()
  source.buffer = audioBuffer
  source.detune.value = detuneForEmotion(emotion)
  source.connect(state.analyser)

  // The 50ms safety pad exists so the FIRST chunk of a stream starts
  // cleanly instead of racing "schedule at exactly currentTime". It must
  // only apply on a fresh start (nothing scheduled, nextChunkStartTime
  // stale/past): re-applying it to every chunk that arrived after its
  // predecessor finished — including a predecessor within 50ms of its
  // natural end — inserted an audible gap per chunk and cumulative drift.
  // Back-to-back chunks chain at nextChunkStartTime exactly (clamped to
  // now, since starting in the past is a WebAudio error).
  const freshStart =
    state.scheduledSources.length === 0 && state.nextChunkStartTime <= context.currentTime
  const startAt = freshStart
    ? context.currentTime + 0.05
    : Math.max(context.currentTime, state.nextChunkStartTime)
  source.start(startAt)
  // detune changes playback RATE (pitch and speed together): a -15c "sad"
  // chunk plays ~0.87% longer than buffer.duration, so chaining the next
  // chunk at the un-scaled duration started it before this one finished —
  // a per-chunk sliver of overlapping voice that accumulated over long
  // emotional replies. Scale by the actual playback rate.
  const playbackRate = Math.pow(2, source.detune.value / 1200)
  state.nextChunkStartTime = startAt + audioBuffer.duration / playbackRate
  state.scheduledSources.push(source)

  // B3 (karaoke timing): fire a DOM event at the moment this chunk
  // actually starts playing so callers (e.g. a future subtitle/karaoke
  // UI) can sync to it. setTimeout tolerance is fine per spec (tens of
  // ms) — precise sub-frame sync isn't the goal here. `beat` (the
  // sentence's speech-beat classification, vrm/speechBeats) rides the
  // same event so the beat gesture lands with the voice.
  if (seq !== undefined) {
    const delayMs = Math.max(0, (startAt - context.currentTime) * 1000)
    const timeoutId = window.setTimeout(() => {
      state.sentenceStartTimeoutIds = state.sentenceStartTimeoutIds.filter((id) => id !== timeoutId)
      window.dispatchEvent(new CustomEvent('kurisu:sentence-audio-start', { detail: { seq, beat } }))
    }, delayMs)
    state.sentenceStartTimeoutIds.push(timeoutId)
  }

  source.onended = () => {
    state.scheduledSources = state.scheduledSources.filter((s) => s !== source)
    if (state.streamDraining && state.scheduledSources.length === 0) {
      stopLipSync()
    }
  }
}

const endLipSyncStream = () => {
  const state = stateRef.current
  state.streamDraining = true
  if (state.scheduledSources.length === 0) {
    stopLipSync()
  }
}

// Live app-wide volume update (item 4): mirrors the existing
// handleVolumeChange pattern for the settings-panel voice slider
// (VRMViewer's 'kurisu:volume-change' listener) — re-reads
// loadAppVolume() and pushes it straight to the gain node if the graph
// happens to be live right now. A no-op (nothing to update yet) when no
// playback session is in progress; the node picks up the current
// localStorage value from loadAppVolume() the next time
// ensureLipSyncGraph builds it regardless.
const applyAppVolume = () => {
  const state = stateRef.current
  if (state.appVolumeGain) {
    state.appVolumeGain.gain.value = loadAppVolume()
  }
}

  return {
    ensureAudioContext,
    resetLipSyncExpressions,
    /** Re-reads localStorage `kurisu_volume` and applies it live — call
     * from a 'kurisu-volume-changed' window listener. */
    applyAppVolume,
    /** Per-frame viseme analysis tick — call from the render loop. */
    update: analyseAndAnimate,
    /** True while ANY voice audio is live: stream, drain/fade tail, or a
     * one-shot presence clip. The bridge's isSpeaking must use this, not
     * `state.streaming` (see the isAudible definition for why). */
    isAudible,
    stopLipSync,
    fadeOutAndStopLipSync,
    startLipSync,
    beginStream,
    enqueueLipSyncChunk,
    endLipSyncStream,
  }
}
