/**
 * Audio identity: everything here is synthesized on the fly via WebAudio —
 * no shipped audio assets, so no licensing concerns and no bundle weight.
 * Three independent buses (voice/fx/ambience) so a user can duck the lab
 * hum without muting Kurisu, or vice versa. Voice volume is stored here but
 * applied by VRMViewer's own lip-sync gain node (that graph is
 * created/torn down per-utterance and already owns the speech signal path);
 * fx and ambience live entirely in this module's own persistent context.
 */

const VOLUME_KEY = 'kurisu_volume_v1'
const AMBIENCE_ENABLED_KEY = 'kurisu_ambience_enabled'

export interface VolumePrefs {
  voice: number
  fx: number
  ambience: number
}

const DEFAULT_VOLUMES: VolumePrefs = { voice: 1.0, fx: 0.5, ambience: 0.25 }

const clamp01 = (value: number) => Math.min(1, Math.max(0, value))

const isLikelyTouchDevice = (): boolean => {
  if (typeof window === 'undefined') return false
  const nav = typeof navigator !== 'undefined' ? navigator : undefined
  return (
    'ontouchstart' in window ||
    Boolean(nav && 'maxTouchPoints' in nav && nav.maxTouchPoints > 0)
  )
}

export const loadVolumePrefs = (): VolumePrefs => {
  if (typeof window === 'undefined') return { ...DEFAULT_VOLUMES }
  try {
    const raw = window.localStorage.getItem(VOLUME_KEY)
    if (!raw) return { ...DEFAULT_VOLUMES }
    const parsed = JSON.parse(raw)
    return {
      voice: typeof parsed.voice === 'number' ? clamp01(parsed.voice) : DEFAULT_VOLUMES.voice,
      fx: typeof parsed.fx === 'number' ? clamp01(parsed.fx) : DEFAULT_VOLUMES.fx,
      ambience:
        typeof parsed.ambience === 'number' ? clamp01(parsed.ambience) : DEFAULT_VOLUMES.ambience,
    }
  } catch {
    return { ...DEFAULT_VOLUMES }
  }
}

/** Ambience defaults off on touch devices (same reasoning ChatInterface
 * already applies to audioEnabled: unexpected sound on a phone/tablet reads
 * as broken, not atmospheric), on everywhere else. */
export const loadAmbienceEnabled = (): boolean => {
  if (typeof window === 'undefined') return false
  const stored = window.localStorage.getItem(AMBIENCE_ENABLED_KEY)
  if (stored !== null) return stored === 'true'
  return !isLikelyTouchDevice()
}

export const saveAmbienceEnabled = (enabled: boolean): void => {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(AMBIENCE_ENABLED_KEY, String(enabled))
}

export const saveVolumePrefs = (prefs: VolumePrefs): void => {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(VOLUME_KEY, JSON.stringify(prefs))
  } catch {
    // Non-fatal — losing a volume preference isn't worth surfacing an error.
  }
}

type SoundKind = 'send' | 'receive' | 'error' | 'speak-start'

export type AmbiencePreset = 'machine' | 'street'

/** How far the ambience bus drops while Kurisu speaks (~-6dB), the game-audio
 * "duck music under dialogue" staple. */
const DUCK_FACTOR = 0.5

class Soundscape {
  private context: AudioContext | null = null
  private fxGain: GainNode | null = null
  private ambienceGain: GainNode | null = null
  private ducked = false
  private ambienceNodes: { source: AudioBufferSourceNode; lfo: OscillatorNode; filter: BiquadFilterNode } | null = null
  private activePreset: AmbiencePreset = 'machine'
  // Sparse distant-event blips ('street' preset only) — a recurring
  // setTimeout chain, torn down alongside the rest of the ambience graph.
  private blipTimeoutId: number | null = null
  private prefs: VolumePrefs = loadVolumePrefs()

  /** Must be called from a user-gesture handler the first time (send click,
   * mute toggle, etc.) — browsers block AudioContext until then. Safe to
   * call repeatedly; only constructs the graph once. */
  private ensureContext(): AudioContext | null {
    if (this.context) return this.context
    const AudioCtx = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!AudioCtx) return null
    const context = new AudioCtx()
    const fxGain = context.createGain()
    fxGain.gain.value = this.prefs.fx
    fxGain.connect(context.destination)

    const ambienceGain = context.createGain()
    ambienceGain.gain.value = this.effectiveAmbience()
    ambienceGain.connect(context.destination)

    this.context = context
    this.fxGain = fxGain
    this.ambienceGain = ambienceGain
    return context
  }

  getVolumePrefs(): VolumePrefs {
    return { ...this.prefs }
  }

  private effectiveAmbience(): number {
    return this.prefs.ambience * (this.ducked ? DUCK_FACTOR : 1)
  }

  /** Sidechain-style ducking: fast attack when speech starts (ambience gets
   * out of the way before the first word lands), slow release when it ends
   * (a snap back up right after the last syllable reads as a glitch). */
  setDucked(ducked: boolean): void {
    if (this.ducked === ducked) return
    this.ducked = ducked
    if (!this.context || !this.ambienceGain) return
    const now = this.context.currentTime
    const gain = this.ambienceGain.gain
    gain.cancelScheduledValues(now)
    // setTargetAtTime time constants: ~63% of the way per constant, so these
    // land within ~3x the constant — ~90ms attack, ~750ms release.
    gain.setTargetAtTime(this.effectiveAmbience(), now, ducked ? 0.03 : 0.25)
  }

  setVolume(bus: keyof VolumePrefs, value: number): void {
    const clamped = clamp01(value)
    this.prefs = { ...this.prefs, [bus]: clamped }
    saveVolumePrefs(this.prefs)
    if (bus === 'fx' && this.fxGain) this.fxGain.gain.value = clamped
    if (bus === 'ambience' && this.ambienceGain && this.context) {
      // Respect any active duck, and clear pending duck ramps so the user's
      // slider drag wins immediately.
      this.ambienceGain.gain.cancelScheduledValues(this.context.currentTime)
      this.ambienceGain.gain.value = this.effectiveAmbience()
    }
    if (bus === 'voice' && typeof window !== 'undefined') {
      // VRMViewer owns the actual voice gain node (its lip-sync graph is
      // built/torn down per utterance, independent of this module's
      // persistent context) — notify it to pick up the new value live.
      window.dispatchEvent(new Event('kurisu:volume-change'))
    }
  }

  /** Short synthesized blip. Each kind gets a distinct pitch/envelope so
   * they're distinguishable by ear without needing to look at the screen. */
  play(kind: SoundKind): void {
    const context = this.ensureContext()
    if (!context || !this.fxGain || this.prefs.fx <= 0) return

    const now = context.currentTime
    const osc = context.createOscillator()
    const envelope = context.createGain()
    osc.connect(envelope)
    envelope.connect(this.fxGain)

    const specs: Record<SoundKind, { freq: number; freqEnd?: number; duration: number; type: OscillatorType }> = {
      send: { freq: 660, freqEnd: 880, duration: 0.09, type: 'sine' },
      receive: { freq: 880, freqEnd: 660, duration: 0.11, type: 'sine' },
      'speak-start': { freq: 520, duration: 0.06, type: 'sine' },
      error: { freq: 220, freqEnd: 160, duration: 0.22, type: 'triangle' },
    }
    const spec = specs[kind]
    osc.type = spec.type
    osc.frequency.setValueAtTime(spec.freq, now)
    if (spec.freqEnd) {
      osc.frequency.exponentialRampToValueAtTime(spec.freqEnd, now + spec.duration)
    }
    envelope.gain.setValueAtTime(0.0001, now)
    envelope.gain.exponentialRampToValueAtTime(0.18, now + 0.012)
    envelope.gain.exponentialRampToValueAtTime(0.0001, now + spec.duration)

    osc.start(now)
    osc.stop(now + spec.duration + 0.02)
  }

  /** Low, filtered-noise room tone with a slow amplitude LFO so it reads as
   * ambient machinery/street rather than a flat, obviously-looping tone.
   * Idempotent — calling this while already running (with the same preset)
   * is a no-op; switching presets while running is handled by
   * setAmbiencePreset, not by calling this again.
   *
   * `preset` defaults to 'machine' — the original (and only) ambience this
   * method ever produced — so the existing no-arg call sites (ChatInterface)
   * keep behaving exactly as before. */
  startAmbience(preset: AmbiencePreset = 'machine'): void {
    const context = this.ensureContext()
    if (!context || !this.ambienceGain || this.ambienceNodes) return
    this.activePreset = preset

    const bufferSeconds = 2
    const buffer = context.createBuffer(1, context.sampleRate * bufferSeconds, context.sampleRate)
    const data = buffer.getChannelData(0)
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.random() * 2 - 1
    }
    const source = context.createBufferSource()
    source.buffer = buffer
    source.loop = true

    const filter = context.createBiquadFilter()
    filter.type = 'lowpass'
    filter.frequency.value = preset === 'street' ? 600 : 120
    filter.Q.value = 0.7

    const lfo = context.createOscillator()
    // 'machine': ~14s cycle, subtle depth — reads as room tone.
    // 'street': slower/deeper — a lazier, more open-air swell.
    lfo.frequency.value = preset === 'street' ? 0.03 : 0.07
    const lfoGain = context.createGain()
    lfoGain.gain.value = preset === 'street' ? 0.2 : 0.35
    lfo.connect(lfoGain)

    const level = context.createGain()
    level.gain.value = 0.5
    lfoGain.connect(level.gain)

    source.connect(filter)
    filter.connect(level)
    level.connect(this.ambienceGain)

    source.start()
    lfo.start()
    this.ambienceNodes = { source, lfo, filter }

    if (preset === 'street') {
      this.scheduleNextBlip(context)
    }
  }

  /** Sparse, stereo-panned distant "city" blips for the 'street' preset —
   * a soft, short filtered-noise chirp panned to a random position, fired
   * every 20-40s. Self-rescheduling; cleared in stopAmbience. */
  private scheduleNextBlip(context: AudioContext): void {
    const delayMs = 20000 + Math.random() * 20000
    this.blipTimeoutId = window.setTimeout(() => {
      this.playDistantBlip(context)
      if (this.ambienceNodes && this.activePreset === 'street') {
        this.scheduleNextBlip(context)
      }
    }, delayMs)
  }

  private playDistantBlip(context: AudioContext): void {
    if (!this.ambienceGain) return
    const now = context.currentTime
    const duration = 0.3 + Math.random() * 0.3

    const bufferSize = Math.max(1, Math.floor(context.sampleRate * duration))
    const buffer = context.createBuffer(1, bufferSize, context.sampleRate)
    const data = buffer.getChannelData(0)
    for (let i = 0; i < bufferSize; i++) {
      const decay = 1 - i / bufferSize
      data[i] = (Math.random() * 2 - 1) * decay
    }

    const source = context.createBufferSource()
    source.buffer = buffer

    const filter = context.createBiquadFilter()
    filter.type = 'bandpass'
    filter.frequency.value = 300 + Math.random() * 500
    filter.Q.value = 1.2

    const envelope = context.createGain()
    envelope.gain.setValueAtTime(0.0001, now)
    envelope.gain.exponentialRampToValueAtTime(0.22, now + duration * 0.15)
    envelope.gain.exponentialRampToValueAtTime(0.0001, now + duration)

    const panner = context.createStereoPanner ? context.createStereoPanner() : null
    source.connect(filter)
    if (panner) {
      panner.pan.value = Math.random() * 2 - 1
      filter.connect(envelope)
      envelope.connect(panner)
      panner.connect(this.ambienceGain)
    } else {
      filter.connect(envelope)
      envelope.connect(this.ambienceGain)
    }

    source.start(now)
    source.stop(now + duration + 0.05)
  }

  /** Live-swap the ambience preset without an audible pop: tears down and
   * rebuilds the ambience graph in place. No-op if ambience isn't running
   * (the next startAmbience call will just use the new preset directly). */
  setAmbiencePreset(preset: AmbiencePreset): void {
    if (this.activePreset === preset && this.ambienceNodes) return
    const wasRunning = Boolean(this.ambienceNodes)
    if (wasRunning) {
      this.stopAmbience()
      this.startAmbience(preset)
    } else {
      this.activePreset = preset
    }
  }

  stopAmbience(): void {
    if (this.blipTimeoutId !== null) {
      window.clearTimeout(this.blipTimeoutId)
      this.blipTimeoutId = null
    }
    if (!this.ambienceNodes) return
    try {
      this.ambienceNodes.source.stop()
      this.ambienceNodes.lfo.stop()
    } catch {
      // Already stopped — fine.
    }
    this.ambienceNodes = null
  }
}

export const soundscape = new Soundscape()

// Auto-duck ambience under her voice: VRMViewer dispatches 'kurisu:speaking'
// on lip-sync stream start/stop (see dispatchSpeakingChange), so the duck
// tracks actual audio playback — including barge-in cuts — not chat state.
if (typeof window !== 'undefined') {
  window.addEventListener('kurisu:speaking', (event) => {
    const speaking = Boolean((event as CustomEvent<{ speaking?: boolean }>).detail?.speaking)
    soundscape.setDucked(speaking)
  })
}
