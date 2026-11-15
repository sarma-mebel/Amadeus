/**
 * Continuous PCM16 mono mic capture for the full-duplex voice loop
 * (PROTOTYPE, opt-in — see duplexTurn.ts). Unlike MicRecorder (which
 * buffers a whole utterance client-side and hands back one WAV blob for a
 * one-shot POST /api/stt), this class streams raw PCM16 frames continuously
 * to a callback for as long as capture is active — endpoint detection runs
 * server-side (amadeus/web/streaming/duplex.py::DuplexController), so the
 * client never decides where an utterance ends.
 *
 * Same getUserMedia constraints as every other capture path in this app
 * (echoCancellation/noiseSuppression/autoGainControl) — her own audio must
 * be suppressed from the input before it reaches the graph, since barge-in
 * detection runs entirely server-side off whatever this class sends
 * (including while she is speaking; the client keeps streaming
 * unconditionally, exactly as the task's barge-in design requires).
 *
 * Uses a ScriptProcessorNode (same choice as MicRecorder — the simplest
 * reliable cross-browser capture path; an AudioWorklet module would need a
 * separate build asset for marginal benefit here) resampled to 16kHz
 * PCM16-LE, matching DuplexController's default sample rate and the STT
 * pipeline's expected format.
 */

export const DUPLEX_SAMPLE_RATE = 16000

/** Resample a Float32 buffer (any input rate) to the target rate via simple
 * linear interpolation, then quantize to PCM16-LE. Mirrors
 * micRecorder.ts's encodeWavFromFloat32 resampling, minus the WAV header —
 * duplex frames are raw PCM, framed by the WebSocket message boundary
 * itself rather than a container format. */
export const resampleToPcm16 = (samples: Float32Array, inputSampleRate: number, targetSampleRate: number): ArrayBuffer => {
  const ratio = inputSampleRate / targetSampleRate
  const outputLength = Math.max(1, Math.round(samples.length / ratio))
  const pcm16 = new Int16Array(outputLength)
  for (let i = 0; i < outputLength; i++) {
    const sourceIndex = i * ratio
    const indexFloor = Math.floor(sourceIndex)
    const indexCeil = Math.min(indexFloor + 1, samples.length - 1)
    const frac = sourceIndex - indexFloor
    const a = samples[indexFloor] ?? 0
    const b = samples[indexCeil] ?? 0
    const value = a + (b - a) * frac
    const clamped = Math.max(-1, Math.min(1, value))
    pcm16[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff
  }
  return pcm16.buffer
}

export class DuplexMic {
  /** Fired once per capture buffer with a raw PCM16-LE frame at
   * DUPLEX_SAMPLE_RATE, ready to send as a WebSocket binary frame. */
  onFrame: ((frame: ArrayBuffer) => void) | null = null
  /** Optional level tap (RMS 0..1) for a UI waveform — same shape as
   * MicRecorder/MicMonitor's onLevel, so callers can reuse levelBus. */
  onLevel: ((rms: number, dtMs: number) => void) | null = null

  private stream: MediaStream | null = null
  private audioContext: AudioContext | null = null
  private sourceNode: MediaStreamAudioSourceNode | null = null
  private processorNode: ScriptProcessorNode | null = null

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    })
    const AudioCtx =
      window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!AudioCtx) throw new Error('AudioContext not supported')
    this.audioContext = new AudioCtx()
    this.sourceNode = this.audioContext.createMediaStreamSource(this.stream)
    // 4096 samples: same buffer size as MicRecorder, balancing main-thread
    // load against the extra latency a bigger buffer would add to
    // server-side endpointing (each buffer is one DuplexController.feed()
    // call — the smaller the buffer, the finer-grained the VAD decisions).
    this.processorNode = this.audioContext.createScriptProcessor(4096, 1, 1)
    const captureSampleRate = this.audioContext.sampleRate
    this.processorNode.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0)
      if (input.length === 0) return
      if (this.onLevel) {
        let sumSquares = 0
        for (let i = 0; i < input.length; i++) sumSquares += input[i] * input[i]
        this.onLevel(Math.sqrt(sumSquares / input.length), (input.length / captureSampleRate) * 1000)
      }
      if (this.onFrame) {
        const pcm16 = resampleToPcm16(input, captureSampleRate, DUPLEX_SAMPLE_RATE)
        this.onFrame(pcm16)
      }
    }
    this.sourceNode.connect(this.processorNode)
    // Must connect to a destination for onaudioprocess to fire in all
    // browsers; a zero-gain node keeps captured audio inaudible.
    const silentGain = this.audioContext.createGain()
    silentGain.gain.value = 0
    this.processorNode.connect(silentGain)
    silentGain.connect(this.audioContext.destination)
    if (this.audioContext.state === 'suspended') {
      void this.audioContext.resume().catch(() => undefined)
    }
  }

  stop(): void {
    this.onFrame = null
    this.onLevel = null
    try {
      this.processorNode?.disconnect()
    } catch {
      // Already torn down.
    }
    try {
      this.sourceNode?.disconnect()
    } catch {
      // Already torn down.
    }
    this.processorNode = null
    this.sourceNode = null
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop()
      this.stream = null
    }
    if (this.audioContext && this.audioContext.state !== 'closed') {
      void this.audioContext.close().catch(() => undefined)
    }
    this.audioContext = null
  }
}
