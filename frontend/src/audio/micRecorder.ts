/** Push-to-talk mic capture: getUserMedia + ScriptProcessorNode recording
 * and in-browser WAV encoding (16kHz mono 16-bit PCM), so the server only
 * ever parses a WAV header — no MediaRecorder webm/opus, no ffmpeg. */

export const STT_SAMPLE_RATE = 16000

/** Encodes a mono Float32 PCM buffer (any input sample rate) into a 16kHz
 * mono 16-bit PCM WAV blob, entirely in-browser. Avoids MediaRecorder's
 * webm/opus output, which would need ffmpeg server-side to decode. */
export const encodeWavFromFloat32 = (samples: Float32Array, inputSampleRate: number): Blob => {
  // Simple linear resampling down (or up) to the target rate.
  const ratio = inputSampleRate / STT_SAMPLE_RATE
  const outputLength = Math.max(1, Math.round(samples.length / ratio))
  const resampled = new Float32Array(outputLength)
  for (let i = 0; i < outputLength; i++) {
    const sourceIndex = i * ratio
    const indexFloor = Math.floor(sourceIndex)
    const indexCeil = Math.min(indexFloor + 1, samples.length - 1)
    const frac = sourceIndex - indexFloor
    const a = samples[indexFloor] ?? 0
    const b = samples[indexCeil] ?? 0
    resampled[i] = a + (b - a) * frac
  }

  const pcm16 = new Int16Array(resampled.length)
  for (let i = 0; i < resampled.length; i++) {
    const clamped = Math.max(-1, Math.min(1, resampled[i]))
    pcm16[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff
  }

  const blockAlign = 2 // mono, 16-bit
  const byteRate = STT_SAMPLE_RATE * blockAlign
  const dataSize = pcm16.length * 2
  const buffer = new ArrayBuffer(44 + dataSize)
  const view = new DataView(buffer)

  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i))
    }
  }

  writeString(0, 'RIFF')
  view.setUint32(4, 36 + dataSize, true)
  writeString(8, 'WAVE')
  writeString(12, 'fmt ')
  view.setUint32(16, 16, true) // PCM fmt chunk size
  view.setUint16(20, 1, true) // audio format = PCM
  view.setUint16(22, 1, true) // channels = mono
  view.setUint32(24, STT_SAMPLE_RATE, true)
  view.setUint32(28, byteRate, true)
  view.setUint16(32, blockAlign, true)
  view.setUint16(34, 16, true) // bits per sample
  writeString(36, 'data')
  view.setUint32(40, dataSize, true)

  let offset = 44
  for (let i = 0; i < pcm16.length; i++) {
    view.setInt16(offset, pcm16[i], true)
    offset += 2
  }

  return new Blob([buffer], { type: 'audio/wav' })
}

/** Captures mic audio via getUserMedia + ScriptProcessorNode (the simplest
 * reliable cross-browser capture path — an AudioWorklet module would need a
 * separate build asset for marginal benefit here) and hands back a WAV blob
 * once stopped. */
export class MicRecorder {
  private stream: MediaStream | null = null
  private audioContext: AudioContext | null = null
  private sourceNode: MediaStreamAudioSourceNode | null = null
  private processorNode: ScriptProcessorNode | null = null
  private chunks: Float32Array[] = []

  /** Optional level tap for hands-free endpointing: called once per capture
   * buffer with the buffer's RMS level (0..1) and its duration in ms.
   * Additive — push-to-talk callers simply leave it null. */
  onLevel: ((rms: number, dtMs: number) => void) | null = null

  async start(): Promise<void> {
    // Echo cancellation matters here: her own audio can overlap a capture
    // (speech tail on barge-in, listening backchannels while the user
    // talks), and without EC it lands in the recording and gets
    // transcribed as if the user said it.
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    })
    const AudioCtx = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!AudioCtx) throw new Error('AudioContext not supported')
    this.audioContext = new AudioCtx()
    this.sourceNode = this.audioContext.createMediaStreamSource(this.stream)
    // 4096-sample buffer: large enough to avoid glitching on a busy main
    // thread, small enough not to add noticeable latency to capture.
    this.processorNode = this.audioContext.createScriptProcessor(4096, 1, 1)
    this.chunks = []
    const captureSampleRate = this.audioContext.sampleRate
    this.processorNode.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0)
      this.chunks.push(new Float32Array(input))
      if (this.onLevel && input.length > 0) {
        let sumSquares = 0
        for (let i = 0; i < input.length; i++) {
          sumSquares += input[i] * input[i]
        }
        this.onLevel(Math.sqrt(sumSquares / input.length), (input.length / captureSampleRate) * 1000)
      }
    }
    this.sourceNode.connect(this.processorNode)
    // Must connect to a destination for onaudioprocess to fire in all
    // browsers; a zero-gain node keeps captured audio inaudible.
    const silentGain = this.audioContext.createGain()
    silentGain.gain.value = 0
    this.processorNode.connect(silentGain)
    silentGain.connect(this.audioContext.destination)
    // Push-to-talk always starts from a user gesture, so the context is
    // already running and this is a no-op. Hands-free may start capture
    // without a gesture (persisted toggle on page load); an active
    // getUserMedia capture exempts the context from the autoplay policy,
    // but nudge it explicitly in case it came up suspended.
    if (this.audioContext.state === 'suspended') {
      void this.audioContext.resume().catch(() => undefined)
    }
  }

  /** Hands-free: drop buffered audio, keeping only the trailing keepMs, while
   * capture keeps running. Called while the endpointer sits in 'idle' so the
   * always-open mic holds a short pre-roll instead of growing unboundedly
   * (the pre-roll keeps the first syllable, heard before the voice-onset
   * gate confirms, in the utterance). Additive — push-to-talk never calls
   * this, so PTT behavior is untouched. */
  trimBuffer(keepMs: number): void {
    if (this.chunks.length === 0) return
    if (keepMs <= 0) {
      this.chunks = []
      return
    }
    const sampleRate = this.audioContext?.sampleRate ?? STT_SAMPLE_RATE
    const keepSamples = Math.ceil((keepMs / 1000) * sampleRate)
    let kept = 0
    let firstKept = this.chunks.length
    for (let i = this.chunks.length - 1; i >= 0; i--) {
      kept += this.chunks[i].length
      firstKept = i
      if (kept >= keepSamples) break
    }
    if (firstKept > 0) this.chunks = this.chunks.slice(firstKept)
  }

  /** Hands-free: snapshot the buffered audio as a WAV blob and clear the
   * buffer WITHOUT stopping the stream — the mic stays open for the next
   * utterance. Additive sibling of stop() (which tears the capture down). */
  takeWav(): Blob | null {
    if (this.chunks.length === 0) return null
    const sourceRate = this.audioContext?.sampleRate ?? STT_SAMPLE_RATE
    const totalLength = this.chunks.reduce((sum, chunk) => sum + chunk.length, 0)
    const merged = new Float32Array(totalLength)
    let offset = 0
    for (const chunk of this.chunks) {
      merged.set(chunk, offset)
      offset += chunk.length
    }
    this.chunks = []
    return encodeWavFromFloat32(merged, sourceRate)
  }

  /** Stops capture and returns the recorded audio as a WAV blob, or null if
   * nothing was recorded. */
  async stop(): Promise<Blob | null> {
    const context = this.audioContext
    const sourceRate = context?.sampleRate ?? STT_SAMPLE_RATE
    this.processorNode?.disconnect()
    this.sourceNode?.disconnect()
    this.stream?.getTracks().forEach((track) => track.stop())
    if (context && context.state !== 'closed') {
      await context.close().catch(() => undefined)
    }
    this.stream = null
    this.audioContext = null
    this.sourceNode = null
    this.processorNode = null

    if (this.chunks.length === 0) return null
    const totalLength = this.chunks.reduce((sum, chunk) => sum + chunk.length, 0)
    const merged = new Float32Array(totalLength)
    let offset = 0
    for (const chunk of this.chunks) {
      merged.set(chunk, offset)
      offset += chunk.length
    }
    this.chunks = []
    return encodeWavFromFloat32(merged, sourceRate)
  }

  cancel(): void {
    this.processorNode?.disconnect()
    this.sourceNode?.disconnect()
    this.stream?.getTracks().forEach((track) => track.stop())
    if (this.audioContext && this.audioContext.state !== 'closed') {
      void this.audioContext.close().catch(() => undefined)
    }
    this.stream = null
    this.audioContext = null
    this.sourceNode = null
    this.processorNode = null
    this.chunks = []
  }
}
