/**
 * Low-power echo-cancelled microphone level monitor.
 *
 * NEVER records: an AnalyserNode is polled for RMS on an interval; no PCM
 * leaves the audio graph, nothing is buffered, encoded, or sent anywhere.
 * Distinct from MicRecorder (which captures WAV for STT), with
 * echoCancellation requested so Kurisu's own output is suppressed from the
 * input before it ever reaches the analyser.
 *
 * Currently unused: auto barge-in originally opened this monitor while she
 * spoke, but since hands-free became a continuous open-mic mode the barge-in
 * detector rides the hands-free capture stream's level tap instead
 * (src/hooks/useHandsFree.ts) — one getUserMedia stream instead of two.
 * Kept as the analyser-only primitive for any future feature that needs mic
 * levels without a recording-capable stream.
 */

const POLL_MS = 50
const FFT_SIZE = 1024

export class MicMonitor {
  onLevel: ((rms: number, dtMs: number) => void) | null = null

  private stream: MediaStream | null = null
  private context: AudioContext | null = null
  private analyser: AnalyserNode | null = null
  private source: MediaStreamAudioSourceNode | null = null
  private timer: number | null = null

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    })
    this.context = new AudioContext()
    this.source = this.context.createMediaStreamSource(this.stream)
    this.analyser = this.context.createAnalyser()
    this.analyser.fftSize = FFT_SIZE
    this.source.connect(this.analyser)

    const buffer = new Float32Array(this.analyser.fftSize)
    this.timer = window.setInterval(() => {
      const analyser = this.analyser
      if (!analyser || !this.onLevel) return
      analyser.getFloatTimeDomainData(buffer)
      let sum = 0
      for (let i = 0; i < buffer.length; i++) sum += buffer[i] * buffer[i]
      this.onLevel(Math.sqrt(sum / buffer.length), POLL_MS)
    }, POLL_MS)
  }

  stop(): void {
    if (this.timer !== null) {
      window.clearInterval(this.timer)
      this.timer = null
    }
    this.onLevel = null
    try {
      this.source?.disconnect()
    } catch {
      // Already torn down.
    }
    this.source = null
    this.analyser = null
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop()
      this.stream = null
    }
    if (this.context) {
      void this.context.close().catch(() => {})
      this.context = null
    }
  }
}
