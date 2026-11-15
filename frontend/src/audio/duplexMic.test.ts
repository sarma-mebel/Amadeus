import { describe, expect, it } from 'vitest'
import { DUPLEX_SAMPLE_RATE, resampleToPcm16 } from './duplexMic'

describe('resampleToPcm16', () => {
  it('produces a PCM16-LE buffer at the target sample count when rates match', () => {
    const samples = new Float32Array([0, 0.5, -0.5, 1, -1])
    const buffer = resampleToPcm16(samples, DUPLEX_SAMPLE_RATE, DUPLEX_SAMPLE_RATE)
    const view = new Int16Array(buffer)
    expect(view.length).toBe(samples.length)
    expect(view[0]).toBe(0)
    expect(view[3]).toBeGreaterThan(30000) // 1.0 -> ~32767
    expect(view[4]).toBeLessThan(-30000) // -1.0 -> ~-32768
  })

  it('downsamples proportionally (48kHz -> 16kHz is a 3x reduction)', () => {
    const samples = new Float32Array(4800).fill(0.2)
    const buffer = resampleToPcm16(samples, 48000, DUPLEX_SAMPLE_RATE)
    const view = new Int16Array(buffer)
    expect(view.length).toBeCloseTo(1600, -1)
  })

  it('upsamples proportionally (8kHz -> 16kHz is a 2x increase)', () => {
    const samples = new Float32Array(800).fill(0.1)
    const buffer = resampleToPcm16(samples, 8000, DUPLEX_SAMPLE_RATE)
    const view = new Int16Array(buffer)
    expect(view.length).toBeCloseTo(1600, -1)
  })

  it('clamps out-of-range samples instead of overflowing', () => {
    const samples = new Float32Array([2.5, -3.0])
    const buffer = resampleToPcm16(samples, DUPLEX_SAMPLE_RATE, DUPLEX_SAMPLE_RATE)
    const view = new Int16Array(buffer)
    expect(view[0]).toBeLessThanOrEqual(32767)
    expect(view[1]).toBeGreaterThanOrEqual(-32768)
  })

  it('never divides by zero on a single-sample buffer', () => {
    const samples = new Float32Array([0.3])
    const buffer = resampleToPcm16(samples, DUPLEX_SAMPLE_RATE, DUPLEX_SAMPLE_RATE)
    expect(new Int16Array(buffer).length).toBeGreaterThanOrEqual(1)
  })
})
