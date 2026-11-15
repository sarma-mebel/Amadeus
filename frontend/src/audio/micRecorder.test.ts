import { describe, expect, it } from 'vitest'
import { STT_SAMPLE_RATE, encodeWavFromFloat32 } from './micRecorder'

const view = async (blob: Blob) => new DataView(await blob.arrayBuffer())

describe('encodeWavFromFloat32', () => {
  it('writes a valid 16kHz mono 16-bit PCM WAV header', async () => {
    const samples = new Float32Array(16000) // 1s at the target rate
    const v = await view(encodeWavFromFloat32(samples, 16000))

    const ascii = (offset: number, length: number) =>
      Array.from({ length }, (_, i) => String.fromCharCode(v.getUint8(offset + i))).join('')
    expect(ascii(0, 4)).toBe('RIFF')
    expect(ascii(8, 4)).toBe('WAVE')
    expect(v.getUint16(22, true)).toBe(1) // mono
    expect(v.getUint32(24, true)).toBe(STT_SAMPLE_RATE)
    expect(v.getUint16(34, true)).toBe(16) // bits per sample
    expect(v.getUint32(40, true)).toBe(16000 * 2) // data size
    expect(v.byteLength).toBe(44 + 16000 * 2)
  })

  it('resamples 48kHz input down to 16kHz', async () => {
    const samples = new Float32Array(48000) // 1s at 48kHz
    const v = await view(encodeWavFromFloat32(samples, 48000))
    expect(v.getUint32(40, true)).toBe(16000 * 2)
  })

  it('clamps out-of-range samples to the int16 range', async () => {
    const samples = new Float32Array([2.0, -2.0, 0.5])
    const v = await view(encodeWavFromFloat32(samples, STT_SAMPLE_RATE))
    expect(v.getInt16(44, true)).toBe(0x7fff)
    expect(v.getInt16(46, true)).toBe(-0x8000)
  })
})
