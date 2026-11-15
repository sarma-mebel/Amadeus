// @vitest-environment jsdom
import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useMicRecorder } from './useMicRecorder'

// Controllable MicRecorder: start() resolves only when the test says so,
// standing in for getUserMedia latency.
const { recorders, MockRecorder } = vi.hoisted(() => {
  class MockRecorder {
    onLevel: ((rms: number, dtMs: number) => void) | null = null
    cancelCalls = 0
    startResolve: (() => void) | null = null
    start() {
      return new Promise<void>((resolve) => {
        this.startResolve = resolve
      })
    }
    async stop() {
      return new Blob(['x'], { type: 'audio/wav' })
    }
    cancel() {
      this.cancelCalls += 1
    }
  }
  const recorders: InstanceType<typeof MockRecorder>[] = []
  const Tracked = new Proxy(MockRecorder, {
    construct(target, args) {
      const instance = Reflect.construct(target, args)
      recorders.push(instance)
      return instance
    },
  })
  return { recorders, MockRecorder: Tracked }
})

vi.mock('../audio/micRecorder', () => ({ MicRecorder: MockRecorder }))

const options = () => ({
  canUseMic: true,
  onTranscript: vi.fn(),
  shouldBargeIn: () => false,
  onBargeIn: vi.fn(),
  onPermissionDenied: vi.fn(),
  onSttUnavailable: vi.fn(),
  onTapTooShort: vi.fn(),
  textareaRef: { current: null },
})

describe('useMicRecorder', () => {
  beforeEach(() => {
    recorders.length = 0
  })

  it('records and returns to idle across a normal hold-release', async () => {
    const opts = options()
    const { result } = renderHook(() => useMicRecorder(opts))

    await act(async () => {
      const pending = result.current.startRecording()
      await Promise.resolve()
      recorders[0].startResolve?.()
      await pending
    })
    expect(result.current.micStatus).toBe('recording')
  })

  it('does not get stuck recording when released while getUserMedia is pending', async () => {
    // The "mic turns on but can never be turned off" bug: a quick tap fires
    // stop before start's await resolves; stop finds no recorder and resets,
    // then start used to finish anyway and flip to 'recording' with the
    // active flag false — unreachable by every guard.
    const opts = options()
    const { result } = renderHook(() => useMicRecorder(opts))

    await act(async () => {
      const startPending = result.current.startRecording()
      await Promise.resolve()
      // Release before permission/setup completed.
      await result.current.stopRecordingAndTranscribe()
      recorders[0].startResolve?.()
      await startPending
    })

    expect(result.current.micStatus).toBe('idle')
    expect(recorders[0].cancelCalls).toBeGreaterThan(0)
    expect(opts.onTapTooShort).toHaveBeenCalled()

    // And the mic still works afterwards.
    await act(async () => {
      const pending = result.current.startRecording()
      await Promise.resolve()
      recorders[1].startResolve?.()
      await pending
    })
    expect(result.current.micStatus).toBe('recording')
  })
})
