// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, renderHook } from '@testing-library/react'
import { useDuplexCall, type UseDuplexCallOptions } from './useDuplexCall'
import { startDuplexSession, type DuplexHandlers } from '../chat/duplexTurn'
import { acknowledgment } from '../vrm/acknowledgment'
import { kurisuBridge } from '../vrm/kurisuBridge'
import { publishMicLevel, resetMicLevel } from '../audio/levelBus'
import type { Message } from '../chat/messages'
import type { DoneFrame } from '../streaming'

vi.mock('../chat/duplexTurn', () => ({
  startDuplexSession: vi.fn(),
}))
vi.mock('../vrm/acknowledgment', () => ({
  acknowledgment: { arm: vi.fn(), release: vi.fn() },
}))
vi.mock('../vrm/kurisuBridge', () => ({
  kurisuBridge: { reactReset: vi.fn(), vrmAnimations: { play: vi.fn() } },
}))
vi.mock('../audio/levelBus', () => ({
  publishMicLevel: vi.fn(),
  resetMicLevel: vi.fn(),
}))
vi.mock('../soundscape', () => ({
  soundscape: { play: vi.fn(), startAmbience: vi.fn() },
  loadAmbienceEnabled: () => false,
}))
vi.mock('./useSessions', () => ({
  resolveUserId: () => 'user-test',
}))

const doneFrame = (overrides: Partial<DoneFrame['response']> = {}, cancelled = false): DoneFrame =>
  ({
    type: 'done',
    cancelled,
    response: { reply: 'hi', turn_id: 'turn-1', animations: null, ...overrides },
  }) as unknown as DoneFrame

describe('useDuplexCall', () => {
  let messages: Message[]
  let handle: { cancel: ReturnType<typeof vi.fn>; stop: ReturnType<typeof vi.fn> }

  beforeEach(() => {
    vi.clearAllMocks()
    messages = []
    handle = { cancel: vi.fn(), stop: vi.fn() }
    vi.mocked(startDuplexSession).mockReturnValue(handle)
  })

  afterEach(() => {
    cleanup()
  })

  const setup = (overrides: Partial<UseDuplexCallOptions> = {}) => {
    const options: UseDuplexCallOptions = {
      setMessages: (updater) => {
        messages = typeof updater === 'function' ? updater(messages) : updater
      },
      adoptSessionId: vi.fn(),
      setRequiresAudioUnlock: vi.fn(),
      completeTurnMessages: vi.fn((prev: Message[]) => prev),
      onFallback: vi.fn(),
      ...overrides,
    }
    const rendered = renderHook(() => useDuplexCall(options))
    return { options, ...rendered }
  }

  const startAndGetHandlers = (result: { current: { startDuplex: () => void } }): DuplexHandlers => {
    act(() => result.current.startDuplex())
    return vi.mocked(startDuplexSession).mock.calls[0][1]
  }

  it('capability false reports the unavailable fallback and never activates', () => {
    const { options, result } = setup()
    const handlers = startAndGetHandlers(result)

    act(() => handlers.onCapability?.(false))

    expect(result.current.duplexActive).toBe(false)
    expect(options.onFallback).toHaveBeenCalledWith('unavailable')
  })

  it('activates on capability, adopts the session id, and turns transcripts into user bubbles', () => {
    const { options, result } = setup()
    const handlers = startAndGetHandlers(result)

    act(() => {
      handlers.onSessionId?.('sess-9')
      handlers.onCapability?.(true)
      handlers.onTranscript?.('hello there')
    })

    expect(result.current.duplexActive).toBe(true)
    expect(options.adoptSessionId).toHaveBeenCalledWith('sess-9')
    expect(messages).toHaveLength(1)
    expect(messages[0]).toMatchObject({ role: 'user', content: 'hello there' })
  })

  it('arms the acknowledgment on endpoint (processing) and releases it on first token', () => {
    const { result } = setup()
    const handlers = startAndGetHandlers(result)

    act(() => {
      handlers.onCapability?.(true)
      handlers.onState?.('processing')
    })
    expect(acknowledgment.arm).toHaveBeenCalledTimes(1)
    expect(kurisuBridge.reactReset).toHaveBeenCalledTimes(1)
    const armedId = vi.mocked(acknowledgment.arm).mock.calls[0][0]

    act(() => handlers.onToken?.('Hey'))
    expect(acknowledgment.release).toHaveBeenCalledWith(armedId)
    // The assistant bubble exists and (after the coalescing flush) carries
    // the token text.
    expect(messages.some((msg) => msg.role === 'assistant')).toBe(true)
  })

  it('streams tokens into one assistant bubble and folds the turn id at done', async () => {
    vi.useFakeTimers()
    try {
      const { options, result } = setup()
      const handlers = startAndGetHandlers(result)

      act(() => {
        handlers.onCapability?.(true)
        handlers.onTranscript?.('question')
        handlers.onToken?.('Hello ')
        handlers.onToken?.('world.')
        vi.advanceTimersByTime(20)
      })

      const assistant = messages.find((msg) => msg.role === 'assistant')
      expect(assistant?.content).toBe('Hello world.')

      act(() => handlers.onDone?.(doneFrame()))
      expect(options.completeTurnMessages).toHaveBeenCalledWith(
        expect.anything(),
        assistant?.id,
        'turn-1',
      )
    } finally {
      vi.useRealTimers()
    }
  })

  it('a second turn gets its own assistant bubble after the first done', () => {
    const { result } = setup()
    const handlers = startAndGetHandlers(result)

    act(() => {
      handlers.onCapability?.(true)
      handlers.onTranscript?.('one')
      handlers.onToken?.('a')
      handlers.onDone?.(doneFrame())
      handlers.onTranscript?.('two')
      handlers.onToken?.('b')
    })

    expect(messages.filter((msg) => msg.role === 'assistant')).toHaveLength(2)
  })

  it('plays a requested animation at done, but not on a cancelled turn', () => {
    const { result } = setup()
    const handlers = startAndGetHandlers(result)

    act(() => {
      handlers.onCapability?.(true)
      handlers.onToken?.('a')
      handlers.onDone?.(doneFrame({ animations: ['Peace Sign'] }))
    })
    expect(kurisuBridge.vrmAnimations?.play).toHaveBeenCalledWith('Peace Sign')

    act(() => {
      handlers.onToken?.('b')
      handlers.onDone?.(doneFrame({ animations: ['Cheering'] }, true))
    })
    expect(kurisuBridge.vrmAnimations?.play).not.toHaveBeenCalledWith('Cheering')
  })

  it('publishes mic levels and resets them on stop', () => {
    const { result } = setup()
    const handlers = startAndGetHandlers(result)

    act(() => {
      handlers.onCapability?.(true)
      handlers.onMicLevel?.(0.4, 16)
    })
    expect(publishMicLevel).toHaveBeenCalledWith(0.4)

    act(() => result.current.stopDuplex())
    expect(handle.stop).toHaveBeenCalledTimes(1)
    expect(resetMicLevel).toHaveBeenCalled()
    expect(result.current.duplexActive).toBe(false)
  })

  it('a mid-call socket death falls back; a deliberate stop does not', () => {
    const { options, result } = setup()
    const handlers = startAndGetHandlers(result)

    act(() => {
      handlers.onCapability?.(true)
      handlers.onClosed?.()
    })
    expect(options.onFallback).toHaveBeenCalledWith('closed')

    // Fresh session, this time the user hangs up first.
    vi.mocked(startDuplexSession).mockClear()
    act(() => result.current.startDuplex())
    const handlers2 = vi.mocked(startDuplexSession).mock.calls[0][1]
    act(() => {
      handlers2.onCapability?.(true)
      result.current.stopDuplex()
      handlers2.onClosed?.()
    })
    expect(options.onFallback).toHaveBeenCalledTimes(1)
  })

  it('mic_unavailable errors fall back; turn-scoped errors keep the session alive', () => {
    const { options, result } = setup()
    const handlers = startAndGetHandlers(result)

    act(() => {
      handlers.onCapability?.(true)
      handlers.onError?.('stt exploded', 'stt_error')
    })
    expect(options.onFallback).not.toHaveBeenCalled()
    expect(result.current.duplexActive).toBe(true)

    act(() => handlers.onError?.('no mic', 'mic_unavailable'))
    expect(options.onFallback).toHaveBeenCalledWith('mic')
    expect(result.current.duplexActive).toBe(false)
  })
})
