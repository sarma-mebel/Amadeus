// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, renderHook } from '@testing-library/react'
import { useCallMode, type UseCallModeOptions } from './useCallMode'

describe('useCallMode', () => {
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  beforeEach(() => {
    window.localStorage.clear()
    Element.prototype.scrollIntoView = vi.fn()
    Element.prototype.scrollTo = vi.fn()
  })

  const setup = (overrides: Partial<UseCallModeOptions> = {}) => {
    const options: UseCallModeOptions = {
      handsFree: false,
      toggleHandsFree: vi.fn(),
      autoBargeIn: false,
      toggleAutoBargeIn: vi.fn(),
      stopCurrentStream: vi.fn(),
      ...overrides,
    }
    const rendered = renderHook((props: UseCallModeOptions) => useCallMode(props), {
      initialProps: options,
    })
    return { options, ...rendered }
  }

  it('startCall forces hands-free and auto barge-in on and stashes the prior prefs', () => {
    const { options, result } = setup()

    act(() => result.current.startCall())

    expect(result.current.callMode).toBe(true)
    expect(options.toggleHandsFree).toHaveBeenCalledTimes(1)
    expect(options.toggleAutoBargeIn).toHaveBeenCalledTimes(1)
    expect(options.stopCurrentStream).not.toHaveBeenCalled()
  })

  it('does not toggle prefs the user already had on', () => {
    const { options, result } = setup({ handsFree: true, autoBargeIn: true })

    act(() => result.current.startCall())

    expect(result.current.callMode).toBe(true)
    expect(options.toggleHandsFree).not.toHaveBeenCalled()
    expect(options.toggleAutoBargeIn).not.toHaveBeenCalled()
  })

  it('endCall stops the stream and restores only what the call changed', () => {
    const { options, result, rerender } = setup()

    act(() => result.current.startCall())
    // The toggles flipped the live values on — mirror that in props, as the
    // real useHandsFree state would.
    rerender({ ...options, handsFree: true, autoBargeIn: true })

    act(() => result.current.endCall())

    expect(result.current.callMode).toBe(false)
    expect(options.stopCurrentStream).toHaveBeenCalledTimes(1)
    // Both prefs were off before the call — both get toggled back off.
    expect(options.toggleHandsFree).toHaveBeenCalledTimes(2)
    expect(options.toggleAutoBargeIn).toHaveBeenCalledTimes(2)
  })

  it('endCall leaves a pref alone when the user already had it on before the call', () => {
    const { options, result, rerender } = setup({ handsFree: true })

    act(() => result.current.startCall())
    rerender({ ...options, handsFree: true, autoBargeIn: true })

    act(() => result.current.endCall())

    // handsFree was the user's own setting — never toggled; autoBargeIn was
    // forced on by the call and gets restored off.
    expect(options.toggleHandsFree).not.toHaveBeenCalled()
    expect(options.toggleAutoBargeIn).toHaveBeenCalledTimes(2)
  })

  it('startCall and endCall are no-ops when already in / out of a call', () => {
    const { options, result } = setup()

    act(() => result.current.endCall())
    expect(options.stopCurrentStream).not.toHaveBeenCalled()

    act(() => result.current.startCall())
    act(() => result.current.startCall())
    expect(options.toggleHandsFree).toHaveBeenCalledTimes(1)
  })

  it('duplex-preferred start runs the duplex engine and never touches the classic toggles', () => {
    const tryStartDuplex = vi.fn()
    const stopDuplex = vi.fn()
    const { options, result } = setup({
      duplexPreferred: () => true,
      tryStartDuplex,
      stopDuplex,
    })

    act(() => result.current.startCall())
    expect(result.current.callMode).toBe(true)
    expect(tryStartDuplex).toHaveBeenCalledTimes(1)
    expect(options.toggleHandsFree).not.toHaveBeenCalled()
    expect(options.toggleAutoBargeIn).not.toHaveBeenCalled()

    // Hang up a pure-duplex call: stop the loop, and never "restore" the
    // classic toggles into a flipped state (they were never engaged).
    act(() => result.current.endCall())
    expect(stopDuplex).toHaveBeenCalledTimes(1)
    expect(options.toggleHandsFree).not.toHaveBeenCalled()
    expect(options.toggleAutoBargeIn).not.toHaveBeenCalled()
  })

  it('duplexPreferred false keeps the classic engine even when duplex is wired', () => {
    const tryStartDuplex = vi.fn()
    const { options, result } = setup({
      duplexPreferred: () => false,
      tryStartDuplex,
      stopDuplex: vi.fn(),
    })

    act(() => result.current.startCall())
    expect(tryStartDuplex).not.toHaveBeenCalled()
    expect(options.toggleHandsFree).toHaveBeenCalledTimes(1)
  })

  it('fallbackToClassic downgrades a live duplex call in place, and endCall restores', () => {
    const { options, result, rerender } = setup({
      duplexPreferred: () => true,
      tryStartDuplex: vi.fn(),
      stopDuplex: vi.fn(),
    })

    act(() => result.current.startCall())
    expect(options.toggleHandsFree).not.toHaveBeenCalled()

    act(() => result.current.fallbackToClassic())
    expect(options.toggleHandsFree).toHaveBeenCalledTimes(1)
    expect(options.toggleAutoBargeIn).toHaveBeenCalledTimes(1)
    // Second fallback (e.g. a late close event) is a no-op.
    act(() => result.current.fallbackToClassic())
    expect(options.toggleHandsFree).toHaveBeenCalledTimes(1)

    rerender({ ...options, handsFree: true, autoBargeIn: true } as UseCallModeOptions)
    act(() => result.current.endCall())
    // The downgrade engaged the toggles, so hang-up restores them off.
    expect(options.toggleHandsFree).toHaveBeenCalledTimes(2)
    expect(options.toggleAutoBargeIn).toHaveBeenCalledTimes(2)
  })

  it('fallbackToClassic after the call ended is a no-op', () => {
    const { options, result } = setup({
      duplexPreferred: () => true,
      tryStartDuplex: vi.fn(),
      stopDuplex: vi.fn(),
    })

    act(() => result.current.startCall())
    act(() => result.current.endCall())
    act(() => result.current.fallbackToClassic())
    expect(options.toggleHandsFree).not.toHaveBeenCalled()
  })
})
