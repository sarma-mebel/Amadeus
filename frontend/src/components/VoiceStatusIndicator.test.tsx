// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom/vitest'
import { cleanup, render, screen } from '@testing-library/react'
import VoiceStatusIndicator, { type VoiceState } from './VoiceStatusIndicator'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

const STATE_LABELS: Record<VoiceState, string> = {
  idle: 'Voice idle',
  listening: 'Listening',
  transcribing: 'Transcribing',
  thinking: 'Kurisu is thinking',
  speaking: 'Kurisu is speaking',
}

describe('VoiceStatusIndicator', () => {
  for (const [state, label] of Object.entries(STATE_LABELS) as Array<[VoiceState, string]>) {
    it(`renders the ${state} state with its class and aria-label`, () => {
      render(<VoiceStatusIndicator state={state} />)
      const el = screen.getByRole('status')
      expect(el).toHaveClass(`voice-status-${state}`)
      expect(el).toHaveAttribute('aria-label', label)
    })
  }

  it('starts a requestAnimationFrame loop while listening and cancels it on unmount', () => {
    let rafId = 0
    const scheduled: FrameRequestCallback[] = []
    const rafSpy = vi.fn((cb: FrameRequestCallback) => {
      scheduled.push(cb)
      rafId += 1
      return rafId
    })
    const cafSpy = vi.fn()
    vi.stubGlobal('requestAnimationFrame', rafSpy)
    vi.stubGlobal('cancelAnimationFrame', cafSpy)

    const { unmount } = render(<VoiceStatusIndicator state="listening" />)

    expect(rafSpy).toHaveBeenCalledTimes(1)

    unmount()
    expect(cafSpy).toHaveBeenCalledTimes(1)
  })

  it('does not start a requestAnimationFrame loop for non-listening states', () => {
    const rafSpy = vi.fn(() => 0)
    vi.stubGlobal('requestAnimationFrame', rafSpy)
    vi.stubGlobal('cancelAnimationFrame', vi.fn())

    render(<VoiceStatusIndicator state="idle" />)
    expect(rafSpy).not.toHaveBeenCalled()
  })

  it('cancels the previous rAF loop when transitioning away from listening', () => {
    let rafId = 0
    const rafSpy = vi.fn(() => {
      rafId += 1
      return rafId
    })
    const cafSpy = vi.fn()
    vi.stubGlobal('requestAnimationFrame', rafSpy)
    vi.stubGlobal('cancelAnimationFrame', cafSpy)

    const { rerender } = render(<VoiceStatusIndicator state="listening" />)
    expect(rafSpy).toHaveBeenCalledTimes(1)

    rerender(<VoiceStatusIndicator state="speaking" />)
    expect(cafSpy).toHaveBeenCalledTimes(1)
  })
})
