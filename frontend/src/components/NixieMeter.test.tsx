// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom/vitest'
import { act, cleanup, render, screen } from '@testing-library/react'
import NixieMeter, { SETTLE_VALUE } from './NixieMeter'

const stubMatchMedia = (reducedMotion: boolean) => {
  vi.stubGlobal(
    'matchMedia',
    vi.fn().mockReturnValue({
      matches: reducedMotion,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    })
  )
}

describe('NixieMeter', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    stubMatchMedia(false)
  })

  afterEach(() => {
    cleanup()
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('renders all 7 digit tubes plus the decimal tube', () => {
    const { container } = render(<NixieMeter settled={false} />)
    const digitTubes = container.querySelectorAll('.nixie-tube:not(.nixie-tube-dot)')
    const dotTubes = container.querySelectorAll('.nixie-tube-dot')
    const digitCount = SETTLE_VALUE.replace('.', '').length
    expect(digitTubes).toHaveLength(digitCount)
    expect(dotTubes).toHaveLength(1)
  })

  it('settles left-to-right onto 1.048596 when settled becomes true', async () => {
    const { container, rerender } = render(<NixieMeter settled={false} />)
    rerender(<NixieMeter settled />)
    // The settle effect staggers each tube by 70ms; run past the last one.
    // act() so the timeout-driven setDigits flush to the DOM before asserting.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(7 * 70 + 10)
    })
    const digits = Array.from(container.querySelectorAll('.nixie-digit')).map(
      (el) => el.textContent
    )
    expect(digits.join('')).toBe(SETTLE_VALUE.replace('.', ''))
    expect(screen.getByRole('img', { name: `Divergence meter: ${SETTLE_VALUE}` })).toBeTruthy()
  })

  it('renders statically at 1.048596 under prefers-reduced-motion, even before settling', () => {
    stubMatchMedia(true)
    const { container } = render(<NixieMeter settled={false} />)
    const digits = Array.from(container.querySelectorAll('.nixie-digit')).map(
      (el) => el.textContent
    )
    expect(digits.join('')).toBe(SETTLE_VALUE.replace('.', ''))
    // No rolling timers should be scheduled at all under reduced motion.
    expect(vi.getTimerCount()).toBe(0)
  })

  it('clears all rolling interval timers on unmount', () => {
    const { unmount } = render(<NixieMeter settled={false} />)
    expect(vi.getTimerCount()).toBeGreaterThan(0)
    unmount()
    expect(vi.getTimerCount()).toBe(0)
  })

  it('clears settle timeouts on unmount without throwing', () => {
    const { rerender, unmount } = render(<NixieMeter settled={false} />)
    rerender(<NixieMeter settled />)
    expect(() => unmount()).not.toThrow()
    expect(vi.getTimerCount()).toBe(0)
  })
})
