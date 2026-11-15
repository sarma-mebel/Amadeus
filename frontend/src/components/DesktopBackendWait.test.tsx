// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom/vitest'
import { cleanup, render, screen } from '@testing-library/react'
import DesktopBackendWait from './DesktopBackendWait'

describe('DesktopBackendWait', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.stubGlobal(
      'matchMedia',
      vi.fn().mockReturnValue({
        matches: false,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      })
    )
  })

  afterEach(() => {
    cleanup()
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('renders the boot status region with a rolling divergence meter', () => {
    render(<DesktopBackendWait />)
    expect(screen.getByRole('status', { name: /starting backend/i })).toBeTruthy()
    expect(screen.getByRole('img', { name: /divergence meter calibrating/i })).toBeTruthy()
  })

  it('cleans up the meter timers on unmount', () => {
    const { unmount } = render(<DesktopBackendWait />)
    expect(vi.getTimerCount()).toBeGreaterThan(0)
    unmount()
    expect(vi.getTimerCount()).toBe(0)
  })
})
