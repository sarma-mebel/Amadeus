// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom/vitest'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'

// Hoisted so the mock factory (evaluated before the import below runs) can
// close over it. Defaults to browser (false) so tests opt into Tauri
// explicitly, matching every other suite's convention in this repo.
const { tauriFlag } = vi.hoisted(() => ({ tauriFlag: { value: false } }))
vi.mock('../platform', () => ({ isTauri: () => tauriFlag.value }))

import LoadingScreen from './LoadingScreen'

const AUTO_CONNECT_KEY = 'amadeus_auto_connect'

describe('LoadingScreen auto-connect default', () => {
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    tauriFlag.value = false
  })

  beforeEach(() => {
    window.localStorage.clear()
    // /healthz never needs to resolve for these tests — they only assert
    // the checkbox's INITIAL checked state, computed synchronously from
    // localStorage + isTauri() at mount, well before any fetch settles.
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})))
    // The progress-ramp effect drives a real RAF loop purely for visuals;
    // stubbing it out avoids leaving a dangling animation loop running
    // against a promise that intentionally never resolves in this suite.
    vi.stubGlobal('requestAnimationFrame', vi.fn(() => 0))
    vi.stubGlobal('cancelAnimationFrame', vi.fn())
    // jsdom doesn't implement matchMedia; the waveform-canvas effect calls
    // it unconditionally on mount (reduced-motion check), so every render
    // needs this stubbed or mount itself throws.
    vi.stubGlobal(
      'matchMedia',
      vi.fn().mockReturnValue({
        matches: false,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      }),
    )
  })

  const getCheckbox = () =>
    screen.getByRole('checkbox', { name: /auto-connect on launch/i }) as HTMLInputElement

  it('defaults ON in Tauri when nothing is stored', () => {
    tauriFlag.value = true
    render(<LoadingScreen onConnect={() => {}} />)
    expect(getCheckbox().checked).toBe(true)
  })

  it('defaults OFF in the browser when nothing is stored', () => {
    tauriFlag.value = false
    render(<LoadingScreen onConnect={() => {}} />)
    expect(getCheckbox().checked).toBe(false)
  })

  it('an explicit stored "false" wins even in Tauri', () => {
    tauriFlag.value = true
    window.localStorage.setItem(AUTO_CONNECT_KEY, 'false')
    render(<LoadingScreen onConnect={() => {}} />)
    expect(getCheckbox().checked).toBe(false)
  })

  it('an explicit stored "true" wins even in the browser', () => {
    tauriFlag.value = false
    window.localStorage.setItem(AUTO_CONNECT_KEY, 'true')
    render(<LoadingScreen onConnect={() => {}} />)
    expect(getCheckbox().checked).toBe(true)
  })
})

describe('LoadingScreen nixie divergence meter', () => {
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    tauriFlag.value = false
  })

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

  it('renders all 7 digit tubes plus the decimal tube while loading', () => {
    window.localStorage.clear()
    stubMatchMedia(false)
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})))
    vi.stubGlobal('requestAnimationFrame', vi.fn(() => 0))
    vi.stubGlobal('cancelAnimationFrame', vi.fn())

    const { container } = render(<LoadingScreen onConnect={() => {}} />)
    expect(container.querySelectorAll('.nixie-tube:not(.nixie-tube-dot)')).toHaveLength(7)
    expect(container.querySelectorAll('.nixie-tube-dot')).toHaveLength(1)
  })

  it('settles the meter onto 1.048596 once the load gate reports ready', async () => {
    window.localStorage.setItem(AUTO_CONNECT_KEY, 'false')
    stubMatchMedia(false)
    vi.stubGlobal(
      'fetch',
      vi.fn(() =>
        Promise.resolve({
          json: () => Promise.resolve({ status: 'ok', voice: 'ready', version: '0.17.0' }),
        })
      )
    )
    // Drive the progress-ramp rAF loop deterministically. Two things must be
    // faithful or the ramp can never cross the >99 ready gate:
    //  - cancelAnimationFrame must actually cancel: the pre-backendUp tick
    //    closure targets 40%, and if it survives its effect cleanup it fights
    //    the post-flip closure's 100% target — the eased mix converges ~71%.
    //  - each round flushes inside act() so setProgress lands and the effect
    //    re-subscribes with the fresh closure before the next round runs.
    let nextRafId = 1
    let rafCallbacks = new Map<number, FrameRequestCallback>()
    vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
      const id = nextRafId
      nextRafId += 1
      rafCallbacks.set(id, cb)
      return id
    })
    vi.stubGlobal('cancelAnimationFrame', (id: number) => {
      rafCallbacks.delete(id)
    })
    const drain = async () => {
      for (let round = 0; round < 300; round += 1) {
        const pending = Array.from(rafCallbacks.values())
        rafCallbacks = new Map()
        if (!pending.length) break
        await act(async () => {
          for (const cb of pending) cb(performance.now() + round * 50)
          await Promise.resolve()
        })
      }
    }

    const { container } = render(<LoadingScreen onConnect={() => {}} />)
    // Let the healthz fetch promise chain resolve (setBackendUp/setVoiceState).
    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
      await Promise.resolve()
    })
    await drain()

    // The settle stagger is ~490ms of real time and the drain itself does
    // ~300 act() rounds; under full-suite worker contention both stretch,
    // so the waitFor AND the test's own budget need headroom over the 5s
    // default (measured: alone 0.7s, contended full suite blows 5s).
    await waitFor(
      () => {
        const digits = Array.from(container.querySelectorAll('.nixie-digit')).map(
          (el) => el.textContent
        )
        expect(digits.join('')).toBe('1048596')
      },
      { timeout: 10000 }
    )
    expect(screen.getByText('READY')).toBeTruthy()
  }, 20000)

  it('renders the meter statically at 1.048596 under prefers-reduced-motion', () => {
    window.localStorage.clear()
    stubMatchMedia(true)
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})))
    vi.stubGlobal('requestAnimationFrame', vi.fn(() => 0))
    vi.stubGlobal('cancelAnimationFrame', vi.fn())

    const { container } = render(<LoadingScreen onConnect={() => {}} />)
    const digits = Array.from(container.querySelectorAll('.nixie-digit')).map(
      (el) => el.textContent
    )
    expect(digits.join('')).toBe('1048596')
  })

  it('clears the meter rolling timers when LoadingScreen unmounts', () => {
    window.localStorage.clear()
    stubMatchMedia(false)
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})))
    vi.stubGlobal('requestAnimationFrame', vi.fn(() => 0))
    vi.stubGlobal('cancelAnimationFrame', vi.fn())
    vi.useFakeTimers()

    const { unmount } = render(<LoadingScreen onConnect={() => {}} />)
    expect(vi.getTimerCount()).toBeGreaterThan(0)
    unmount()
    expect(vi.getTimerCount()).toBe(0)
    vi.useRealTimers()
  })
})
