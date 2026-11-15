// @vitest-environment jsdom
//
// Covers the reflex-layer dispatch seam added to useAmbientSignals: every
// tracker event additionally fans out as a 'kurisu:ambient' CustomEvent
// (vrm/reflexes.ts routes on its detail), and a dispatch failure counts in
// diagnostics without ever eating the server delivery itself. The
// transition rules live in ambient.ts and are covered by ambient.test.ts —
// these tests drive the real tracker with single obviously-noteworthy
// snapshots (music start, critical battery).
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, renderHook, waitFor } from '@testing-library/react'
import { diagnostics } from '../diagnostics'
import type { AmbientSnapshot } from '../ambient'

vi.mock('../platform', () => ({ isTauri: () => true }))
const invokeMock = vi.fn()
vi.mock('@tauri-apps/api/core', () => ({
  invoke: (...args: unknown[]) => invokeMock(...args),
}))

import { useAmbientSignals } from './useAmbientSignals'

const snapshot = (overrides: Partial<AmbientSnapshot> = {}): AmbientSnapshot => ({
  window_title: null,
  battery_percent: null,
  battery_charging: null,
  media_title: null,
  media_artist: null,
  media_playing: null,
  ...overrides,
})

describe('useAmbientSignals reflex dispatch', () => {
  beforeEach(() => {
    diagnostics.resetForTest()
    invokeMock.mockReset()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('dispatches kurisu:ambient with the event kind/detail alongside delivery', async () => {
    invokeMock.mockResolvedValue(snapshot({ media_playing: true, media_title: 'Hacking to the Gate' }))
    const seen: Array<Record<string, unknown>> = []
    const listener = (ev: Event) => {
      seen.push((ev as CustomEvent<Record<string, unknown>>).detail)
    }
    window.addEventListener('kurisu:ambient', listener)
    const deliver = vi.fn().mockResolvedValue(undefined)
    try {
      renderHook(() => useAmbientSignals({ deliver }))
      await waitFor(() => expect(deliver).toHaveBeenCalledTimes(1))
      expect(seen).toHaveLength(1)
      expect(seen[0]).toMatchObject({ kind: 'music' })
      expect(String(seen[0].detail)).toContain('Hacking to the Gate')
      expect(seen[0].severity).toBeUndefined()
      expect(deliver).toHaveBeenCalledWith('music', expect.stringContaining('Hacking to the Gate'))
    } finally {
      window.removeEventListener('kurisu:ambient', listener)
    }
  })

  it('a critical battery event carries the structured severity discriminator', async () => {
    invokeMock.mockResolvedValue(snapshot({ battery_percent: 5, battery_charging: false }))
    const seen: Array<Record<string, unknown>> = []
    const listener = (ev: Event) => {
      seen.push((ev as CustomEvent<Record<string, unknown>>).detail)
    }
    window.addEventListener('kurisu:ambient', listener)
    const deliver = vi.fn().mockResolvedValue(undefined)
    try {
      renderHook(() => useAmbientSignals({ deliver }))
      await waitFor(() => expect(deliver).toHaveBeenCalledTimes(1))
      expect(seen[0]).toMatchObject({ kind: 'battery', severity: 'critical' })
    } finally {
      window.removeEventListener('kurisu:ambient', listener)
    }
  })

  it('a dispatch failure counts in diagnostics and never eats the delivery', async () => {
    invokeMock.mockResolvedValue(snapshot({ media_playing: true, media_title: 'Believe Me' }))
    const originalDispatch = window.dispatchEvent.bind(window)
    vi.spyOn(window, 'dispatchEvent').mockImplementation((ev: Event) => {
      if (ev.type === 'kurisu:ambient') throw new Error('dispatch dead')
      return originalDispatch(ev)
    })
    const deliver = vi.fn().mockResolvedValue(undefined)
    renderHook(() => useAmbientSignals({ deliver }))
    await waitFor(() => expect(deliver).toHaveBeenCalledTimes(1))
    expect(deliver).toHaveBeenCalledWith('music', expect.stringContaining('Believe Me'))
    const counters = diagnostics.snapshot().counters
    expect(counters['reflex.ambient_dispatch_failed']).toBe(1)
  })
})
