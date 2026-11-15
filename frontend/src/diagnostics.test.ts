import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { diagnostics } from './diagnostics'

describe('diagnostics', () => {
  let warnSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    diagnostics.resetForTest()
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
  })

  afterEach(() => {
    warnSpy.mockRestore()
  })

  it('increments a counter and reflects it in the snapshot', () => {
    diagnostics.increment('pet.poke_failed')
    diagnostics.increment('pet.poke_failed')
    const snapshot = diagnostics.snapshot()
    expect(snapshot.counters['pet.poke_failed']).toBe(2)
  })

  it('tracks per-name counters independently', () => {
    diagnostics.increment('pet.poke_failed')
    diagnostics.increment('presence.request_failed')
    diagnostics.increment('presence.request_failed')
    const snapshot = diagnostics.snapshot()
    expect(snapshot.counters['pet.poke_failed']).toBe(1)
    expect(snapshot.counters['presence.request_failed']).toBe(2)
  })

  it('keeps the most recent detail per counter name', () => {
    diagnostics.increment('pet.poke_failed', 'HTTP 422')
    diagnostics.increment('pet.poke_failed', 'HTTP 500')
    const snapshot = diagnostics.snapshot()
    expect(snapshot.lastDetail['pet.poke_failed']).toBe('HTTP 500')
  })

  it('records a lastSeen timestamp that updates on repeat increments', () => {
    vi.useFakeTimers()
    vi.setSystemTime(1_000)
    diagnostics.increment('presence.request_failed')
    vi.setSystemTime(5_000)
    diagnostics.increment('presence.request_failed')
    const snapshot = diagnostics.snapshot()
    expect(snapshot.lastSeen['presence.request_failed']).toBe(5_000)
    vi.useRealTimers()
  })

  it('appends every increment to the ring buffer in order', () => {
    diagnostics.increment('a', '1')
    diagnostics.increment('b', '2')
    diagnostics.increment('a', '3')
    const snapshot = diagnostics.snapshot()
    expect(snapshot.events.map((e) => [e.name, e.detail])).toEqual([
      ['a', '1'],
      ['b', '2'],
      ['a', '3'],
    ])
  })

  it('caps the ring buffer at ~25 events, dropping the oldest first', () => {
    for (let i = 0; i < 30; i++) {
      diagnostics.increment('spam', String(i))
    }
    const snapshot = diagnostics.snapshot()
    expect(snapshot.events.length).toBe(25)
    // Oldest 5 (0..4) dropped; buffer starts at 5.
    expect(snapshot.events[0].detail).toBe('5')
    expect(snapshot.events[snapshot.events.length - 1].detail).toBe('29')
    // Counter itself is NOT capped — only the ring buffer is.
    expect(snapshot.counters['spam']).toBe(30)
  })

  it('notifies subscribers on every increment with a fresh snapshot', () => {
    const received: number[] = []
    const unsubscribe = diagnostics.subscribe((snapshot) => {
      received.push(snapshot.counters['x'] ?? 0)
    })
    diagnostics.increment('x')
    diagnostics.increment('x')
    unsubscribe()
    diagnostics.increment('x')
    expect(received).toEqual([1, 2])
  })

  it('stops notifying after unsubscribe', () => {
    let calls = 0
    const unsubscribe = diagnostics.subscribe(() => {
      calls += 1
    })
    diagnostics.increment('y')
    unsubscribe()
    diagnostics.increment('y')
    diagnostics.increment('y')
    expect(calls).toBe(1)
  })

  it('console.warns exactly once per counter name, even across many increments', () => {
    diagnostics.increment('noisy')
    diagnostics.increment('noisy')
    diagnostics.increment('noisy')
    expect(warnSpy).toHaveBeenCalledTimes(1)
    expect(warnSpy.mock.calls[0][0]).toContain('noisy')
  })

  it('warns independently per distinct counter name', () => {
    diagnostics.increment('one')
    diagnostics.increment('two')
    diagnostics.increment('one')
    expect(warnSpy).toHaveBeenCalledTimes(2)
  })

  it('includes the detail in the warn-once console message when provided', () => {
    diagnostics.increment('pet.poke_failed', 'HTTP 422')
    expect(warnSpy.mock.calls[0][0]).toContain('HTTP 422')
  })

  it('snapshot() returns an independent copy that does not mutate live', () => {
    diagnostics.increment('z')
    const snap1 = diagnostics.snapshot()
    diagnostics.increment('z')
    expect(snap1.counters['z']).toBe(1)
    expect(diagnostics.snapshot().counters['z']).toBe(2)
  })

  it('resetForTest clears counters, events, and warn-once memory', () => {
    diagnostics.increment('r')
    diagnostics.resetForTest()
    expect(diagnostics.snapshot().counters).toEqual({})
    expect(diagnostics.snapshot().events).toEqual([])
    diagnostics.increment('r')
    // warn-once memory was cleared too, so this fires again (once before
    // reset, once after).
    expect(warnSpy).toHaveBeenCalledTimes(2)
  })
})
