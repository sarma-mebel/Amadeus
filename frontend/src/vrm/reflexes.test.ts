// @vitest-environment jsdom
//
// Pure engine coverage for reflexes.ts: the shipped table's event->action
// routing, the centralized gates, per-reflex cooldowns with a fake clock,
// the CustomEvent matches filters, the never-throws contract, and
// attach/detach symmetry. jsdom needed only for window event dispatch
// (same rationale as attentionEvents.test.ts).
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createReflexEngine, REFLEX_TABLE, type ReflexEngine, type ReflexPrimitives } from './reflexes'
import { diagnostics } from '../diagnostics'

const buildPrimitives = (): ReflexPrimitives & {
  requestBlink: ReturnType<typeof vi.fn>
  snapGaze: ReturnType<typeof vi.fn>
  queueCuriousTilt: ReturnType<typeof vi.fn>
  expressionFlick: ReturnType<typeof vi.fn>
  isSpeaking: ReturnType<typeof vi.fn>
  isDozing: ReturnType<typeof vi.fn>
  isTucked: ReturnType<typeof vi.fn>
  prefersReducedMotion: ReturnType<typeof vi.fn>
  now: ReturnType<typeof vi.fn>
} => ({
  requestBlink: vi.fn(),
  snapGaze: vi.fn(),
  queueCuriousTilt: vi.fn(),
  expressionFlick: vi.fn(),
  isSpeaking: vi.fn().mockReturnValue(false),
  isDozing: vi.fn().mockReturnValue(false),
  isTucked: vi.fn().mockReturnValue(false),
  prefersReducedMotion: vi.fn().mockReturnValue(false),
  now: vi.fn().mockReturnValue(100_000),
})

const ambientEvent = (detail: Record<string, unknown>): CustomEvent =>
  new CustomEvent('kurisu:ambient', { detail })

describe('createReflexEngine (shipped table)', () => {
  let primitives: ReturnType<typeof buildPrimitives>
  let engine: ReflexEngine

  beforeEach(() => {
    diagnostics.resetForTest()
    primitives = buildPrimitives()
    engine = createReflexEngine(primitives)
    engine.attach()
  })

  afterEach(() => {
    engine.detach()
  })

  it('barge-in-catch: kurisu:barge-in fires a full blink + gaze snap', () => {
    window.dispatchEvent(new Event('kurisu:barge-in'))
    expect(primitives.requestBlink).toHaveBeenCalledWith(1)
    expect(primitives.snapGaze).toHaveBeenCalledTimes(1)
  })

  it('barge-in-catch is deliberately NOT gated on speaking (being interrupted mid-speech is the trigger)', () => {
    primitives.isSpeaking.mockReturnValue(true)
    window.dispatchEvent(new Event('kurisu:barge-in'))
    expect(primitives.requestBlink).toHaveBeenCalledWith(1)
  })

  it('window-focus-reengage: focus fires a gaze snap + soft blink', () => {
    window.dispatchEvent(new Event('focus'))
    expect(primitives.snapGaze).toHaveBeenCalledTimes(1)
    expect(primitives.requestBlink).toHaveBeenCalledWith(0.7)
  })

  it('battery-critical-glance: only a critical battery ambient event fires the tilt + sad flick', () => {
    window.dispatchEvent(ambientEvent({ kind: 'battery', severity: 'low' }))
    expect(primitives.queueCuriousTilt).not.toHaveBeenCalled()

    window.dispatchEvent(ambientEvent({ kind: 'battery', severity: 'critical' }))
    expect(primitives.queueCuriousTilt).toHaveBeenCalledTimes(1)
    expect(primitives.expressionFlick).toHaveBeenCalledWith('sad', 0.2, 1800)
  })

  it('music-tilt: a music ambient event routes to the curious tilt, focus kind does not', () => {
    window.dispatchEvent(ambientEvent({ kind: 'focus', detail: 'long focus stretch' }))
    expect(primitives.queueCuriousTilt).not.toHaveBeenCalled()

    window.dispatchEvent(ambientEvent({ kind: 'music', detail: 'now playing' }))
    expect(primitives.queueCuriousTilt).toHaveBeenCalledTimes(1)
    expect(primitives.expressionFlick).not.toHaveBeenCalled() // music never pulls a face
  })

  it('untuck-blink: a tuck-change event fires only when she is now UNtucked', () => {
    primitives.isTucked.mockReturnValue(true) // change landed on tucked -> this was a tuck
    window.dispatchEvent(new Event('kurisu:pet-tuck-change'))
    expect(primitives.requestBlink).not.toHaveBeenCalled()

    primitives.isTucked.mockReturnValue(false) // now untucked -> this was the untuck
    window.dispatchEvent(new Event('kurisu:pet-tuck-change'))
    expect(primitives.requestBlink).toHaveBeenCalledWith(1)
    expect(primitives.snapGaze).toHaveBeenCalledTimes(1)
  })

  it.each([
    ['prefersReducedMotion', 'reducedMotion'],
    ['isSpeaking', 'notWhileSpeaking'],
    ['isDozing', 'notWhileDozing'],
    ['isTucked', 'notWhileTucked'],
  ] as const)('%s gate blocks music-tilt (all four gates on that row)', (probe) => {
    primitives[probe].mockReturnValue(true)
    window.dispatchEvent(ambientEvent({ kind: 'music' }))
    expect(primitives.queueCuriousTilt).not.toHaveBeenCalled()
  })

  it('reducedMotion gate blocks barge-in-catch', () => {
    primitives.prefersReducedMotion.mockReturnValue(true)
    window.dispatchEvent(new Event('kurisu:barge-in'))
    expect(primitives.requestBlink).not.toHaveBeenCalled()
    expect(primitives.snapGaze).not.toHaveBeenCalled()
  })

  it('speaking and dozing gates block window-focus-reengage', () => {
    primitives.isSpeaking.mockReturnValue(true)
    window.dispatchEvent(new Event('focus'))
    primitives.isSpeaking.mockReturnValue(false)
    primitives.isDozing.mockReturnValue(true)
    window.dispatchEvent(new Event('focus'))
    expect(primitives.snapGaze).not.toHaveBeenCalled()
  })

  it('cooldown: a second event inside the window is a silent no-op, and re-allows after expiry', () => {
    primitives.now.mockReturnValue(100_000)
    window.dispatchEvent(new Event('kurisu:barge-in'))
    primitives.now.mockReturnValue(100_000 + 7_999) // barge-in cooldown is 8000
    window.dispatchEvent(new Event('kurisu:barge-in'))
    expect(primitives.snapGaze).toHaveBeenCalledTimes(1)

    primitives.now.mockReturnValue(100_000 + 8_000)
    window.dispatchEvent(new Event('kurisu:barge-in'))
    expect(primitives.snapGaze).toHaveBeenCalledTimes(2)
  })

  it('cooldowns are PER-reflex: two reflexes on the same event name cool down independently', () => {
    window.dispatchEvent(ambientEvent({ kind: 'music' }))
    expect(primitives.queueCuriousTilt).toHaveBeenCalledTimes(1)
    // music-tilt is now cooling down; battery-critical-glance is not.
    window.dispatchEvent(ambientEvent({ kind: 'battery', severity: 'critical' }))
    expect(primitives.queueCuriousTilt).toHaveBeenCalledTimes(2)
  })

  it('a gated event does NOT consume the cooldown', () => {
    primitives.isSpeaking.mockReturnValue(true)
    window.dispatchEvent(ambientEvent({ kind: 'music' }))
    primitives.isSpeaking.mockReturnValue(false)
    window.dispatchEvent(ambientEvent({ kind: 'music' }))
    expect(primitives.queueCuriousTilt).toHaveBeenCalledTimes(1)
  })

  it('an ambient event with no detail matches nothing and never throws', () => {
    window.dispatchEvent(new Event('kurisu:ambient'))
    expect(primitives.queueCuriousTilt).not.toHaveBeenCalled()
    expect(primitives.expressionFlick).not.toHaveBeenCalled()
  })

  it('detach removes the listeners', () => {
    engine.detach()
    window.dispatchEvent(new Event('kurisu:barge-in'))
    expect(primitives.requestBlink).not.toHaveBeenCalled()
    engine.attach() // afterEach detaches again
  })
})

describe('createReflexEngine (custom defs)', () => {
  beforeEach(() => {
    diagnostics.resetForTest()
  })

  it('a throwing action is swallowed and counted, and the engine keeps working', () => {
    const primitives = buildPrimitives()
    const engine = createReflexEngine(primitives, [
      {
        id: 'boom',
        event: 'kurisu:boom',
        cooldownMs: 0,
        gates: [],
        action: () => {
          throw new Error('primitive died')
        },
      },
      {
        id: 'fine',
        event: 'kurisu:boom',
        cooldownMs: 0,
        gates: [],
        action: (p) => p.snapGaze(),
      },
    ])
    engine.attach()
    try {
      expect(() => window.dispatchEvent(new Event('kurisu:boom'))).not.toThrow()
      const snapshot = diagnostics.snapshot()
      expect(snapshot.counters['reflex.action_failed']).toBe(1)
      expect(snapshot.lastDetail['reflex.action_failed']).toContain('boom')
      expect(primitives.snapGaze).toHaveBeenCalledTimes(1) // sibling reflex unaffected
    } finally {
      engine.detach()
    }
  })

  it('fire() applies gates and cooldowns identically to a real event', () => {
    const primitives = buildPrimitives()
    const action = vi.fn()
    const engine = createReflexEngine(primitives, [
      { id: 'manual', event: 'kurisu:manual', cooldownMs: 5_000, gates: ['notWhileSpeaking'], action },
    ])

    primitives.isSpeaking.mockReturnValue(true)
    engine.fire('manual')
    expect(action).not.toHaveBeenCalled()

    primitives.isSpeaking.mockReturnValue(false)
    engine.fire('manual')
    expect(action).toHaveBeenCalledTimes(1)
    engine.fire('manual') // inside cooldown
    expect(action).toHaveBeenCalledTimes(1)

    primitives.now.mockReturnValue(100_000 + 5_000)
    engine.fire('manual')
    expect(action).toHaveBeenCalledTimes(2)
  })

  it('fire() with an unknown id is a silent no-op', () => {
    const engine = createReflexEngine(buildPrimitives(), [])
    expect(() => engine.fire('nope')).not.toThrow()
  })

  it('every shipped table row has a positive cooldown and a unique id', () => {
    const ids = REFLEX_TABLE.map((def) => def.id)
    expect(new Set(ids).size).toBe(ids.length)
    for (const def of REFLEX_TABLE) {
      expect(def.cooldownMs).toBeGreaterThan(0)
    }
  })
})
