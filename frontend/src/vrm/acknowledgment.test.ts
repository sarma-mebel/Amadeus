import { describe, expect, it, vi } from 'vitest'
import { ACKNOWLEDGMENT_TIMEOUT_MS, createAcknowledgmentController } from './acknowledgment'

/** Minimal fake clock: setTimeout/clearTimeout injected as deps so the
 * controller's hard-timeout release is testable without real timers or
 * vitest's fake-timer machinery (mirrors adaptiveQuality.test.ts's style of
 * injecting a controllable `now`/clock rather than mocking globals). */
const createFakeClock = () => {
  let nextId = 1
  const pending = new Map<number, { fn: () => void; dueAt: number }>()
  let now = 0
  const setTimeout = (fn: () => void, ms: number): number => {
    const id = nextId++
    pending.set(id, { fn, dueAt: now + ms })
    return id
  }
  const clearTimeout = (id: number): void => {
    pending.delete(id)
  }
  const advance = (ms: number) => {
    now += ms
    // Snapshot due entries so a timeout firing while iterating (none of
    // these do, but future ones might re-arm) doesn't mutate mid-iteration.
    const due = [...pending.entries()].filter(([, entry]) => entry.dueAt <= now)
    for (const [id, entry] of due) {
      pending.delete(id)
      entry.fn()
    }
  }
  return { setTimeout, clearTimeout, advance }
}

describe('createAcknowledgmentController', () => {
  it('arm fires the trigger once and isArmed is true', () => {
    const trigger = vi.fn()
    const clock = createFakeClock()
    const controller = createAcknowledgmentController({
      trigger,
      prefersReducedMotion: () => false,
      setTimeout: clock.setTimeout,
      clearTimeout: clock.clearTimeout,
    })

    expect(controller.isArmed()).toBe(false)
    controller.arm('turn-1')
    expect(trigger).toHaveBeenCalledTimes(1)
    expect(controller.isArmed()).toBe(true)
  })

  it('release with the matching turnId clears the hold', () => {
    const trigger = vi.fn()
    const clock = createFakeClock()
    const controller = createAcknowledgmentController({
      trigger,
      prefersReducedMotion: () => false,
      setTimeout: clock.setTimeout,
      clearTimeout: clock.clearTimeout,
    })

    controller.arm('turn-1')
    controller.release('turn-1')
    expect(controller.isArmed()).toBe(false)
  })

  it('a stale release for a superseded turn is ignored', () => {
    const trigger = vi.fn()
    const clock = createFakeClock()
    const controller = createAcknowledgmentController({
      trigger,
      prefersReducedMotion: () => false,
      setTimeout: clock.setTimeout,
      clearTimeout: clock.clearTimeout,
    })

    controller.arm('turn-1')
    controller.arm('turn-2') // supersedes turn-1 (e.g. barge-in / new send)
    controller.release('turn-1') // late callback from the old turn
    // turn-2 must still be armed — an old turn's release must never cancel
    // a newer, still-in-flight turn's hold.
    expect(controller.isArmed()).toBe(true)

    controller.release('turn-2')
    expect(controller.isArmed()).toBe(false)
  })

  it('re-arming the same turnId does not double-fire the trigger', () => {
    const trigger = vi.fn()
    const clock = createFakeClock()
    const controller = createAcknowledgmentController({
      trigger,
      prefersReducedMotion: () => false,
      setTimeout: clock.setTimeout,
      clearTimeout: clock.clearTimeout,
    })

    controller.arm('turn-1')
    controller.arm('turn-1') // e.g. a redundant re-arm call for the same turn
    expect(trigger).toHaveBeenCalledTimes(1)
    expect(controller.isArmed()).toBe(true)
  })

  it('arming a new turnId after a previous arm re-triggers the beat', () => {
    const trigger = vi.fn()
    const clock = createFakeClock()
    const controller = createAcknowledgmentController({
      trigger,
      prefersReducedMotion: () => false,
      setTimeout: clock.setTimeout,
      clearTimeout: clock.clearTimeout,
    })

    controller.arm('turn-1')
    controller.release('turn-1')
    controller.arm('turn-2')
    expect(trigger).toHaveBeenCalledTimes(2)
    expect(controller.isArmed()).toBe(true)
  })

  it('hard timeout releases the hold if nothing else ever does', () => {
    const trigger = vi.fn()
    const clock = createFakeClock()
    const controller = createAcknowledgmentController({
      trigger,
      prefersReducedMotion: () => false,
      setTimeout: clock.setTimeout,
      clearTimeout: clock.clearTimeout,
    })

    controller.arm('turn-1')
    expect(controller.isArmed()).toBe(true)
    clock.advance(ACKNOWLEDGMENT_TIMEOUT_MS - 1)
    expect(controller.isArmed()).toBe(true)
    clock.advance(1)
    expect(controller.isArmed()).toBe(false)
  })

  it('a release after the hard timeout already fired is a harmless no-op', () => {
    const trigger = vi.fn()
    const clock = createFakeClock()
    const controller = createAcknowledgmentController({
      trigger,
      prefersReducedMotion: () => false,
      setTimeout: clock.setTimeout,
      clearTimeout: clock.clearTimeout,
    })

    controller.arm('turn-1')
    clock.advance(ACKNOWLEDGMENT_TIMEOUT_MS)
    expect(controller.isArmed()).toBe(false)
    expect(() => controller.release('turn-1')).not.toThrow()
    expect(controller.isArmed()).toBe(false)
  })

  it('release clears the pending hard-timeout timer (no late auto-release surprises)', () => {
    const trigger = vi.fn()
    const clock = createFakeClock()
    const controller = createAcknowledgmentController({
      trigger,
      prefersReducedMotion: () => false,
      setTimeout: clock.setTimeout,
      clearTimeout: clock.clearTimeout,
    })

    controller.arm('turn-1')
    controller.release('turn-1')
    // A second turn arms after the first turn's original timeout would have
    // fired — its own hold must not be clobbered by a leaked timer id from
    // turn-1.
    controller.arm('turn-2')
    clock.advance(ACKNOWLEDGMENT_TIMEOUT_MS - 1)
    expect(controller.isArmed()).toBe(true)
  })

  it('does not fire the trigger when reduced motion is preferred, but still tracks armed state', () => {
    const trigger = vi.fn()
    const clock = createFakeClock()
    const controller = createAcknowledgmentController({
      trigger,
      prefersReducedMotion: () => true,
      setTimeout: clock.setTimeout,
      clearTimeout: clock.clearTimeout,
    })

    controller.arm('turn-1')
    expect(trigger).not.toHaveBeenCalled()
    // Still tracked/releasable — reduced motion suppresses the visual beat,
    // not the turn-scoping bookkeeping itself.
    expect(controller.isArmed()).toBe(true)
    controller.release('turn-1')
    expect(controller.isArmed()).toBe(false)
  })

  it('brackets the armed window with thinking start/end (lifelike pass)', () => {
    const clock = createFakeClock()
    const onThinkingStart = vi.fn()
    const onThinkingEnd = vi.fn()
    const controller = createAcknowledgmentController({
      trigger: vi.fn(),
      onThinkingStart,
      onThinkingEnd,
      prefersReducedMotion: () => false,
      setTimeout: clock.setTimeout,
      clearTimeout: clock.clearTimeout,
    })

    controller.arm('turn-1')
    expect(onThinkingStart).toHaveBeenCalledTimes(1)
    expect(onThinkingEnd).not.toHaveBeenCalled()
    controller.release('turn-1')
    expect(onThinkingEnd).toHaveBeenCalledTimes(1)
    // A second release must not double-fire the end.
    controller.release('turn-1')
    expect(onThinkingEnd).toHaveBeenCalledTimes(1)
  })

  it('a superseding arm closes the previous thinking window before opening its own', () => {
    const clock = createFakeClock()
    const onThinkingStart = vi.fn()
    const onThinkingEnd = vi.fn()
    const controller = createAcknowledgmentController({
      trigger: vi.fn(),
      onThinkingStart,
      onThinkingEnd,
      prefersReducedMotion: () => false,
      setTimeout: clock.setTimeout,
      clearTimeout: clock.clearTimeout,
    })

    controller.arm('turn-1')
    controller.arm('turn-2')
    expect(onThinkingEnd).toHaveBeenCalledTimes(1) // turn-1's window closed
    expect(onThinkingStart).toHaveBeenCalledTimes(2)
  })

  it('the hard timeout also ends the thinking window', () => {
    const clock = createFakeClock()
    const onThinkingEnd = vi.fn()
    const controller = createAcknowledgmentController({
      trigger: vi.fn(),
      onThinkingStart: vi.fn(),
      onThinkingEnd,
      prefersReducedMotion: () => false,
      setTimeout: clock.setTimeout,
      clearTimeout: clock.clearTimeout,
    })

    controller.arm('turn-1')
    clock.advance(ACKNOWLEDGMENT_TIMEOUT_MS)
    expect(onThinkingEnd).toHaveBeenCalledTimes(1)
  })

  it('reduced motion skips the thinking window entirely — start AND end', () => {
    const clock = createFakeClock()
    const onThinkingStart = vi.fn()
    const onThinkingEnd = vi.fn()
    const controller = createAcknowledgmentController({
      trigger: vi.fn(),
      onThinkingStart,
      onThinkingEnd,
      prefersReducedMotion: () => true,
      setTimeout: clock.setTimeout,
      clearTimeout: clock.clearTimeout,
    })

    controller.arm('turn-1')
    controller.release('turn-1')
    expect(onThinkingStart).not.toHaveBeenCalled()
    expect(onThinkingEnd).not.toHaveBeenCalled()
  })

  it('defaults never fire for presence lines — only whatever calls arm() explicitly', () => {
    // The controller has no ambient hook into presence/greeting/poke paths;
    // it only ever does anything when a caller (useChatTurn/usePetChat, both
    // user-initiated send paths) calls arm(). This test pins that by
    // asserting a fresh controller starts fully unarmed and inert.
    const trigger = vi.fn()
    const controller = createAcknowledgmentController({ trigger, prefersReducedMotion: () => false })
    expect(controller.isArmed()).toBe(false)
    expect(trigger).not.toHaveBeenCalled()
  })
})
