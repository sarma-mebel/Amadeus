import { describe, expect, it } from 'vitest'
import {
  advanceDoze,
  createDozeMachineState,
  DOZING_AFTER_DROWSY_S,
  DROWSY_AFTER_IDLE_S,
  isDozing,
  isNightHour,
  WAKING_DURATION_S,
  wakeDoze,
} from './petDoze'

const advanceSeconds = (
  state: ReturnType<typeof createDozeMachineState>,
  totalSeconds: number,
  inputs: { hour: number; isIdle: boolean },
  stepSeconds = 1,
) => {
  let remaining = totalSeconds
  while (remaining > 0) {
    const step = Math.min(stepSeconds, remaining)
    advanceDoze(state, step, inputs)
    remaining -= step
  }
}

describe('isNightHour', () => {
  it('treats 23:00-05:59 as night', () => {
    expect(isNightHour(23)).toBe(true)
    expect(isNightHour(0)).toBe(true)
    expect(isNightHour(3)).toBe(true)
    expect(isNightHour(5)).toBe(true)
  })

  it('treats 06:00-22:59 as day', () => {
    expect(isNightHour(6)).toBe(false)
    expect(isNightHour(14)).toBe(false)
    expect(isNightHour(22)).toBe(false)
  })
})

describe('advanceDoze', () => {
  it('starts awake', () => {
    const state = createDozeMachineState()
    expect(state.phase).toBe('awake')
    expect(state.eyeCloseAmount).toBe(0)
  })

  it('never drifts toward drowsy during the day, no matter how idle', () => {
    const state = createDozeMachineState()
    advanceSeconds(state, DROWSY_AFTER_IDLE_S + 60, { hour: 14, isIdle: true }, 5)
    expect(state.phase).toBe('awake')
  })

  it('never drifts toward drowsy at night while activity keeps happening', () => {
    const state = createDozeMachineState()
    advanceSeconds(state, DROWSY_AFTER_IDLE_S + 60, { hour: 2, isIdle: false }, 5)
    expect(state.phase).toBe('awake')
  })

  it('goes awake -> drowsy after ~10 idle minutes at night', () => {
    const state = createDozeMachineState()
    advanceSeconds(state, DROWSY_AFTER_IDLE_S - 1, { hour: 2, isIdle: true }, 5)
    expect(state.phase).toBe('awake')
    advanceDoze(state, 2, { hour: 2, isIdle: true })
    expect(state.phase).toBe('drowsy')
  })

  it('goes drowsy -> dozing after ~2 more idle minutes', () => {
    const state = createDozeMachineState()
    // Fine-grained (1s) steps so the phase transition lands exactly on the
    // boundary instead of overshooting by a partial step — a coarser step
    // size can carry a little of the triggering tick's time into the new
    // phase (harmless in the render loop, where steps are sub-frame, but
    // it would make an exact boundary assertion here flaky).
    advanceSeconds(state, DROWSY_AFTER_IDLE_S, { hour: 2, isIdle: true }, 1)
    expect(state.phase).toBe('drowsy')
    expect(state.phaseElapsedS).toBe(0)
    advanceSeconds(state, DOZING_AFTER_DROWSY_S - 1, { hour: 2, isIdle: true }, 1)
    expect(state.phase).toBe('drowsy')
    advanceDoze(state, 2, { hour: 2, isIdle: true })
    expect(state.phase).toBe('dozing')
  })

  it('eyeCloseAmount reaches ~1 by the time she is fully dozing', () => {
    const state = createDozeMachineState()
    advanceSeconds(state, DROWSY_AFTER_IDLE_S + DOZING_AFTER_DROWSY_S + 1, { hour: 2, isIdle: true }, 5)
    expect(state.phase).toBe('dozing')
    expect(state.eyeCloseAmount).toBeCloseTo(1, 1)
  })

  it('activity while drowsy triggers a waking transition, not an instant snap to awake', () => {
    const state = createDozeMachineState()
    advanceSeconds(state, DROWSY_AFTER_IDLE_S + 1, { hour: 2, isIdle: true }, 5)
    expect(state.phase).toBe('drowsy')
    advanceDoze(state, 0.1, { hour: 2, isIdle: false })
    expect(state.phase).toBe('waking')
  })

  it('activity while dozing triggers waking, not an instant snap to awake', () => {
    const state = createDozeMachineState()
    advanceSeconds(state, DROWSY_AFTER_IDLE_S + DOZING_AFTER_DROWSY_S + 1, { hour: 2, isIdle: true }, 5)
    expect(state.phase).toBe('dozing')
    advanceDoze(state, 0.1, { hour: 2, isIdle: false })
    expect(state.phase).toBe('waking')
  })

  it('waking always completes its full fixed duration before returning to awake', () => {
    const state = createDozeMachineState()
    advanceSeconds(state, DROWSY_AFTER_IDLE_S + DOZING_AFTER_DROWSY_S + 1, { hour: 2, isIdle: true }, 5)
    advanceDoze(state, 0.1, { hour: 2, isIdle: false }) // -> waking
    expect(state.phase).toBe('waking')
    advanceDoze(state, WAKING_DURATION_S - 0.1, { hour: 2, isIdle: true })
    expect(state.phase).toBe('waking')
    advanceDoze(state, 0.2, { hour: 2, isIdle: true })
    expect(state.phase).toBe('awake')
    expect(state.eyeCloseAmount).toBe(0)
  })

  it('waking from a partial drowsy droop interpolates from the droop, never snapping shut', () => {
    const state = createDozeMachineState()
    // Just into drowsy: the lids have only partially drooped.
    advanceSeconds(state, DROWSY_AFTER_IDLE_S, { hour: 2, isIdle: true }, 1)
    advanceSeconds(state, 10, { hour: 2, isIdle: true }, 1)
    expect(state.phase).toBe('drowsy')
    const droop = state.eyeCloseAmount
    expect(droop).toBeGreaterThan(0)
    expect(droop).toBeLessThan(0.5)

    // Activity wakes her: the very next waking frame must start at (just
    // under) the droop, not snap to fully closed for a frame first.
    advanceDoze(state, 0.01, { hour: 2, isIdle: false })
    expect(state.phase).toBe('waking')
    expect(state.eyeCloseAmount).toBeLessThanOrEqual(droop)
    // And it eases down from there over the wake duration.
    advanceDoze(state, WAKING_DURATION_S / 2, { hour: 2, isIdle: true })
    expect(state.eyeCloseAmount).toBeLessThan(droop)
    expect(state.eyeCloseAmount).toBeGreaterThan(0)
  })

  it('waking from a full doze still interpolates down from fully closed', () => {
    const state = createDozeMachineState()
    advanceSeconds(state, DROWSY_AFTER_IDLE_S + DOZING_AFTER_DROWSY_S + 1, { hour: 2, isIdle: true }, 5)
    expect(state.phase).toBe('dozing')
    expect(state.eyeCloseAmount).toBe(1)
    wakeDoze(state)
    advanceDoze(state, WAKING_DURATION_S / 2, { hour: 2, isIdle: true })
    expect(state.eyeCloseAmount).toBeCloseTo(0.5, 1)
  })

  it('crossing into daytime hours while drowsy also triggers waking', () => {
    const state = createDozeMachineState()
    advanceSeconds(state, DROWSY_AFTER_IDLE_S + 1, { hour: 2, isIdle: true }, 5)
    expect(state.phase).toBe('drowsy')
    advanceDoze(state, 0.1, { hour: 8, isIdle: true })
    expect(state.phase).toBe('waking')
  })

  it('resets idle progress toward drowsy when activity interrupts an awake stretch', () => {
    const state = createDozeMachineState()
    advanceSeconds(state, DROWSY_AFTER_IDLE_S - 60, { hour: 2, isIdle: true }, 5)
    expect(state.phase).toBe('awake')
    advanceDoze(state, 1, { hour: 2, isIdle: false }) // interrupts
    expect(state.phaseElapsedS).toBeLessThan(2)
    // Even after the ORIGINAL elapsed time has passed again, she should not
    // yet be drowsy because the clock was reset.
    advanceSeconds(state, 61, { hour: 2, isIdle: true }, 5)
    expect(state.phase).toBe('awake')
  })
})

describe('isDozing', () => {
  it('is true only in the dozing phase', () => {
    const state = createDozeMachineState()
    expect(isDozing(state)).toBe(false)
    state.phase = 'drowsy'
    expect(isDozing(state)).toBe(false)
    state.phase = 'dozing'
    expect(isDozing(state)).toBe(true)
    state.phase = 'waking'
    expect(isDozing(state)).toBe(false)
  })
})

describe('wakeDoze', () => {
  it('returns true and transitions to waking when called while dozing', () => {
    const state = createDozeMachineState()
    state.phase = 'dozing'
    state.phaseElapsedS = 42
    const result = wakeDoze(state)
    expect(result).toBe(true)
    expect(state.phase).toBe('waking')
    expect(state.phaseElapsedS).toBe(0)
  })

  it('returns false but still transitions to waking when called while merely drowsy', () => {
    const state = createDozeMachineState()
    state.phase = 'drowsy'
    const result = wakeDoze(state)
    expect(result).toBe(false)
    expect(state.phase).toBe('waking')
  })

  it('returns false and is a no-op when already awake', () => {
    const state = createDozeMachineState()
    const result = wakeDoze(state)
    expect(result).toBe(false)
    expect(state.phase).toBe('awake')
  })

  it('returns false and is a no-op when already waking', () => {
    const state = createDozeMachineState()
    state.phase = 'waking'
    state.phaseElapsedS = 0.5
    const result = wakeDoze(state)
    expect(result).toBe(false)
    expect(state.phase).toBe('waking')
    expect(state.phaseElapsedS).toBe(0.5) // untouched — did not reset an in-progress stir
  })
})
