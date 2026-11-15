// Pure decision-logic coverage for idleThrottle.ts: the cursor-gaze wake
// tracker's side-effecting movement detection, the render loop's idle
// predicate composed from injected disqualifier checks, and the doze-aware
// frame-interval selector. No THREE/DOM needed — every collaborator is a
// plain stub.
import { describe, expect, it } from 'vitest'
import {
  createCursorGazeWakeTracker,
  createIdleFrameIntervalS,
  createIsIdleNow,
  CURSOR_GAZE_WAKE_THRESHOLD,
  type IsIdleNowDeps,
} from './idleThrottle'
import { createCursorGazeState } from './cursorGaze'
import { createDozeMachineState, type DozeMachineState } from './petDoze'

describe('createCursorGazeWakeTracker', () => {
  it('reports no movement while inactive', () => {
    const state = createCursorGazeState()
    const moved = createCursorGazeWakeTracker(state)
    state.targetLateral = 1
    state.targetVertical = 1
    // active stays false (default) -> always reports false regardless of movement.
    expect(moved()).toBe(false)
  })

  it('reports movement beyond the threshold once active', () => {
    const state = createCursorGazeState()
    state.active = true
    const moved = createCursorGazeWakeTracker(state)
    // First call establishes baseline (already matches current target) -> false.
    expect(moved()).toBe(false)
    state.targetLateral += CURSOR_GAZE_WAKE_THRESHOLD + 0.01
    expect(moved()).toBe(true)
  })

  it('does not report movement at or under the threshold', () => {
    const state = createCursorGazeState()
    state.active = true
    const moved = createCursorGazeWakeTracker(state)
    moved() // baseline
    state.targetVertical += CURSOR_GAZE_WAKE_THRESHOLD
    expect(moved()).toBe(false)
  })

  it('updates its baseline on every call (side effect), so a small drift never accumulates into a false positive', () => {
    const state = createCursorGazeState()
    state.active = true
    const moved = createCursorGazeWakeTracker(state)
    moved()
    // Several tiny moves, each under threshold on its own.
    for (let i = 0; i < 5; i++) {
      state.targetLateral += CURSOR_GAZE_WAKE_THRESHOLD * 0.3
      expect(moved()).toBe(false)
    }
  })
})

describe('createIsIdleNow', () => {
  const baseDeps = (overrides: Partial<IsIdleNowDeps> = {}): IsIdleNowDeps => ({
    isLipSyncAudible: () => false,
    isMidGesture: () => false,
    isReframing: () => false,
    isLightingLerpActive: () => false,
    isUserDrivingCamera: () => false,
    isPointerOverPetWindow: () => false,
    cursorGazeMovedBeyondThreshold: () => false,
    getLastUserInputAt: () => -10000,
    getLastPointerMoveAt: () => -10000,
    getLastTypingAt: () => -10000,
    getLastInteractionAt: () => -10000,
    idleActivityWindowMs: 3000,
    ...overrides,
  })

  it('is idle when nothing disqualifies it and all activity is outside the window', () => {
    const isIdleNow = createIsIdleNow(baseDeps())
    expect(isIdleNow(10000)).toBe(true)
  })

  it('is not idle while lip sync is audible', () => {
    const isIdleNow = createIsIdleNow(baseDeps({ isLipSyncAudible: () => true }))
    expect(isIdleNow(10000)).toBe(false)
  })

  it('is not idle mid-gesture', () => {
    const isIdleNow = createIsIdleNow(baseDeps({ isMidGesture: () => true }))
    expect(isIdleNow(10000)).toBe(false)
  })

  it('is not idle while reframing or the lighting lerp is active', () => {
    expect(createIsIdleNow(baseDeps({ isReframing: () => true }))(10000)).toBe(false)
    expect(createIsIdleNow(baseDeps({ isLightingLerpActive: () => true }))(10000)).toBe(false)
  })

  it('is not idle while the user is driving the camera or hovering the pet window', () => {
    expect(createIsIdleNow(baseDeps({ isUserDrivingCamera: () => true }))(10000)).toBe(false)
    expect(createIsIdleNow(baseDeps({ isPointerOverPetWindow: () => true }))(10000)).toBe(false)
  })

  it('is not idle when the cursor-gaze tracker reports movement', () => {
    const isIdleNow = createIsIdleNow(baseDeps({ cursorGazeMovedBeyondThreshold: () => true }))
    expect(isIdleNow(10000)).toBe(false)
  })

  it('is not idle when any activity timestamp is within the window', () => {
    const now = 10000
    expect(
      createIsIdleNow(baseDeps({ getLastUserInputAt: () => now - 100 }))(now)
    ).toBe(false)
    expect(
      createIsIdleNow(baseDeps({ getLastPointerMoveAt: () => now - 100 }))(now)
    ).toBe(false)
    expect(createIsIdleNow(baseDeps({ getLastTypingAt: () => now - 100 }))(now)).toBe(false)
    expect(
      createIsIdleNow(baseDeps({ getLastInteractionAt: () => now - 100 }))(now)
    ).toBe(false)
  })

  it('is idle again once activity falls exactly outside the window', () => {
    const now = 10000
    const isIdleNow = createIsIdleNow(baseDeps({ getLastUserInputAt: () => now - 3000 }))
    expect(isIdleNow(now)).toBe(true)
  })

  it('calls cursorGazeMovedBeyondThreshold unconditionally even when an earlier check already disqualifies idle', () => {
    let calls = 0
    const isIdleNow = createIsIdleNow(
      baseDeps({
        isLipSyncAudible: () => true,
        cursorGazeMovedBeyondThreshold: () => {
          calls += 1
          return false
        },
      })
    )
    isIdleNow(10000)
    expect(calls).toBe(1)
  })
})

describe('createIdleFrameIntervalS', () => {
  const dozingState = (): DozeMachineState => {
    const state = createDozeMachineState()
    state.phase = 'dozing'
    return state
  }

  it('returns 1/10 while dozing, regardless of variant', () => {
    expect(createIdleFrameIntervalS(dozingState(), 'pet')()).toBeCloseTo(1 / 10)
    expect(createIdleFrameIntervalS(dozingState(), 'full')()).toBeCloseTo(1 / 10)
  })

  it('returns 1/15 for the pet variant while awake', () => {
    expect(createIdleFrameIntervalS(createDozeMachineState(), 'pet')()).toBeCloseTo(1 / 15)
  })

  it('returns 1/30 for the full variant while awake', () => {
    expect(createIdleFrameIntervalS(createDozeMachineState(), 'full')()).toBeCloseTo(1 / 30)
  })

  it('re-reads doze state live rather than baking in the value at construction time', () => {
    const state = createDozeMachineState()
    const interval = createIdleFrameIntervalS(state, 'pet')
    expect(interval()).toBeCloseTo(1 / 15)
    state.phase = 'dozing'
    expect(interval()).toBeCloseTo(1 / 10)
  })
})
