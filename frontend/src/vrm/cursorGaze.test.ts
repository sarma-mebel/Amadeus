import { describe, expect, it } from 'vitest'
import {
  CURSOR_LINGER_COOLDOWN_MS,
  CURSOR_LINGER_MS,
  createCursorGazeState,
  createCursorLingerState,
  cursorToNormalizedOffset,
  updateCursorLinger,
} from './cursorGaze'

describe('cursorToNormalizedOffset', () => {
  it('maps window center to (0, 0)', () => {
    const result = cursorToNormalizedOffset({
      cursor_x: 500,
      cursor_y: 300,
      win_x: 0,
      win_y: 0,
      win_w: 1000,
      win_h: 600,
    })
    expect(result.lateral).toBeCloseTo(0)
    expect(result.vertical).toBeCloseTo(0)
  })

  it('maps the top-left corner to (-1, 1) — left and up', () => {
    const result = cursorToNormalizedOffset({
      cursor_x: 0,
      cursor_y: 0,
      win_x: 0,
      win_y: 0,
      win_w: 1000,
      win_h: 600,
    })
    expect(result.lateral).toBeCloseTo(-1)
    expect(result.vertical).toBeCloseTo(1)
  })

  it('maps the bottom-right corner to (1, -1) — right and down', () => {
    const result = cursorToNormalizedOffset({
      cursor_x: 1000,
      cursor_y: 600,
      win_x: 0,
      win_y: 0,
      win_w: 1000,
      win_h: 600,
    })
    expect(result.lateral).toBeCloseTo(1)
    expect(result.vertical).toBeCloseTo(-1)
  })

  it('clamps a cursor outside the window rect', () => {
    const result = cursorToNormalizedOffset({
      cursor_x: -500,
      cursor_y: 5000,
      win_x: 0,
      win_y: 0,
      win_w: 1000,
      win_h: 600,
    })
    expect(result.lateral).toBe(-1)
    expect(result.vertical).toBe(-1)
  })

  it('accounts for a non-zero window origin', () => {
    const result = cursorToNormalizedOffset({
      cursor_x: 1600,
      cursor_y: 900,
      win_x: 1500,
      win_y: 800,
      win_w: 200,
      win_h: 200,
    })
    expect(result.lateral).toBeCloseTo(0)
    expect(result.vertical).toBeCloseTo(0)
  })

  it('returns zero for a degenerate (zero-size) window rect', () => {
    const result = cursorToNormalizedOffset({
      cursor_x: 10,
      cursor_y: 10,
      win_x: 0,
      win_y: 0,
      win_w: 0,
      win_h: 0,
    })
    expect(result.lateral).toBe(0)
    expect(result.vertical).toBe(0)
  })
})

describe('createCursorGazeState', () => {
  it('starts inactive at the origin', () => {
    const state = createCursorGazeState()
    expect(state.active).toBe(false)
    expect(state.targetLateral).toBe(0)
    expect(state.targetVertical).toBe(0)
    expect(state.lastUpdateAt).toBe(0)
  })
})

describe('updateCursorLinger', () => {
  const ctx = (x: number, y: number) => ({
    cursor_x: x,
    cursor_y: y,
    win_x: 100,
    win_y: 100,
    win_w: 300,
    win_h: 400,
  })

  it('fires after a still cursor lingers inside the window', () => {
    const state = createCursorLingerState()
    expect(updateCursorLinger(state, ctx(200, 200), 0)).toBe(false)
    // Still: below the linger threshold, then past it.
    expect(updateCursorLinger(state, ctx(202, 201), 1000)).toBe(false)
    expect(updateCursorLinger(state, ctx(203, 200), CURSOR_LINGER_MS + 200)).toBe(true)
  })

  it('movement or leaving the window restarts the clock', () => {
    const state = createCursorLingerState()
    updateCursorLinger(state, ctx(200, 200), 0)
    // Big move: clock restarts, so the old start time no longer counts.
    expect(updateCursorLinger(state, ctx(320, 380), CURSOR_LINGER_MS)).toBe(false)
    expect(
      updateCursorLinger(state, ctx(321, 380), CURSOR_LINGER_MS + 1000),
    ).toBe(false)
    // Outside the window entirely: never fires no matter how long.
    const out = createCursorLingerState()
    updateCursorLinger(out, ctx(1000, 1000), 0)
    expect(updateCursorLinger(out, ctx(1000, 1000), CURSOR_LINGER_MS * 4)).toBe(false)
  })

  it('cooldown makes a parked cursor a one-time event', () => {
    const state = createCursorLingerState()
    updateCursorLinger(state, ctx(200, 200), 0)
    expect(updateCursorLinger(state, ctx(200, 200), CURSOR_LINGER_MS + 1)).toBe(true)
    // Still parked long past another linger interval: cooldown holds.
    expect(
      updateCursorLinger(state, ctx(200, 200), CURSOR_LINGER_MS * 3),
    ).toBe(false)
    // After the cooldown, a continued linger may fire again.
    expect(
      updateCursorLinger(
        state,
        ctx(200, 200),
        CURSOR_LINGER_COOLDOWN_MS + CURSOR_LINGER_MS * 3,
      ),
    ).toBe(true)
  })
})
