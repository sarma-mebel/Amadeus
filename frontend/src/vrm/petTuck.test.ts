// @vitest-environment jsdom
/**
 * Tuck-away geometry + state persistence: the pure halves (tuckedX,
 * loadTuckState validation). The slides need a real Tauri window and are
 * exercised on the desktop build.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import {
  isPetTucked,
  loadTuckState,
  PET_TUCKED_KEY,
  PEEK_EXTRA_PX,
  TUCK_SLIVER_PX,
  tuckedX,
  type TuckState,
} from './petTuck'

const state = (over: Partial<TuckState> = {}): TuckState => ({
  side: 'left',
  work_x: 0,
  work_w: 2000,
  win_w: 380,
  win_y: 540,
  ...over,
})

describe('tuckedX', () => {
  it('leaves exactly the sliver visible on either side', () => {
    // Left tuck: window hangs off the left; sliver's worth remains.
    expect(tuckedX(state(), TUCK_SLIVER_PX)).toBe(0 - 380 + TUCK_SLIVER_PX)
    // Right tuck: window hangs off the right.
    expect(tuckedX(state({ side: 'right' }), TUCK_SLIVER_PX)).toBe(
      2000 - TUCK_SLIVER_PX,
    )
    // Peek widens the visible strip.
    expect(tuckedX(state(), TUCK_SLIVER_PX + PEEK_EXTRA_PX)).toBe(
      0 - 380 + TUCK_SLIVER_PX + PEEK_EXTRA_PX,
    )
    // Fully visible == flush against the tucked edge (the untuck target).
    expect(tuckedX(state(), 380)).toBe(0)
    expect(tuckedX(state({ side: 'right' }), 380)).toBe(2000 - 380)
  })

  it('respects a non-zero work-area origin (secondary monitor)', () => {
    const s = state({ work_x: 2000, work_w: 1600 })
    expect(tuckedX(s, TUCK_SLIVER_PX)).toBe(2000 - 380 + TUCK_SLIVER_PX)
    expect(tuckedX({ ...s, side: 'right' }, TUCK_SLIVER_PX)).toBe(
      2000 + 1600 - TUCK_SLIVER_PX,
    )
  })
})

describe('loadTuckState', () => {
  beforeEach(() => window.localStorage.clear())
  afterEach(() => window.localStorage.clear())

  it('round-trips a valid state and drives isPetTucked', () => {
    expect(isPetTucked()).toBe(false)
    window.localStorage.setItem(PET_TUCKED_KEY, JSON.stringify(state()))
    expect(loadTuckState()).toEqual(state())
    expect(isPetTucked()).toBe(true)
  })

  it('rejects junk without throwing', () => {
    window.localStorage.setItem(PET_TUCKED_KEY, 'not json')
    expect(loadTuckState()).toBeNull()
    window.localStorage.setItem(PET_TUCKED_KEY, JSON.stringify({ side: 'up' }))
    expect(loadTuckState()).toBeNull()
    window.localStorage.setItem(
      PET_TUCKED_KEY,
      JSON.stringify({ side: 'left', work_x: 'x' }),
    )
    expect(loadTuckState()).toBeNull()
    expect(isPetTucked()).toBe(false)
  })
})
