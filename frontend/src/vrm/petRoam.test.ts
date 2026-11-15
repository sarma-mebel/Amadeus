// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { diagnostics } from '../diagnostics'
import { kurisuBridge, type VrmClipInfo } from './kurisuBridge'
import {
  FIRST_ROAM_DELAY_MS,
  INTERACTION_QUIET_MS,
  MAX_CONSECUTIVE_ROAM_FAILURES,
  MAX_STROLL_MS,
  MIN_STROLL_DISTANCE_PX,
  MIN_STROLL_MS,
  PET_ROAM_CHANGE_EVENT,
  PET_ROAM_KEY,
  ROAM_MAX_INTERVAL_MS,
  ROAM_MIN_INTERVAL_MS,
  STROLL_SPEED_PX_PER_S,
  STROLL_WATCHDOG_MS,
  findEagerWalkClip,
  isEligibleToRoam,
  isPetRoamEnabled,
  isRoamMotionRecent,
  nextRoamDelayMs,
  pickRoamTarget,
  resetRoamMotionForTest,
  setPetRoamEnabled,
  startPetRoam,
  strollDurationMs,
  type PetRoamChangeDetail,
  type RoamContext,
  type RoamEligibility,
} from './petRoam'

const eligible = (overrides: Partial<RoamEligibility> = {}): RoamEligibility => ({
  enabled: true,
  busy: false,
  speaking: false,
  listening: false,
  dozing: false,
  dndActive: false,
  courtesyHidden: false,
  msSinceInteraction: INTERACTION_QUIET_MS,
  ...overrides,
})

describe('isEligibleToRoam', () => {
  it('is true when nothing else is going on', () => {
    expect(isEligibleToRoam(eligible())).toBe(true)
  })

  it.each([
    ['enabled', { enabled: false }],
    ['busy', { busy: true }],
    ['speaking', { speaking: true }],
    ['listening', { listening: true }],
    ['dozing', { dozing: true }],
    ['dndActive', { dndActive: true }],
    ['courtesyHidden', { courtesyHidden: true }],
  ] as const)('yields to %s', (_name, overrides) => {
    expect(isEligibleToRoam(eligible(overrides))).toBe(false)
  })

  it('requires the full interaction-quiet window (boundary)', () => {
    expect(isEligibleToRoam(eligible({ msSinceInteraction: INTERACTION_QUIET_MS - 1 }))).toBe(false)
    expect(isEligibleToRoam(eligible({ msSinceInteraction: INTERACTION_QUIET_MS }))).toBe(true)
  })
})

describe('nextRoamDelayMs', () => {
  it('stays inside the cadence bounds', () => {
    expect(nextRoamDelayMs(() => 0)).toBe(ROAM_MIN_INTERVAL_MS)
    expect(nextRoamDelayMs(() => 0.999999)).toBeLessThanOrEqual(ROAM_MAX_INTERVAL_MS)
    expect(nextRoamDelayMs(() => 0.999999)).toBeGreaterThan(ROAM_MIN_INTERVAL_MS)
    for (const r of [0.1, 0.5, 0.9]) {
      const delay = nextRoamDelayMs(() => r)
      expect(delay).toBeGreaterThanOrEqual(ROAM_MIN_INTERVAL_MS)
      expect(delay).toBeLessThanOrEqual(ROAM_MAX_INTERVAL_MS)
    }
  })

  it('keeps the first-enable delay past the quiet window but well under the cadence', () => {
    // > quiet window: the enabling click itself is an interaction, so a
    // first tick inside the window would always find her ineligible.
    // < cadence floor: the whole point is visible feedback after toggle-on.
    expect(FIRST_ROAM_DELAY_MS).toBeGreaterThan(INTERACTION_QUIET_MS)
    expect(FIRST_ROAM_DELAY_MS).toBeLessThan(ROAM_MIN_INTERVAL_MS)
  })
})

describe('pickRoamTarget', () => {
  const ctx = (overrides: Partial<RoamContext> = {}): RoamContext => ({
    win_x: 800,
    win_y: 200,
    win_w: 380,
    win_h: 560,
    work_x: 0,
    work_y: 0,
    work_w: 2000,
    work_h: 1100,
    ...overrides,
  })

  it('lands on the work-area bottom edge (the perch edge)', () => {
    const target = pickRoamTarget(ctx(), () => 0)
    expect(target).not.toBeNull()
    expect(target!.y).toBe(1100 - 560)
  })

  it('honors the minimum stroll distance and direction sign', () => {
    // random 0: both directions feasible -> 0 < 0.5 picks left; distance
    // floor is MIN_STROLL_DISTANCE_PX.
    const left = pickRoamTarget(ctx(), () => 0)
    expect(left!.direction).toBe(-1)
    expect(left!.distancePx).toBeGreaterThanOrEqual(MIN_STROLL_DISTANCE_PX)
    expect(left!.x).toBe(800 - left!.distancePx)
    const right = pickRoamTarget(ctx(), () => 0.6)
    expect(right!.direction).toBe(1)
    expect(right!.x).toBe(800 + right!.distancePx)
  })

  it('never leaves the work area (clamped by available room)', () => {
    // Window near the left wall: only right has room.
    const nearLeft = pickRoamTarget(ctx({ win_x: 10 }), () => 0)
    expect(nearLeft!.direction).toBe(1)
    // random -> 1-ish maxes the distance; must stay within [minX, maxX].
    const far = pickRoamTarget(ctx({ win_x: 1900 - 380 }), () => 0.999999)
    expect(far).not.toBeNull()
    expect(far!.x).toBeGreaterThanOrEqual(0)
    expect(far!.x).toBeLessThanOrEqual(2000 - 380)
  })

  it('returns null when there is no room to wander', () => {
    // Window as wide as the work area.
    expect(pickRoamTarget(ctx({ win_w: 2000 }), () => 0)).toBeNull()
    // Free span exists but both directions are under the minimum distance.
    expect(
      pickRoamTarget(ctx({ work_w: 380 + MIN_STROLL_DISTANCE_PX, win_x: 100 }), () => 0),
    ).toBeNull()
  })
})

describe('strollDurationMs', () => {
  it('clamps to the min/max glide duration', () => {
    expect(strollDurationMs(1)).toBe(MIN_STROLL_MS)
    expect(strollDurationMs(1_000_000)).toBe(MAX_STROLL_MS)
  })

  it('paces mid-range distances at the stroll speed', () => {
    const distance = 450
    expect(strollDurationMs(distance)).toBe(Math.round((distance / STROLL_SPEED_PX_PER_S) * 1000))
  })
})

describe('kurisu_pet_roam flag', () => {
  afterEach(() => {
    window.localStorage.removeItem(PET_ROAM_KEY)
  })

  it('defaults OFF and only the literal true enables', () => {
    expect(isPetRoamEnabled()).toBe(false)
    window.localStorage.setItem(PET_ROAM_KEY, 'yes')
    expect(isPetRoamEnabled()).toBe(false)
    setPetRoamEnabled(true)
    expect(isPetRoamEnabled()).toBe(true)
    setPetRoamEnabled(false)
    expect(isPetRoamEnabled()).toBe(false)
  })
})

const clipInfo = (overrides: Partial<VrmClipInfo>): VrmClipInfo => ({
  name: 'Clip',
  filename: 'clip.vrma',
  directory: null,
  role: 'idle',
  tags: [],
  disabled: false,
  additive: false,
  isFbx: false,
  lazy: false,
  loaded: true,
  duration: 2,
  looping: false,
  ...overrides,
})

describe('findEagerWalkClip', () => {
  afterEach(() => {
    kurisuBridge.vrmAnimations = null
  })

  it('finds an eager, enabled, loaded role-walk clip', () => {
    kurisuBridge.vrmAnimations = {
      clips: {},
      mixer: null,
      play: vi.fn(),
      list: vi.fn(),
      current: () => null,
      describe: () => [
        clipInfo({ name: 'Stroll', role: 'walk' as VrmClipInfo['role'] }),
      ],
    }
    expect(findEagerWalkClip()?.name).toBe('Stroll')
  })

  it('never picks a lazy/unvetted or disabled walk clip', () => {
    kurisuBridge.vrmAnimations = {
      clips: {},
      mixer: null,
      play: vi.fn(),
      list: vi.fn(),
      current: () => null,
      describe: () => [
        clipInfo({ name: 'Walking', lazy: true, loaded: false }),
        clipInfo({ name: 'Walk In Circle', disabled: true }),
      ],
    }
    expect(findEagerWalkClip()).toBeNull()
  })

  it('returns null with no animation controls registered', () => {
    expect(findEagerWalkClip()).toBeNull()
  })
})

describe('startPetRoam driver', () => {
  const context: RoamContext = {
    win_x: 800,
    win_y: 540,
    win_w: 380,
    win_h: 560,
    work_x: 0,
    work_y: 0,
    work_w: 2000,
    work_h: 1100,
  }

  let roamEvents: PetRoamChangeDetail[]
  let recordEvents: (event: Event) => void

  beforeEach(() => {
    vi.useFakeTimers()
    diagnostics.resetForTest()
    resetRoamMotionForTest()
    window.localStorage.setItem(PET_ROAM_KEY, 'true')
    roamEvents = []
    recordEvents = (event: Event) => {
      roamEvents.push((event as CustomEvent<PetRoamChangeDetail>).detail)
    }
    window.addEventListener(PET_ROAM_CHANGE_EVENT, recordEvents)
  })

  afterEach(() => {
    window.removeEventListener(PET_ROAM_CHANGE_EVENT, recordEvents)
    window.localStorage.removeItem(PET_ROAM_KEY)
    vi.useRealTimers()
  })

  it('does nothing outside Tauri', async () => {
    const invokeFn = vi.fn()
    const handle = startPetRoam({
      getEligibility: () => eligible(),
      invokeFn,
      isTauriFn: () => false,
      random: () => 0,
    })
    await vi.advanceTimersByTimeAsync(ROAM_MAX_INTERVAL_MS * 2)
    expect(invokeFn).not.toHaveBeenCalled()
    expect(handle.strolling()).toBe(false)
    handle.stop()
  })

  it('strolls on the cadence: context, tween, events, and perch refresh', async () => {
    const calls: Array<{ cmd: string; args?: Record<string, unknown> }> = []
    const invokeFn = vi.fn(async (cmd: string, args?: Record<string, unknown>) => {
      calls.push({ cmd, args })
      if (cmd === 'pet_roam_context') return context
      if (cmd === 'animate_pet_window_to') return true
      if (cmd === 'pet_perch_state') return { perched: true, edge: 'bottom' }
      return undefined
    })
    const handle = startPetRoam({
      getEligibility: () => eligible(),
      invokeFn,
      isTauriFn: () => true,
      random: () => 0,
    })
    // Nothing before the first tick — which is the short first-enable delay,
    // not the steady-state cadence (FIRST_ROAM_DELAY_MS < ROAM_MIN_INTERVAL_MS).
    await vi.advanceTimersByTimeAsync(FIRST_ROAM_DELAY_MS - 1)
    expect(invokeFn).not.toHaveBeenCalled()
    await vi.advanceTimersByTimeAsync(1)
    expect(calls[0]?.cmd).toBe('pet_roam_context')
    expect(calls[1]?.cmd).toBe('animate_pet_window_to')
    // random 0 -> stroll left by exactly MIN_STROLL_DISTANCE_PX, along the
    // bottom edge, at the clamped slow-glide duration.
    expect(calls[1]?.args).toEqual({
      x: 800 - MIN_STROLL_DISTANCE_PX,
      y: 1100 - 560,
      durationMs: strollDurationMs(MIN_STROLL_DISTANCE_PX),
    })
    expect(roamEvents[0]).toEqual({ roaming: true, direction: -1 })
    expect(roamEvents[1]).toEqual({ roaming: false, direction: -1 })
    // Recent roam motion suppresses the 'dragged' poke downstream.
    expect(isRoamMotionRecent()).toBe(true)
    // Perch refresh after the Rust-side flush window.
    await vi.advanceTimersByTimeAsync(2_000)
    expect(calls.some((c) => c.cmd === 'pet_perch_state')).toBe(true)
    handle.stop()
  })

  it('skips the attempt (no invokes) when ineligible, then recovers next tick', async () => {
    let speaking = true
    const invokeFn = vi.fn(async (cmd: string) => {
      if (cmd === 'pet_roam_context') return context
      if (cmd === 'animate_pet_window_to') return true
      return undefined
    })
    const handle = startPetRoam({
      getEligibility: () => eligible({ speaking }),
      invokeFn,
      isTauriFn: () => true,
      random: () => 0,
    })
    await vi.advanceTimersByTimeAsync(ROAM_MIN_INTERVAL_MS)
    expect(invokeFn).not.toHaveBeenCalled()
    speaking = false
    await vi.advanceTimersByTimeAsync(ROAM_MIN_INTERVAL_MS)
    expect(invokeFn).toHaveBeenCalled()
    handle.stop()
  })

  it('respects the opt-in flag re-read at each attempt', async () => {
    window.localStorage.setItem(PET_ROAM_KEY, 'false')
    const invokeFn = vi.fn()
    const handle = startPetRoam({
      getEligibility: () => eligible(),
      invokeFn,
      isTauriFn: () => true,
      random: () => 0,
    })
    await vi.advanceTimersByTimeAsync(ROAM_MIN_INTERVAL_MS)
    expect(invokeFn).not.toHaveBeenCalled()
    window.localStorage.setItem(PET_ROAM_KEY, 'true')
    await vi.advanceTimersByTimeAsync(ROAM_MIN_INTERVAL_MS)
    expect(invokeFn).toHaveBeenCalled()
    handle.stop()
  })

  it('cancels mid-stroll the moment a yield condition begins', async () => {
    let speaking = false
    let resolveAnimate: ((value: boolean) => void) | null = null
    const invokeFn = vi.fn(async (cmd: string) => {
      if (cmd === 'pet_roam_context') return context
      if (cmd === 'animate_pet_window_to') {
        return new Promise<boolean>((resolve) => {
          resolveAnimate = resolve
        })
      }
      return undefined
    })
    const handle = startPetRoam({
      getEligibility: () => eligible({ speaking }),
      invokeFn,
      isTauriFn: () => true,
      random: () => 0,
    })
    await vi.advanceTimersByTimeAsync(ROAM_MIN_INTERVAL_MS)
    expect(handle.strolling()).toBe(true)
    // She starts speaking mid-glide: the watchdog must fire the Rust cancel.
    speaking = true
    await vi.advanceTimersByTimeAsync(STROLL_WATCHDOG_MS)
    expect(invokeFn.mock.calls.some((c) => c[0] === 'cancel_pet_window_animation')).toBe(true)
    // The Rust tween resolves false (cancelled); the stroll cleans up.
    resolveAnimate!(false)
    await vi.advanceTimersByTimeAsync(0)
    expect(handle.strolling()).toBe(false)
    expect(roamEvents.at(-1)).toEqual({ roaming: false, direction: -1 })
    handle.stop()
  })

  it('cancelStroll (interaction path) fires the Rust cancel immediately', async () => {
    let resolveAnimate: ((value: boolean) => void) | null = null
    const invokeFn = vi.fn(async (cmd: string) => {
      if (cmd === 'pet_roam_context') return context
      if (cmd === 'animate_pet_window_to') {
        return new Promise<boolean>((resolve) => {
          resolveAnimate = resolve
        })
      }
      return undefined
    })
    const handle = startPetRoam({
      getEligibility: () => eligible(),
      invokeFn,
      isTauriFn: () => true,
      random: () => 0,
    })
    await vi.advanceTimersByTimeAsync(ROAM_MIN_INTERVAL_MS)
    expect(handle.strolling()).toBe(true)
    handle.cancelStroll('interaction')
    await vi.advanceTimersByTimeAsync(0)
    expect(invokeFn.mock.calls.some((c) => c[0] === 'cancel_pet_window_animation')).toBe(true)
    resolveAnimate!(false)
    await vi.advanceTimersByTimeAsync(0)
    expect(handle.strolling()).toBe(false)
    handle.stop()
  })

  it('disables itself for the session after repeated invoke failures', async () => {
    const invokeFn = vi.fn(async () => {
      throw new Error('not allowed by ACL')
    })
    const handle = startPetRoam({
      getEligibility: () => eligible(),
      invokeFn,
      isTauriFn: () => true,
      random: () => 0,
    })
    for (let i = 0; i < MAX_CONSECUTIVE_ROAM_FAILURES; i += 1) {
      await vi.advanceTimersByTimeAsync(ROAM_MIN_INTERVAL_MS)
    }
    const snapshot = diagnostics.snapshot()
    expect(snapshot.counters['pet.roam_invoke_failed']).toBe(MAX_CONSECUTIVE_ROAM_FAILURES)
    expect(snapshot.counters['pet.roam_disabled']).toBe(1)
    // Disabled: no further attempts ever fire.
    invokeFn.mockClear()
    await vi.advanceTimersByTimeAsync(ROAM_MAX_INTERVAL_MS * 2)
    expect(invokeFn).not.toHaveBeenCalled()
    handle.stop()
  })

  it('stop() halts scheduling and cancels an in-flight stroll', async () => {
    let resolveAnimate: ((value: boolean) => void) | null = null
    const invokeFn = vi.fn(async (cmd: string) => {
      if (cmd === 'pet_roam_context') return context
      if (cmd === 'animate_pet_window_to') {
        return new Promise<boolean>((resolve) => {
          resolveAnimate = resolve
        })
      }
      return undefined
    })
    const handle = startPetRoam({
      getEligibility: () => eligible(),
      invokeFn,
      isTauriFn: () => true,
      random: () => 0,
    })
    await vi.advanceTimersByTimeAsync(ROAM_MIN_INTERVAL_MS)
    expect(handle.strolling()).toBe(true)
    handle.stop()
    await vi.advanceTimersByTimeAsync(0)
    expect(invokeFn.mock.calls.some((c) => c[0] === 'cancel_pet_window_animation')).toBe(true)
    resolveAnimate!(false)
    await vi.advanceTimersByTimeAsync(ROAM_MAX_INTERVAL_MS * 2)
    // No new context fetches after stop.
    expect(invokeFn.mock.calls.filter((c) => c[0] === 'pet_roam_context').length).toBe(1)
  })
})

describe('pickRoamTarget multi-monitor', () => {
  const twoMonitorCtx = (): RoamContext => ({
    win_x: 800,
    win_y: 540,
    win_w: 380,
    win_h: 560,
    work_x: 0,
    work_y: 0,
    work_w: 2000,
    work_h: 1100,
    monitors: [
      { x: 0, y: 0, w: 2000, h: 1100 }, // containing (home) monitor
      { x: 2000, y: 100, w: 1600, h: 900 }, // side monitor
    ],
  })

  const sequence = (...values: number[]) => {
    let i = 0
    return () => values[Math.min(i++, values.length - 1)]
  }

  it('wanders to the other monitor when the roll allows it', () => {
    // rolls: cross-chance (pass), monitor index, x position along its span
    const target = pickRoamTarget(twoMonitorCtx(), sequence(0.0, 0.0, 0.5))
    expect(target).not.toBeNull()
    expect(target!.x).toBeGreaterThanOrEqual(2000)
    expect(target!.x).toBeLessThanOrEqual(2000 + 1600 - 380)
    // Bottom edge of the TARGET monitor, not the home one.
    expect(target!.y).toBe(100 + 900 - 560)
    expect(target!.direction).toBe(1)
    expect(target!.distancePx).toBeGreaterThan(0)
  })

  it('stays home when the roll says no', () => {
    const target = pickRoamTarget(twoMonitorCtx(), sequence(0.9, 0.5, 0.5))
    expect(target).not.toBeNull()
    // Same-monitor pick: inside the home work area.
    expect(target!.x + 380).toBeLessThanOrEqual(2000)
    expect(target!.y).toBe(1100 - 560)
  })

  it('single monitor or missing field behaves exactly as before', () => {
    const noField = pickRoamTarget(
      { ...twoMonitorCtx(), monitors: undefined },
      sequence(0.0, 0.5, 0.5),
    )
    expect(noField).not.toBeNull()
    expect(noField!.y).toBe(1100 - 560)

    const onlyHome = pickRoamTarget(
      { ...twoMonitorCtx(), monitors: [{ x: 0, y: 0, w: 2000, h: 1100 }] },
      sequence(0.0, 0.5, 0.5),
    )
    expect(onlyHome).not.toBeNull()
    expect(onlyHome!.y).toBe(1100 - 560)
  })

  it('skips other monitors too narrow for the window', () => {
    const cramped: RoamContext = {
      ...twoMonitorCtx(),
      monitors: [
        { x: 0, y: 0, w: 2000, h: 1100 },
        { x: 2000, y: 0, w: 400, h: 900 }, // barely wider than the window
      ],
    }
    const target = pickRoamTarget(cramped, sequence(0.0, 0.0, 0.5))
    expect(target).not.toBeNull()
    expect(target!.y).toBe(1100 - 560) // fell through to the home monitor
  })
})
