// Pure decision-logic coverage for petPerch.ts: sit-clip search (eager/
// vetted only), the wind-damping fallback, and the perch/unperch restore
// path. No THREE/DOM needed — describeClips/playClip are plain stubs and
// springboneWindState is a plain object matching SpringboneWindState's
// shape.
import { describe, expect, it, vi } from 'vitest'
import { createPetPerchHandler, PERCH_SIT_CLIP_NAME_PATTERN, type PetPerchHandlerDeps } from './petPerch'
import type { VrmClipInfo } from './kurisuBridge'

const clip = (overrides: Partial<VrmClipInfo> = {}): VrmClipInfo => ({
  name: 'Idle A',
  filename: 'idle_a.vrma',
  directory: null,
  role: 'idle',
  tags: [],
  disabled: false,
  additive: false,
  isFbx: false,
  lazy: false,
  loaded: true,
  duration: 2,
  looping: true,
  ...overrides,
})

const makeDeps = (overrides: Partial<PetPerchHandlerDeps> = {}): PetPerchHandlerDeps => ({
  describeClips: () => [],
  playClip: vi.fn(),
  resumeIdle: vi.fn(),
  springboneWindState: { windPhase: 0, windStrengthPref: 1 } as PetPerchHandlerDeps['springboneWindState'],
  readWindStrengthPref: () => 1,
  setPerched: vi.fn(),
  ...overrides,
})

describe('PERCH_SIT_CLIP_NAME_PATTERN', () => {
  it('matches names containing "sit" as a word boundary', () => {
    expect(PERCH_SIT_CLIP_NAME_PATTERN.test('Sitting Idle')).toBe(true)
    expect(PERCH_SIT_CLIP_NAME_PATTERN.test('sit_loop')).toBe(true)
    expect(PERCH_SIT_CLIP_NAME_PATTERN.test('Visitor')).toBe(false)
  })
})

describe('createPetPerchHandler', () => {
  it('calls setPerched(true) and plays the first eager, non-disabled, loaded sit clip found', () => {
    const playClip = vi.fn()
    const clips = [
      clip({ name: 'Wave', role: 'gesture' }),
      clip({ name: 'Library Sit', lazy: true }), // lazy — must never be promoted
      clip({ name: 'Disabled Sit', disabled: true }),
      clip({ name: 'Sitting Pose' }), // eager, not disabled, loaded — the match
    ]
    const setPerched = vi.fn()
    const handler = createPetPerchHandler(makeDeps({ describeClips: () => clips, playClip, setPerched }))

    handler.handlePerchChanged({ perched: true })

    expect(setPerched).toHaveBeenCalledWith(true)
    expect(playClip).toHaveBeenCalledWith('Sitting Pose', { loop: true })
  })

  it('falls back to damping springbone wind when no vetted sit clip exists', () => {
    const springboneWindState = { windPhase: 0, windStrengthPref: 1 } as PetPerchHandlerDeps['springboneWindState']
    const handler = createPetPerchHandler(
      makeDeps({ describeClips: () => [clip({ name: 'Wave' })], springboneWindState, readWindStrengthPref: () => 1 })
    )

    handler.handlePerchChanged({ perched: true })

    expect(springboneWindState.windStrengthPref).toBeCloseTo(0.3)
  })

  it('falls back to wind damping when describeClips returns undefined (bridge not yet registered)', () => {
    const springboneWindState = { windPhase: 0, windStrengthPref: 1 } as PetPerchHandlerDeps['springboneWindState']
    const handler = createPetPerchHandler(
      makeDeps({ describeClips: () => undefined, springboneWindState, readWindStrengthPref: () => 1 })
    )

    expect(() => handler.handlePerchChanged({ perched: true })).not.toThrow()
    expect(springboneWindState.windStrengthPref).toBeCloseTo(0.3)
  })

  it('on unperch, resumes idle only if a sit clip had actually been played', () => {
    const resumeIdle = vi.fn()
    const handler = createPetPerchHandler(makeDeps({ resumeIdle }))

    // Never perched with a sit clip -> unperch is a no-op for resumeIdle.
    handler.handlePerchChanged({ perched: false })
    expect(resumeIdle).not.toHaveBeenCalled()
  })

  it('on unperch after a sit clip played, resumes idle and does not touch wind (no override was set)', () => {
    const resumeIdle = vi.fn()
    const springboneWindState = { windPhase: 0, windStrengthPref: 1 } as PetPerchHandlerDeps['springboneWindState']
    const readWindStrengthPref = vi.fn(() => 1)
    const handler = createPetPerchHandler(
      makeDeps({
        describeClips: () => [clip({ name: 'Sit Loop' })],
        resumeIdle,
        springboneWindState,
        readWindStrengthPref,
      })
    )

    handler.handlePerchChanged({ perched: true })
    handler.handlePerchChanged({ perched: false })

    expect(resumeIdle).toHaveBeenCalledTimes(1)
  })

  it('on unperch after the wind-damping fallback, restores the real wind preference', () => {
    const springboneWindState = { windPhase: 0, windStrengthPref: 1 } as PetPerchHandlerDeps['springboneWindState']
    const readWindStrengthPref = vi.fn(() => 0.7)
    const handler = createPetPerchHandler(
      makeDeps({ describeClips: () => [], springboneWindState, readWindStrengthPref })
    )

    handler.handlePerchChanged({ perched: true })
    expect(springboneWindState.windStrengthPref).toBeCloseTo(0.7 * 0.3)

    handler.handlePerchChanged({ perched: false })
    expect(springboneWindState.windStrengthPref).toBe(0.7)
  })

  it('treats a missing `perched` field as false', () => {
    const setPerched = vi.fn()
    const handler = createPetPerchHandler(makeDeps({ setPerched }))
    handler.handlePerchChanged(undefined)
    expect(setPerched).toHaveBeenCalledWith(false)
  })
})
