// @vitest-environment jsdom
//
// Pure decision-logic coverage for animationScheduler.ts: the gesture-clip
// name validation helpers (module-level, no scheduler needed), plus
// pool-selection/no-repeat/cooldown behavior of createAnimationScheduler
// itself. jsdom is needed only because the scheduler reaches for
// window.setTimeout/clearTimeout for its idle-variety timer — none of this
// touches WebGL/the DOM otherwise, so a plain THREE.AnimationMixer bound to
// an empty Object3D root stands in for the real scene.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as THREE from 'three'
import type { RefObject } from 'react'
import {
  createAnimationScheduler,
  EMOTION_GESTURE_MAP,
  findUnresolvedGestureClipNames,
  gestureMapClipNames,
  TALK_BODY_STORAGE_KEY,
  warnUnresolvedGestureClipNames,
  type AnimationSchedulerDeps,
  type ClipMeta,
} from './animationScheduler'

describe('gestureMapClipNames', () => {
  it('flattens every pool in EMOTION_GESTURE_MAP into a de-duplicated name list', () => {
    const names = gestureMapClipNames()
    // "Annoyance" appears in both angry and annoyed pools -> de-duplicated.
    expect(names.filter((n) => n === 'Annoyance')).toHaveLength(1)
    expect(names).toContain('Peace Sign')
    expect(names).toContain('Embarrassment')
    expect(new Set(names).size).toBe(names.length)
  })
})

describe('findUnresolvedGestureClipNames', () => {
  const identityResolve = (name: string) => name

  it('returns nothing when every referenced name resolves to a loaded clip', () => {
    const loaded = new Set(['Peace Sign', 'Clapping'])
    expect(findUnresolvedGestureClipNames(['Peace Sign', 'Clapping'], identityResolve, loaded)).toEqual([])
  })

  it('flags a name that never resolves (rename/typo/stray lazy:true)', () => {
    const loaded = new Set(['Peace Sign'])
    expect(findUnresolvedGestureClipNames(['Peace Sign', 'Cliping'], identityResolve, loaded)).toEqual([
      'Cliping',
    ])
  })

  it('honors alias resolution — a name that resolves via the resolver counts as loaded', () => {
    const loaded = new Set(['Peace Sign'])
    const resolveViaAlias = (name: string) => (name === 'peace_sign' ? 'Peace Sign' : name)
    expect(findUnresolvedGestureClipNames(['peace_sign'], resolveViaAlias, loaded)).toEqual([])
  })

  it('de-duplicates repeated references', () => {
    const loaded = new Set<string>()
    expect(findUnresolvedGestureClipNames(['Missing', 'Missing'], identityResolve, loaded)).toEqual([
      'Missing',
    ])
  })
})

describe('warnUnresolvedGestureClipNames', () => {
  it('console.warns once with the full unresolved list, and stays silent when everything resolves', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const identityResolve = (name: string) => name

    warnUnresolvedGestureClipNames(['A', 'B'], identityResolve, new Set(['A', 'B']))
    expect(warnSpy).not.toHaveBeenCalled()

    warnUnresolvedGestureClipNames(['A', 'Missing'], identityResolve, new Set(['A']))
    expect(warnSpy).toHaveBeenCalledTimes(1)
    expect(warnSpy.mock.calls[0][1]).toEqual(['Missing'])

    warnSpy.mockRestore()
  })
})

describe('createAnimationScheduler', () => {
  const makeClip = (name: string) => new THREE.AnimationClip(name, 1, [])

  const makeDeps = (overrides: Partial<AnimationSchedulerDeps> = {}): AnimationSchedulerDeps => {
    const root = new THREE.Object3D()
    const mixer = new THREE.AnimationMixer(root)
    const clips: Record<string, THREE.AnimationClip> = {
      IdleA: makeClip('IdleA'),
      IdleB: makeClip('IdleB'),
      Greeting: makeClip('Greeting'),
      'Peace Sign': makeClip('Peace Sign'),
      'Sad Pose': makeClip('Sad Pose'),
    }
    const clipMeta: Record<string, ClipMeta> = {}
    const currentActionRef: RefObject<THREE.AnimationAction | null> = { current: null }
    const currentClipNameRef: RefObject<string | null> = { current: null }
    const currentModeRef: RefObject<'idle' | 'actual'> = { current: 'idle' }
    const scheduledActualTimeoutRef: RefObject<number | null> = { current: null }
    const mixerFinishedListenerRef: RefObject<
      ((event: THREE.Event & { action: THREE.AnimationAction }) => void) | null
    > = { current: null }

    return {
      mixer,
      clips,
      clipMeta,
      clipAliases: new Map(),
      idleNames: ['IdleA', 'IdleB'],
      actionNames: ['Greeting', 'Peace Sign', 'Sad Pose'],
      talkClipNames: [],
      isSpeechStreaming: () => false,
      getLastInteractionAt: () => performance.now(),
      onFlusteredBeat: () => {},
      debugLog: () => {},
      currentActionRef,
      currentClipNameRef,
      currentModeRef,
      scheduledActualTimeoutRef,
      mixerFinishedListenerRef,
      ...overrides,
    }
  }

  beforeEach(() => {
    vi.spyOn(Math, 'random').mockReturnValue(0)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('playRandomIdle picks from idleNames and records the current clip name', () => {
    const deps = makeDeps()
    const scheduler = createAnimationScheduler(deps)
    scheduler.playRandomIdle()
    // Math.random mocked to 0 -> first candidate in the (unfiltered, no
    // current clip yet) pool.
    expect(deps.currentClipNameRef.current).toBe('IdleA')
    expect(deps.currentModeRef.current).toBe('idle')
  })

  it('playRandomIdle excludes the currently-playing idle clip (no immediate repeat)', () => {
    const deps = makeDeps()
    const scheduler = createAnimationScheduler(deps)
    deps.currentClipNameRef.current = 'IdleA'
    scheduler.playRandomIdle()
    // With IdleA excluded, only IdleB remains regardless of the mocked
    // Math.random() = 0.
    expect(deps.currentClipNameRef.current).toBe('IdleB')
  })

  it('resolveClipName resolves via clipAliases (case/extension-insensitive) before falling through to the raw input', () => {
    const deps = makeDeps({ clipAliases: new Map([['peace sign', 'Peace Sign']]) })
    const scheduler = createAnimationScheduler(deps)
    expect(scheduler.resolveClipName('Peace Sign')).toBe('Peace Sign') // already-loaded name short-circuits
    expect(scheduler.resolveClipName('peace sign')).toBe('Peace Sign')
    expect(scheduler.resolveClipName('Totally Unknown')).toBe('Totally Unknown')
  })

  it('kurisuReact plays the highest-eligible-pool gesture once per call and enforces MAX_GESTURES_PER_TURN', () => {
    const deps = makeDeps()
    const scheduler = createAnimationScheduler(deps)
    scheduler.kurisuReact('happy', 0.9) // clears the 0.7 threshold for Peace Sign
    expect(deps.currentClipNameRef.current).toBe('Peace Sign')
    expect(deps.currentModeRef.current).toBe('actual')

    // Second gesture within the same turn but before the cooldown window
    // and past MAX_GESTURES_PER_TURN (2) should stop firing new gestures.
    scheduler.kurisuReact('sad', 0.9)
    const secondClip = deps.currentClipNameRef.current
    scheduler.kurisuReact('sad', 0.9)
    expect(deps.currentClipNameRef.current).toBe(secondClip) // third call: capped out, no change
  })

  it('kurisuReact does nothing for neutral/relaxed or a below-threshold intensity', () => {
    const deps = makeDeps()
    const scheduler = createAnimationScheduler(deps)
    scheduler.kurisuReact('neutral', 0.99)
    expect(deps.currentClipNameRef.current).toBeNull()
    scheduler.kurisuReact('happy', 0.1) // below every threshold in the happy pool
    expect(deps.currentClipNameRef.current).toBeNull()
  })

  describe('mouth-only talking (kurisu_talk_body, default off)', () => {
    const talkDeps = (overrides: Partial<AnimationSchedulerDeps> = {}) =>
      makeDeps({
        clips: {
          IdleA: makeClip('IdleA'),
          IdleB: makeClip('IdleB'),
          TalkA: makeClip('TalkA'),
          'Peace Sign': makeClip('Peace Sign'),
        },
        talkClipNames: ['TalkA'],
        ...overrides,
      })

    afterEach(() => {
      window.localStorage.removeItem(TALK_BODY_STORAGE_KEY)
    })

    it('speaking does not start the talk-loop clip by default', () => {
      const deps = talkDeps()
      const scheduler = createAnimationScheduler(deps)
      scheduler.onSpeakingChange(true)
      expect(deps.currentClipNameRef.current).toBeNull()
    })

    it('speaking crossfades a playing one-shot back to idle instead of freezing it', () => {
      const deps = talkDeps()
      const scheduler = createAnimationScheduler(deps)
      deps.currentModeRef.current = 'actual'
      scheduler.onSpeakingChange(true)
      expect(deps.currentClipNameRef.current).toBe('IdleA')
      expect(deps.currentModeRef.current).toBe('idle')
    })

    it('opting back in restores the talk-loop clip on speaking', () => {
      window.localStorage.setItem(TALK_BODY_STORAGE_KEY, 'true')
      const deps = talkDeps()
      const scheduler = createAnimationScheduler(deps)
      scheduler.onSpeakingChange(true)
      expect(deps.currentClipNameRef.current).toBe('TalkA')
    })

    it('suppresses emotion gesture clips mid-speech, but only mid-speech', () => {
      const deps = talkDeps({ isSpeechStreaming: () => true })
      const scheduler = createAnimationScheduler(deps)
      scheduler.kurisuReact('happy', 0.9)
      expect(deps.currentClipNameRef.current).toBeNull()

      window.localStorage.setItem(TALK_BODY_STORAGE_KEY, 'true')
      scheduler.kurisuReact('happy', 0.9)
      expect(deps.currentClipNameRef.current).toBe('Peace Sign')
    })
  })

  it('kurisuReactReset clears the per-turn gesture count so a new turn can gesture again', () => {
    const deps = makeDeps()
    const scheduler = createAnimationScheduler(deps)
    scheduler.kurisuReact('happy', 0.9)
    scheduler.kurisuReact('sad', 0.9)
    // Capped out for this turn.
    const capped = deps.currentClipNameRef.current
    scheduler.kurisuReact('sad', 0.9)
    expect(deps.currentClipNameRef.current).toBe(capped)

    scheduler.kurisuReactReset()
    deps.currentClipNameRef.current = null
    scheduler.kurisuReact('happy', 0.9)
    expect(deps.currentClipNameRef.current).toBe('Peace Sign')
  })

  it('warns in DEV when EMOTION_GESTURE_MAP references a clip name outside the loaded set', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    // Only two of the many EMOTION_GESTURE_MAP-referenced names are loaded
    // in this fixture (Peace Sign, Sad Pose), so DEV-mode init should warn
    // about the rest (Clapping, Joy, ...).
    const deps = makeDeps()
    createAnimationScheduler(deps)

    expect(warnSpy).toHaveBeenCalled()
    const [, unresolved] = warnSpy.mock.calls[0] as [string, string[]]
    expect(unresolved).toEqual(
      expect.arrayContaining(gestureMapClipNames().filter((name) => !(name in deps.clips)))
    )
    expect(unresolved).not.toContain('Peace Sign')
    expect(unresolved).not.toContain('Sad Pose')
    warnSpy.mockRestore()
  })
})

describe('EMOTION_GESTURE_MAP', () => {
  it('every pool entry has a name and a threshold in [0, 1]', () => {
    for (const pool of Object.values(EMOTION_GESTURE_MAP)) {
      for (const gesture of pool ?? []) {
        expect(typeof gesture.name).toBe('string')
        expect(gesture.name.length).toBeGreaterThan(0)
        expect(gesture.threshold).toBeGreaterThanOrEqual(0)
        expect(gesture.threshold).toBeLessThanOrEqual(1)
      }
    }
  })
})
