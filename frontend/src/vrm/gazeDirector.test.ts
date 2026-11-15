import { afterEach, describe, expect, it, vi } from 'vitest'
import * as THREE from 'three'
import { createGazeDirector, createGazeRuntimeState } from './gazeDirector'

/** advanceGaze mixes wall-clock (performance.now) with frame deltaTime, so
 * the tests pin performance.now to a controllable value and advance both
 * together (same spirit as acknowledgment.test.ts's injected fake clock). */
const useFakeNow = () => {
  // Starts nonzero: thinkingSince uses 0 as its "not thinking" sentinel,
  // and real performance.now() is never 0 by the time a turn is sent.
  let now = 1000
  const spy = vi.spyOn(performance, 'now').mockImplementation(() => now)
  return {
    spy,
    advance: (ms: number) => {
      now += ms
    },
  }
}

const makeDirector = (onGazeShift?: (magnitude: number) => void) => {
  const camera = new THREE.PerspectiveCamera()
  camera.position.set(0, 1.4, 1.2)
  const gazeStateRef = { current: createGazeRuntimeState() }
  gazeStateRef.current.proxy = new THREE.Object3D()
  gazeStateRef.current.proxy.position.copy(camera.position)
  const director = createGazeDirector({
    camera,
    gazeStateRef,
    isSpeaking: () => false,
    onGazeShift,
  })
  return { director, camera, gazeStateRef }
}

/** Run advanceGaze over simulated frames, advancing the fake clock in step. */
const runFrames = (
  director: ReturnType<typeof makeDirector>['director'],
  clock: ReturnType<typeof useFakeNow>,
  frames: number,
  frameMs = 16,
) => {
  for (let i = 0; i < frames; i++) {
    clock.advance(frameMs)
    director.advanceGaze(frameMs / 1000)
  }
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('thinking aversion (LLM-wait beat)', () => {
  it('holds eye contact through the acknowledgment window before averting', () => {
    const clock = useFakeNow()
    const { director, camera, gazeStateRef } = makeDirector()

    director.beginThinking()
    // Inside the hold window (< 550ms minimum): proxy stays pinned near the
    // camera — she heard you, she's not looking away yet.
    runFrames(director, clock, 20) // 320ms
    const duringHold = gazeStateRef.current.proxy!.position.distanceTo(camera.position)
    expect(duringHold).toBeLessThan(0.08)
  })

  it('averts up-and-away after the hold, then re-engages on endThinking', () => {
    const clock = useFakeNow()
    const { director, camera, gazeStateRef } = makeDirector()

    director.beginThinking()
    clock.advance(1000) // safely past the randomized 550-900ms hold
    runFrames(director, clock, 60) // ~1s of frames for the proxy lerp to settle
    const proxy = gazeStateRef.current.proxy!
    // Identity camera quaternion: up is +Y, so the "considering" look sits
    // measurably above the camera and off to one side.
    expect(proxy.position.y).toBeGreaterThan(camera.position.y + 0.05)
    expect(Math.abs(proxy.position.x - camera.position.x)).toBeGreaterThan(0.05)

    director.endThinking()
    // Re-engage: lock-on suppresses wandering and the offset is gone, so
    // the proxy eases back onto the camera.
    expect(director.social.lockOnUntil).toBeGreaterThan(performance.now())
    runFrames(director, clock, 60)
    expect(proxy.position.distanceTo(camera.position)).toBeLessThan(0.08)
  })

  it('self-caps a thinking window whose end signal never arrives', () => {
    const clock = useFakeNow()
    const { director } = makeDirector()

    director.beginThinking()
    clock.advance(13000) // past THINKING_MAX_MS (12s)
    runFrames(director, clock, 1)
    // Expired: the state self-cleared, so the normal idle gaze cycle owns
    // the proxy again — a dropped thinking-end can never wedge the stare.
    expect(director.social.thinkingSince).toBe(0)
  })

  it('fires onGazeShift when the aversion engages and again on re-engage', () => {
    const clock = useFakeNow()
    const onGazeShift = vi.fn()
    const { director } = makeDirector(onGazeShift)

    director.beginThinking()
    clock.advance(1000)
    runFrames(director, clock, 5)
    expect(onGazeShift).toHaveBeenCalledTimes(1) // engage — exactly once, not per frame

    director.endThinking()
    expect(onGazeShift).toHaveBeenCalledTimes(2) // re-engage shift
  })

  it('endThinking without a live thinking window is a no-op', () => {
    useFakeNow()
    const onGazeShift = vi.fn()
    const { director } = makeDirector(onGazeShift)
    director.endThinking()
    expect(onGazeShift).not.toHaveBeenCalled()
    expect(director.social.lockOnUntil).toBe(0)
  })

  it('emotional aversion still wins over the thinking look', () => {
    const clock = useFakeNow()
    const { director, camera, gazeStateRef } = makeDirector()

    director.beginThinking()
    clock.advance(1000)
    director.applyEmotionGaze('sad', 0.8) // look-down must beat look-up
    runFrames(director, clock, 60)
    expect(gazeStateRef.current.proxy!.position.y).toBeLessThan(camera.position.y)
  })
})
