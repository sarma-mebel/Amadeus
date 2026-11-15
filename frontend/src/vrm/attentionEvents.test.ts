// @vitest-environment jsdom
//
// Pure decision-logic coverage for attentionEvents.ts: the typing-return
// gaze/blink snap, the mic-speaking lock-on + silence-gap gating, and the
// procedural-body/gaze-director passthroughs, driven via attach() +
// window.dispatchEvent (jsdom needed only for window event dispatch, same
// rationale as animationScheduler.test.ts).
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as THREE from 'three'
import { createAttentionEvents, type AttentionEventsDeps } from './attentionEvents'
import { createGazeRuntimeState, type GazeSocialState } from './gazeDirector'
import { createBlinkState } from './expressionDriver'

const buildGazeSocial = (): GazeSocialState => ({
  typingUntil: 0,
  lastTypingAt: 0,
  lockOnUntil: 0,
  aversionUntil: 0,
  aversionLateral: 0,
  aversionVertical: 0,
  saccadeOffset: new THREE.Vector3(),
  saccadeTarget: new THREE.Vector3(),
  nextSaccadeIn: 0.3,
  thinkingSince: 0,
  thinkingHoldMs: 700,
  thinkingLateral: 0,
  thinkingVertical: 0,
  thinkingRetargetIn: 0,
})

describe('createAttentionEvents', () => {
  let deps: AttentionEventsDeps
  let controller: ReturnType<typeof createAttentionEvents>
  let nowMs: number

  const proceduralBody = {
    queueListeningNod: vi.fn(),
    queueCuriousTilt: vi.fn(),
    queueBeatGesture: vi.fn(),
  }
  const gazeDirector = {
    beginThinking: vi.fn(),
    endThinking: vi.fn(),
  }

  beforeEach(() => {
    nowMs = 100000
    proceduralBody.queueListeningNod.mockClear()
    proceduralBody.queueCuriousTilt.mockClear()
    proceduralBody.queueBeatGesture.mockClear()
    gazeDirector.beginThinking.mockClear()
    gazeDirector.endThinking.mockClear()
    deps = {
      gazeSocial: buildGazeSocial(),
      gazeStateRef: { current: createGazeRuntimeState() },
      blinkStateRef: { current: createBlinkState() },
      gazeDirector,
      proceduralBody,
      now: () => nowMs,
      random: () => 0.5,
    }
    controller = createAttentionEvents(deps)
    controller.attach()
  })

  afterEach(() => {
    controller.detach()
  })

  describe('kurisu:activity', () => {
    it('updates lastTypingAt/typingUntil on every event', () => {
      window.dispatchEvent(new CustomEvent('kurisu:activity'))
      expect(deps.gazeSocial.lastTypingAt).toBe(nowMs)
      expect(deps.gazeSocial.typingUntil).toBe(nowMs + 2500)
    })

    it('does not snap gaze back when returning within the 8s idle threshold', () => {
      deps.gazeSocial.lastTypingAt = nowMs - 7999
      deps.gazeStateRef.current.phase = 'offset'
      window.dispatchEvent(new CustomEvent('kurisu:activity'))
      expect(deps.gazeStateRef.current.phase).toBe('offset')
    })

    it('snaps an offset gaze back to onCamera and primes an early blink after an 8s+ idle return', () => {
      deps.gazeSocial.lastTypingAt = nowMs - 8001
      deps.gazeStateRef.current.phase = 'offset'
      deps.blinkStateRef.current.phase = 'idle'
      deps.blinkStateRef.current.elapsed = 5
      deps.blinkStateRef.current.nextBlinkAt = 10

      window.dispatchEvent(new CustomEvent('kurisu:activity'))

      expect(deps.gazeStateRef.current.phase).toBe('onCamera')
      expect(deps.gazeStateRef.current.nextChangeAt).toBeCloseTo(2 + 0.5 * 4)
      expect(deps.blinkStateRef.current.nextBlinkAt).toBeCloseTo(5.15)
    })

    it('never delays an already-earlier scheduled blink', () => {
      deps.gazeSocial.lastTypingAt = nowMs - 8001
      deps.blinkStateRef.current.phase = 'idle'
      deps.blinkStateRef.current.elapsed = 0
      deps.blinkStateRef.current.nextBlinkAt = 0.05 // already sooner than elapsed + 0.15

      window.dispatchEvent(new CustomEvent('kurisu:activity'))

      expect(deps.blinkStateRef.current.nextBlinkAt).toBe(0.05)
    })
  })

  describe('kurisu:user-speaking', () => {
    it('extends lockOnUntil to now + 1600ms', () => {
      window.dispatchEvent(new CustomEvent('kurisu:user-speaking'))
      expect(deps.gazeSocial.lockOnUntil).toBe(nowMs + 1600)
    })

    it('never shortens an existing later lock-on', () => {
      deps.gazeSocial.lockOnUntil = nowMs + 5000
      window.dispatchEvent(new CustomEvent('kurisu:user-speaking'))
      expect(deps.gazeSocial.lockOnUntil).toBe(nowMs + 5000)
    })

    it('snaps gaze back only on the FIRST speaking event after an 8s+ silence gap', () => {
      deps.gazeStateRef.current.phase = 'offset'
      // First event ever (lastSpeakingAt starts at 0) -> silentFor is huge -> snaps.
      window.dispatchEvent(new CustomEvent('kurisu:user-speaking'))
      expect(deps.gazeStateRef.current.phase).toBe('onCamera')

      // Immediately following event (same nowMs) -> silentFor is 0 -> no-op,
      // but phase is already onCamera so this doesn't distinguish; verify
      // via a fresh offset + short gap instead.
      deps.gazeStateRef.current.phase = 'offset'
      window.dispatchEvent(new CustomEvent('kurisu:user-speaking'))
      expect(deps.gazeStateRef.current.phase).toBe('offset')
    })
  })

  it('kurisu:listening-nod queues a listening nod', () => {
    window.dispatchEvent(new CustomEvent('kurisu:listening-nod'))
    expect(proceduralBody.queueListeningNod).toHaveBeenCalledTimes(1)
  })

  it('kurisu:cursor-linger queues a curious tilt', () => {
    window.dispatchEvent(new CustomEvent('kurisu:cursor-linger'))
    expect(proceduralBody.queueCuriousTilt).toHaveBeenCalledTimes(1)
  })

  it('kurisu:thinking-start/end bracket the gaze director thinking beat', () => {
    window.dispatchEvent(new CustomEvent('kurisu:thinking-start'))
    expect(gazeDirector.beginThinking).toHaveBeenCalledTimes(1)
    window.dispatchEvent(new CustomEvent('kurisu:thinking-end'))
    expect(gazeDirector.endThinking).toHaveBeenCalledTimes(1)
  })

  describe('kurisu:sentence-audio-start', () => {
    it('queues a beat gesture for question/negation/emphasis beats', () => {
      for (const beat of ['question', 'negation', 'emphasis'] as const) {
        proceduralBody.queueBeatGesture.mockClear()
        window.dispatchEvent(new CustomEvent('kurisu:sentence-audio-start', { detail: { beat } }))
        expect(proceduralBody.queueBeatGesture).toHaveBeenCalledWith(beat)
      }
    })

    it('ignores unrecognized or missing beat kinds', () => {
      window.dispatchEvent(new CustomEvent('kurisu:sentence-audio-start', { detail: { beat: 'other' } }))
      window.dispatchEvent(new CustomEvent('kurisu:sentence-audio-start', { detail: {} }))
      expect(proceduralBody.queueBeatGesture).not.toHaveBeenCalled()
    })
  })

  it('detach() removes every listener so subsequent events are no-ops', () => {
    controller.detach()
    window.dispatchEvent(new CustomEvent('kurisu:listening-nod'))
    expect(proceduralBody.queueListeningNod).not.toHaveBeenCalled()
  })
})
