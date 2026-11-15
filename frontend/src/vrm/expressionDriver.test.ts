import { describe, expect, it, vi } from 'vitest'
import type { VRM } from '@pixiv/three-vrm'
import {
  createBlinkState,
  createExpressionEasingState,
  createExpressionDriver,
  createMoodBaselineState,
  VRM_EXPRESSION_NAMES,
} from './expressionDriver'

const makeVrm = () => {
  const values = new Map<string, number>()
  const vrm = {
    expressionManager: {
      getValue: (name: string) => values.get(name) ?? 0,
      setValue: (name: string, value: number) => {
        values.set(name, value)
      },
    },
  } as unknown as VRM
  return { vrm, values }
}

const makeDriver = (vrm: VRM | null, onEmotionCue = vi.fn(), isLipSyncActive?: () => boolean) => {
  const expressionStateRef = { current: createExpressionEasingState() }
  const moodBaselineStateRef = { current: createMoodBaselineState() }
  const blinkStateRef = { current: createBlinkState() }
  const driver = createExpressionDriver({
    getVrm: () => vrm,
    expressionStateRef,
    moodBaselineStateRef,
    blinkStateRef,
    onEmotionCue,
    isLipSyncActive,
  })
  return { driver, expressionStateRef, moodBaselineStateRef, blinkStateRef, onEmotionCue }
}

describe('createExpressionDriver', () => {
  it('set() eases the named expression in and zeroes the others', () => {
    const { vrm, values } = makeVrm()
    const { driver } = makeDriver(vrm)

    driver.set('happy', 0.8, 1000)
    driver.advanceExpression(0.2) // full 0.2s ease-in window

    expect(values.get('happy')).toBeCloseTo(0.8)
    VRM_EXPRESSION_NAMES.filter((name) => name !== 'happy').forEach((name) => {
      expect(values.get(name) ?? 0).toBe(0)
    })
  })

  it('set() forwards the normalized emotion to the gaze cue hook', () => {
    const { vrm } = makeVrm()
    const { driver, onEmotionCue } = makeDriver(vrm)
    driver.set('  Sad  ', 0.6)
    expect(onEmotionCue).toHaveBeenCalledWith('sad', 0.6)
  })

  it('holds, releases, then settles onto the relaxed baseline', () => {
    const { vrm, values } = makeVrm()
    const { driver, expressionStateRef } = makeDriver(vrm)

    driver.set('happy', 0.8, 1000)
    driver.advanceExpression(0.2) // in -> hold
    driver.advanceExpression(1.0) // hold expires -> out
    driver.advanceExpression(0.6) // out completes -> settle on relaxed

    const state = expressionStateRef.current
    expect(state.name).toBe('relaxed')
    expect(state.phase).toBe('hold')
    expect(state.holdDuration).toBe(Number.POSITIVE_INFINITY)
    expect(values.get('happy')).toBe(0)
    // Baseline sits near 0.4 (mood EMA drifts it only slightly per reply).
    expect(values.get('relaxed')).toBeGreaterThan(0.35)
    expect(values.get('relaxed')).toBeLessThan(0.5)
  })

  it('happy replies nudge the mood baseline target up; sad ones down', () => {
    const { vrm } = makeVrm()
    const { driver, moodBaselineStateRef } = makeDriver(vrm)

    driver.set('happy', 1)
    expect(moodBaselineStateRef.current.target).toBeGreaterThan(0.4)

    const afterHappy = moodBaselineStateRef.current.target
    driver.set('sad', 1)
    expect(moodBaselineStateRef.current.target).toBeLessThan(afterHappy)
  })

  it('warns and no-ops before the VRM is available', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const { driver, onEmotionCue } = makeDriver(null)
    driver.set('happy', 0.8)
    expect(warn).toHaveBeenCalled()
    expect(onEmotionCue).not.toHaveBeenCalled()
    warn.mockRestore()
  })

  it('advanceBlink runs a closing -> closed -> opening cycle on the blink expression', () => {
    const { vrm, values } = makeVrm()
    const { driver, blinkStateRef } = makeDriver(vrm)
    blinkStateRef.current.nextBlinkAt = 1
    blinkStateRef.current.pendingDoubleBlink = false

    driver.advanceBlink(1.1) // idle timer expires
    expect(blinkStateRef.current.phase).toBe('closing')

    driver.advanceBlink(0.07) // fully closed
    expect(values.get('blink')).toBeCloseTo(1)
    expect(blinkStateRef.current.phase).toBe('closed')

    driver.advanceBlink(0.05) // hold expires
    expect(blinkStateRef.current.phase).toBe('opening')

    driver.advanceBlink(0.12) // fully open again
    expect(values.get('blink')).toBeCloseTo(0)
  })

  it('suppresses new blinks while a strong surprised expression is showing', () => {
    const { vrm } = makeVrm()
    const { driver, blinkStateRef } = makeDriver(vrm)
    driver.set('surprised', 0.9, 5000)
    blinkStateRef.current.nextBlinkAt = 0.5
    driver.advanceBlink(1)
    expect(blinkStateRef.current.phase).toBe('idle')
  })

  it("a 'neutral' set releases to the relaxed baseline instead of pinning forever", () => {
    const { vrm, values } = makeVrm()
    const { driver, expressionStateRef } = makeDriver(vrm)

    driver.set('neutral', 0.9, 500)
    driver.advanceExpression(0.2) // ease-in completes
    expect(values.get('neutral')).toBeCloseTo(0.9)
    driver.advanceExpression(0.5) // hold elapses -> out
    driver.advanceExpression(0.6) // out completes -> settles on relaxed

    expect(expressionStateRef.current.name).toBe('relaxed')
    expect(values.get('neutral')).toBe(0)
    expect(values.get('relaxed')).toBeCloseTo(expressionStateRef.current.relaxedBaseline)
  })

  it("leaves 'relaxed' untouched while lip sync owns it", () => {
    const { vrm, values } = makeVrm()
    const { driver } = makeDriver(vrm, vi.fn(), () => true)

    values.set('relaxed', 0.12) // the lip-sync tick's jaw-slack duck
    driver.set('happy', 0.8, 1000)
    driver.advanceExpression(0.2)

    expect(values.get('happy')).toBeCloseTo(0.8)
    expect(values.get('relaxed')).toBeCloseTo(0.12)
  })

  it('requestBlink starts an immediate blink once spaced far enough from the last', () => {
    const { vrm } = makeVrm()
    const { driver, blinkStateRef } = makeDriver(vrm)
    blinkStateRef.current.nextBlinkAt = 10 // ambient timer far away

    driver.advanceBlink(1) // accumulate idle time (doubles as since-last-blink)
    driver.requestBlink()
    expect(blinkStateRef.current.phase).toBe('closing')
  })

  it('requestBlink respects the spacing guard right after a blink', () => {
    const { vrm } = makeVrm()
    const { driver, blinkStateRef } = makeDriver(vrm)
    blinkStateRef.current.nextBlinkAt = 10

    driver.advanceBlink(0.3) // only 0.3s since the machine went idle
    driver.requestBlink()
    expect(blinkStateRef.current.phase).toBe('idle')
  })

  it('requestBlink never interrupts a blink already in flight', () => {
    const { vrm } = makeVrm()
    const { driver, blinkStateRef } = makeDriver(vrm)
    blinkStateRef.current.nextBlinkAt = 0.5
    driver.advanceBlink(1) // timer blink begins
    expect(blinkStateRef.current.phase).toBe('closing')
    driver.advanceBlink(0.03) // mid-closing
    const elapsedMidBlink = blinkStateRef.current.elapsed
    driver.requestBlink()
    expect(blinkStateRef.current.phase).toBe('closing')
    expect(blinkStateRef.current.elapsed).toBe(elapsedMidBlink) // not restarted
  })

  it('requestBlink honors probability 0 and the surprised suppression', () => {
    const { vrm } = makeVrm()
    const { driver, blinkStateRef } = makeDriver(vrm)
    blinkStateRef.current.nextBlinkAt = 10
    driver.advanceBlink(1)

    driver.requestBlink(0) // dice roll can never pass
    expect(blinkStateRef.current.phase).toBe('idle')

    driver.set('surprised', 0.9, 5000) // wide-eyed hold suppresses blinks
    driver.requestBlink()
    expect(blinkStateRef.current.phase).toBe('idle')
  })

  it("resolves canonical names to a VRM0 pack's custom casing ('Surprised')", () => {
    const values = new Map<string, number>()
    const vrm = {
      expressionManager: {
        expressionMap: { Surprised: {}, happy: {}, angry: {}, sad: {}, relaxed: {}, neutral: {} },
        getValue: (name: string) => values.get(name) ?? 0,
        setValue: (name: string, value: number) => {
          values.set(name, value)
        },
      },
    } as unknown as VRM
    const { driver } = makeDriver(vrm)

    driver.set('surprised', 0.9, 1000)
    driver.advanceExpression(0.2)

    expect(values.get('Surprised')).toBeCloseTo(0.9)
    expect(values.has('surprised')).toBe(false)
  })
})
