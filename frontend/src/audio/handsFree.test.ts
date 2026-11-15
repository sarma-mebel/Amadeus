// @vitest-environment jsdom
import { describe, expect, it } from 'vitest'
import {
  HANDS_FREE_EVENT,
  HANDS_FREE_MAX_SAMPLE_GAP_MS,
  createHandsFreeEndpointer,
  loadHandsFreeEnabled,
  saveHandsFreeEnabled,
  type HandsFreeFeedResult,
} from './handsFree'

// ~50ms per feed (between MicMonitor's 50ms poll and the ~85ms capture
// buffer at 48kHz — the machine only cares about the timestamps).
const DT = 50
const LOUD = 0.1 // typical AGC'd speech
const HOVER = 0.012 // between release (0.010) and acquire (0.015) thresholds
const QUIET = 0.001 // room noise

type Endpointer = ReturnType<typeof createHandsFreeEndpointer>

/** Feeds a constant level from fromMs to toMs; throws if any sample fires an
 * endpoint/discard (the span is expected to be uneventful). Returns the last
 * 'listening' result. */
const feedQuietly = (ep: Endpointer, level: number, fromMs: number, toMs: number) => {
  let last: HandsFreeFeedResult | null = null
  for (let ts = fromMs; ts <= toMs; ts += DT) {
    const result = ep.feed(level, ts)
    if (result.kind !== 'listening') {
      throw new Error(`unexpected ${result.kind} at ts=${ts}`)
    }
    last = result
  }
  if (!last) throw new Error('empty span')
  return last
}

/** Feeds a constant level until an endpoint/discard fires; throws if the
 * span ends without one. */
const feedUntilEvent = (ep: Endpointer, level: number, fromMs: number, toMs: number) => {
  for (let ts = fromMs; ts <= toMs; ts += DT) {
    const result = ep.feed(level, ts)
    if (result.kind !== 'listening') return result
  }
  throw new Error(`no event fired between ${fromMs} and ${toMs}`)
}

describe('createHandsFreeEndpointer', () => {
  it('stays idle through silence indefinitely (no timeout — the mic just stays open)', () => {
    const ep = createHandsFreeEndpointer()
    const last = feedQuietly(ep, QUIET, 0, 60_000)
    expect(last).toEqual({ kind: 'listening', phase: 'idle' })
  })

  it('does not acquire on a level between release and acquire thresholds', () => {
    const ep = createHandsFreeEndpointer()
    // Hysteresis is one-way: HOVER sustains a run but never starts one.
    expect(feedQuietly(ep, HOVER, 0, 5_000)).toEqual({ kind: 'listening', phase: 'idle' })
  })

  it('drops a blip shorter than the min voiced duration without ever endpointing', () => {
    const ep = createHandsFreeEndpointer()
    feedQuietly(ep, QUIET, 0, 500)
    // 150ms of loud (a click / short cough): pending, but never voiced.
    const during = feedQuietly(ep, LOUD, 550, 700)
    expect(during).toEqual({ kind: 'listening', phase: 'pending' })
    // Back to silence: straight to idle, and nothing ever fires.
    const after = feedQuietly(ep, QUIET, 750, 5_000)
    expect(after).toEqual({ kind: 'listening', phase: 'idle' })
  })

  it('endpoints a normal utterance after the trailing-silence window', () => {
    const ep = createHandsFreeEndpointer()
    // 1s of speech.
    const voiced = feedQuietly(ep, LOUD, 0, 1_000)
    expect(voiced).toEqual({ kind: 'listening', phase: 'voiced' })
    // Silence just short of the window: still listening.
    expect(feedQuietly(ep, QUIET, 1_050, 1_750)).toEqual({ kind: 'listening', phase: 'voiced' })
    // Crossing 800ms of trailing silence: endpoint.
    const result = feedUntilEvent(ep, QUIET, 1_800, 2_000)
    expect(result.kind).toBe('endpoint')
    if (result.kind === 'endpoint') {
      expect(result.reason).toBe('silence')
      expect(result.utteranceMs).toBeGreaterThanOrEqual(1_800)
    }
  })

  it('keeps the utterance alive when speech resumes inside the silence window', () => {
    const ep = createHandsFreeEndpointer()
    feedQuietly(ep, LOUD, 0, 1_000)
    feedQuietly(ep, QUIET, 1_050, 1_650) // 600ms pause — inside the window
    expect(feedQuietly(ep, LOUD, 1_700, 2_400)).toEqual({ kind: 'listening', phase: 'voiced' })
    // Now a full silence window ends it, spanning both speech runs.
    const result = feedUntilEvent(ep, QUIET, 2_450, 3_400)
    expect(result.kind).toBe('endpoint')
    if (result.kind === 'endpoint') expect(result.utteranceMs).toBeGreaterThanOrEqual(3_000)
  })

  it('treats hysteresis-zone levels as voice mid-utterance (no early endpoint)', () => {
    const ep = createHandsFreeEndpointer()
    feedQuietly(ep, LOUD, 0, 500)
    // Trailing-off speech hovering between the thresholds for 2s: still voiced.
    expect(feedQuietly(ep, HOVER, 550, 2_500)).toEqual({ kind: 'listening', phase: 'voiced' })
    expect(feedUntilEvent(ep, QUIET, 2_550, 3_500).kind).toBe('endpoint')
  })

  it('discards an utterance below the min-utterance floor instead of sending it', () => {
    const ep = createHandsFreeEndpointer()
    // 300ms of voice: passes the 250ms onset gate but totals < 600ms.
    feedQuietly(ep, LOUD, 0, 300)
    const result = feedUntilEvent(ep, QUIET, 350, 1_300)
    expect(result).toEqual({ kind: 'discard', reason: 'too-short' })
    // Machine is reusable: a real utterance afterwards endpoints normally.
    feedQuietly(ep, LOUD, 1_350, 2_350)
    expect(feedUntilEvent(ep, QUIET, 2_400, 3_400).kind).toBe('endpoint')
  })

  it('caps runaway utterances at the max duration', () => {
    const ep = createHandsFreeEndpointer()
    let capped: HandsFreeFeedResult | null = null
    for (let ts = 0; ts <= 31_000; ts += DT) {
      const result = ep.feed(LOUD, ts)
      if (result.kind !== 'listening') {
        capped = result
        break
      }
    }
    if (!capped || capped.kind !== 'endpoint') {
      throw new Error(`expected max-utterance endpoint, got ${JSON.stringify(capped)}`)
    }
    expect(capped.reason).toBe('max-utterance')
    expect(capped.utteranceMs).toBeGreaterThanOrEqual(30_000)
    expect(capped.utteranceMs).toBeLessThan(30_200)
    // Self-reset: continued voice starts a fresh utterance that can endpoint.
    feedQuietly(ep, LOUD, 31_050, 32_050)
    expect(feedUntilEvent(ep, QUIET, 32_100, 33_100).kind).toBe('endpoint')
  })

  it('clamps a large timestamp gap instead of booking it as one slab of silence', () => {
    const ep = createHandsFreeEndpointer()
    feedQuietly(ep, LOUD, 0, 1_000)
    // One sample after a 5s stall: dt clamps to the gap cap, far below the
    // 800ms silence window — no endpoint yet.
    const afterGap = ep.feed(QUIET, 6_000)
    expect(afterGap).toEqual({ kind: 'listening', phase: 'voiced' })
    expect(HANDS_FREE_MAX_SAMPLE_GAP_MS).toBeLessThan(800)
  })

  it('reset() forgets the utterance and the timestamp baseline', () => {
    const ep = createHandsFreeEndpointer()
    feedQuietly(ep, LOUD, 0, 1_000)
    ep.reset()
    // First loud sample after reset re-enters pending (dt baseline gone, so
    // no phantom voiced time either).
    expect(ep.feed(LOUD, 10_000)).toEqual({ kind: 'listening', phase: 'pending' })
    // And silence right after drops to idle — the pre-reset second of voice
    // must not count toward the onset gate.
    expect(ep.feed(QUIET, 10_050)).toEqual({ kind: 'listening', phase: 'idle' })
  })
})

describe('hands-free preference', () => {
  it('defaults off, persists, and broadcasts changes', () => {
    window.localStorage.clear()
    expect(loadHandsFreeEnabled()).toBe(false)
    const seen: boolean[] = []
    const listener = (event: Event) =>
      seen.push(Boolean((event as CustomEvent<{ enabled: boolean }>).detail?.enabled))
    window.addEventListener(HANDS_FREE_EVENT, listener)
    saveHandsFreeEnabled(true)
    expect(loadHandsFreeEnabled()).toBe(true)
    saveHandsFreeEnabled(false)
    expect(loadHandsFreeEnabled()).toBe(false)
    window.removeEventListener(HANDS_FREE_EVENT, listener)
    expect(seen).toEqual([true, false])
  })
})
