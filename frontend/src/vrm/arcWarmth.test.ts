// @vitest-environment jsdom
import { describe, expect, it, vi } from 'vitest'
import { clamp01, computeWarmthFromRelationship, DEFAULT_WARMTH, startArcWarmth } from './arcWarmth'

describe('clamp01', () => {
  it('clamps to [0, 1]', () => {
    expect(clamp01(-0.5)).toBe(0)
    expect(clamp01(1.5)).toBe(1)
    expect(clamp01(0.42)).toBe(0.42)
  })
})

describe('computeWarmthFromRelationship', () => {
  it('returns the default when relationship is missing/not an object', () => {
    expect(computeWarmthFromRelationship(undefined)).toBe(DEFAULT_WARMTH)
    expect(computeWarmthFromRelationship(null)).toBe(DEFAULT_WARMTH)
    expect(computeWarmthFromRelationship('nope')).toBe(DEFAULT_WARMTH)
  })

  it('returns the default for an empty relationship object (brand-new profile)', () => {
    expect(computeWarmthFromRelationship({})).toBe(DEFAULT_WARMTH)
  })

  it('applies the formula: 0.6*trust + 0.4*(curiosity+1)/2', () => {
    // trust=1, curiosity=1 -> 0.6*1 + 0.4*1 = 1.0
    expect(
      computeWarmthFromRelationship({
        trust_level: 1,
        arc: { curiosity_vs_annoyance: 1 },
      }),
    ).toBeCloseTo(1.0)

    // trust=0, curiosity=-1 -> 0.6*0 + 0.4*0 = 0.0
    expect(
      computeWarmthFromRelationship({
        trust_level: 0,
        arc: { curiosity_vs_annoyance: -1 },
      }),
    ).toBeCloseTo(0.0)

    // trust=0.5, curiosity=0 -> 0.6*0.5 + 0.4*0.5 = 0.5
    expect(
      computeWarmthFromRelationship({
        trust_level: 0.5,
        arc: { curiosity_vs_annoyance: 0 },
      }),
    ).toBeCloseTo(0.5)

    // trust=0.8, curiosity=0.4 -> 0.6*0.8 + 0.4*0.7 = 0.48 + 0.28 = 0.76
    expect(
      computeWarmthFromRelationship({
        trust_level: 0.8,
        arc: { curiosity_vs_annoyance: 0.4 },
      }),
    ).toBeCloseTo(0.76)
  })

  it('falls back to a neutral curiosity term (0) when arc is missing but trust is present', () => {
    // trust=1, curiosity defaults to 0 -> 0.6*1 + 0.4*0.5 = 0.8
    expect(computeWarmthFromRelationship({ trust_level: 1 })).toBeCloseTo(0.8)
  })

  it('falls back to a neutral trust term (0.5) when trust is missing but arc is present', () => {
    // trust defaults to 0.5, curiosity=1 -> 0.6*0.5 + 0.4*1 = 0.7
    expect(
      computeWarmthFromRelationship({ arc: { curiosity_vs_annoyance: 1 } }),
    ).toBeCloseTo(0.7)
  })

  it('clamps out-of-range inputs', () => {
    expect(
      computeWarmthFromRelationship({
        trust_level: 2, // malformed/out-of-range
        arc: { curiosity_vs_annoyance: 1 },
      }),
    ).toBe(1)
  })

  it('ignores non-finite/malformed numeric fields', () => {
    expect(
      computeWarmthFromRelationship({
        trust_level: Number.NaN,
        arc: { curiosity_vs_annoyance: 'high' },
      }),
    ).toBe(DEFAULT_WARMTH)
  })
})

describe('startArcWarmth', () => {
  const makeResponse = (relationship: unknown) =>
    ({
      ok: true,
      json: async () => ({ overview: { relationship } }),
    }) as Response

  it('fetches once at start and updates get() on success', async () => {
    const fetchOverview = vi.fn().mockResolvedValue(
      makeResponse({ trust_level: 1, arc: { curiosity_vs_annoyance: 1 } }),
    )
    const warmth = startArcWarmth({ fetchOverview })
    expect(warmth.get()).toBe(DEFAULT_WARMTH) // before the async fetch resolves
    await vi.waitFor(() => expect(warmth.get()).toBeCloseTo(1.0))
    expect(fetchOverview).toHaveBeenCalledTimes(1)
    warmth.stop()
  })

  it('keeps the default when the fetch fails', async () => {
    const fetchOverview = vi.fn().mockRejectedValue(new Error('network down'))
    const warmth = startArcWarmth({ fetchOverview })
    await vi.waitFor(() => expect(fetchOverview).toHaveBeenCalledTimes(1))
    expect(warmth.get()).toBe(DEFAULT_WARMTH)
    warmth.stop()
  })

  it('keeps the default when the response is not ok', async () => {
    const fetchOverview = vi.fn().mockResolvedValue({ ok: false, json: async () => ({}) } as Response)
    const warmth = startArcWarmth({ fetchOverview })
    await vi.waitFor(() => expect(fetchOverview).toHaveBeenCalledTimes(1))
    expect(warmth.get()).toBe(DEFAULT_WARMTH)
    warmth.stop()
  })

  it('refreshes on the configured interval', async () => {
    vi.useFakeTimers()
    let call = 0
    const fetchOverview = vi.fn().mockImplementation(async () => {
      call += 1
      return makeResponse(
        call === 1
          ? { trust_level: 0, arc: { curiosity_vs_annoyance: -1 } }
          : { trust_level: 1, arc: { curiosity_vs_annoyance: 1 } },
      )
    })
    const warmth = startArcWarmth({ fetchOverview, refreshIntervalMs: 1000 })
    await vi.advanceTimersByTimeAsync(0)
    expect(warmth.get()).toBeCloseTo(0)
    await vi.advanceTimersByTimeAsync(1000)
    expect(warmth.get()).toBeCloseTo(1)
    expect(fetchOverview).toHaveBeenCalledTimes(2)
    warmth.stop()
    vi.useRealTimers()
  })

  it('stop() prevents further scheduled refreshes', async () => {
    vi.useFakeTimers()
    const fetchOverview = vi.fn().mockResolvedValue(makeResponse({}))
    const warmth = startArcWarmth({ fetchOverview, refreshIntervalMs: 1000 })
    await vi.advanceTimersByTimeAsync(0)
    warmth.stop()
    await vi.advanceTimersByTimeAsync(10000)
    expect(fetchOverview).toHaveBeenCalledTimes(1)
    vi.useRealTimers()
  })
})
