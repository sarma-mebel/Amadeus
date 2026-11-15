import { describe, expect, it } from 'vitest'
import { deriveHealth, deriveLlmHealthBanner, summarizeTools } from './statusApi'
import type { BackendStatus } from './statusApi'

/** Minimal valid BackendStatus for deriveHealth — only llm/tts are read,
 * everything else is filled with the healthy defaults. */
const baseStatus = (overrides: Partial<BackendStatus> = {}): BackendStatus => ({
  version: '0.0.0-test',
  llm: { provider: 'dummy', class: 'DummyProvider', model: null, is_dummy: false },
  tts: {
    requested: null,
    loaded: null,
    class: null,
    engine: null,
    cache_signature: null,
    warming: false,
    fallback_from: null,
    error: null,
  },
  stt: { provider: 'dummy', loaded: false, unavailable_reason: null },
  memory: {
    long_term: false,
    fact_extraction: false,
    episodes: false,
    greeting: false,
    embedder: null,
    summary_provider_active: false,
  },
  ...overrides,
})

describe('summarizeTools', () => {
  it('returns null when the backend sent no tools block (older backend)', () => {
    expect(summarizeTools(undefined)).toBeNull()
    expect(summarizeTools(null as unknown as BackendStatus['tools'])).toBeNull()
    expect(summarizeTools({} as BackendStatus['tools'])).toBeNull()
  })

  it('summarizes the registry shape', () => {
    const summary = summarizeTools({
      enabled: true,
      registered: ['calculator', 'web_search', 'reminders'],
      unavailable: [{ name: 'literature', reason: 'biopython not installed' }],
    })
    expect(summary).toEqual({
      enabled: true,
      registeredLabel: 'calculator · web_search · reminders',
      unavailable: [{ name: 'literature', reason: 'biopython not installed' }],
    })
  })

  it('labels an empty registry explicitly', () => {
    const summary = summarizeTools({ enabled: true, registered: [], unavailable: [] })
    expect(summary?.registeredLabel).toBe('none registered')
  })

  it('tolerates malformed unavailable entries and missing reasons', () => {
    const summary = summarizeTools({
      enabled: true,
      registered: ['calculator'],
      unavailable: [
        { name: 'web_search' } as unknown as { name: string; reason: string },
        null as unknown as { name: string; reason: string },
        { name: 42, reason: 'x' } as unknown as { name: string; reason: string },
      ],
    })
    expect(summary?.unavailable).toEqual([{ name: 'web_search', reason: 'unavailable' }])
  })

  it('still summarizes the legacy boolean shape', () => {
    const summary = summarizeTools({
      enabled: true,
      desktop_files: true,
      literature: false,
      reminders: true,
      web_search: false,
    })
    expect(summary?.registeredLabel).toBe('calculator · files · reminders')
    expect(summary?.unavailable).toEqual([])
  })

  it('legacy shape with nothing optional reads calculator only', () => {
    const summary = summarizeTools({
      enabled: true,
      desktop_files: false,
      literature: false,
      reminders: false,
      web_search: false,
    })
    expect(summary?.registeredLabel).toBe('calculator only')
  })
})

describe('deriveLlmHealthBanner', () => {
  it('prefers llm_health when present — a degraded state wins', () => {
    expect(deriveLlmHealthBanner({ state: 'invalid_key' }, false)).toBe('invalid_key')
    expect(deriveLlmHealthBanner({ state: 'rate_limited' }, false)).toBe('rate_limited')
    expect(deriveLlmHealthBanner({ state: 'error' }, false)).toBe('error')
    expect(deriveLlmHealthBanner({ state: 'dummy' }, false)).toBe('dummy')
  })

  it('llm_health "ok" means no banner, even if a stale llmIsDummy read says otherwise', () => {
    // e.g. mid-flip right after Settings saved a real key — llm_health has
    // already self-healed to 'ok' but the onboarding-gate probe hasn't
    // re-fired yet. llm_health is the fresher, process-wide truth.
    expect(deriveLlmHealthBanner({ state: 'ok' }, true)).toBeNull()
  })

  it('falls back to llmIsDummy when llm_health is absent (older backend)', () => {
    expect(deriveLlmHealthBanner(undefined, true)).toBe('dummy')
    expect(deriveLlmHealthBanner(undefined, false)).toBeNull()
    expect(deriveLlmHealthBanner(null, true)).toBe('dummy')
  })
})

describe('deriveHealth', () => {
  it('is ok when everything (including tts_health) is healthy', () => {
    expect(deriveHealth(baseStatus({ tts_health: { state: 'ok' } }))).toBe('ok')
  })

  it('is ok when tts_health is absent (older backend) and nothing else is degraded', () => {
    expect(deriveHealth(baseStatus())).toBe('ok')
  })

  it('is ok when tts_health is "warming" or "off" — normal, not alarming states', () => {
    expect(deriveHealth(baseStatus({ tts_health: { state: 'warming' } }))).toBe('ok')
    expect(deriveHealth(baseStatus({ tts_health: { state: 'off' } }))).toBe('ok')
  })

  it('is degraded when tts_health is "degraded"', () => {
    expect(deriveHealth(baseStatus({ tts_health: { state: 'degraded' } }))).toBe('degraded')
  })

  it('is degraded when tts_health is "failing"', () => {
    expect(deriveHealth(baseStatus({ tts_health: { state: 'failing' } }))).toBe('degraded')
  })
})
