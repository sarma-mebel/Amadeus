// @vitest-environment jsdom
/**
 * DiagnosticsSection: latency percentile columns (present only when the
 * backend sends p50_ms/p95_ms) and recall-health amber highlighting (only
 * when a degradation counter is nonzero — all-zeros stays visually quiet,
 * that's the documented healthy state).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom/vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'

import DiagnosticsSection from './DiagnosticsSection'
import { diagnostics } from '../../diagnostics'

type DurationsPayload = Record<string, Record<string, number>>
type SelfEvalPayload = {
  last_run_at: string | null
  scores: Record<string, number> | null
  regression: boolean
  regressions: Record<string, { latest: number; median: number; drop: number }> | null
}

type LlmHealthPayload = { state: string; detail?: string | null; since?: string | null }
type TtsHealthPayload = {
  state: string
  detail?: string | null
  since?: string | null
  consecutive_failures?: number
}

/** Minimal /api/status payload fetchBackendStatus accepts (it requires
 * truthy llm and tts blocks; everything else is optional). */
const statusPayload = (
  counters: Record<string, number>,
  durations?: DurationsPayload,
  selfEval?: SelfEvalPayload,
  llmHealth?: LlmHealthPayload,
  ttsHealth?: TtsHealthPayload,
): Record<string, unknown> => ({
  version: '0.0.0-test',
  llm: { provider: 'dummy', class: 'DummyProvider', model: null, is_dummy: true },
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
  counters,
  ...(durations !== undefined ? { durations } : {}),
  ...(selfEval !== undefined ? { self_eval: selfEval } : {}),
  ...(llmHealth !== undefined ? { llm_health: llmHealth } : {}),
  ...(ttsHealth !== undefined ? { tts_health: ttsHealth } : {}),
  memory: {
    long_term: false,
    fact_extraction: false,
    episodes: false,
    greeting: false,
    embedder: null,
    summary_provider_active: false,
  },
})

const stubStatus = (payload: unknown) => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: RequestInfo | URL) => {
      const path = String(url)
      if (path.endsWith('/api/status')) {
        return new Response(JSON.stringify(payload), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
      }
      throw new Error(`Unexpected fetch: ${path}`)
    }),
  )
}

describe('DiagnosticsSection', () => {
  beforeEach(() => {
    diagnostics.resetForTest()
  })

  // No vitest `globals` in this repo's config, so RTL's automatic afterEach
  // cleanup never registers — same convention as SettingsPanel.test.tsx.
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    diagnostics.resetForTest()
  })

  it('renders p50/p95 latency columns when the backend sends them', async () => {
    stubStatus(
      statusPayload(
        {},
        {
          'turn.first_audio_ms': {
            count: 10,
            total_ms: 12000,
            max_ms: 3000,
            last_ms: 900,
            p50_ms: 1100,
            p95_ms: 2800,
          },
        },
      ),
    )
    render(<DiagnosticsSection />)
    await screen.findByText('turn.first_audio_ms')
    expect(screen.getByRole('columnheader', { name: 'p50' })).toBeInTheDocument()
    expect(screen.getByRole('columnheader', { name: 'p95' })).toBeInTheDocument()
    expect(screen.getByText('1100ms')).toBeInTheDocument()
    expect(screen.getByText('2800ms')).toBeInTheDocument()
  })

  it('omits the percentile columns on an old backend without the fields', async () => {
    stubStatus(
      statusPayload(
        {},
        {
          'turn.first_audio_ms': { count: 2, total_ms: 2000, max_ms: 1200, last_ms: 800 },
        },
      ),
    )
    render(<DiagnosticsSection />)
    await screen.findByText('turn.first_audio_ms')
    expect(screen.queryByRole('columnheader', { name: 'p50' })).not.toBeInTheDocument()
    expect(screen.queryByRole('columnheader', { name: 'p95' })).not.toBeInTheDocument()
    // The original four stat columns are still there.
    expect(screen.getByRole('columnheader', { name: 'Avg' })).toBeInTheDocument()
    expect(screen.getByRole('columnheader', { name: 'Max' })).toBeInTheDocument()
  })

  it('amber-highlights nonzero recall-health counters and shows the hint', async () => {
    stubStatus(
      statusPayload({
        'memory.recall_embed_timeouts': 4,
        'memory.vector_store_load_corrupt': 1,
        'memory.turns': 20,
      }),
    )
    render(<DiagnosticsSection />)
    const timeoutCell = await screen.findByText('memory.recall_embed_timeouts')
    expect(timeoutCell.closest('tr')).toHaveClass('settings-diagnostics-alert')
    expect(
      screen.getByText(/per-turn memory recall may be timing out/),
    ).toBeInTheDocument()
    const corruptCell = screen.getByText('memory.vector_store_load_corrupt')
    expect(corruptCell.closest('tr')).toHaveClass('settings-diagnostics-alert')
    expect(screen.getByText(/quarantined as corrupt/)).toBeInTheDocument()
    // An ordinary counter never gets the alert treatment.
    expect(screen.getByText('memory.turns').closest('tr')).not.toHaveClass(
      'settings-diagnostics-alert',
    )
  })

  it('stays visually quiet when recall-health counters are zero', async () => {
    stubStatus(
      statusPayload({
        'memory.recall_embed_timeouts': 0,
        'memory.profile_load_corrupt': 0,
      }),
    )
    render(<DiagnosticsSection />)
    await screen.findByText('memory.recall_embed_timeouts')
    await waitFor(() => {
      expect(document.querySelector('.settings-diagnostics-alert')).toBeNull()
    })
    expect(screen.queryByText(/timing out/)).not.toBeInTheDocument()
    expect(screen.queryByText(/quarantined/)).not.toBeInTheDocument()
  })

  it('renders nothing for self-eval when the field is absent (older backend)', async () => {
    stubStatus(statusPayload({}))
    render(<DiagnosticsSection />)
    await screen.findByText('No server-side failures recorded yet.')
    expect(screen.queryByText('Nightly self-eval')).not.toBeInTheDocument()
  })

  it('renders nothing for self-eval when the job has never run', async () => {
    stubStatus(
      statusPayload({}, undefined, {
        last_run_at: null,
        scores: null,
        regression: false,
        regressions: null,
      }),
    )
    render(<DiagnosticsSection />)
    await screen.findByText('No server-side failures recorded yet.')
    expect(screen.queryByText('Nightly self-eval')).not.toBeInTheDocument()
  })

  it('renders last self-eval run time and scores', async () => {
    stubStatus(
      statusPayload({}, undefined, {
        last_run_at: '2026-07-26T04:05:00Z',
        scores: { greeting_loop_no_repeat: 9, in_character_consistency: 8 },
        regression: false,
        regressions: null,
      }),
    )
    render(<DiagnosticsSection />)
    await screen.findByText('Nightly self-eval')
    expect(screen.getByText('greeting_loop_no_repeat')).toBeInTheDocument()
    expect(screen.getByText('in_character_consistency')).toBeInTheDocument()
    expect(screen.getByText('9')).toBeInTheDocument()
    expect(screen.getByText('8')).toBeInTheDocument()
    expect(
      screen.queryByText(/Conversation quality dropped/),
    ).not.toBeInTheDocument()
  })

  it('amber-highlights a regressed scenario and shows the regression hint', async () => {
    stubStatus(
      statusPayload({}, undefined, {
        last_run_at: '2026-07-26T04:05:00Z',
        scores: { greeting_loop_no_repeat: 3, in_character_consistency: 8 },
        regression: true,
        regressions: {
          greeting_loop_no_repeat: { latest: 3, median: 9, drop: 6 },
        },
      }),
    )
    render(<DiagnosticsSection />)
    await screen.findByText('Nightly self-eval')
    const regressedCell = screen.getByText('greeting_loop_no_repeat')
    expect(regressedCell.closest('tr')).toHaveClass('settings-diagnostics-alert')
    const okCell = screen.getByText('in_character_consistency')
    expect(okCell.closest('tr')).not.toHaveClass('settings-diagnostics-alert')
    expect(screen.getByText(/Conversation quality dropped/)).toBeInTheDocument()
  })

  it('renders nothing for provider health when both fields are absent (older backend)', async () => {
    stubStatus(statusPayload({}))
    render(<DiagnosticsSection />)
    await screen.findByText('No server-side failures recorded yet.')
    expect(screen.queryByText('Provider health')).not.toBeInTheDocument()
  })

  it('renders llm_health and tts_health state plainly when ok', async () => {
    stubStatus(
      statusPayload({}, undefined, undefined, { state: 'ok' }, { state: 'ok' }),
    )
    render(<DiagnosticsSection />)
    await screen.findByText('Provider health')
    expect(screen.getByText('LLM')).toBeInTheDocument()
    expect(screen.getByText('TTS (voice)')).toBeInTheDocument()
    const ttsRow = screen.getByText('TTS (voice)').closest('tr')
    expect(ttsRow).not.toHaveClass('settings-diagnostics-alert')
    expect(screen.queryByText(/voice degraded/)).not.toBeInTheDocument()
    expect(screen.queryByText(/voice failing/)).not.toBeInTheDocument()
  })

  it('does not amber-highlight tts_health "warming" or "off" — normal states', async () => {
    stubStatus(
      statusPayload({}, undefined, undefined, undefined, { state: 'warming' }),
    )
    render(<DiagnosticsSection />)
    await screen.findByText('Provider health')
    const ttsRow = screen.getByText('TTS (voice)').closest('tr')
    expect(ttsRow).not.toHaveClass('settings-diagnostics-alert')
    expect(screen.queryByText(/voice degraded|voice failing/)).not.toBeInTheDocument()
  })

  it('amber-highlights tts_health "degraded" with a short hint', async () => {
    stubStatus(
      statusPayload({}, undefined, undefined, undefined, {
        state: 'degraded',
        detail: 'cuda oom',
      }),
    )
    render(<DiagnosticsSection />)
    await screen.findByText('Provider health')
    const ttsRow = screen.getByText('TTS (voice)').closest('tr')
    expect(ttsRow).toHaveClass('settings-diagnostics-alert')
    expect(screen.getByText(/voice degraded — check GPU\/VRAM, see logs/)).toBeInTheDocument()
  })

  it('amber-highlights tts_health "failing" with a short hint', async () => {
    stubStatus(
      statusPayload({}, undefined, undefined, undefined, {
        state: 'failing',
        consecutive_failures: 3,
      }),
    )
    render(<DiagnosticsSection />)
    await screen.findByText('Provider health')
    const ttsRow = screen.getByText('TTS (voice)').closest('tr')
    expect(ttsRow).toHaveClass('settings-diagnostics-alert')
    expect(screen.getByText(/voice failing — check GPU\/VRAM, see logs/)).toBeInTheDocument()
  })
})

describe('DiagnosticsSection self-test', () => {
  beforeEach(() => {
    diagnostics.resetForTest()
  })
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  const selfTestReport = (overrides: Record<string, unknown> = {}) => ({
    legs: [
      {
        key: 'llm',
        label: 'Chat model',
        status: 'fail',
        detail: 'Running on the dummy provider — replies are canned.',
        fix: 'Add your API key and restart.',
        duration_ms: 12,
      },
      {
        key: 'memory',
        label: 'Memory',
        status: 'pass',
        detail: 'Stored a fact and recalled it.',
        duration_ms: 30,
      },
    ],
    summary: {
      status: 'fail',
      headline: 'Chat model is not working.',
      next_step: 'Add your API key and restart.',
      passed: 1,
      failed: 1,
      warned: 0,
      skipped: 0,
    },
    duration_ms: 42,
    ...overrides,
  })

  /** Status fetch plus the self-test POST, so the button has something to
   * receive. Tracks whether the POST was actually issued. */
  const stubSelfTest = (report: unknown) => {
    const calls: string[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
        const path = String(url)
        calls.push(`${init?.method ?? 'GET'} ${path}`)
        const body = path.includes('/api/self_test') ? report : statusPayload({})
        return new Response(JSON.stringify(body), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
      }),
    )
    return calls
  }

  it('does not run the self-test until the button is pressed', async () => {
    // It round-trips the live LLM and synthesizes audio, so firing on mount
    // would spend tokens and GPU time on every Settings open.
    const calls = stubSelfTest(selfTestReport())
    render(<DiagnosticsSection />)
    await waitFor(() => expect(calls.some((c) => c.includes('/api/status'))).toBe(true))
    expect(calls.some((c) => c.includes('/api/self_test'))).toBe(false)
  })

  it('shows the headline, the next step, and each leg on demand', async () => {
    stubSelfTest(selfTestReport())
    render(<DiagnosticsSection />)
    screen.getByRole('button', { name: /run self-test/i }).click()

    await waitFor(() =>
      expect(screen.getByText('Chat model is not working.')).toBeInTheDocument(),
    )
    // The fix text is the whole point — a failure without a next step is
    // just a counter with nicer formatting.
    expect(screen.getAllByText('Add your API key and restart.').length).toBeGreaterThan(0)
    expect(screen.getByText(/Stored a fact and recalled it/)).toBeInTheDocument()
  })

  it('reports a passing run without a next step', async () => {
    stubSelfTest(
      selfTestReport({
        legs: [
          {
            key: 'llm',
            label: 'Chat model',
            status: 'pass',
            detail: 'GeminiProvider replied.',
            duration_ms: 10,
          },
        ],
        summary: {
          status: 'pass',
          headline: 'Everything checked out.',
          next_step: null,
          passed: 1,
          failed: 0,
          warned: 0,
          skipped: 0,
        },
      }),
    )
    render(<DiagnosticsSection />)
    screen.getByRole('button', { name: /run self-test/i }).click()
    await waitFor(() =>
      expect(screen.getByText('Everything checked out.')).toBeInTheDocument(),
    )
  })

  it('says so plainly when the backend cannot be reached', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: RequestInfo | URL) => {
        if (String(url).includes('/api/self_test')) throw new Error('offline')
        return new Response(JSON.stringify(statusPayload({})), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
      }),
    )
    render(<DiagnosticsSection />)
    screen.getByRole('button', { name: /run self-test/i }).click()
    await waitFor(() =>
      expect(screen.getByText(/Could not reach the backend/)).toBeInTheDocument(),
    )
    expect(diagnostics.snapshot().counters['self_test.request_failed']).toBe(1)
  })
})
