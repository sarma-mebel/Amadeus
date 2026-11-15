/** Client for GET /api/status — the backend's runtime provider truth. */

import { apiFetch } from './api'

/** One capability checked by POST /api/self_test (amadeus/web/self_test.py).
 * `fix` is the actionable next step and is present on every fail/skip — a
 * finding without one is just a counter with nicer formatting. */
export interface SelfTestLeg {
  key: string
  label: string
  status: 'pass' | 'warn' | 'fail' | 'skip'
  detail: string
  fix?: string
  duration_ms?: number
  provider?: string
  embedder?: string
  audio_bytes?: number
}

export interface SelfTestReport {
  legs: SelfTestLeg[]
  summary: {
    status: 'pass' | 'warn' | 'fail'
    headline: string
    /** The single most fundamental failure's fix; null when all clear. */
    next_step: string | null
    passed: number
    failed: number
    warned: number
    skipped: number
  }
  duration_ms: number
}

export interface BackendStatus {
  version: string
  /** Optional: true on hosted public demos — settings/persona mutation
   * endpoints are locked server-side, so the UI hides those controls. */
  demo_mode?: boolean
  /** Optional: the active character identity (BYO character packs). Absent
   * on older backends; the UI falls back to the built-in Kurisu strings. */
  character?: {
    name?: string
    full_name?: string
    custom?: boolean
    /** models.json id of this character's avatar, for auto-selection. */
    model_id?: string | null
  }
  llm: {
    provider: string
    class: string
    model: string | null
    is_dummy: boolean
  }
  tts: {
    requested: string | null
    loaded: string | null
    class: string | null
    engine: string | null
    /** Optional: set when the loaded engine is a known-slow fallback (e.g.
     * qwen3 without faster-qwen3-tts, ~10x slower). Absent on older backends. */
    engine_warning?: string | null
    cache_signature: string | null
    warming: boolean
    fallback_from: string | null
    error: string | null
  }
  stt: {
    provider: string
    loaded: boolean
    unavailable_reason: string | null
  }
  /** Process-lifetime health counters (amadeus/telemetry.py). Optional:
   * absent on older backends. */
  counters?: Record<string, number>
  /** Per-stage latency stats (amadeus/telemetry.py record_duration), keyed
   * by stage name — e.g. "stt.request_ms", "llm.first_token_ms",
   * "llm.stream_total_ms", "tts.first_audio_ms", "turn.first_audio_ms" (the
   * headline silence-to-voice number). Optional: absent on older backends,
   * and a fresh process reports {} until the first turn completes. */
  durations?: Record<string, { count: number; total_ms: number; max_ms: number; last_ms: number }>
  /** Process-wide LLM provider health (amadeus/provider_health.py) —
   * self-healing back to 'ok' on the next successful call. Optional: absent
   * on older backends. */
  llm_health?: {
    state: 'ok' | 'rate_limited' | 'invalid_key' | 'error' | 'dummy'
    detail?: string | null
    since?: string | null
  }
  /** Process-wide TTS provider health (amadeus/provider_health.py) — the
   * same "is it OK right now" job llm_health does for the LLM side, layered
   * over the pre-existing tts.warming/tts.*_synth_failures signals rather
   * than replacing them. 'degraded' = at least one recent synthesis
   * failure; 'failing' = 3+ CONSECUTIVE failures with no success in
   * between (the voice is effectively dead, not just having a bad moment);
   * 'off' = no provider loaded and not warming (voice disabled/
   * unconfigured — a normal state, never alarming). Optional: absent on
   * older backends. */
  tts_health?: {
    state: 'ok' | 'warming' | 'degraded' | 'failing' | 'off'
    detail?: string | null
    since?: string | null
    consecutive_failures?: number
  }
  memory: {
    long_term: boolean
    fact_extraction: boolean
    episodes: boolean
    greeting: boolean
    embedder: string | null
    summary_provider_active: boolean
  }
  // Optional: absent on backends older than the tools block. Two shapes
  // exist: the legacy boolean flags (v0.8.8) and the newer registry shape
  // (registered/unavailable) — all fields optional so either payload parses.
  tools?: {
    enabled: boolean
    // Legacy boolean flags (pre-registry backends).
    desktop_files?: boolean
    literature?: boolean
    reminders?: boolean
    web_search?: boolean
    // Registry shape: names actually registered this run, plus tools that
    // were configured on but could not load (missing key/package) with why.
    registered?: string[]
    unavailable?: Array<{ name: string; reason: string }>
  }
  /** Nightly conversation-quality self-check (amadeus/self_eval.py,
   * config.self_eval, default off). Optional: absent on older backends;
   * all-None/false when the job is disabled or hasn't run yet. */
  self_eval?: {
    last_run_at: string | null
    scores: Record<string, number> | null
    regression: boolean
    regressions: Record<string, { latest: number; median: number; drop: number }> | null
  }
}

export interface ToolsSummary {
  enabled: boolean
  /** Human-readable list of live tools, or null when unknown/none. */
  registeredLabel: string | null
  /** Tools that were configured on but failed to load, with the reason. */
  unavailable: Array<{ name: string; reason: string }>
}

/** Normalizes both tools payload shapes (legacy booleans vs registry) into
 * one summary for the status popover. Returns null when the backend didn't
 * send a tools block at all, so callers render nothing extra. */
export const summarizeTools = (tools: BackendStatus['tools']): ToolsSummary | null => {
  if (!tools || typeof tools !== 'object' || typeof tools.enabled !== 'boolean') return null

  const unavailable = Array.isArray(tools.unavailable)
    ? tools.unavailable
        .filter((entry): entry is { name: string; reason: string } =>
          Boolean(entry && typeof entry === 'object' && typeof entry.name === 'string'),
        )
        .map((entry) => ({
          name: entry.name,
          reason: typeof entry.reason === 'string' && entry.reason ? entry.reason : 'unavailable',
        }))
    : []

  let registeredLabel: string | null = null
  if (Array.isArray(tools.registered)) {
    const names = tools.registered.filter((name): name is string => typeof name === 'string' && name.length > 0)
    registeredLabel = names.length > 0 ? names.join(' · ') : 'none registered'
  } else {
    // Legacy boolean payload: reconstruct the old chip line.
    const optional = [
      tools.desktop_files && 'files',
      tools.literature && 'papers',
      tools.reminders && 'reminders',
      tools.web_search && 'web',
    ].filter(Boolean) as string[]
    registeredLabel = optional.length > 0 ? `calculator · ${optional.join(' · ')}` : 'calculator only'
  }

  return { enabled: tools.enabled, registeredLabel, unavailable }
}

/** Null on ANY failure (older backend without the endpoint, network error,
 * malformed payload) — callers hide the status UI entirely in that case, so
 * the chip stays forward- and backward-compatible. */
export const fetchBackendStatus = async (): Promise<BackendStatus | null> => {
  try {
    const response = await apiFetch('/api/status')
    if (!response.ok) return null
    const data = await response.json()
    if (!data || typeof data !== 'object' || !data.llm || !data.tts) return null
    return data as BackendStatus
  } catch {
    return null
  }
}

/** The persistent LLM-health banner's state key: which one of the mutually
 * exclusive banner variants (if any) should show right now. Mirrors
 * llm_health.state's vocabulary, plus 'dummy' also covering the pre-
 * llm_health signal (llmIsDummy from /api/settings) for older backends. */
export type LlmHealthBannerState = 'dummy' | 'invalid_key' | 'rate_limited' | 'error' | null

/** Which banner (if any) should show, given the newer llm_health block
 * (preferred — self-healing, process-wide truth) and the older llmIsDummy
 * boolean (the pre-existing onboarding-gate signal, still the only source
 * on a backend too old to send llm_health). llm_health, when present, is
 * authoritative: an 'ok' state means no banner even if a stale llmIsDummy
 * read is still true (e.g. mid-flip while Settings just saved a real key).
 * Absent llm_health (older backend) falls back to llmIsDummy exactly like
 * the original dummy-only banner did. */
export const deriveLlmHealthBanner = (
  llmHealth: BackendStatus['llm_health'] | null | undefined,
  llmIsDummy: boolean,
): LlmHealthBannerState => {
  if (llmHealth?.state) {
    return llmHealth.state === 'ok' ? null : llmHealth.state
  }
  return llmIsDummy ? 'dummy' : null
}

/** Amber when anything is silently degraded; green otherwise. */
export const deriveHealth = (status: BackendStatus): 'ok' | 'degraded' => {
  const { llm, tts } = status
  if (llm.is_dummy) return 'degraded'
  if (tts.fallback_from) return 'degraded'
  if (tts.engine_warning) return 'degraded'
  if (tts.warming) return 'degraded'
  if (tts.requested && !tts.loaded) return 'degraded'
  // qwen3 without the CUDA-graph engine runs ~10x slower than realtime —
  // technically working, practically degraded.
  if (tts.loaded === 'qwen3_tts' && tts.engine && tts.engine !== 'faster') return 'degraded'
  // Streamed sentences have been failing over to slower synthesis this
  // process lifetime (e.g. CUDA OOM) — the voice is limping, say so.
  if ((status.counters?.['tts.stream_synth_failures'] ?? 0) > 0) return 'degraded'
  // tts_health (amadeus/provider_health.py): 'failing' means 3+ consecutive
  // synthesis failures right now, a stronger signal than the lifetime
  // counter above — always degraded. 'degraded' there is already covered by
  // the counter check for any backend new enough to send tts_health, but
  // included for backends where synthesis has failed without yet tripping
  // a counter this checks (belt-and-braces, cheap).
  if (status.tts_health?.state === 'failing' || status.tts_health?.state === 'degraded') {
    return 'degraded'
  }
  return 'ok'
}
