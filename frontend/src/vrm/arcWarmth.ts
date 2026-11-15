/**
 * Arc-driven warmth: how attentive/affectionate the pet's ambient body
 * language reads, derived from the SAME relationship state the backend
 * already persists (trust_level + the KurisuArc floats — see
 * amadeus/memory/profile.py) rather than inventing a second, purely
 * client-side notion of "how close are we". Fetched once at pet mount via
 * memoryRequest('overview') (../memoryApi) and refreshed hourly — this is a
 * slow-moving relationship signal, not a per-turn one, so polling every 60
 * minutes is plenty and keeps this off the hot path entirely.
 *
 * `warmth` is deliberately a single scalar, not the raw relationship object:
 * every consumer (cursorGaze engagement, petIdleBehaviors cadence/clip
 * preference) just wants "how warm does this feel right now", and computing
 * it once here means the formula lives in exactly one place instead of
 * being re-derived per call site.
 *
 * IMPORTANT: computeWarmth's formula is mirrored server-side (see the task
 * spec / persona.py's own warmth-adjacent thresholds) — if this changes,
 * the two must be kept in sync intentionally, not drift apart silently.
 */

import { memoryRequest } from '../memoryApi'

export const clamp01 = (value: number): number => Math.max(0, Math.min(1, value))

/** Default when the overview is unavailable (network failure, long-term
 * memory disabled, relationship still empty `{}` for a brand-new profile) —
 * a mild, unremarkable warmth rather than either extreme, so a pet that
 * can't reach the backend yet doesn't read as either coldly indifferent or
 * implausibly affectionate on first run. */
export const DEFAULT_WARMTH = 0.3

/** Parses the `overview.relationship` shape memory_overview() returns
 * (amadeus/memory/introspection.py) into the two scalars the formula needs.
 * `relationship` may be `{}` (no profile yet) or absent fields within it —
 * every read is defensive and falls back to a neutral value rather than
 * throwing, so a partially-populated or shape-drifted response degrades to
 * DEFAULT_WARMTH instead of crashing the pet's mount effect. */
export const computeWarmthFromRelationship = (relationship: unknown): number => {
  if (!relationship || typeof relationship !== 'object') return DEFAULT_WARMTH
  const rel = relationship as Record<string, unknown>
  const trust = typeof rel.trust_level === 'number' && Number.isFinite(rel.trust_level) ? rel.trust_level : null
  const arc = rel.arc && typeof rel.arc === 'object' ? (rel.arc as Record<string, unknown>) : null
  const curiosity =
    arc && typeof arc.curiosity_vs_annoyance === 'number' && Number.isFinite(arc.curiosity_vs_annoyance)
      ? arc.curiosity_vs_annoyance
      : null
  if (trust === null && curiosity === null) return DEFAULT_WARMTH
  // Missing halves fall back to a neutral midpoint for THAT term only (0.5
  // trust, 0 curiosity) rather than collapsing the whole computation to the
  // flat default — a response with trust_level but no arc yet (or vice
  // versa) still produces a meaningful warmth instead of always reading as
  // "brand new".
  const trustTerm = trust ?? 0.5
  const curiosityTerm = curiosity ?? 0
  const warmth = 0.6 * trustTerm + 0.4 * ((curiosityTerm + 1) / 2)
  return clamp01(warmth)
}

const REFRESH_INTERVAL_MS = 60 * 60 * 1000

export interface ArcWarmth {
  /** Current warmth in [0, 1] — DEFAULT_WARMTH until the first fetch lands
   * (or forever, if every fetch fails). */
  get: () => number
  stop: () => void
}

/**
 * Starts the mount-time fetch + hourly refresh. No-op-safe by construction:
 * every failure path (network error, non-OK response, malformed JSON, a
 * `relationship: {}` new-profile response) leaves `get()` returning whatever
 * it last successfully computed (DEFAULT_WARMTH if nothing has ever
 * succeeded) rather than throwing or freezing warmth at a stale extreme.
 */
export const startArcWarmth = (deps?: {
  now?: () => number
  refreshIntervalMs?: number
  /** Injectable for tests — defaults to the real memoryRequest. */
  fetchOverview?: () => Promise<Response>
}): ArcWarmth => {
  const refreshIntervalMs = deps?.refreshIntervalMs ?? REFRESH_INTERVAL_MS
  const fetchOverview = deps?.fetchOverview ?? (() => memoryRequest('overview'))

  let warmth = DEFAULT_WARMTH
  let stopped = false
  let intervalId: number | null = null

  const refresh = async () => {
    try {
      const response = await fetchOverview()
      if (!response.ok) return
      const data = await response.json()
      if (stopped) return
      warmth = computeWarmthFromRelationship(data?.overview?.relationship)
    } catch {
      // Best-effort, per module doc — warmth just holds its last value.
    }
  }

  void refresh()
  intervalId = window.setInterval(() => {
    void refresh()
  }, refreshIntervalMs)

  return {
    get: () => warmth,
    stop: () => {
      stopped = true
      if (intervalId !== null) {
        window.clearInterval(intervalId)
        intervalId = null
      }
    },
  }
}
