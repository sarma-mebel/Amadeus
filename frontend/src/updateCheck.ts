/** Update-available nudge against the GitHub releases feed.
 *
 * Nudge only — no auto-download, no auto-install: the UI just links to the
 * releases page. Deliberately failure-silent everywhere: the repo may be
 * private (404), the machine offline, the API rate-limited, or the build a
 * dev checkout (version "dev") — in every such case the check simply
 * reports "no update" (at most one console.debug per launch).
 *
 * Network discipline: results are cached in localStorage for 24h
 * (`kurisu_update_check`), so the GitHub API is hit at most about once a
 * day per machine — far under the 60/hr/IP unauthenticated limit. The
 * current backend version always comes in as an argument (from the
 * /api/status payload the app already fetches): the backend is the thing
 * being updated, so a build-time frontend constant would be wrong.
 */

import { isTauri } from './platform'

export const RELEASES_URL = 'https://github.com/zxKyouma/Amadeus/releases'
const LATEST_RELEASE_API =
  'https://api.github.com/repos/zxKyouma/Amadeus/releases/latest'

export const UPDATE_CHECK_STORAGE_KEY = 'kurisu_update_check'
/** Pre-cache key (a bare skipped tag); read as a dismissedTag fallback so
 * an already-skipped version doesn't come back after the upgrade. */
const LEGACY_SKIP_STORAGE_KEY = 'kurisu_skip_version'

const CACHE_TTL_MS = 24 * 60 * 60 * 1000
const FETCH_TIMEOUT_MS = 5000

/** Strictly numeric dot-separated version, optional leading "v". Returns
 * null for anything else — including prerelease suffixes ("0.13.0-rc1"):
 * conservative by design, a prerelease never triggers a nudge. */
const parseVersion = (value: string): number[] | null => {
  const stripped = value.trim().replace(/^v/i, '')
  if (!stripped) return null
  const nums: number[] = []
  for (const part of stripped.split('.')) {
    if (!/^\d+$/.test(part)) return null
    nums.push(Number.parseInt(part, 10))
  }
  return nums
}

/** Numeric segment-wise compare of "v1.2.3"-style versions ("1.0" == "1.0.0").
 * Returns >0 when a is newer, <0 when b is newer, 0 when equal or when
 * either side is unparseable (e.g. "dev", prereleases, garbage). */
export const compareVersions = (a: string, b: string): number => {
  const va = parseVersion(a)
  const vb = parseVersion(b)
  if (!va || !vb) return 0
  const length = Math.max(va.length, vb.length)
  for (let i = 0; i < length; i++) {
    const diff = (va[i] ?? 0) - (vb[i] ?? 0)
    if (diff !== 0) return diff
  }
  return 0
}

interface UpdateCheckCache {
  /** Epoch ms of the last successful releases fetch. */
  checkedAt?: number
  /** tag_name of the latest release as of checkedAt. */
  latestTag?: string | null
  /** Tag the user dismissed — that exact version never nudges again; a
   * newer one does. */
  dismissedTag?: string | null
}

const loadCache = (): UpdateCheckCache => {
  try {
    const raw = localStorage.getItem(UPDATE_CHECK_STORAGE_KEY)
    const parsed: unknown = raw ? JSON.parse(raw) : {}
    const cache: UpdateCheckCache =
      parsed && typeof parsed === 'object' ? (parsed as UpdateCheckCache) : {}
    if (!cache.dismissedTag) {
      const legacy = localStorage.getItem(LEGACY_SKIP_STORAGE_KEY)
      if (legacy) cache.dismissedTag = legacy
    }
    return cache
  } catch {
    return {}
  }
}

const saveCache = (cache: UpdateCheckCache): void => {
  try {
    localStorage.setItem(UPDATE_CHECK_STORAGE_KEY, JSON.stringify(cache))
  } catch {
    // Storage unavailable — the check just re-runs next launch.
  }
}

/** Permanently silence the nudge for this exact tag (a newer release will
 * still nudge). */
export const dismissUpdate = (tag: string): void => {
  saveCache({ ...loadCache(), dismissedTag: tag })
}

// At most one debug line per launch, no matter how the check fails.
let failureLogged = false
const logFailureOnce = (reason: unknown): void => {
  if (failureLogged) return
  failureLogged = true
  console.debug('update check skipped:', reason)
}

const fetchLatestTag = async (): Promise<string | null> => {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS)
  try {
    const response = await fetch(LATEST_RELEASE_API, {
      headers: { Accept: 'application/vnd.github+json' },
      signal: controller.signal,
    })
    if (!response.ok) {
      // 404 private repo, 403 rate limit, ...
      logFailureOnce(`HTTP ${response.status}`)
      return null
    }
    const data: unknown = await response.json()
    const tag = (data as { tag_name?: unknown } | null)?.tag_name
    return typeof tag === 'string' && tag ? tag : null
  } catch (error) {
    logFailureOnce(error) // offline, timeout abort, malformed JSON ...
    return null
  } finally {
    clearTimeout(timer)
  }
}

export interface UpdateCheckOptions {
  /** Hosted public demo — the viewer can't update the server; never nudge. */
  demoMode?: boolean
  /** Test hook: "current time" for the 24h cache decision. */
  now?: number
}

/** Returns the newer, non-dismissed release tag (e.g. "v0.13.0") or null.
 * Consults the 24h localStorage cache first and only hits the network when
 * it's stale; a failed fetch falls back to the last cached answer. Never
 * throws. */
export const checkForUpdate = async (
  currentVersion: string | null | undefined,
  options: UpdateCheckOptions = {},
): Promise<string | null> => {
  if (options.demoMode) return null
  if (!currentVersion || !parseVersion(currentVersion)) return null // "dev" etc.

  const now = options.now ?? Date.now()
  let cache = loadCache()
  const age = typeof cache.checkedAt === 'number' ? now - cache.checkedAt : NaN
  const fresh = Number.isFinite(age) && age >= 0 && age < CACHE_TTL_MS
  if (!fresh) {
    const tag = await fetchLatestTag()
    if (tag !== null) {
      cache = { ...cache, checkedAt: now, latestTag: tag }
      saveCache(cache)
    }
    // On failure keep any stale cached tag: offline machines still get the
    // nudge they already knew about, and the next launch retries.
  }

  const latest = cache.latestTag
  if (!latest || cache.dismissedTag === latest) return null
  return compareVersions(latest, currentVersion) > 0 ? latest : null
}

export const openReleasesPage = (): void => {
  if (isTauri()) {
    // The Rust shell plugin is registered and shell:allow-open is granted;
    // invoke directly so we don't need the @tauri-apps/plugin-shell npm pkg.
    void import('@tauri-apps/api/core')
      .then(({ invoke }) => invoke('plugin:shell|open', { path: RELEASES_URL }))
      .catch(() => window.open(RELEASES_URL, '_blank', 'noopener'))
    return
  }
  window.open(RELEASES_URL, '_blank', 'noopener')
}

// Once per app launch, across React strict-mode double-mounts.
let updateCheckStarted = false

/** Run the check once per launch; resolves the newer non-dismissed tag or
 * null. Later callers (settings reopen, status re-polls) get null without
 * touching the network. */
export const runOnceUpdateCheck = async (
  currentVersion: string | null | undefined,
  options: UpdateCheckOptions = {},
): Promise<string | null> => {
  if (updateCheckStarted) return null
  updateCheckStarted = true
  return checkForUpdate(currentVersion, options)
}

/** Test hook: reset the once-per-launch latch and the log-once flag. */
export const __resetUpdateCheckForTests = (): void => {
  updateCheckStarted = false
  failureLogged = false
}
