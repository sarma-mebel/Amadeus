// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  __resetUpdateCheckForTests,
  checkForUpdate,
  compareVersions,
  dismissUpdate,
  runOnceUpdateCheck,
  UPDATE_CHECK_STORAGE_KEY,
} from './updateCheck'

const mockLatest = (tag: string | null, ok = true, status = 200) => {
  const fetchMock = vi.fn(async () => ({
    ok,
    status,
    json: async () => (tag ? { tag_name: tag } : {}),
  }))
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

const readCache = () =>
  JSON.parse(window.localStorage.getItem(UPDATE_CHECK_STORAGE_KEY) ?? '{}')

const writeCache = (cache: Record<string, unknown>) =>
  window.localStorage.setItem(UPDATE_CHECK_STORAGE_KEY, JSON.stringify(cache))

beforeEach(() => {
  window.localStorage.clear()
  vi.spyOn(console, 'debug').mockImplementation(() => {})
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  __resetUpdateCheckForTests()
})

describe('compareVersions', () => {
  it('compares numeric segments, stripping a v prefix', () => {
    expect(compareVersions('v0.8.7', '0.8.6')).toBeGreaterThan(0)
    expect(compareVersions('0.8.6', 'v0.8.7')).toBeLessThan(0)
    expect(compareVersions('v0.8.6', '0.8.6')).toBe(0)
  })

  it('orders numerically, not lexically', () => {
    expect(compareVersions('0.10.0', '0.9.9')).toBeGreaterThan(0)
  })

  it('treats different-length versions as padded with zeros', () => {
    expect(compareVersions('1.0', '1.0.0')).toBe(0)
    expect(compareVersions('1.0.1', '1.0')).toBeGreaterThan(0)
  })

  it('is neutral on unparseable versions', () => {
    expect(compareVersions('dev', '0.8.6')).toBe(0)
    expect(compareVersions('0.8.6', 'garbage')).toBe(0)
    expect(compareVersions('', '0.8.6')).toBe(0)
  })

  it('is conservative about prerelease suffixes (no nudge for an rc)', () => {
    expect(compareVersions('0.13.0-rc1', '0.12.1')).toBe(0)
    expect(compareVersions('v0.13.0-beta.2', '0.12.1')).toBe(0)
  })
})

describe('checkForUpdate', () => {
  it('returns the tag when a newer release exists', async () => {
    mockLatest('v0.9.0')
    expect(await checkForUpdate('0.8.6')).toBe('v0.9.0')
  })

  it('returns null when up to date or ahead', async () => {
    mockLatest('v0.8.6')
    expect(await checkForUpdate('0.8.6')).toBeNull()
    window.localStorage.clear()
    mockLatest('v0.8.5')
    expect(await checkForUpdate('0.8.6')).toBeNull()
  })

  it('is silent on 404/rate-limit responses (private repo)', async () => {
    mockLatest(null, false, 404)
    expect(await checkForUpdate('0.8.6')).toBeNull()
  })

  it('is silent on network failure', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new Error('offline')
      }),
    )
    expect(await checkForUpdate('0.8.6')).toBeNull()
  })

  it('skips dev/unknown/unparseable builds without fetching', async () => {
    const fetchSpy = vi.fn()
    vi.stubGlobal('fetch', fetchSpy)
    expect(await checkForUpdate('dev')).toBeNull()
    expect(await checkForUpdate(null)).toBeNull()
    expect(await checkForUpdate('not-a-version')).toBeNull()
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it('skips demo mode without fetching', async () => {
    const fetchSpy = vi.fn()
    vi.stubGlobal('fetch', fetchSpy)
    expect(await checkForUpdate('0.8.6', { demoMode: true })).toBeNull()
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it('passes an abort signal to fetch (5s timeout guard)', async () => {
    const fetchMock = mockLatest('v0.9.0')
    await checkForUpdate('0.8.6')
    const init = fetchMock.mock.calls[0][1] as RequestInit | undefined
    expect(init?.signal).toBeInstanceOf(AbortSignal)
  })

  describe('24h cache', () => {
    it('records checkedAt + latestTag after a successful fetch', async () => {
      mockLatest('v0.9.0')
      await checkForUpdate('0.8.6', { now: 1_000_000 })
      expect(readCache()).toMatchObject({ checkedAt: 1_000_000, latestTag: 'v0.9.0' })
    })

    it('answers from a fresh cache without touching the network', async () => {
      writeCache({ checkedAt: 1_000_000, latestTag: 'v0.9.0' })
      const fetchSpy = vi.fn()
      vi.stubGlobal('fetch', fetchSpy)
      const tag = await checkForUpdate('0.8.6', { now: 1_000_000 + 60 * 60 * 1000 })
      expect(tag).toBe('v0.9.0')
      expect(fetchSpy).not.toHaveBeenCalled()
    })

    it('refetches once the cache is older than 24h', async () => {
      writeCache({ checkedAt: 1_000_000, latestTag: 'v0.9.0' })
      const fetchMock = mockLatest('v0.10.0')
      const now = 1_000_000 + 25 * 60 * 60 * 1000
      const tag = await checkForUpdate('0.8.6', { now })
      expect(tag).toBe('v0.10.0')
      expect(fetchMock).toHaveBeenCalledTimes(1)
      expect(readCache()).toMatchObject({ checkedAt: now, latestTag: 'v0.10.0' })
    })

    it('falls back to the stale cached tag when the refetch fails', async () => {
      writeCache({ checkedAt: 1_000_000, latestTag: 'v0.9.0' })
      vi.stubGlobal(
        'fetch',
        vi.fn(async () => {
          throw new Error('offline')
        }),
      )
      const tag = await checkForUpdate('0.8.6', { now: 1_000_000 + 25 * 60 * 60 * 1000 })
      expect(tag).toBe('v0.9.0')
      // Failed fetch must not refresh checkedAt (next launch retries).
      expect(readCache().checkedAt).toBe(1_000_000)
    })
  })

  describe('dismiss semantics', () => {
    it('never nudges again for a dismissed tag', async () => {
      mockLatest('v0.9.0')
      expect(await checkForUpdate('0.8.6')).toBe('v0.9.0')
      dismissUpdate('v0.9.0')
      expect(await checkForUpdate('0.8.6')).toBeNull()
    })

    it('a newer release than the dismissed one nudges again', async () => {
      writeCache({ checkedAt: 0, latestTag: 'v0.9.0', dismissedTag: 'v0.9.0' })
      mockLatest('v0.10.0')
      const tag = await checkForUpdate('0.8.6', { now: 25 * 60 * 60 * 1000 })
      expect(tag).toBe('v0.10.0')
      // The dismissal survives the cache refresh.
      expect(readCache().dismissedTag).toBe('v0.9.0')
    })

    it('honors the legacy kurisu_skip_version key', async () => {
      window.localStorage.setItem('kurisu_skip_version', 'v0.9.0')
      mockLatest('v0.9.0')
      expect(await checkForUpdate('0.8.6')).toBeNull()
    })
  })

  it('survives garbage in the cache slot', async () => {
    window.localStorage.setItem(UPDATE_CHECK_STORAGE_KEY, 'not json{{{')
    mockLatest('v0.9.0')
    expect(await checkForUpdate('0.8.6')).toBe('v0.9.0')
  })
})

describe('runOnceUpdateCheck', () => {
  it('only checks once per launch', async () => {
    mockLatest('v0.9.0')
    expect(await runOnceUpdateCheck('0.8.6')).toBe('v0.9.0')
    expect(await runOnceUpdateCheck('0.8.6')).toBeNull()
  })
})
