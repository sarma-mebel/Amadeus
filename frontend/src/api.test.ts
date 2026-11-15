import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  API_TOKEN_STORAGE_KEY,
  AUTH_ERROR_EVENT,
  apiFetch,
  loadApiToken,
  resolveWsUrl,
  saveApiToken,
} from './api'

const makeLocalStorage = () => {
  const store = new Map<string, string>()
  return {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => {
      store.set(key, value)
    },
    removeItem: (key: string) => {
      store.delete(key)
    },
  }
}

let dispatched: Event[]

beforeEach(() => {
  dispatched = []
  vi.stubGlobal('window', {
    localStorage: makeLocalStorage(),
    dispatchEvent: (event: Event) => {
      dispatched.push(event)
      return true
    },
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

const stubFetch = (status = 200) => {
  const fetchMock = vi.fn(async () => ({ status, ok: status >= 200 && status < 300 }))
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

describe('token storage', () => {
  it('round-trips and trims the token', () => {
    saveApiToken('  secret-token  ')
    expect(loadApiToken()).toBe('secret-token')
    expect(window.localStorage.getItem(API_TOKEN_STORAGE_KEY)).toBe('secret-token')
  })

  it('clears storage when saving an empty/whitespace token', () => {
    saveApiToken('secret')
    saveApiToken('   ')
    expect(loadApiToken()).toBe('')
    expect(window.localStorage.getItem(API_TOKEN_STORAGE_KEY)).toBeNull()
  })
})

describe('apiFetch auth wiring', () => {
  it('sends no Authorization header when no token is set', async () => {
    const fetchMock = stubFetch()
    await apiFetch('/api/status')
    // URL may be absolute when VITE_API_BASE_URL is configured — assert the
    // path suffix and that no init (hence no auth header) was added.
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit | undefined]
    expect(url.endsWith('/api/status')).toBe(true)
    expect(init).toBeUndefined()
  })

  it('attaches Bearer token to every request when configured', async () => {
    saveApiToken('secret-token')
    const fetchMock = stubFetch()
    await apiFetch('/api/status')
    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(new Headers(init.headers).get('Authorization')).toBe('Bearer secret-token')
  })

  it('preserves caller headers while adding Authorization', async () => {
    saveApiToken('secret-token')
    const fetchMock = stubFetch()
    await apiFetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
    })
    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    const headers = new Headers(init.headers)
    expect(headers.get('Content-Type')).toBe('application/json')
    expect(headers.get('Authorization')).toBe('Bearer secret-token')
    expect(init.method).toBe('POST')
    expect(init.body).toBe('{}')
  })

  it('dispatches the auth-error event on a 401 response', async () => {
    saveApiToken('wrong-token')
    stubFetch(401)
    const response = await apiFetch('/api/chat')
    expect(response.status).toBe(401)
    expect(dispatched.map((event) => event.type)).toContain(AUTH_ERROR_EVENT)
  })

  it('does not dispatch the auth-error event on success or non-401 errors', async () => {
    stubFetch(200)
    await apiFetch('/api/chat')
    stubFetch(503)
    await apiFetch('/api/tts')
    expect(dispatched).toHaveLength(0)
  })
})

describe('resolveWsUrl', () => {
  it('leaves the WS URL untouched when no token is set', () => {
    expect(resolveWsUrl('/api/chat/stream').endsWith('/api/chat/stream')).toBe(true)
  })

  it('appends the token as an encoded query param', () => {
    saveApiToken('se cret+t/oken')
    expect(
      resolveWsUrl('/api/chat/stream').endsWith(
        `/api/chat/stream?token=${encodeURIComponent('se cret+t/oken')}`,
      ),
    ).toBe(true)
  })
})
