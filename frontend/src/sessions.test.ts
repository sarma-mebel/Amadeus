import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  createSession,
  deleteSession,
  fetchSessionTranscript,
  formatRelativeTime,
  listSessions,
} from './sessions'
import { saveApiToken } from './api'

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

beforeEach(() => {
  vi.stubGlobal('window', {
    localStorage: makeLocalStorage(),
    dispatchEvent: () => true,
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

const stubFetch = (status: number, payload?: unknown) => {
  const fetchMock = vi.fn(async () => ({
    status,
    ok: status >= 200 && status < 300,
    json: async () => payload,
  }))
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

describe('listSessions', () => {
  it('returns null on 404 (older backend without sessions routes)', async () => {
    stubFetch(404)
    expect(await listSessions('user-1')).toBeNull()
  })

  it('throws on other HTTP errors', async () => {
    stubFetch(500)
    await expect(listSessions('user-1')).rejects.toThrow('HTTP 500')
  })

  it('parses the sessions payload and encodes the user id', async () => {
    const fetchMock = stubFetch(200, {
      sessions: [
        {
          session_id: 's1',
          title: 'Time travel talk',
          created_at: '2026-07-01T10:00:00Z',
          updated_at: '2026-07-02T10:00:00Z',
          message_count: 8,
        },
      ],
    })
    const sessions = await listSessions('user one')
    expect(String(fetchMock.mock.calls[0][0]).endsWith('/api/sessions?user_id=user%20one')).toBe(true)
    expect(sessions).toEqual([
      {
        session_id: 's1',
        title: 'Time travel talk',
        created_at: '2026-07-01T10:00:00Z',
        updated_at: '2026-07-02T10:00:00Z',
        message_count: 8,
      },
    ])
  })

  it('skips malformed entries and defaults missing fields', async () => {
    stubFetch(200, {
      sessions: [
        null,
        'garbage',
        { title: 'no id' },
        { session_id: 's2', message_count: 'three' },
      ],
    })
    const sessions = await listSessions('u')
    expect(sessions).toEqual([
      {
        session_id: 's2',
        title: 'Untitled conversation',
        created_at: '',
        updated_at: '',
        message_count: 0,
      },
    ])
  })

  it('sends the Bearer header through the shared fetch helper', async () => {
    saveApiToken('sekrit')
    const fetchMock = stubFetch(200, { sessions: [] })
    await listSessions('u')
    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(new Headers(init.headers).get('Authorization')).toBe('Bearer sekrit')
  })
})

describe('createSession', () => {
  it('POSTs the user id and returns the new session id', async () => {
    const fetchMock = stubFetch(200, { session_id: 'fresh' })
    expect(await createSession('u1')).toBe('fresh')
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url.endsWith('/api/sessions/new')).toBe(true)
    expect(init.method).toBe('POST')
    expect(JSON.parse(String(init.body))).toEqual({ user_id: 'u1' })
  })

  it('throws when the response has no session_id', async () => {
    stubFetch(200, {})
    await expect(createSession('u1')).rejects.toThrow('session_id')
  })

  it('throws on HTTP errors', async () => {
    stubFetch(500)
    await expect(createSession('u1')).rejects.toThrow('HTTP 500')
  })
})

describe('deleteSession', () => {
  it('DELETEs the session with the user id, encoding both', async () => {
    const fetchMock = stubFetch(200, { ok: true })
    await deleteSession('s/1', 'u 1')
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url.endsWith('/api/sessions/s%2F1?user_id=u%201')).toBe(true)
    expect(init.method).toBe('DELETE')
  })

  it('throws on HTTP errors', async () => {
    stubFetch(404)
    await expect(deleteSession('s1', 'u1')).rejects.toThrow('HTTP 404')
  })
})

describe('fetchSessionTranscript', () => {
  it('returns chronological messages with ja/emotion normalized', async () => {
    const fetchMock = stubFetch(200, {
      messages: [
        {
          role: 'user',
          content: 'hi',
          ja: null,
          emotion: null,
          timestamp: '2026-07-01T10:00:00Z',
        },
        {
          role: 'assistant',
          content: 'Hello.',
          ja: 'こんにちは。',
          emotion: 'happy',
          timestamp: '2026-07-01T10:00:05Z',
        },
      ],
    })
    const messages = await fetchSessionTranscript('s1', 'u1')
    expect(String(fetchMock.mock.calls[0][0]).endsWith('/api/sessions/s1/transcript?user_id=u1')).toBe(true)
    expect(messages).toEqual([
      { role: 'user', content: 'hi', ja: null, emotion: null, timestamp: '2026-07-01T10:00:00Z', turn_id: null },
      {
        role: 'assistant',
        content: 'Hello.',
        ja: 'こんにちは。',
        emotion: 'happy',
        timestamp: '2026-07-01T10:00:05Z',
        turn_id: null,
      },
    ])
  })

  it('drops malformed entries and unknown roles', async () => {
    stubFetch(200, {
      messages: [
        null,
        { role: 'system', content: 'nope' },
        { role: 'assistant' },
        { role: 'user', content: 'kept', ja: '', emotion: '', timestamp: '' },
      ],
    })
    const messages = await fetchSessionTranscript('s1', 'u1')
    expect(messages).toEqual([
      { role: 'user', content: 'kept', ja: null, emotion: null, timestamp: '', turn_id: null },
    ])
  })

  it('throws on HTTP errors', async () => {
    stubFetch(500)
    await expect(fetchSessionTranscript('s1', 'u1')).rejects.toThrow('HTTP 500')
  })
})

describe('formatRelativeTime', () => {
  const now = Date.parse('2026-07-10T12:00:00Z')

  it('buckets recent timestamps', () => {
    expect(formatRelativeTime('2026-07-10T11:59:40Z', now)).toBe('just now')
    expect(formatRelativeTime('2026-07-10T11:35:00Z', now)).toBe('25m ago')
    expect(formatRelativeTime('2026-07-10T07:00:00Z', now)).toBe('5h ago')
    expect(formatRelativeTime('2026-07-08T12:00:00Z', now)).toBe('2d ago')
  })

  it('is empty on unparseable timestamps', () => {
    expect(formatRelativeTime('', now)).toBe('')
    expect(formatRelativeTime('garbage', now)).toBe('')
  })
})
