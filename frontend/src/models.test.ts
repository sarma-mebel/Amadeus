// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  DEFAULT_MODEL,
  __resetModelRegistryCache,
  fetchModelRegistry,
  loadModelId,
  MODEL_CHANGE_EVENT,
  dispatchModelChange,
  resolveModelFile,
  saveModelId,
  type ModelDefinition,
} from './models'

const jsonResponse = (payload: unknown, status = 200) =>
  new Response(JSON.stringify(payload), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })

beforeEach(() => {
  window.localStorage.clear()
  __resetModelRegistryCache()
})

afterEach(() => {
  window.localStorage.clear()
  vi.unstubAllGlobals()
  __resetModelRegistryCache()
})

describe('loadModelId', () => {
  it('falls back to the default id when nothing is stored', () => {
    expect(loadModelId()).toBe(DEFAULT_MODEL.id)
  })

  it('falls back to the default id on garbage (empty string)', () => {
    window.localStorage.setItem('kurisu_model_v1', '')
    expect(loadModelId()).toBe(DEFAULT_MODEL.id)
  })

  it('returns a previously saved id', () => {
    saveModelId('some_outfit')
    expect(loadModelId()).toBe('some_outfit')
  })
})

describe('resolveModelFile', () => {
  const registry: ModelDefinition[] = [
    DEFAULT_MODEL,
    { id: 'casual', label: 'Casual', file: '/static/models/casual.vrm' },
  ]

  it('resolves a matching entry', () => {
    expect(resolveModelFile('casual', registry)).toBe('/static/models/casual.vrm')
  })

  it('falls back to the default file on a miss', () => {
    expect(resolveModelFile('nonexistent', registry)).toBe(DEFAULT_MODEL.file)
  })
})

describe('fetchModelRegistry', () => {
  it('returns only the default model on a 404', async () => {
    const fetchMock = vi.fn(async () => jsonResponse({}, 404))
    vi.stubGlobal('fetch', fetchMock)

    const registry = await fetchModelRegistry()
    expect(registry).toEqual([DEFAULT_MODEL])
  })

  it('returns only the default model on a network error', async () => {
    const fetchMock = vi.fn(async () => {
      throw new Error('network down')
    })
    vi.stubGlobal('fetch', fetchMock)

    const registry = await fetchModelRegistry()
    expect(registry).toEqual([DEFAULT_MODEL])
  })

  it('returns only the default model on malformed JSON', async () => {
    const fetchMock = vi.fn(
      async () =>
        new Response('not json{{{', {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
    )
    vi.stubGlobal('fetch', fetchMock)

    const registry = await fetchModelRegistry()
    expect(registry).toEqual([DEFAULT_MODEL])
  })

  it('merges valid extra entries with the default', async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({
        models: [{ id: 'casual', label: 'Casual', file: '/static/models/casual.vrm' }],
      })
    )
    vi.stubGlobal('fetch', fetchMock)

    const registry = await fetchModelRegistry()
    expect(registry).toEqual([
      DEFAULT_MODEL,
      { id: 'casual', label: 'Casual', file: '/static/models/casual.vrm' },
    ])
  })

  it('rejects an entry with a traversal path', async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({
        models: [{ id: 'evil', label: 'Evil', file: '../evil.vrm' }],
      })
    )
    vi.stubGlobal('fetch', fetchMock)

    const registry = await fetchModelRegistry()
    expect(registry).toEqual([DEFAULT_MODEL])
  })

  it('rejects an entry with an absolute external URL', async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({
        models: [{ id: 'evil', label: 'Evil', file: 'http://x/evil.vrm' }],
      })
    )
    vi.stubGlobal('fetch', fetchMock)

    const registry = await fetchModelRegistry()
    expect(registry).toEqual([DEFAULT_MODEL])
  })

  it('ignores a duplicate id (first wins)', async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({
        models: [
          { id: 'casual', label: 'Casual', file: '/static/models/casual.vrm' },
          { id: 'casual', label: 'Casual 2', file: '/static/models/casual2.vrm' },
        ],
      })
    )
    vi.stubGlobal('fetch', fetchMock)

    const registry = await fetchModelRegistry()
    expect(registry).toEqual([
      DEFAULT_MODEL,
      { id: 'casual', label: 'Casual', file: '/static/models/casual.vrm' },
    ])
  })

  it('ignores an entry colliding with the default id', async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({
        models: [{ id: DEFAULT_MODEL.id, label: 'Fake default', file: '/static/models/fake.vrm' }],
      })
    )
    vi.stubGlobal('fetch', fetchMock)

    const registry = await fetchModelRegistry()
    expect(registry).toEqual([DEFAULT_MODEL])
  })

  it('rejects malformed entries (missing/empty fields)', async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({
        models: [
          { id: '', label: 'No id', file: '/static/models/x.vrm' },
          { id: 'ok', label: '', file: '/static/models/y.vrm' },
          { id: 'ok2', label: 'Ok2' },
          { id: 123, label: 'Bad id type', file: '/static/models/z.vrm' },
        ],
      })
    )
    vi.stubGlobal('fetch', fetchMock)

    const registry = await fetchModelRegistry()
    expect(registry).toEqual([DEFAULT_MODEL])
  })

  it('caches the promise so fetch is only called once across repeated calls', async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({
        models: [{ id: 'casual', label: 'Casual', file: '/static/models/casual.vrm' }],
      })
    )
    vi.stubGlobal('fetch', fetchMock)

    const first = await fetchModelRegistry()
    const second = await fetchModelRegistry()
    expect(first).toEqual(second)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })
})

describe('dispatchModelChange', () => {
  it('dispatches a CustomEvent with the id as detail', () => {
    const handler = vi.fn()
    window.addEventListener(MODEL_CHANGE_EVENT, handler)
    dispatchModelChange('casual')
    expect(handler).toHaveBeenCalledTimes(1)
    const event = handler.mock.calls[0][0] as CustomEvent<string>
    expect(event.detail).toBe('casual')
    window.removeEventListener(MODEL_CHANGE_EVENT, handler)
  })
})
