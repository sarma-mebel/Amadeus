// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom/vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import UpdateNudge from './UpdateNudge'
import { __resetUpdateCheckForTests, UPDATE_CHECK_STORAGE_KEY } from '../updateCheck'

const jsonResponse = (payload: unknown, status = 200) =>
  new Response(JSON.stringify(payload), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })

const statusPayload = (overrides: Record<string, unknown> = {}) => ({
  version: '0.12.1',
  llm: { provider: 'gemini', class: 'GeminiProvider', model: 'g', is_dummy: false },
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
    summary_provider_active: true,
  },
  ...overrides,
})

/** Answers /api/status with the given payload and the GitHub latest-release
 * endpoint with the given tag; everything else 404s. */
const stubFetch = (
  status: Record<string, unknown> | null,
  latestTag: string | null,
) =>
  vi.fn(async (url: RequestInfo | URL) => {
    const path = String(url)
    if (path.endsWith('/api/status')) {
      return status ? jsonResponse(status) : jsonResponse({}, 404)
    }
    if (path.includes('api.github.com')) {
      return latestTag ? jsonResponse({ tag_name: latestTag }) : jsonResponse({}, 404)
    }
    return jsonResponse({}, 404)
  })

describe('UpdateNudge', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.spyOn(console, 'debug').mockImplementation(() => {})
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
    __resetUpdateCheckForTests()
  })

  it('renders the banner when a newer release exists', async () => {
    vi.stubGlobal('fetch', stubFetch(statusPayload(), 'v0.13.0'))
    render(<UpdateNudge />)
    expect(await screen.findByRole('status')).toHaveTextContent('v0.13.0 is available')
    expect(screen.getByRole('button', { name: /Releases/ })).toBeInTheDocument()
  })

  it('renders nothing when already up to date', async () => {
    const fetchMock = stubFetch(statusPayload(), 'v0.12.1')
    vi.stubGlobal('fetch', fetchMock)
    const { container } = render(<UpdateNudge />)
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    await waitFor(() => expect(container).toBeEmptyDOMElement())
  })

  it('renders nothing in demo mode', async () => {
    const fetchMock = stubFetch(statusPayload({ demo_mode: true }), 'v0.13.0')
    vi.stubGlobal('fetch', fetchMock)
    const { container } = render(<UpdateNudge />)
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    await waitFor(() => expect(container).toBeEmptyDOMElement())
    // The GitHub API must not even be consulted for a hosted demo viewer.
    const githubCalls = fetchMock.mock.calls.filter(([url]) =>
      String(url).includes('api.github.com'),
    )
    expect(githubCalls).toHaveLength(0)
  })

  it('renders nothing when the backend version is unparseable', async () => {
    const fetchMock = stubFetch(statusPayload({ version: 'dev' }), 'v0.13.0')
    vi.stubGlobal('fetch', fetchMock)
    const { container } = render(<UpdateNudge />)
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    await waitFor(() => expect(container).toBeEmptyDOMElement())
  })

  it('renders nothing when the status fetch fails', async () => {
    const fetchMock = stubFetch(null, 'v0.13.0')
    vi.stubGlobal('fetch', fetchMock)
    const { container } = render(<UpdateNudge />)
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    await waitFor(() => expect(container).toBeEmptyDOMElement())
  })

  it('dismissing hides the banner and persists the tag', async () => {
    vi.stubGlobal('fetch', stubFetch(statusPayload(), 'v0.13.0'))
    render(<UpdateNudge />)
    await screen.findByRole('status')
    await userEvent.click(
      screen.getByRole('button', { name: 'Dismiss update notice' }),
    )
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
    const cache = JSON.parse(
      window.localStorage.getItem(UPDATE_CHECK_STORAGE_KEY) ?? '{}',
    )
    expect(cache.dismissedTag).toBe('v0.13.0')
  })

  it('stays hidden on a remount once the tag was dismissed', async () => {
    vi.stubGlobal('fetch', stubFetch(statusPayload(), 'v0.13.0'))
    const first = render(<UpdateNudge />)
    await first.findByRole('status')
    await userEvent.click(
      screen.getByRole('button', { name: 'Dismiss update notice' }),
    )
    first.unmount()

    const second = render(<UpdateNudge />)
    // Give the async status -> update-check chain time to settle.
    await waitFor(() => expect(second.container).toBeEmptyDOMElement())
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
  })
})
