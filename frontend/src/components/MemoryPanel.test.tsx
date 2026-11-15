// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom/vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import MemoryPanel from './MemoryPanel'
import { resolveUserId } from '../hooks/useSessions'
import { markSeen } from '../milestones'
import { markIntroDone, isIntroDone } from '../introMode'

const baseOverview = {
  status: 'ok',
  overview: {
    profile: {
      facts: { name: 'Alex', occupation: 'researcher' },
      open_threads: ['finish the CMB paper'],
      session_count: 4,
      first_met: '2026-06-01',
      last_episode: 'Talked about time travel.',
    },
    facts: ['Prefers coffee over tea', 'Works late at night'],
    episodes: ['We argued about the Copenhagen interpretation.'],
    counts: { facts: 2, episodes: 1, reflections: 0 },
  },
}

const jsonResponse = (payload: unknown) =>
  new Response(JSON.stringify(payload), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })

describe('MemoryPanel', () => {
  let fetchMock: ReturnType<typeof vi.fn>
  let memoryBodies: Array<Record<string, unknown>>

  // No vitest `globals` in this repo's config, so RTL's automatic
  // afterEach cleanup never registers — without this, renders stack up
  // across tests and every query starts matching twice.
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  const stubMemoryApi = (
    respond: (body: Record<string, unknown>) => Response | Promise<Response> = () =>
      jsonResponse(baseOverview),
  ) => {
    memoryBodies = []
    fetchMock = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
      const path = String(url)
      if (path.endsWith('/api/memory') && init?.method === 'POST') {
        const body = JSON.parse(String(init.body)) as Record<string, unknown>
        memoryBodies.push(body)
        return respond(body)
      }
      if (path.includes('/api/diary')) {
        return jsonResponse({ diary: null })
      }
      throw new Error(`Unexpected fetch: ${path}`)
    })
    vi.stubGlobal('fetch', fetchMock)
  }

  beforeEach(() => {
    window.localStorage.clear()
    stubMemoryApi()
  })

  it('loads the overview and renders all sections', async () => {
    render(<MemoryPanel onClose={() => {}} />)

    expect(await screen.findByText('About you')).toBeInTheDocument()
    expect(screen.getByText('Alex')).toBeInTheDocument()
    expect(screen.getByText('Prefers coffee over tea')).toBeInTheDocument()
    expect(screen.getByText('We argued about the Copenhagen interpretation.')).toBeInTheDocument()
    expect(screen.getByText('finish the CMB paper')).toBeInTheDocument()
    // The overview request carried the action and ids.
    expect(memoryBodies[0]).toMatchObject({ action: 'overview' })
  })

  it('optimistically removes a fact and sends the exact text to forget', async () => {
    const user = userEvent.setup()
    render(<MemoryPanel onClose={() => {}} />)

    const forgetButton = await screen.findByRole('button', {
      name: 'Forget: Prefers coffee over tea',
    })
    await user.click(forgetButton)

    expect(screen.queryByText('Prefers coffee over tea')).not.toBeInTheDocument()
    await waitFor(() => {
      expect(memoryBodies).toContainEqual(
        expect.objectContaining({ action: 'forget', text: 'Prefers coffee over tea' }),
      )
    })
  })

  it('submits a new memory through the remember input', async () => {
    const user = userEvent.setup()
    render(<MemoryPanel onClose={() => {}} />)

    const input = await screen.findByRole('textbox', { name: 'Remember something' })
    await user.type(input, 'My birthday is June 10')
    await user.click(screen.getByRole('button', { name: 'Remember' }))

    await waitFor(() => {
      expect(memoryBodies).toContainEqual(
        expect.objectContaining({ action: 'remember', text: 'My birthday is June 10' }),
      )
    })
    // Optimistically prepended into the Memories list.
    expect(await screen.findByText('My birthday is June 10')).toBeInTheDocument()
  })

  it('shows the designed EmptyState when she remembers nothing', async () => {
    stubMemoryApi(() =>
      jsonResponse({
        status: 'ok',
        overview: {
          profile: { facts: {}, open_threads: [] },
          facts: [],
          episodes: [],
          counts: { facts: 0, episodes: 0, reflections: 0 },
        },
      }),
    )
    const { container } = render(<MemoryPanel onClose={() => {}} />)

    expect(await screen.findByText('Nothing recorded yet.')).toBeInTheDocument()
    expect(
      screen.getByText("She's still getting to know you — memories show up here once you talk."),
    ).toBeInTheDocument()
    expect(container.querySelector('.empty-state-icon svg')).toBeInTheDocument()
  })

  it('shows the designed EmptyState on the synapse map tab when the graph has no nodes', async () => {
    const user = userEvent.setup()
    stubMemoryApi((body) =>
      body.action === 'graph'
        ? jsonResponse({ graph: { nodes: [], edges: [] } })
        : jsonResponse(baseOverview),
    )
    const { container } = render(<MemoryPanel onClose={() => {}} />)

    await screen.findByText('About you')
    await user.click(screen.getByRole('tab', { name: 'Synapse map' }))

    expect(await screen.findByText('No synapses formed yet.')).toBeInTheDocument()
    expect(container.querySelector('.empty-state-icon svg')).toBeInTheDocument()
  })

  it('points at Settings → Features when long-term memory is off', async () => {
    stubMemoryApi(() => jsonResponse({ status: 'unavailable' }))
    render(<MemoryPanel onClose={() => {}} />)

    expect(await screen.findByText(/Settings → Features/)).toBeInTheDocument()
    // No remember input in the unavailable state.
    expect(screen.queryByRole('textbox', { name: 'Remember something' })).not.toBeInTheDocument()
  })

  it('closes on Escape', async () => {
    const onClose = vi.fn()
    const user = userEvent.setup()
    render(<MemoryPanel onClose={onClose} />)
    await screen.findByText('About you')

    await user.keyboard('{Escape}')
    expect(onClose).toHaveBeenCalled()
  })

  describe('relationship arc', () => {
    const overviewWithRelationship = {
      status: 'ok',
      overview: {
        ...baseOverview.overview,
        relationship: {
          trust_level: 0.62,
          arc: {
            curiosity_vs_annoyance: 0.14,
            logic_vs_emotion: -0.05,
            pride_vs_vulnerability: -0.08,
          },
          tone_history: [{ date: '2026-07-10', tone: 0.3 }],
          tone_trend: 'low',
        },
      },
    }

    it('renders arc values and pole labels from a mocked overview', async () => {
      stubMemoryApi(() => jsonResponse(overviewWithRelationship))
      render(<MemoryPanel onClose={() => {}} />)

      expect(await screen.findByText('Relationship')).toBeInTheDocument()
      expect(screen.getByText('62%')).toBeInTheDocument()
      expect(screen.getByText('annoyance')).toBeInTheDocument()
      expect(screen.getByText('curiosity')).toBeInTheDocument()
      expect(screen.getByText('emotion')).toBeInTheDocument()
      expect(screen.getByText('logic')).toBeInTheDocument()
      expect(screen.getByText('vulnerability')).toBeInTheDocument()
      expect(screen.getByText('pride')).toBeInTheDocument()
      expect(screen.getByText("you've seemed low lately")).toBeInTheDocument()
    })

    it('renders no Relationship section when the key is absent', async () => {
      render(<MemoryPanel onClose={() => {}} />)

      await screen.findByText('About you')
      expect(screen.queryByText('Relationship')).not.toBeInTheDocument()
    })

    it('reset button issues the reset_relationship action and refetches', async () => {
      let overviewCalls = 0
      stubMemoryApi((body) => {
        if (body.action === 'overview') {
          overviewCalls += 1
          return jsonResponse(overviewWithRelationship)
        }
        if (body.action === 'reset_relationship') {
          return jsonResponse({ status: 'ok', detail: 'Reset.' })
        }
        throw new Error(`Unexpected action: ${String(body.action)}`)
      })
      const user = userEvent.setup()
      render(<MemoryPanel onClose={() => {}} />)

      const resetButton = await screen.findByRole('button', { name: 'Reset arc' })
      await user.click(resetButton)
      const confirmButton = await screen.findByRole('button', {
        name: 'Confirm resetting the relationship arc',
      })
      await user.click(confirmButton)

      await waitFor(() => {
        expect(memoryBodies).toContainEqual(
          expect.objectContaining({ action: 'reset_relationship' }),
        )
      })
      await waitFor(() => expect(overviewCalls).toBe(2))
    })

    it('reset re-arms milestone celebrations and conversational onboarding (item 9)', async () => {
      const userId = resolveUserId()
      markSeen([{ kind: 'sessions', value: 10, label: '10 sessions' }], userId)
      markIntroDone(userId)
      expect(isIntroDone(userId)).toBe(true)
      expect(
        window.localStorage.getItem(`kurisu_milestones_seen_v1:${userId}`),
      ).toContain('sessions:10')

      stubMemoryApi((body) => {
        if (body.action === 'reset_relationship') {
          return jsonResponse({ status: 'ok', detail: 'Reset.' })
        }
        return jsonResponse(overviewWithRelationship)
      })
      const user = userEvent.setup()
      render(<MemoryPanel onClose={() => {}} />)

      const resetButton = await screen.findByRole('button', { name: 'Reset arc' })
      await user.click(resetButton)
      await user.click(
        await screen.findByRole('button', { name: 'Confirm resetting the relationship arc' }),
      )

      await waitFor(() => {
        expect(memoryBodies).toContainEqual(
          expect.objectContaining({ action: 'reset_relationship' }),
        )
      })

      await waitFor(() => expect(isIntroDone(userId)).toBe(false))
      expect(
        window.localStorage.getItem(`kurisu_milestones_seen_v1:${userId}`),
      ).toBeNull()
    })
  })

  describe('fact_records / pin / edit / forget_id (M4)', () => {
    const overviewWithRecords = {
      status: 'ok',
      overview: {
        profile: { facts: {}, open_threads: [] },
        fact_records: [
          { id: 'f1', text: 'Prefers coffee over tea', pinned: false, timestamp: '2026-07-01' },
          { id: 'f2', text: 'Works late at night', pinned: true, timestamp: '2026-07-02' },
        ],
        episode_records: [
          { id: 'e1', text: 'We argued about the Copenhagen interpretation.', pinned: false, timestamp: '2026-07-03' },
        ],
      },
    }

    it('renders fact_records, sorting pinned first', async () => {
      stubMemoryApi(() => jsonResponse(overviewWithRecords))
      render(<MemoryPanel onClose={() => {}} />)

      await screen.findByText('Prefers coffee over tea')
      const items = screen.getAllByRole('listitem')
      const factItems = items.filter((item) =>
        ['Prefers coffee over tea', 'Works late at night'].some((text) =>
          item.textContent?.includes(text),
        ),
      )
      expect(factItems[0]).toHaveTextContent('Works late at night')
      expect(factItems[1]).toHaveTextContent('Prefers coffee over tea')
    })

    it('pin toggle sends the pin action with the record id and flips optimistically', async () => {
      stubMemoryApi(() => jsonResponse(overviewWithRecords))
      const user = userEvent.setup()
      render(<MemoryPanel onClose={() => {}} />)

      await screen.findByText('Prefers coffee over tea')
      const pinButton = screen.getByRole('button', { name: 'Pin: Prefers coffee over tea' })
      await user.click(pinButton)

      await waitFor(() => {
        expect(memoryBodies).toContainEqual(
          expect.objectContaining({ action: 'pin', record_id: 'f1' }),
        )
      })
      expect(screen.getByRole('button', { name: 'Unpin: Prefers coffee over tea' })).toHaveAttribute(
        'aria-pressed',
        'true',
      )
    })

    it('edit swaps the row for an input and sends the edit action', async () => {
      stubMemoryApi(() => jsonResponse(overviewWithRecords))
      const user = userEvent.setup()
      render(<MemoryPanel onClose={() => {}} />)

      await screen.findByText('Prefers coffee over tea')
      await user.click(screen.getByRole('button', { name: 'Edit: Prefers coffee over tea' }))

      const input = screen.getByRole('textbox', { name: 'Edit memory: Prefers coffee over tea' })
      await user.clear(input)
      await user.type(input, 'Prefers tea now')
      await user.click(screen.getByRole('button', { name: 'Save' }))

      await waitFor(() => {
        expect(memoryBodies).toContainEqual(
          expect.objectContaining({ action: 'edit', record_id: 'f1', text: 'Prefers tea now' }),
        )
      })
      expect(await screen.findByText('Prefers tea now')).toBeInTheDocument()
    })

    it('deletes a record with an id via forget_id', async () => {
      stubMemoryApi(() => jsonResponse(overviewWithRecords))
      const user = userEvent.setup()
      render(<MemoryPanel onClose={() => {}} />)

      await screen.findByText('Prefers coffee over tea')
      await user.click(screen.getByRole('button', { name: 'Forget: Prefers coffee over tea' }))

      expect(screen.queryByText('Prefers coffee over tea')).not.toBeInTheDocument()
      await waitFor(() => {
        expect(memoryBodies).toContainEqual(
          expect.objectContaining({ action: 'forget_id', record_id: 'f1' }),
        )
      })
    })

    it('falls back to legacy forget-by-text when a record has no id', async () => {
      stubMemoryApi(() =>
        jsonResponse({
          status: 'ok',
          overview: {
            profile: { facts: {}, open_threads: [] },
            facts: ['A legacy fact with no id'],
            episodes: [],
          },
        }),
      )
      const user = userEvent.setup()
      render(<MemoryPanel onClose={() => {}} />)

      await screen.findByText('A legacy fact with no id')
      await user.click(screen.getByRole('button', { name: 'Forget: A legacy fact with no id' }))

      await waitFor(() => {
        expect(memoryBodies).toContainEqual(
          expect.objectContaining({ action: 'forget', text: 'A legacy fact with no id' }),
        )
      })
      // No pin/edit buttons without a record id.
      expect(screen.queryByRole('button', { name: /^Pin:/ })).not.toBeInTheDocument()
    })

    it('keeps the row and shows the protected message when forget_id returns protected', async () => {
      stubMemoryApi((body) => {
        if (body.action === 'forget_id') {
          return jsonResponse({ status: 'protected', detail: 'That memory is pinned — unpin it first.' })
        }
        return jsonResponse(overviewWithRecords)
      })
      const user = userEvent.setup()
      render(<MemoryPanel onClose={() => {}} />)

      await screen.findByText('Works late at night')
      await user.click(screen.getByRole('button', { name: 'Forget: Works late at night' }))

      await waitFor(() => {
        expect(memoryBodies).toContainEqual(
          expect.objectContaining({ action: 'forget_id', record_id: 'f2' }),
        )
      })
      // Row restored (not removed) with the inline protected message.
      expect(await screen.findByText('Works late at night')).toBeInTheDocument()
      expect(await screen.findByText('That memory is pinned — unpin it first.')).toBeInTheDocument()
    })
  })

  describe('her diary (M3)', () => {
    it('renders the diary section when present, with the footer note', async () => {
      fetchMock = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
        const path = String(url)
        if (path.endsWith('/api/memory') && init?.method === 'POST') {
          memoryBodies.push(JSON.parse(String(init.body)))
          return jsonResponse(baseOverview)
        }
        if (path.includes('/api/diary')) {
          return jsonResponse({
            diary: { date: '2026-07-10', text: 'Today he asked about the CMB paper again.', generated_at: '2026-07-10T08:00:00Z', consumed: false },
          })
        }
        throw new Error(`Unexpected fetch: ${path}`)
      })
      memoryBodies = []
      vi.stubGlobal('fetch', fetchMock)

      render(<MemoryPanel onClose={() => {}} />)

      expect(await screen.findByText('Her diary')).toBeInTheDocument()
      expect(screen.getByText('Today he asked about the CMB paper again.')).toBeInTheDocument()
      expect(
        screen.getByText("Her own note about her day — you weren't supposed to see this."),
      ).toBeInTheDocument()
    })

    it('renders nothing for a null diary', async () => {
      stubMemoryApi() // default stub answers /api/diary with { diary: null }
      render(<MemoryPanel onClose={() => {}} />)

      await screen.findByText('About you')
      expect(screen.queryByText('Her diary')).not.toBeInTheDocument()
    })

    it('renders nothing when the diary fetch errors', async () => {
      fetchMock = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
        const path = String(url)
        if (path.endsWith('/api/memory') && init?.method === 'POST') {
          return jsonResponse(baseOverview)
        }
        if (path.includes('/api/diary')) {
          throw new Error('network error')
        }
        throw new Error(`Unexpected fetch: ${path}`)
      })
      vi.stubGlobal('fetch', fetchMock)

      render(<MemoryPanel onClose={() => {}} />)

      await screen.findByText('About you')
      expect(screen.queryByText('Her diary')).not.toBeInTheDocument()
    })

    it('renders nothing for a malformed diary payload', async () => {
      fetchMock = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
        const path = String(url)
        if (path.endsWith('/api/memory') && init?.method === 'POST') {
          return jsonResponse(baseOverview)
        }
        if (path.includes('/api/diary')) {
          return jsonResponse({ diary: { date: '2026-07-10' } }) // missing text
        }
        throw new Error(`Unexpected fetch: ${path}`)
      })
      vi.stubGlobal('fetch', fetchMock)

      render(<MemoryPanel onClose={() => {}} />)

      await screen.findByText('About you')
      expect(screen.queryByText('Her diary')).not.toBeInTheDocument()
    })

    it('shows the diary even when the rest of the overview is empty', async () => {
      fetchMock = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
        const path = String(url)
        if (path.endsWith('/api/memory') && init?.method === 'POST') {
          return jsonResponse({
            status: 'ok',
            overview: { profile: { facts: {}, open_threads: [] }, facts: [], episodes: [] },
          })
        }
        if (path.includes('/api/diary')) {
          return jsonResponse({
            diary: { date: '2026-07-10', text: 'A quiet day.', generated_at: '2026-07-10T08:00:00Z', consumed: false },
          })
        }
        throw new Error(`Unexpected fetch: ${path}`)
      })
      vi.stubGlobal('fetch', fetchMock)

      render(<MemoryPanel onClose={() => {}} />)

      expect(await screen.findByText('Nothing recorded yet.')).toBeInTheDocument()
      expect(await screen.findByText('Her diary')).toBeInTheDocument()
      expect(screen.getByText('A quiet day.')).toBeInTheDocument()
    })
  })
})
