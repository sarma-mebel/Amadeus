// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom/vitest'
import { act, cleanup, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import ChatInterface from './ChatInterface'
import TranscriptList from './TranscriptList'
import { kurisuBridge } from '../vrm/kurisuBridge'
import { resolveUserId } from '../hooks/useSessions'

const chatReply = {
  reply: 'Hmph. It works.',
  session_id: 'sess-smoke-1',
  timings: { llm_ms: 5, tts_ms: null, total_ms: 9 },
  sentences: [],
}

const jsonResponse = (payload: unknown, status = 200) =>
  new Response(JSON.stringify(payload), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })

/** Shared fetch stub for the regenerate/edit/rewind tests: answers
 * /api/chat with `replyForChat()` (so successive turns can return different
 * text/turn_id), and /rewind + /amend with simple ok responses, recording
 * every call so tests can assert on request bodies. Every other route
 * (mount-time probes) 404s like the smoke-test stub above. */
const buildTurnAwareFetchMock = (replyForChat: () => Record<string, unknown>) =>
  vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
    const path = String(url)
    if (path.endsWith('/api/chat') && init?.method === 'POST') {
      return jsonResponse(replyForChat())
    }
    if (path.includes('/rewind') && init?.method === 'POST') {
      return jsonResponse({ session_id: 'sess-smoke-1', status: 'ok', messages: [] })
    }
    if (path.includes('/amend') && init?.method === 'POST') {
      return jsonResponse({ session_id: 'sess-smoke-1', status: 'ok' })
    }
    return jsonResponse({}, 404)
  })

describe('ChatInterface smoke', () => {
  let fetchMock: ReturnType<typeof vi.fn>
  let wsAttempts: number

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  beforeEach(() => {
    window.localStorage.clear()
    wsAttempts = 0

    // jsdom gaps: the transcript autoscroll uses scrollIntoView/scrollTo.
    Element.prototype.scrollIntoView = vi.fn()
    Element.prototype.scrollTo = vi.fn()

    // Simulate an unreachable streaming endpoint: streamChat catches the
    // constructor throw and reports a connect failure, which must drop the
    // turn onto the POST path.
    class ExplodingWebSocket {
      constructor() {
        wsAttempts += 1
        throw new Error('no websocket in this test')
      }
    }
    vi.stubGlobal('WebSocket', ExplodingWebSocket)

    fetchMock = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
      const path = String(url)
      if (path.endsWith('/api/chat') && init?.method === 'POST') {
        return jsonResponse(chatReply)
      }
      // Every mount-time probe (status, sessions, STT,
      // reminders, greeting, update check) has an older-backend fallback
      // path — a 404 exercises exactly that.
      return jsonResponse({}, 404)
    })
    vi.stubGlobal('fetch', fetchMock)
  })

  it('sends a message via WS-fallback-to-POST and renders the reply', async () => {
    const user = userEvent.setup()
    render(<ChatInterface entered />)

    const input = await screen.findByPlaceholderText('Type a message…')
    await user.type(input, 'Hello Kurisu{Enter}')

    // The user bubble appears immediately…
    expect(await screen.findByText('Hello Kurisu')).toBeInTheDocument()

    // …the WS path was attempted and failed…
    await waitFor(() => expect(wsAttempts).toBeGreaterThan(0))

    // …and the POST fallback delivered the assistant reply.
    expect(await screen.findByText('Hmph. It works.')).toBeInTheDocument()
    const postCall = fetchMock.mock.calls.find(
      ([url, init]) => String(url).endsWith('/api/chat') && init?.method === 'POST',
    )
    expect(postCall).toBeTruthy()
    const body = JSON.parse(String(postCall![1]!.body))
    expect(body.message).toBe('Hello Kurisu')
    expect(body.defer_audio).toBe(true)

    // The adopted session id persists for the next turn.
    await waitFor(() =>
      expect(window.localStorage.getItem('session_id')).toBe('sess-smoke-1'),
    )
  })

  it('keeps the transcript row list memoized (React.memo)', () => {
    // The transcript extraction exists so unrelated ChatInterface state
    // changes (settings panel, mic state, HUD…) no longer re-walk the row
    // array on every ~16ms token flush — that only holds while
    // TranscriptList stays wrapped in React.memo. Guard against an
    // accidental unwrap; the row behavior itself is covered by the
    // regenerate/edit/rewind tests above, which all render through it.
    expect((TranscriptList as { $$typeof?: symbol }).$$typeof).toBe(Symbol.for('react.memo'))
  })

  it('renders the immersion-mode toggle when onToggleUiMode is provided and calls it on click', async () => {
    const user = userEvent.setup()
    const onToggleUiMode = vi.fn()
    render(<ChatInterface entered onToggleUiMode={onToggleUiMode} />)

    const toggle = await screen.findByTitle('Immersive mode: hide the log, keep her')
    expect(toggle).toBeInTheDocument()
    expect(toggle).toHaveAttribute('aria-pressed', 'false')

    await user.click(toggle)
    expect(onToggleUiMode).toHaveBeenCalledTimes(1)
  })

  it('does not render the immersion-mode toggle when onToggleUiMode is omitted', async () => {
    render(<ChatInterface entered />)
    await screen.findByPlaceholderText('Type a message…')
    expect(screen.queryByText('◨')).not.toBeInTheDocument()
  })

  it('shows the VN dialogue box with the latest assistant line once a turn completes', async () => {
    const user = userEvent.setup()
    render(<ChatInterface entered uiMode="vn" />)

    const root = (await screen.findByPlaceholderText('Type a message…')).closest(
      '.chat-interface',
    )
    expect(root).toHaveAttribute('data-mode', 'vn')

    // No assistant messages yet and not loading — no dialogue box at all.
    expect(document.querySelector('.vn-dialogue')).not.toBeInTheDocument()

    const input = await screen.findByPlaceholderText('Type a message…')
    await user.type(input, 'Hello Kurisu{Enter}')

    await waitFor(() => expect(wsAttempts).toBeGreaterThan(0))

    await waitFor(() => {
      const dialogue = document.querySelector('.vn-dialogue-text')
      expect(dialogue).toHaveTextContent('Hmph. It works.')
    })
  })

  it('renders no VN dialogue box when there are no assistant messages and it is not loading', async () => {
    render(<ChatInterface entered uiMode="vn" />)
    await screen.findByPlaceholderText('Type a message…')
    expect(document.querySelector('.vn-dialogue')).not.toBeInTheDocument()
  })

  it('shows the voice status indicator as idle on mount, then thinking while a turn is in flight', async () => {
    const user = userEvent.setup()

    // Hold the POST /api/chat response open so the turn stays in the
    // isLoading-but-not-streaming window long enough to observe 'thinking'.
    let resolvePost!: () => void
    const postGate = new Promise<void>((resolve) => {
      resolvePost = resolve
    })
    fetchMock.mockImplementation(async (url: RequestInfo | URL, init?: RequestInit) => {
      const path = String(url)
      if (path.endsWith('/api/chat') && init?.method === 'POST') {
        await postGate
        return jsonResponse(chatReply)
      }
      return jsonResponse({}, 404)
    })

    render(<ChatInterface entered />)

    const headerIndicator = () => document.querySelector('.chat-header .voice-status')

    // Idle before anything happens.
    await waitFor(() => expect(headerIndicator()).toHaveClass('voice-status-idle'))

    const input = await screen.findByPlaceholderText('Type a message…')
    await user.type(input, 'Hello Kurisu{Enter}')

    await waitFor(() => expect(wsAttempts).toBeGreaterThan(0))

    // The turn is loading but no stream/audio has started — 'thinking'.
    await waitFor(() => expect(headerIndicator()).toHaveClass('voice-status-thinking'))
    expect(headerIndicator()).toHaveAttribute('aria-label', 'Kurisu is thinking')

    resolvePost()

    // Once the reply lands, the indicator drops back to idle.
    await screen.findByText('Hmph. It works.')
    await waitFor(() => expect(headerIndicator()).toHaveClass('voice-status-idle'))
  })

  it('regenerates the last reply: rewind then re-send, ending with a 2/2 swipe counter', async () => {
    const user = userEvent.setup()
    let turn = 0
    fetchMock = buildTurnAwareFetchMock(() => {
      turn += 1
      return {
        reply: turn === 1 ? 'First answer.' : 'Second answer.',
        session_id: 'sess-smoke-1',
        turn_id: turn === 1 ? 't1' : 't2',
        timings: { llm_ms: 5, tts_ms: null, total_ms: 9 },
        sentences: [],
            }
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<ChatInterface entered />)
    const input = await screen.findByPlaceholderText('Type a message…')
    await user.type(input, 'Hello Kurisu{Enter}')
    await waitFor(() => expect(wsAttempts).toBeGreaterThan(0))
    await screen.findByText('First answer.')

    const regenerateButton = await screen.findByLabelText('Regenerate this reply')
    await user.click(regenerateButton)

    // Rewind POST fired with the first turn's id.
    await waitFor(() => {
      const rewindCall = fetchMock.mock.calls.find(([url]) => String(url).includes('/rewind'))
      expect(rewindCall).toBeTruthy()
      const body = JSON.parse(String(rewindCall![1]!.body))
      expect(body.turn_id).toBe('t1')
    })

    // Same prompt re-sent to /api/chat.
    await waitFor(() => {
      const chatCalls = fetchMock.mock.calls.filter(
        ([url, init]) => String(url).endsWith('/api/chat') && init?.method === 'POST',
      )
      expect(chatCalls.length).toBe(2)
      const body = JSON.parse(String(chatCalls[1][1]!.body))
      expect(body.message).toBe('Hello Kurisu')
    })

    await screen.findByText('Second answer.')
    await waitFor(() => expect(screen.getByText('2/2')).toBeInTheDocument())
  })

  it('swipes to the previous variant and amends it to the server', async () => {
    const user = userEvent.setup()
    let turn = 0
    fetchMock = buildTurnAwareFetchMock(() => {
      turn += 1
      return {
        reply: turn === 1 ? 'First answer.' : 'Second answer.',
        session_id: 'sess-smoke-1',
        turn_id: turn === 1 ? 't1' : 't2',
        timings: { llm_ms: 5, tts_ms: null, total_ms: 9 },
        sentences: [],
            }
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<ChatInterface entered />)
    const input = await screen.findByPlaceholderText('Type a message…')
    await user.type(input, 'Hello Kurisu{Enter}')
    await waitFor(() => expect(wsAttempts).toBeGreaterThan(0))
    await screen.findByText('First answer.')

    await user.click(await screen.findByLabelText('Regenerate this reply'))
    await screen.findByText('Second answer.')
    await waitFor(() => expect(screen.getByText('2/2')).toBeInTheDocument())

    const prevButton = await screen.findByLabelText('Show the previous reply')
    await user.click(prevButton)

    await screen.findByText('First answer.')
    await waitFor(() => {
      const amendCall = fetchMock.mock.calls.find(([url]) => String(url).includes('/amend'))
      expect(amendCall).toBeTruthy()
      const body = JSON.parse(String(amendCall![1]!.body))
      expect(body.content).toBe('First answer.')
    })
  })

  it('edits the last user message: rewinds, re-sends the edited text, drops the old reply', async () => {
    const user = userEvent.setup()
    let turn = 0
    fetchMock = buildTurnAwareFetchMock(() => {
      turn += 1
      return {
        reply: turn === 1 ? 'First answer.' : 'Edited answer.',
        session_id: 'sess-smoke-1',
        turn_id: turn === 1 ? 't1' : 't2',
        timings: { llm_ms: 5, tts_ms: null, total_ms: 9 },
        sentences: [],
            }
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<ChatInterface entered />)
    const input = await screen.findByPlaceholderText('Type a message…')
    await user.type(input, 'Hello Kurisu{Enter}')
    await waitFor(() => expect(wsAttempts).toBeGreaterThan(0))
    await screen.findByText('First answer.')

    const editButton = await screen.findByLabelText('Edit this message')
    await user.click(editButton)

    const textarea = await screen.findByDisplayValue('Hello Kurisu')
    await user.clear(textarea)
    await user.type(textarea, 'Edited Kurisu question{Enter}')

    await waitFor(() => {
      const rewindCall = fetchMock.mock.calls.find(([url]) => String(url).includes('/rewind'))
      expect(rewindCall).toBeTruthy()
      const body = JSON.parse(String(rewindCall![1]!.body))
      expect(body.turn_id).toBe('t1')
    })

    await waitFor(() => {
      const chatCalls = fetchMock.mock.calls.filter(
        ([url, init]) => String(url).endsWith('/api/chat') && init?.method === 'POST',
      )
      expect(chatCalls.length).toBe(2)
      const body = JSON.parse(String(chatCalls[1][1]!.body))
      expect(body.message).toBe('Edited Kurisu question')
    })

    await screen.findByText('Edited answer.')
    expect(screen.queryByText('First answer.')).not.toBeInTheDocument()
  })

  it('rewinds to an earlier message: truncates the transcript and refills the input', async () => {
    const user = userEvent.setup()
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    let turn = 0
    fetchMock = buildTurnAwareFetchMock(() => {
      turn += 1
      return {
        reply: `Reply ${turn}.`,
        session_id: 'sess-smoke-1',
        turn_id: `t${turn}`,
        timings: { llm_ms: 5, tts_ms: null, total_ms: 9 },
        sentences: [],
            }
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<ChatInterface entered />)
    const input = await screen.findByPlaceholderText('Type a message…')

    await user.type(input, 'First message{Enter}')
    await waitFor(() => expect(wsAttempts).toBeGreaterThan(0))
    await screen.findByText('Reply 1.')

    // The first turn's WS failure puts streaming into a short cooldown
    // (streamCooldownRef), so this second turn goes straight to POST
    // without another WS attempt — only the reply matters here.
    await user.type(input, 'Second message{Enter}')
    await screen.findByText('Reply 2.')

    const transcript = document.querySelector('.messages-container') as HTMLElement
    const firstUserBubble = await screen.findByText('First message')
    const rewindButton = firstUserBubble
      .closest('.message')!
      .querySelector('.rewind-button') as HTMLButtonElement
    expect(rewindButton).toBeTruthy()
    await user.click(rewindButton)

    expect(confirmSpy).toHaveBeenCalled()

    // Scope queries to the transcript container — the rewound text also
    // becomes the (out-of-scope) input value, checked separately below.
    await waitFor(() => {
      expect(within(transcript).queryByText('First message')).not.toBeInTheDocument()
    })
    expect(within(transcript).queryByText('Second message')).not.toBeInTheDocument()
    expect(within(transcript).queryByText('Reply 1.')).not.toBeInTheDocument()
    expect(within(transcript).queryByText('Reply 2.')).not.toBeInTheDocument()
    expect((input as HTMLTextAreaElement).value).toBe('First message')

    confirmSpy.mockRestore()
  })

  it('restamps the user turnId after a regenerate: Edit button reappears and rewind uses the new turn id (item 2)', async () => {
    const user = userEvent.setup()
    let turn = 0
    fetchMock = buildTurnAwareFetchMock(() => {
      turn += 1
      return {
        reply: turn === 1 ? 'First answer.' : 'Second answer.',
        session_id: 'sess-smoke-1',
        turn_id: turn === 1 ? 't1' : 't2',
        timings: { llm_ms: 5, tts_ms: null, total_ms: 9 },
        sentences: [],
            }
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<ChatInterface entered />)
    const input = await screen.findByPlaceholderText('Type a message…')
    await user.type(input, 'Hello Kurisu{Enter}')
    await waitFor(() => expect(wsAttempts).toBeGreaterThan(0))
    await screen.findByText('First answer.')

    await user.click(await screen.findByLabelText('Regenerate this reply'))
    await screen.findByText('Second answer.')

    // canEdit gates on turnId === lastTurnId — before the fix, the user
    // message kept its stale t1 stamp (stampTurn's `!msg.turnId` skip left
    // an already-stamped user message alone) and the button never
    // reappeared after the regenerate completed.
    const editButton = await screen.findByLabelText('Edit this message')
    expect(editButton).toBeInTheDocument()

    // Rewind (which also keys off the user message's turnId) must use the
    // new t2 id, not the stale t1 from the original turn.
    const userBubble = screen.getByText('Hello Kurisu')
    const rewindButton = userBubble.closest('.message')!.querySelector('.rewind-button') as HTMLButtonElement
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    await user.click(rewindButton)

    await waitFor(() => {
      const rewindCalls = fetchMock.mock.calls.filter(([url]) => String(url).includes('/rewind'))
      // The regenerate's own rewind (t1) already fired; the row-level
      // rewind button's call is the most recent one.
      const lastRewind = rewindCalls[rewindCalls.length - 1]
      expect(JSON.parse(String(lastRewind![1]!.body)).turn_id).toBe('t2')
    })
    confirmSpy.mockRestore()
  })

  it('does not fire two rewinds on a double-click regenerate (item 3)', async () => {
    const user = userEvent.setup()
    let turn = 0
    let resolveRewind!: () => void
    const rewindGate = new Promise<void>((resolve) => {
      resolveRewind = resolve
    })
    fetchMock = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
      const path = String(url)
      if (path.endsWith('/api/chat') && init?.method === 'POST') {
        turn += 1
        return jsonResponse({
          reply: turn === 1 ? 'First answer.' : 'Second answer.',
          session_id: 'sess-smoke-1',
          turn_id: turn === 1 ? 't1' : 't2',
          timings: { llm_ms: 5, tts_ms: null, total_ms: 9 },
          sentences: [],
                })
      }
      if (path.includes('/rewind') && init?.method === 'POST') {
        // Held open so both clicks of a double-click land before either
        // resolves — a real re-entrancy race, not just a fast synchronous
        // guard check.
        await rewindGate
        return jsonResponse({ session_id: 'sess-smoke-1', status: 'ok' })
      }
      return jsonResponse({}, 404)
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<ChatInterface entered />)
    const input = await screen.findByPlaceholderText('Type a message…')
    await user.type(input, 'Hello Kurisu{Enter}')
    await waitFor(() => expect(wsAttempts).toBeGreaterThan(0))
    await screen.findByText('First answer.')

    const regenerateButton = await screen.findByLabelText('Regenerate this reply')
    await user.click(regenerateButton)
    // Second click while the first rewind is still in flight — the button
    // is also expected to be disabled (busy) by then, but click it directly
    // to prove the guard itself (not just the disabled attribute) holds.
    await user.click(regenerateButton)

    resolveRewind()
    await screen.findByText('Second answer.')

    const rewindCalls = fetchMock.mock.calls.filter(([url]) => String(url).includes('/rewind'))
    expect(rewindCalls.length).toBe(1)
    const chatCalls = fetchMock.mock.calls.filter(
      ([url, init]) => String(url).endsWith('/api/chat') && init?.method === 'POST',
    )
    expect(chatCalls.length).toBe(2)
  })

  it('restores the previous reply on a terminal (non-connect) regenerate failure instead of destroying it (item 1)', async () => {
    const user = userEvent.setup()
    fetchMock = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
      const path = String(url)
      if (path.endsWith('/api/chat') && init?.method === 'POST') {
        return jsonResponse({
          reply: 'First answer.',
          session_id: 'sess-smoke-1',
          turn_id: 't1',
          timings: { llm_ms: 5, tts_ms: null, total_ms: 9 },
          sentences: [],
                })
      }
      if (path.includes('/rewind') && init?.method === 'POST') {
        return jsonResponse({ session_id: 'sess-smoke-1', status: 'ok' })
      }
      return jsonResponse({}, 404)
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<ChatInterface entered />)
    const input = await screen.findByPlaceholderText('Type a message…')
    await user.type(input, 'Hello Kurisu{Enter}')
    await waitFor(() => expect(wsAttempts).toBeGreaterThan(0))
    await screen.findByText('First answer.')

    // Now make the regenerate's re-send fail outright (network error) —
    // the POST catch path must restore "First answer." before appending
    // the error bubble, not silently drop the user's last good reply.
    fetchMock.mockImplementation(async (url: RequestInfo | URL, init?: RequestInit) => {
      const path = String(url)
      if (path.endsWith('/api/chat') && init?.method === 'POST') {
        throw new Error('network down')
      }
      if (path.includes('/rewind') && init?.method === 'POST') {
        return jsonResponse({ session_id: 'sess-smoke-1', status: 'ok' })
      }
      return jsonResponse({}, 404)
    })

    await user.click(await screen.findByLabelText('Regenerate this reply'))

    // The restored reply reappears (its content survived the destroy-then-
    // fail cycle) alongside the new error bubble — neither replaces the
    // other.
    await screen.findByText('First answer.')
    expect(await screen.findByText(/Connection failed/)).toBeInTheDocument()
  })

  it('an error bubble does not hide the previous reply\'s swipe/regenerate/edit actions (item 6)', async () => {
    const user = userEvent.setup()
    fetchMock = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
      const path = String(url)
      if (path.endsWith('/api/chat') && init?.method === 'POST') {
        return jsonResponse({
          reply: 'First answer.',
          session_id: 'sess-smoke-1',
          turn_id: 't1',
          timings: { llm_ms: 5, tts_ms: null, total_ms: 9 },
          sentences: [],
                })
      }
      return jsonResponse({}, 404)
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<ChatInterface entered />)
    const input = await screen.findByPlaceholderText('Type a message…')
    await user.type(input, 'Hello Kurisu{Enter}')
    await waitFor(() => expect(wsAttempts).toBeGreaterThan(0))
    await screen.findByText('First answer.')

    // A second turn whose POST fails outright — produces a trailing error
    // bubble after the real reply.
    fetchMock.mockImplementation(async (url: RequestInfo | URL, init?: RequestInit) => {
      const path = String(url)
      if (path.endsWith('/api/chat') && init?.method === 'POST') {
        throw new Error('network down')
      }
      return jsonResponse({}, 404)
    })
    await user.type(input, 'Second message{Enter}')
    await screen.findByText(/Connection failed/)

    // The real reply's regenerate button (and the first user message's
    // Edit button) must still be present — lastAssistantIndex/lastTurnId
    // must skip the isError bubble when computing "the last real turn".
    expect(await screen.findByLabelText('Regenerate this reply')).toBeInTheDocument()
    expect(await screen.findByLabelText('Edit this message')).toBeInTheDocument()
  })

  it('caps folded variants at 5, dropping the oldest (item 9)', async () => {
    const user = userEvent.setup()
    let turn = 0
    fetchMock = buildTurnAwareFetchMock(() => {
      turn += 1
      return {
        reply: `Answer ${turn}.`,
        session_id: 'sess-smoke-1',
        turn_id: `t${turn}`,
        timings: { llm_ms: 5, tts_ms: null, total_ms: 9 },
        sentences: [],
            }
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<ChatInterface entered />)
    const input = await screen.findByPlaceholderText('Type a message…')
    await user.type(input, 'Hello Kurisu{Enter}')
    await waitFor(() => expect(wsAttempts).toBeGreaterThan(0))
    await screen.findByText('Answer 1.')

    // Regenerate 6 times: 7 total variants (Answer 1..7) would be stashed
    // without a cap; MAX_VARIANTS=5 keeps only the newest 5.
    for (let i = 0; i < 6; i += 1) {
      const regenerateButton = await screen.findByLabelText('Regenerate this reply')
      await user.click(regenerateButton)
      await screen.findByText(`Answer ${i + 2}.`)
    }

    await waitFor(() => expect(screen.getByText('5/5')).toBeInTheDocument())

    // Swiping back 4 times (to the oldest kept variant) reaches Answer 3 —
    // Answer 1 and Answer 2 were dropped from the cap.
    for (let i = 0; i < 4; i += 1) {
      await user.click(await screen.findByLabelText('Show the previous reply'))
    }
    await screen.findByText('Answer 3.')
    expect(screen.queryByText('Answer 1.')).not.toBeInTheDocument()
    expect(screen.queryByText('Answer 2.')).not.toBeInTheDocument()
    // Can't go back any further — the oldest kept variant is 1/5.
    expect(await screen.findByLabelText('Show the previous reply')).toBeDisabled()
  })

  describe('milestone celebration (M2)', () => {
    afterEach(() => {
      vi.useRealTimers()
    })

    it('shows the milestone banner and plays the mapped clip once, not again on repeat', async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true })
      const play = vi.fn()
      kurisuBridge.vrmAnimations = {
        clips: {},
        mixer: null,
        play,
        list: vi.fn(),
        describe: vi.fn(() => []),
        current: vi.fn(() => null),
      }

      fetchMock = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
        const path = String(url)
        if (path.endsWith('/api/greeting') && init?.method === 'POST') {
          return jsonResponse({
            session_id: 'sess-smoke-1',
            greeting: null,
            milestones: [{ kind: 'sessions', value: 10, label: '10 sessions together!' }],
          })
        }
        return jsonResponse({}, 404)
      })
      vi.stubGlobal('fetch', fetchMock)

      render(<ChatInterface entered />)

      await screen.findByPlaceholderText('Type a message…')
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0)
      })

      expect(await screen.findByText('10 sessions together!')).toBeInTheDocument()
      expect(screen.getByText('MILESTONE')).toBeInTheDocument()

      // The clip only plays after the 1.5s delay, not immediately.
      expect(play).not.toHaveBeenCalled()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1500)
      })
      expect(play).toHaveBeenCalledWith('Joy')
      expect(play).toHaveBeenCalledTimes(1)

      // Auto-dismiss after 12s.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(12000)
      })
      expect(screen.queryByText('10 sessions together!')).not.toBeInTheDocument()

      kurisuBridge.vrmAnimations = null
    })

    it('does not re-celebrate an already-seen milestone', async () => {
      // The seen-milestones latch is namespaced by user id (item 9) so a
      // relationship reset — which clears it — can't leak into another
      // identity's history; seed the same namespaced key resolveUserId()
      // (and the component under test) will actually read.
      const userId = resolveUserId()
      window.localStorage.setItem(
        `kurisu_milestones_seen_v1:${userId}`,
        JSON.stringify(['sessions:10']),
      )
      const play = vi.fn()
      kurisuBridge.vrmAnimations = {
        clips: {},
        mixer: null,
        play,
        list: vi.fn(),
        describe: vi.fn(() => []),
        current: vi.fn(() => null),
      }

      fetchMock = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
        const path = String(url)
        if (path.endsWith('/api/greeting') && init?.method === 'POST') {
          return jsonResponse({
            session_id: 'sess-smoke-1',
            greeting: null,
            milestones: [{ kind: 'sessions', value: 10, label: '10 sessions together!' }],
          })
        }
        return jsonResponse({}, 404)
      })
      vi.stubGlobal('fetch', fetchMock)

      render(<ChatInterface entered />)
      await screen.findByPlaceholderText('Type a message…')

      // Give the best-effort greeting fetch a tick to resolve.
      await waitFor(() => {
        const greetingCall = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/api/greeting'))
        expect(greetingCall).toBeTruthy()
      })
      expect(screen.queryByText('10 sessions together!')).not.toBeInTheDocument()
      expect(play).not.toHaveBeenCalled()
    })
  })

  describe('empty-state chips persist through a greeting (A2)', () => {
    it('keeps the starter chips until the first USER message, even after an assistant greeting bubble arrives', async () => {
      // A greeting bubble mounts on entry (usePresence -> /api/greeting).
      // The warm starter chips must survive it and only vanish once the user
      // themselves speaks.
      fetchMock = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
        const path = String(url)
        if (path.endsWith('/api/greeting') && init?.method === 'POST') {
          return jsonResponse({
            session_id: 'sess-smoke-1',
            greeting: 'You came back. ...not that I was counting.',
          })
        }
        if (path.endsWith('/api/chat') && init?.method === 'POST') {
          return jsonResponse(chatReply)
        }
        return jsonResponse({}, 404)
      })
      vi.stubGlobal('fetch', fetchMock)

      const user = userEvent.setup()
      render(<ChatInterface entered />)

      // Greeting bubble lands...
      expect(
        await screen.findByText('You came back. ...not that I was counting.'),
      ).toBeInTheDocument()

      // ...and the starter chips are still on screen (A2: gate is "no user
      // message yet", not "no messages at all").
      expect(await screen.findByText('What are you working on?')).toBeInTheDocument()

      // Once the user sends, the chips go away.
      const input = await screen.findByPlaceholderText('Type a message…')
      await user.type(input, 'Hello Kurisu{Enter}')
      await screen.findByText('Hmph. It works.')
      await waitFor(() =>
        expect(screen.queryByText('What are you working on?')).not.toBeInTheDocument(),
      )

      kurisuBridge.vrmAnimations = null
    })
  })

  describe('empty transcript hero (EmptyState)', () => {
    it('shows the warm empty-state with zero messages and hides it once a message exists', async () => {
      const user = userEvent.setup()
      render(<ChatInterface entered />)

      // No greeting configured in this suite's default fetch stub, so the
      // transcript starts genuinely empty and the EmptyState hero renders.
      expect(
        await screen.findByText(
          "Connection established. She'll pretend she wasn't waiting — say something anyway.",
        ),
      ).toBeInTheDocument()

      const input = await screen.findByPlaceholderText('Type a message…')
      await user.type(input, 'Hello Kurisu{Enter}')
      await screen.findByText('Hmph. It works.')

      expect(
        screen.queryByText(
          "Connection established. She'll pretend she wasn't waiting — say something anyway.",
        ),
      ).not.toBeInTheDocument()
    })
  })

  describe('conversational onboarding chips (M6)', () => {
    it('shows a synapse chip once a new profile fact appears after a turn', async () => {
      let overviewCalls = 0
      // The POST /api/chat reply is held open until the test has observed
      // isLoading=true via the DOM (voice status "thinking") — resolving
      // it immediately would let React 18 batch the whole
      // isLoading(true)->(false) cycle into a single act() flush, so the
      // intro-mode isLoading-transition effect would never see a
      // true->false edge (same technique the "thinking" indicator test
      // above uses).
      let resolvePost!: () => void
      const postGate = new Promise<void>((resolve) => {
        resolvePost = resolve
      })
      fetchMock = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
        const path = String(url)
        if (path.endsWith('/api/settings')) {
          return jsonResponse({ providers: { llm_is_dummy: false }, is_first_run: false })
        }
        if (path.endsWith('/api/memory') && init?.method === 'POST') {
          overviewCalls += 1
          // Mount fires exactly one shared overview fetch — the gauge and
          // intro-mode activation both consume it (item 10d) rather than
          // each firing their own — before any turn: a fresh, empty
          // profile. Only the post-turn refetch (after the 4s intro-mode
          // delay) should see the new fact.
          if (overviewCalls <= 1) {
            return jsonResponse({
              status: 'ok',
              overview: {
                profile: { facts: {}, open_threads: [], session_count: 1 },
                facts: [],
                episodes: [],
              },
            })
          }
          return jsonResponse({
            status: 'ok',
            overview: {
              profile: { facts: { name: 'Alex' }, open_threads: [], session_count: 1 },
              facts: [],
              episodes: [],
            },
          })
        }
        if (path.endsWith('/api/chat') && init?.method === 'POST') {
          await postGate
          return jsonResponse(chatReply)
        }
        return jsonResponse({}, 404)
      })
      vi.stubGlobal('fetch', fetchMock)

      const user = userEvent.setup()
      render(<ChatInterface entered />)

      const input = await screen.findByPlaceholderText('Type a message…')
      // Let the mount-time shared overview fetch (gauge + intro-activation)
      // resolve before sending, so intro mode is confirmed active first.
      await waitFor(() => expect(overviewCalls).toBeGreaterThanOrEqual(1))
      await user.type(input, 'Hello Kurisu{Enter}')
      await waitFor(() => expect(wsAttempts).toBeGreaterThan(0))

      const headerIndicator = () => document.querySelector('.chat-header .voice-status')
      await waitFor(() => expect(headerIndicator()).toHaveClass('voice-status-thinking'))

      resolvePost()
      await screen.findByText('Hmph. It works.')

      await waitFor(() => expect(overviewCalls).toBeGreaterThanOrEqual(2), { timeout: 6000 })

      expect(await screen.findByText('SYNAPSE FORMED · name', {}, { timeout: 6000 })).toBeInTheDocument()

      // Auto-removed after 8s.
      await waitFor(
        () => expect(screen.queryByText('SYNAPSE FORMED · name')).not.toBeInTheDocument(),
        { timeout: 10000 },
      )
    }, 20000)
  })

  describe('streaming liveness caret (A3)', () => {
    const noop = () => {}
    const baseTranscriptProps = {
      groupStarts: [true, true],
      lastTurnId: 'turn-1',
      busy: false,
      restoredCount: 0,
      assistantName: 'Kurisu',
      editing: null,
      onEditTextChange: noop,
      onEditSave: noop,
      onEditCancel: noop,
      onRetry: noop,
      onRegenerate: noop,
      onSwipe: noop,
      onEdit: noop,
      onRewind: noop,
    }
    const messages = [
      { role: 'user' as const, content: 'Hi', timestamp: 1, id: 'm-user' },
      {
        role: 'assistant' as const,
        content: 'Streaming repl',
        timestamp: 2,
        id: 'm-asst',
        turnId: 'turn-1',
      },
    ]

    it('renders the caret on the last assistant row while streaming', () => {
      render(
        <TranscriptList
          {...baseTranscriptProps}
          messages={messages}
          lastAssistantIndex={1}
          isStreaming
        />,
      )
      expect(document.querySelector('.streaming-caret')).toBeInTheDocument()
    })

    it('omits the caret once the stream ends', () => {
      render(
        <TranscriptList
          {...baseTranscriptProps}
          messages={messages}
          lastAssistantIndex={1}
          isStreaming={false}
        />,
      )
      expect(document.querySelector('.streaming-caret')).not.toBeInTheDocument()
    })
  })
})
