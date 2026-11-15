// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom/vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { ComponentType } from 'react'
import PetView from './PetView'

/**
 * Smoke coverage for the desktop-pet overlay. PetView is voice-first with no
 * transcript, so its "core wiring" is: the mic button drives usePetChat's
 * voice pipeline (wake-on-press, start/stop recording), the typed composer
 * rides the exact same send path (sendText), and the caption area renders
 * usePetChat's lastReply/lastSent/hint state. usePetChat itself owns every
 * hard-won race fix (VRM-load, doze, lip sync) and is covered by its own
 * unit/hook tests — this file only asserts PetView wires its return values
 * to the right DOM, not usePetChat's internals.
 *
 * VRMViewer is lazy-loaded (`lazy(() => import('./VRMViewer'))`) and pulls
 * in the full three.js/VRM stack, so it's mocked the same way a heavy child
 * is mocked in ChatInterface's neighboring tests — via vi.mock, not a real
 * render.
 */

const sendTextMock = vi.fn()
const stopCurrentStreamMock = vi.fn()
const toggleScreenCompanionMock = vi.fn()
const toggleReadTogetherMock = vi.fn()
const toggleWatchTogetherMock = vi.fn()
const firePokeMock = vi.fn(async () => undefined)
const wakeOnMicPressMock = vi.fn()
const startRecordingMock = vi.fn(async () => undefined)
const stopRecordingAndTranscribeMock = vi.fn(async () => undefined)
const cancelRecordingMock = vi.fn()
const toggleHandsFreeMock = vi.fn()
const focusStartMock = vi.fn(async () => undefined)
const focusEndMock = vi.fn(async () => undefined)

/** Mutable base return value for the mocked usePetChat — individual tests
 * override fields via `usePetChatMock.mockReturnValue({ ...basePetChat, ... })`. */
const basePetChat = {
  isBusy: false,
  lastReply: null as string | null,
  lastSent: null as { text: string } | null,
  lastEmotion: null as string | null,
  stopCurrentStream: stopCurrentStreamMock,
  sendText: sendTextMock,
  mic: {
    micStatus: 'idle' as 'idle' | 'recording' | 'transcribing',
    micAvailable: true as boolean | null,
    micPermissionDenied: false,
    micUnavailableReason: null as string | null,
    micHint: null as string | null,
    showMicHint: vi.fn(),
    startRecording: startRecordingMock,
    stopRecordingAndTranscribe: stopRecordingAndTranscribeMock,
    cancelRecording: cancelRecordingMock,
  },
  handsFree: {
    handsFree: false,
    toggleHandsFree: toggleHandsFreeMock,
    handsFreeStatus: 'idle' as 'idle' | 'listening',
    autoBargeIn: false,
    toggleAutoBargeIn: vi.fn(),
  },
  screenCompanion: false,
  toggleScreenCompanion: toggleScreenCompanionMock,
  lastPeekAt: null as number | null,
  readTogether: false,
  toggleReadTogether: toggleReadTogetherMock,
  lastPageAt: null as number | null,
  watchTogether: false,
  toggleWatchTogether: toggleWatchTogetherMock,
  lastWatchFrameAt: null as number | null,
  firePoke: firePokeMock,
  wakeOnMicPress: wakeOnMicPressMock,
  courtesyActive: false,
  voiceWarmingHint: null as string | null,
  focus: {
    active: false,
    until: null as string | null,
    remainingSeconds: null as number | null,
    start: focusStartMock,
    end: focusEndMock,
  },
}

const usePetChatMock = vi.fn(() => basePetChat)

vi.mock('../hooks/usePetChat', () => ({
  usePetChat: () => usePetChatMock(),
}))

// VRMViewer is a lazy chunk pulling in three.js/VRM — mock it to a plain
// marker div so Suspense resolves synchronously-enough for RTL's
// findBy queries without ever touching WebGL/GLTF loading.
vi.mock('./VRMViewer', () => ({
  default: (() => <div data-testid="vrm-viewer-stub" />) as ComponentType,
}))

describe('PetView', () => {
  afterEach(() => {
    cleanup()
    usePetChatMock.mockReset()
    usePetChatMock.mockReturnValue(basePetChat)
    vi.clearAllMocks()
  })

  beforeEach(() => {
    window.localStorage.clear()
    usePetChatMock.mockReturnValue(basePetChat)
  })

  it('renders the pet overlay with the avatar stage, mic button, and composer', async () => {
    render(<PetView />)

    expect(await screen.findByTestId('vrm-viewer-stub')).toBeInTheDocument()
    expect(screen.getByTitle('Hold to talk')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('Type to her…')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Send message' })).toBeDisabled()
  })

  it('voice-first path: pressing the mic wakes doze and starts recording, releasing stops and transcribes', async () => {
    render(<PetView />)

    const micButton = screen.getByTitle('Hold to talk')
    // pointerdown: wake-on-press fires before recording starts (usePetChat's
    // documented ordering — waking is the point of the interaction).
    const downEvent = new Event('pointerdown', { bubbles: true, cancelable: true })
    micButton.dispatchEvent(downEvent)
    expect(wakeOnMicPressMock).toHaveBeenCalledTimes(1)
    expect(startRecordingMock).toHaveBeenCalledTimes(1)

    const upEvent = new Event('pointerup', { bubbles: true, cancelable: true })
    micButton.dispatchEvent(upEvent)
    expect(stopRecordingAndTranscribeMock).toHaveBeenCalledTimes(1)
  })

  it('mic button is disabled when the mic is unavailable, with the unavailable reason as the tooltip', () => {
    usePetChatMock.mockReturnValue({
      ...basePetChat,
      mic: {
        ...basePetChat.mic,
        micAvailable: false,
        micUnavailableReason: 'Speech-to-text not installed',
      },
    })
    render(<PetView />)

    const micButton = screen.getByTitle('Speech-to-text not installed')
    expect(micButton).toBeDisabled()
  })

  it('typed composer sends through the same sendText pipeline as voice, then clears the draft', async () => {
    const user = userEvent.setup()
    render(<PetView />)

    const input = screen.getByPlaceholderText('Type to her…')
    await user.type(input, 'Hello Kurisu')
    const sendButton = screen.getByRole('button', { name: 'Send message' })
    expect(sendButton).toBeEnabled()

    await user.click(sendButton)

    expect(sendTextMock).toHaveBeenCalledWith('Hello Kurisu')
    expect(input).toHaveValue('')
  })

  it('typed composer also sends on Enter (without Shift) and never on Shift+Enter', async () => {
    const user = userEvent.setup()
    render(<PetView />)

    const input = screen.getByPlaceholderText('Type to her…')
    await user.type(input, 'Line one')
    await user.keyboard('{Shift>}{Enter}{/Shift}')
    expect(sendTextMock).not.toHaveBeenCalled()

    await user.type(input, 'Line two{Enter}')
    expect(sendTextMock).toHaveBeenCalledWith('Line one\nLine two')
  })

  it('does not submit an empty or whitespace-only draft', async () => {
    const user = userEvent.setup()
    render(<PetView />)

    const sendButton = screen.getByRole('button', { name: 'Send message' })
    expect(sendButton).toBeDisabled()

    const input = screen.getByPlaceholderText('Type to her…')
    await user.type(input, '   ')
    expect(screen.getByRole('button', { name: 'Send message' })).toBeDisabled()
    expect(sendTextMock).not.toHaveBeenCalled()
  })

  it('renders her latest reply as a caption', async () => {
    usePetChatMock.mockReturnValue({ ...basePetChat, lastReply: 'Hmph. Fine, I helped.' })
    render(<PetView />)

    expect(await screen.findByText('Hmph. Fine, I helped.')).toBeInTheDocument()
  })

  it('renders the voice-honesty hint when present', async () => {
    usePetChatMock.mockReturnValue({
      ...basePetChat,
      voiceWarmingHint: 'Her voice is still warming up…',
    })
    render(<PetView />)

    expect(await screen.findByText('Her voice is still warming up…')).toBeInTheDocument()
  })

  it('renders the mic hint when present', async () => {
    usePetChatMock.mockReturnValue({
      ...basePetChat,
      mic: { ...basePetChat.mic, micHint: 'Still listening…' },
    })
    render(<PetView />)

    expect(await screen.findByText('Still listening…')).toBeInTheDocument()
  })

  it('shows the recording state on the mic button while active', () => {
    usePetChatMock.mockReturnValue({
      ...basePetChat,
      mic: { ...basePetChat.mic, micStatus: 'recording' },
    })
    render(<PetView />)

    const micButton = screen.getByTitle('Hold to talk')
    expect(micButton).toHaveAttribute('aria-pressed', 'true')
    expect(micButton).toHaveClass('recording')
  })

  it('hands-free toggle reflects active state and calls through to usePetChat', async () => {
    const user = userEvent.setup()
    render(<PetView />)

    const toggle = screen.getByRole('button', { name: 'Hands-free' })
    expect(toggle).toHaveAttribute('aria-pressed', 'false')

    await user.click(toggle)
    expect(toggleHandsFreeMock).toHaveBeenCalledTimes(1)
  })

  it('shows a LISTENING status pill while hands-free is actively listening', () => {
    usePetChatMock.mockReturnValue({
      ...basePetChat,
      handsFree: { ...basePetChat.handsFree, handsFreeStatus: 'listening' },
    })
    render(<PetView />)

    expect(screen.getByText('LISTENING')).toBeInTheDocument()
  })
})
