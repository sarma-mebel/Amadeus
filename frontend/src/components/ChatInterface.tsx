import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { apiFetch, AUTH_ERROR_EVENT, loadApiToken } from '../api'
import { resolveApiPath } from '../config'
import { t, useLocale } from '../i18n'
import { isTauri } from '../platform'
import TranscriptList from './TranscriptList'
import { loadAudioPreference, saveAudioEnabled } from '../audioPrefs'
import { presenceRequest } from '../presenceApi'
import SettingsPanel from './SettingsPanel'
import SessionSidebar from './SessionSidebar'
import { fetchSessionTranscript } from '../sessions'
import MemoryPanel from './MemoryPanel'
import EmptyState from './EmptyState'
import { loadStoredTranscript } from '../transcript'
import { useFullscreen } from '../fullscreen'
import './ChatInterface.css'
import { dismissUpdate, openReleasesPage } from '../updateCheck'
import StatusChip from './StatusChip'
import { deriveLlmHealthBanner, type LlmHealthBannerState } from '../statusApi'
import { kurisuBridge } from '../vrm/kurisuBridge'
import type { Message } from '../chat/messages'
import { resolveUserId, useSessions } from '../hooks/useSessions'
import { useBackendStatus } from '../hooks/useBackendStatus'
import { useCallMode } from '../hooks/useCallMode'
import { useTranscriptView } from '../hooks/useTranscriptView'
import { useOnboardingGate } from '../hooks/useOnboardingGate'
import { useChatTurn } from '../hooks/useChatTurn'
import { useTurnActions } from '../hooks/useTurnActions'
import { useMic } from '../hooks/useMic'
import { useHandsFree } from '../hooks/useHandsFree'
import { useDuplexCall, type DuplexFallbackReason } from '../hooks/useDuplexCall'
import { useWakeWord } from '../hooks/useWakeWord'
import { loadDuplexEnabled } from '../chat/duplexTurn'
import { usePresence } from '../hooks/usePresence'
import { useVision } from '../hooks/useVision'
import VoiceStatusIndicator, { type VoiceState } from './VoiceStatusIndicator'
import RelationshipGauge from './RelationshipGauge'
import { getTimeOfDayBucket, type TimeOfDayBucket } from '../chat/uiHelpers'

// Tier 4b: in-character starter prompts shown in the empty state instead of
// a bare placeholder line. Clicking one fills + focuses the input rather
// than sending immediately, so the user can still edit before committing.
const EMPTY_STATE_SUGGESTIONS = [
  'What are you working on?',
  "Explain time dilation like I'm Okabe",
  'How does the Amadeus system work?',
]

interface ChatInterfaceProps {
  /** False while the Amadeus connection screen still overlays the app.
   * Gates the greeting so her voiced opener plays after the tap that
   * unlocks the AudioContext, not silently behind the loading page. */
  entered?: boolean
  /** Immersion ("VN") mode: 'vn' hides the full transcript in favor of a
   * bottom dialogue overlay showing only her latest line. Defaults to
   * 'full' so existing callers (tests, PetView-free usage) are unaffected. */
  uiMode?: 'full' | 'vn'
  /** Present only when the parent renders the toggle button (App.tsx);
   * omitted entirely hides the button rather than disabling it. */
  onToggleUiMode?: () => void
}

const ChatInterface = ({ entered = true, uiMode = 'full', onToggleUiMode }: ChatInterfaceProps) => {
  useLocale()
  const detectTouchDevice = () => {
    if (typeof window === 'undefined') return false
    const navigatorInfo = typeof navigator !== 'undefined' ? navigator : undefined
    return (
      'ontouchstart' in window ||
      Boolean(navigatorInfo && ('maxTouchPoints' in navigatorInfo && navigatorInfo.maxTouchPoints > 0)) ||
      Boolean((navigatorInfo as unknown as { msMaxTouchPoints?: number })?.msMaxTouchPoints)
    )
  }

  const isTouchDevice = useMemo(detectTouchDevice, [])
  // ensureAudioContext() in VRMViewer silently no-ops without this; disable
  // (not hide) the toggle so it's clear audio isn't available rather than
  // looking like a normal mute.
  const audioSupported =
    typeof window !== 'undefined' &&
    Boolean(window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext)

  // Session continuity: restore the recent transcript instead of a cold,
  // empty chat on every reload. restoredCountRef marks where history ends
  // and the live session begins, so a divider can separate them.
  const [messages, setMessages] = useState<Message[]>(() => loadStoredTranscript())
  const restoredCountRef = useRef(messages.length)
  const [input, setInput] = useState('')
  // loadAudioPreference (src/audioPrefs.ts) returns the RAW stored value —
  // the null case ("never set") matters here: touch devices default the
  // toggle off and show the audio-unlock affordance only until the user
  // has chosen either way.
  const [audioEnabled, setAudioEnabled] = useState(() => {
    if (typeof window === 'undefined') return true
    const saved = loadAudioPreference()
    if (saved !== null) return saved === 'true'
    return detectTouchDevice() ? false : true
  })
  const [requiresAudioUnlock, setRequiresAudioUnlock] = useState(() => {
    if (typeof window === 'undefined') return false
    const saved = loadAudioPreference()
    if (saved !== null) return false
    return detectTouchDevice()
  })
  const [memoryOpen, setMemoryOpen] = useState(false)
  const [isFullscreen, toggleFullscreen] = useFullscreen()
  // Backend status chip, /healthz voice-warmup poll, the update toast, and
  // the character-switch/auto-avatar wiring all live in useBackendStatus —
  // see src/hooks/useBackendStatus.ts. startNewConversationRef is its
  // bridge to useSessions' startNewConversation (assigned below).
  const {
    backendStatus,
    voiceWarming,
    updateAvailable,
    setUpdateAvailable,
    refreshBackendStatus,
    startNewConversationRef,
  } = useBackendStatus()

  // Conversation sidebar/session CRUD state lives in the useSessions hook
  // (support probe, list refresh, new/switch/delete/export) — see
  // src/hooks/useSessions.ts.
  const {
    sessionsSupported,
    sidebarOpen,
    setSidebarOpen,
    sessions,
    sessionsLoading,
    sessionsError,
    currentSessionId,
    adoptSessionId,
    startNewConversation,
    switchToSession,
    removeSession,
    exportSession,
    toggleSidebar,
  } = useSessions({
    restoredCountRef,
    setMessages,
    // Lazy: cancelDeferredTts comes from useChatTurn, wired further down
    // (it needs adoptSessionId from this hook) — only invoked in handlers.
    cancelDeferredTts: () => cancelDeferredTts(),
  })
  startNewConversationRef.current = startNewConversation
  // Transcript text for the sidebar's content search (the sidebar caches
  // per session, so this fetches each transcript at most once per update).
  const fetchTranscriptTextForSearch = useCallback(async (sessionId: string) => {
    const transcript = await fetchSessionTranscript(sessionId, resolveUserId())
    return transcript.map((message) => message.content).join('\n')
  }, [])
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Vision (screenshot attachment): status probe, pending-image chip, and
  // capture/paste/drop handlers all live in useVision — see
  // src/hooks/useVision.ts.
  const {
    pendingImage,
    setPendingImage,
    pendingImageRef,
    turnImageRef,
    visionAvailable,
    handleCaptureScreen,
    handlePaste,
    handleDrop,
  } = useVision()
  // Autoscroll only while the user is at (or near) the bottom, so scrolling
  // up to reread history isn't fought by incoming stream tokens.
  const pinnedToBottomRef = useRef(true)

  /** Handles /remember, /forget and /memory locally via the memory API.
   * Returns true when the input was a memory command (already handled). */
  const handleMemoryCommand = async (trimmed: string): Promise<boolean> => {
    let action: 'remember' | 'forget' | 'overview' | null = null
    let text = ''
    if (trimmed === '/memory') {
      action = 'overview'
    } else if (trimmed.startsWith('/remember ')) {
      action = 'remember'
      text = trimmed.slice('/remember '.length).trim()
    } else if (trimmed.startsWith('/forget ')) {
      action = 'forget'
      text = trimmed.slice('/forget '.length).trim()
    }
    if (!action) return false

    const appendSystem = (content: string) =>
      setMessages((prev) => [...prev, { role: 'system', content, timestamp: Date.now() }])

    try {
      const response = await apiFetch('/api/memory', presenceRequest({ action, text }))
      const data = await response.json()
      if (action === 'overview') {
        const profile = data.overview?.profile ?? {}
        const facts: Record<string, string> = profile.facts ?? {}
        const stored: string[] = data.overview?.facts ?? []
        const lines: string[] = ['What Kurisu remembers:']
        Object.entries(facts).forEach(([key, value]) => lines.push(`• ${key}: ${value}`))
        stored.forEach((fact) => lines.push(`• ${fact}`))
        if (profile.last_episode) lines.push(`Last time: ${profile.last_episode}`)
        if (profile.session_count) {
          lines.push(
            `Sessions: ${profile.session_count}${profile.first_met ? ` since ${profile.first_met}` : ''}`,
          )
        }
        if (lines.length === 1) lines.push('Nothing yet — try /remember <something>.')
        appendSystem(lines.join('\n'))
      } else {
        appendSystem(data.detail ?? data.status ?? 'Done.')
      }
    } catch {
      appendSystem('Memory command failed — is the server reachable?')
    }
    return true
  }

  // The first-run/onboarding gate: llmIsDummy (+ ref), the settings-modal
  // open/onboarding flags, the shared /api/settings probe, the mount-time
  // first-run check, the post-onboarding regreet, and the send path's
  // ensureRealLlmConfigured all live in useOnboardingGate — see
  // src/hooks/useOnboardingGate.ts. tryRequestGreeting closes over the
  // usePresence binding below lazily (only invoked from close handlers).
  const {
    llmIsDummy,
    llmIsDummyRef,
    settingsOpen,
    setSettingsOpen,
    settingsIsOnboarding,
    setSettingsIsOnboarding,
    probeSettings,
    maybeRegreetAfterOnboarding,
    waitForFirstRunProbe,
    ensureRealLlmConfigured,
  } = useOnboardingGate({
    messages,
    setMessages,
    tryRequestGreeting: () => tryRequestGreeting(),
  })

  // LLM-health banner: generalizes the old dummy-only banner to every
  // amadeus/provider_health.py state (dummy/invalid_key/rate_limited/
  // error), sourced from backendStatus.llm_health (falls back to
  // llmIsDummy on older backends — see deriveLlmHealthBanner). Dismissible
  // like before, but a dismissal only suppresses the CURRENT state: a new
  // failure state after a dismiss (e.g. rate_limited -> invalid_key, or
  // 'ok' clearing and later re-degrading) must resurface, so the dismissed
  // value tracks a state key rather than a plain boolean.
  const llmHealthBannerState = deriveLlmHealthBanner(backendStatus?.llm_health, llmIsDummy)
  const [dismissedBannerState, setDismissedBannerState] = useState<LlmHealthBannerState>(null)
  const llmHealthBanner =
    llmHealthBannerState && llmHealthBannerState !== dismissedBannerState
      ? llmHealthBannerState
      : null

  // The send path — sendMessage (stream-first with POST fallback), the
  // cancel machinery (stopCurrentStream/cancelDeferredTts, turn epoch,
  // stream cooldown), and isLoading/isStreaming plus their live ref
  // mirrors all live in useChatTurn — see src/hooks/useChatTurn.ts.
  // requestFarewell/showMicHint and the useTurnActions bookkeeping close
  // over bindings initialized further down lazily (they're only invoked
  // mid-turn, never during render).
  const {
    isLoading,
    isStreaming,
    isLoadingRef,
    isStreamingRef,
    lastTimings,
    toolActivity,
    sendMessage,
    stopCurrentStream,
    cancelDeferredTts,
  } = useChatTurn({
    setMessages,
    input,
    setInput,
    audioEnabled,
    setAudioEnabled,
    setRequiresAudioUnlock,
    adoptSessionId,
    pinnedToBottomRef,
    pendingImageRef,
    turnImageRef,
    setPendingImage,
    ensureRealLlmConfigured,
    handleMemoryCommand,
    requestFarewell: (message) => requestFarewell(message),
    showMicHint: (hint) => showMicHint(hint),
    completeTurnMessages: (prev, assistantId, turnId) =>
      completeTurnMessages(prev, assistantId, turnId),
    takeRestoredVariantMessage: () => takeRestoredVariantMessage(),
    // A turn failure is exactly when llm_health may have just flipped away
    // from 'ok' — re-fetch /api/status right away instead of waiting on its
    // own poll cadence, so the LLM-health banner below reacts promptly.
    onTurnFailure: () => refreshBackendStatus(),
  })

  // Push-to-talk voice input: STT availability probe, hint UI, Kurisu's
  // speaking flag, and the recording lifecycle all live in the useMic hook
  // — see src/hooks/useMic.ts. The callbacks close over sendMessage/
  // stopCurrentStream (defined below) lazily, so declaration order is safe.
  const {
    micAvailable,
    micUnavailableReason,
    micPermissionDenied,
    kurisuSpeaking,
    micHint,
    showMicHint,
    micStatus,
    startRecording,
    stopRecordingAndTranscribe,
    cancelRecording,
  } = useMic({
    isStreaming,
    onTranscript: (text) => {
      window.dispatchEvent(new Event('kurisu:activity'))
      // Auto-send: a spoken turn is a turn, not a draft. The transcript
      // shows up as the user message immediately, like any voice assistant.
      void sendMessage(text)
    },
    stopCurrentStream: () => stopCurrentStream(),
    textareaRef,
  })

  // Hands-free conversation mode: the mic stays open (echo-cancelled) and a
  // VAD endpointer sends each utterance when the user pauses — see
  // src/hooks/useHandsFree.ts.
  const { handsFree, toggleHandsFree, handsFreeStatus, autoBargeIn, toggleAutoBargeIn } =
    useHandsFree({
      micAvailable,
      micPermissionDenied,
      manualMicActive: micStatus !== 'idle',
      isBusy: isLoading || isStreaming,
      onTranscript: (text) => {
        window.dispatchEvent(new Event('kurisu:activity'))
        void sendMessage(text)
      },
      showHint: showMicHint,
      onBargeIn: () => {
        // Same interrupt + face-only huff as the manual mic-button barge-in.
        const wasSpeaking = kurisuBridge.lipSync?.isSpeaking?.() === true
        stopCurrentStream()
        if (wasSpeaking) kurisuBridge.expressions?.set('angry', 0.35, 1600)
      },
    })

  // Full-duplex call engine (opt-in via the Settings duplex toggle): call
  // mode upgrades to continuous server-endpointed conversation, with an
  // in-place downgrade to the classic hands-free loop when duplex can't
  // run — see src/hooks/useDuplexCall.ts. The fallback closure is injected
  // via ref because it needs fallbackToClassic from useCallMode below.
  const duplexFallbackRef = useRef<(reason: DuplexFallbackReason) => void>(() => {})
  const { duplexActive, duplexState, startDuplex, stopDuplex } = useDuplexCall({
    setMessages,
    adoptSessionId,
    setRequiresAudioUnlock,
    completeTurnMessages: (prev, assistantId, turnId) =>
      completeTurnMessages(prev, assistantId, turnId),
    onFallback: (reason) => duplexFallbackRef.current(reason),
  })

  // Call mode: one session-scoped flag that runs the conversation like a
  // phone call — see src/hooks/useCallMode.ts.
  const { callMode, startCall, endCall, fallbackToClassic } = useCallMode({
    handsFree,
    toggleHandsFree,
    autoBargeIn,
    toggleAutoBargeIn,
    stopCurrentStream: () => stopCurrentStream(),
    duplexPreferred: () => loadDuplexEnabled(),
    tryStartDuplex: startDuplex,
    stopDuplex,
  })
  duplexFallbackRef.current = (reason) => {
    fallbackToClassic()
    showMicHint(
      reason === 'closed'
        ? t('Full-duplex dropped — switched to classic call mode')
        : t('Full-duplex unavailable — classic call mode'),
    )
  }

  // Wake word (opt-in via Settings → Features + server wake_word.enabled):
  // a hands-off listener that starts the normal call when she hears the
  // wake phrase. Paused while any call is active (the call owns the mic);
  // a server-side "no" latches off until the user re-toggles.
  useWakeWord({
    callActive: callMode || duplexActive,
    canUseMic: micAvailable === true && !micPermissionDenied,
    onWake: () => {
      window.dispatchEvent(new Event('kurisu:activity'))
      if (!callMode) startCall()
    },
    onUnavailable: (kind) => {
      showMicHint(
        kind === 'mic'
          ? t('Wake word needs microphone access')
          : t('Wake word unavailable on this server — see Settings → Features'),
      )
    },
  })

  // Intro mode (M6), relationship gauge, milestones, greeting/farewell/poke/
  // idle-chatter/reminders and the morning brief card all live in the
  // usePresence hook (src/hooks/usePresence.ts) — wired in below, once
  // isLoadingRef/isStreamingRef exist.

  // Time-of-day tint (Tier 3): same buckets the 3D scene uses, re-evaluated
  // every 10 minutes so a long-open tab still drifts across boundaries.
  // Purely cosmetic — a stale bucket for a few minutes is invisible.
  const [timeOfDay, setTimeOfDay] = useState<TimeOfDayBucket>(() => getTimeOfDayBucket())
  useEffect(() => {
    const intervalId = window.setInterval(() => {
      setTimeOfDay(getTimeOfDayBucket())
    }, 10 * 60 * 1000)
    return () => window.clearInterval(intervalId)
  }, [])

  const latencyDebug =
    typeof window !== 'undefined' && window.localStorage.getItem('kurisu_debug') === '1'

  // Autoscroll/pin machinery, textarea autosize, transcript persistence and
  // the derived groupStarts/lastAssistantText views all live in the
  // useTranscriptView hook — see src/hooks/useTranscriptView.ts.
  const { messageListRef, handleMessagesScroll, groupStarts, lastAssistantText } =
    useTranscriptView({
      messages,
      isLoading,
      isStreaming,
      input,
      textareaRef,
      pinnedToBottomRef,
    })

  // Window close: best-effort session finalize beacon. `keepalive` lets the
  // request outlive the page; fetch (not sendBeacon) so the Bearer header
  // can ride along on token-protected deployments. `pagehide` rather than
  // `beforeunload` — the latter is unreliable in Tauri v2 webviews.
  useEffect(() => {
    const onPageHide = () => {
      try {
        const token = loadApiToken()
        // Body via presenceRequest (src/presenceApi.ts) so the farewell
        // identifies the user the same way chat does (resolveUserId, never
        // a possibly-null raw localStorage read); keepalive and the Bearer
        // header are this call site's own concerns, layered on top.
        void fetch(resolveApiPath('/api/farewell'), {
          ...presenceRequest({ reason: 'window_close' }),
          keepalive: true,
          headers: {
            'Content-Type': 'application/json',
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
          },
        })
      } catch {
        // Teardown path — never block or surface errors here.
      }
    }
    window.addEventListener('pagehide', onPageHide)
    return () => window.removeEventListener('pagehide', onPageHide)
  }, [])

  // Greeting/farewell/poke/idle-chatter/reminders-poll, the morning brief
  // card, the relationship gauge, milestone celebration, and conversational
  // onboarding (M6) chip tracking all live in usePresence — everything that
  // isn't a direct reply to something the user typed. llmIsDummy/probeSettings/
  // the first-run gate stay here since they're shared with the send path and
  // the Settings panel.
  const {
    requestFarewell,
    tryRequestGreeting,
    morningBrief,
    dismissMorningBrief,
    relationshipTrust,
    relationshipSessions,
    milestoneBanner,
    setMilestoneBanner,
    gaugeCelebrating,
    synapseChips,
  } = usePresence({
    entered,
    setMessages,
    adoptSessionId,
    isLoadingRef,
    isStreamingRef,
    isLoading,
    llmIsDummyRef,
    waitForFirstRunProbe,
    memoryOpen,
  })

  // Short, HUD-flavored status tokens (mono/tracked in CSS) — continues the
  // boot screen's diagnostic-log language instead of sentence-case prose.
  const baseStatusCore = isLoading
    ? t('SYNTHESIZING')
    : audioEnabled
        ? t('LISTENING')
        : t('MUTED · TEXT ONLY')
  // While the GPU build's voice model warms up in the background, replies
  // are subtitle-only — surface that so silence doesn't read as broken.
  const baseStatus = voiceWarming ? `${baseStatusCore} · ${t('VOICE WARMING')}` : baseStatusCore

  const statusText = baseStatus

  // Consolidated voice-state for the header indicator (replaces the old
  // lone speaking-dot) — priority order matters: she can't be "speaking"
  // and "listening" at once, but kurisuSpeaking should win over a stale
  // isLoading flag during the handoff between a turn finishing and audio
  // starting to play.
  const voiceState: VoiceState = kurisuSpeaking
    ? 'speaking'
    : duplexActive
      ? duplexState === 'processing'
        ? 'thinking'
        : duplexState === 'assistant_speaking'
          ? 'speaking'
          : 'listening'
      : micStatus === 'transcribing' || handsFreeStatus === 'transcribing'
        ? 'transcribing'
        : micStatus === 'recording' || handsFreeStatus === 'listening'
          ? 'listening'
          : isLoading && !isStreaming
            ? 'thinking'
            : 'idle'


  // The last assistant message's turnId, and its index — edit (only the
  // user message belonging to the very last turn is editable) and
  // regenerate (only the last assistant message gets the ↻/swipe UI) both
  // key off this instead of re-scanning per row. Error bubbles are skipped:
  // a failed send shouldn't hide the previous (real) reply's swipe/
  // regenerate UI or the preceding user message's Edit button.
  const { lastAssistantIndex, lastTurnId } = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i].role === 'assistant' && !messages[i].isError) {
        return { lastAssistantIndex: i, lastTurnId: messages[i].turnId }
      }
    }
    return { lastAssistantIndex: -1, lastTurnId: undefined as string | undefined }
  }, [messages])

  // The starter chips stay up until the user has actually spoken — a restored
  // transcript on mount plus the auto-greeting from usePresence would otherwise
  // race an assistant bubble in and yank the warm empty-state before the user
  // acts. Gate on "no user message yet", not "no messages at all".
  const hasUserMessage = useMemo(
    () => messages.some((message) => message.role === 'user'),
    [messages],
  )

  // ---- Conversation management moved to src/hooks/useSessions.ts ---------

  // Any request answering 401 (bad/missing API token) surfaces exactly one
  // in-theme notice pointing at the settings field, instead of every call
  // site failing silently in its own way.
  useEffect(() => {
    const handleAuthError = () => {
      setMessages((prev) => {
        const last = prev[prev.length - 1]
        if (last?.role === 'system' && last.content.includes('API token')) return prev
        return [
          ...prev,
          {
            role: 'system',
            content:
              'Access denied (401) — the server rejected the API token. Open Settings → Access and enter the correct token.',
            timestamp: Date.now(),
          },
        ]
      })
    }
    window.addEventListener(AUTH_ERROR_EVENT, handleAuthError)
    return () => window.removeEventListener(AUTH_ERROR_EVENT, handleAuthError)
  }, [])

  // Retry/regenerate/swipe/edit/rewind — plus the pendingVariantsRef stash
  // and the completeTurnMessages/takeRestoredVariantMessage bookkeeping the
  // send path folds in at stream-done/POST-completion/terminal-failure time
  // — all live in useTurnActions; see src/hooks/useTurnActions.ts.
  const {
    editing,
    turnActionBusy,
    handleRetry,
    handleRegenerate,
    handleSwipe,
    handleStartEdit,
    handleRewind,
    handleEditTextChange,
    handleEditSave,
    handleEditCancel,
    completeTurnMessages,
    takeRestoredVariantMessage,
  } = useTurnActions({
    messages,
    setMessages,
    restoredCountRef,
    isLoading,
    isStreaming,
    isLoadingRef,
    isStreamingRef,
    lastAssistantIndex,
    sendMessage,
    stopCurrentStream,
    cancelDeferredTts,
    setInput,
    textareaRef,
  })

  const isComposing = (event: React.KeyboardEvent) => {
    const nativeEvent = event.nativeEvent as KeyboardEvent & { isComposing?: boolean }
    return Boolean(nativeEvent.isComposing)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && !isComposing(e)) {
      e.preventDefault()
      sendMessage()
    }
  }

  // Shared by the full empty-state and the chips-footer that survives the
  // auto-greeting until the user's first message (A2) — one source of chips.
  const renderSuggestionChips = () =>
    EMPTY_STATE_SUGGESTIONS.map((suggestion) => (
      <button
        key={suggestion}
        type="button"
        role="listitem"
        className="suggestion-chip"
        onClick={() => {
          setInput(t(suggestion))
          window.dispatchEvent(new Event('kurisu:activity'))
          requestAnimationFrame(() => textareaRef.current?.focus())
        }}
      >
        {t(suggestion)}
      </button>
    ))

  return (
    <div className="chat-interface" data-tod={timeOfDay} data-mode={uiMode ?? 'full'}>
      <div className="chat-header">
        <div className="chat-title">
          <h2>
            {backendStatus?.character?.full_name || 'Makise Kurisu'}
            <VoiceStatusIndicator state={voiceState} />
            <RelationshipGauge
              trust={relationshipTrust}
              sessionCount={relationshipSessions}
              onClick={() => setMemoryOpen(true)}
              celebrating={gaugeCelebrating}
            />
          </h2>
          <span className="chat-subtitle">AMADEUS Interface</span>
          {synapseChips.length > 0 && (
            <div className="synapse-chip-row" aria-live="polite">
              {synapseChips.map((chip) => (
                <span key={chip.id} className="synapse-chip">
                  {chip.key === '__final__' ? chip.value : `SYNAPSE FORMED · ${chip.key}`}
                </span>
              ))}
            </div>
          )}
        </div>
      <div className="chat-toolbar">
        <span className={`chat-status ${isLoading ? 'busy' : ''}`}>{statusText}</span>
        <StatusChip status={backendStatus} />
        {latencyDebug && lastTimings && (
          <span className="chat-timestamp latency-debug">{lastTimings}</span>
        )}
        {sessionsSupported === true && (
          <button
            type="button"
            className="settings-gear-button"
            onClick={toggleSidebar}
            aria-label={sidebarOpen ? t('Close conversations') : t('Open conversations')}
            aria-expanded={sidebarOpen}
            title={t('Conversations')}
          >
            ▤
          </button>
        )}
        <button
          type="button"
          className="settings-gear-button"
          onClick={toggleFullscreen}
          aria-label={isFullscreen ? t('Exit full screen') : t('Enter full screen')}
          title={t('Full screen (F11)')}
        >
          {isFullscreen ? '🗗' : '⛶'}
        </button>
        {onToggleUiMode && (
          <button
            type="button"
            className="settings-gear-button"
            onClick={onToggleUiMode}
            aria-pressed={uiMode === 'vn'}
            aria-label={uiMode === 'vn' ? t('Show full conversation log') : t('Immersive mode')}
            title={
              uiMode === 'vn'
                ? t('Show full conversation log')
                : t('Immersive mode: hide the log, keep her')
            }
          >
            ◨
          </button>
        )}
        {isTauri() && (
          <button
            type="button"
            className="settings-gear-button"
            onClick={() => {
              void (async () => {
                try {
                  const { invoke } = await import('@tauri-apps/api/core')
                  await invoke('toggle_pet_window')
                } catch {
                  // Backend not ready yet, or command unavailable.
                }
              })()
            }}
            aria-label={t('Toggle desktop pet')}
            title={t('Desktop pet: float her above your other windows')}
          >
            ᓚ
          </button>
        )}
        <button
          type="button"
          className="settings-gear-button"
          onClick={() => setMemoryOpen(true)}
          aria-label={t('Memory')}
          title={t('Her memories')}
        >
          ❖
        </button>
        {!backendStatus?.demo_mode && (
          <button
            type="button"
            className="settings-gear-button"
            onClick={() => {
              setSettingsIsOnboarding(false)
              setSettingsOpen(true)
            }}
            aria-label={t('Open settings')}
            title={t('Settings')}
          >
            ⚙
          </button>
        )}
      </div>
      </div>

      {sidebarOpen && sessionsSupported === true && (
        <SessionSidebar
          sessions={sessions}
          currentSessionId={currentSessionId}
          loading={sessionsLoading}
          error={sessionsError}
          onSelect={(sessionId) => void switchToSession(sessionId)}
          onNew={() => void startNewConversation()}
          onDelete={(sessionId) => void removeSession(sessionId)}
          onExport={(sessionId) => void exportSession(sessionId)}
          onClose={() => setSidebarOpen(false)}
          fetchTranscriptText={fetchTranscriptTextForSearch}
          assistantName={backendStatus?.character?.name || 'Kurisu'}
        />
      )}

      <div
        className="messages-container"
        ref={messageListRef}
        onScroll={handleMessagesScroll}
        aria-live="polite"
      >
        {/* Full warm empty-state (icon + hero copy + chips) only while the
            transcript is genuinely empty. */}
        {messages.length === 0 && (
          <EmptyState
            icon="chat"
            title={t(
              "Connection established. She'll pretend she wasn't waiting — say something anyway.",
            )}
            className="empty-state"
          >
            <div className="empty-state-suggestions" role="list">
              {renderSuggestionChips()}
            </div>
          </EmptyState>
        )}
        {/* Row rendering lives in the memoized TranscriptList so unrelated
            ChatInterface state changes (settings panel, mic state, HUD…)
            no longer re-walk the row array — only these props can. */}
        <TranscriptList
          messages={messages}
          groupStarts={groupStarts}
          lastAssistantIndex={lastAssistantIndex}
          lastTurnId={lastTurnId}
          busy={isLoading || isStreaming || turnActionBusy}
          isStreaming={isStreaming}
          restoredCount={restoredCountRef.current}
          assistantName={backendStatus?.character?.name || 'Kurisu'}
          editing={editing}
          onEditTextChange={handleEditTextChange}
          onEditSave={handleEditSave}
          onEditCancel={handleEditCancel}
          onRetry={handleRetry}
          onRegenerate={handleRegenerate}
          onSwipe={handleSwipe}
          onEdit={handleStartEdit}
          onRewind={handleRewind}
        />
        {/* Once a greeting/restored bubble exists but the user hasn't spoken
            yet, keep the starter chips as a footer so the warm invitation
            survives the auto-greeting race (A2). */}
        {messages.length > 0 && !hasUserMessage && (
          <div className="empty-state-suggestions chips-footer" role="list">
            {renderSuggestionChips()}
          </div>
        )}
        {isLoading && toolActivity && (
          <div className="message assistant loading">
            <div className="message-content tool-activity" role="status" aria-label={toolActivity}>
              <span className="thinking-indicator">
                <span className="thinking-label">{toolActivity}</span>
                <span className="thinking-dots" aria-hidden="true">
                  <span />
                  <span />
                  <span />
                </span>
              </span>
            </div>
          </div>
        )}
        {isLoading && !toolActivity && (
          <div className="message assistant loading">
            <div className="message-content">
              <span className="thinking-indicator" role="status" aria-label={t('Kurisu is thinking')}>
                <span className="thinking-label">{t('SYNTHESIZING')}</span>
                <span className="thinking-dots" aria-hidden="true">
                  <span />
                  <span />
                  <span />
                </span>
              </span>
            </div>
          </div>
        )}
      </div>


      {updateAvailable && (
        <div className="update-toast" role="status">
          <span className="update-toast-label">
            {t('UPDATE AVAILABLE')} · {updateAvailable}
          </span>
          <button type="button" onClick={openReleasesPage}>
            {t('VIEW RELEASE')}
          </button>
          <button
            type="button"
            onClick={() => {
              dismissUpdate(updateAvailable)
              setUpdateAvailable(null)
            }}
          >
            {t('SKIP')}
          </button>
          <button
            type="button"
            aria-label={t('Dismiss update notice')}
            onClick={() => setUpdateAvailable(null)}
          >
            ×
          </button>
        </div>
      )}

      {llmHealthBanner && !settingsOpen && (
        <div
          className="morning-brief-card dummy-mode-banner"
          role="alert"
          aria-label={
            llmHealthBanner === 'dummy'
              ? t('Offline demo mode')
              : llmHealthBanner === 'invalid_key'
                ? t('API KEY REJECTED')
                : llmHealthBanner === 'rate_limited'
                  ? t('QUOTA EXHAUSTED')
                  : t('PROVIDER ERROR')
          }
        >
          <div className="morning-brief-header">
            <span className="morning-brief-title">
              {llmHealthBanner === 'dummy' && t('OFFLINE DEMO MODE — NO AI CONNECTED')}
              {llmHealthBanner === 'invalid_key' && t('API KEY REJECTED')}
              {llmHealthBanner === 'rate_limited' && t('QUOTA EXHAUSTED')}
              {llmHealthBanner === 'error' && t('PROVIDER ERROR')}
            </span>
            {/* The original dummy banner (an explicit onboarding/config
                choice, not a transient failure) stays permanent — no
                dismiss — exactly as before. The other three states are
                transient provider failures: dismissible, but the dismissal
                only suppresses this specific state (see
                dismissedBannerState) so a NEW failure after dismissing an
                old one still resurfaces. */}
            {llmHealthBanner !== 'dummy' && (
              <button
                type="button"
                aria-label={t('Dismiss')}
                onClick={() => setDismissedBannerState(llmHealthBanner)}
              >
                ×
              </button>
            )}
          </div>
          <p className="morning-brief-text">
            {llmHealthBanner === 'dummy' &&
              t(
                'Replies right now are canned placeholders, not the real Kurisu. Add a free Gemini key (or an OpenAI key) and she wakes up properly.',
              )}
            {llmHealthBanner === 'invalid_key' &&
              t('The provider is rejecting the API key — fix it in Settings.')}
            {llmHealthBanner === 'rate_limited' &&
              t(
                'Provider quota is exhausted for now — wait for the window to reset, or switch key/provider in Settings.',
              )}
            {llmHealthBanner === 'error' &&
              `${t('The provider is failing right now')}${
                backendStatus?.llm_health?.detail ? `: ${backendStatus.llm_health.detail}` : '.'
              }`}
          </p>
          {llmHealthBanner !== 'error' && (
            <button
              type="button"
              className="dummy-mode-banner-action"
              onClick={() => {
                setSettingsIsOnboarding(llmHealthBanner === 'dummy')
                setSettingsOpen(true)
              }}
            >
              {llmHealthBanner === 'dummy' ? t('Set up an API key →') : t('Open Settings →')}
            </button>
          )}
        </div>
      )}
      {morningBrief && (
        <div className="morning-brief-card" role="note" aria-label={t('Morning brief')}>
          <div className="morning-brief-header">
            <span className="morning-brief-title">{t('MORNING BRIEF')} · {morningBrief.date}</span>
            <button
              type="button"
              aria-label={t('Dismiss morning brief')}
              onClick={dismissMorningBrief}
            >
              ×
            </button>
          </div>
          <p className="morning-brief-text">{morningBrief.text}</p>
        </div>
      )}
      {milestoneBanner && (
        <div className="morning-brief-card milestone-banner" role="note" aria-label={t('Milestone')}>
          <div className="morning-brief-header">
            <span className="morning-brief-title">{t('MILESTONE')}</span>
            <button
              type="button"
              aria-label={t('Dismiss milestone')}
              onClick={() => setMilestoneBanner(null)}
            >
              ×
            </button>
          </div>
          <p className="morning-brief-text">{milestoneBanner.label}</p>
        </div>
      )}
      {pendingImage && (
        <div className="pending-image-chip" role="status">
          <img src={pendingImage.previewUrl} alt={t('Attached screenshot preview')} />
          <span className="pending-image-label">
            {t('Image attached')} · {pendingImage.width}×{pendingImage.height}
          </span>
          <button
            type="button"
            aria-label={t('Remove attached image')}
            onClick={() => setPendingImage(null)}
          >
            ×
          </button>
        </div>
      )}
      {uiMode === 'vn' && (lastAssistantText !== '' || (isLoading && !isStreaming)) && (
        <div className="vn-dialogue" role="log" aria-live="polite">
          <span className="vn-dialogue-name">
            {(backendStatus?.character?.name || 'Kurisu').toUpperCase()}
          </span>
          {isLoading && !isStreaming ? (
            <span className="vn-dialogue-thinking">…</span>
          ) : (
            <p className="vn-dialogue-text">{lastAssistantText}</p>
          )}
          {voiceState === 'speaking' && micAvailable === true && !micPermissionDenied && (
            <span className="vn-interrupt-hint">
              {callMode ? t('just start talking to interrupt') : t('hold the mic to interrupt')}
            </span>
          )}
        </div>
      )}
      {callMode && (
        <div className="call-hud" role="status" aria-label={t('Call in progress')}>
          <span className={`call-hud-state call-hud-${voiceState}`}>
            <span className="call-hud-dot" aria-hidden="true" />
            {voiceState === 'speaking'
              ? t('SPEAKING')
              : voiceState === 'transcribing'
                ? t('HEARD YOU')
                : voiceState === 'listening'
                  ? t('LISTENING')
                  : voiceState === 'thinking'
                    ? t('THINKING')
                    : t('ON CALL')}
          </span>
          <button type="button" className="call-hud-end" onClick={endCall}>
            {t('End call')}
          </button>
        </div>
      )}
      <div
        className="input-container"
        onDrop={handleDrop}
        onDragOver={(event) => {
          if (visionAvailable && event.dataTransfer?.types.includes('Files')) {
            event.preventDefault()
          }
        }}
      >
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => {
            setInput(e.target.value)
            window.dispatchEvent(new Event('kurisu:activity'))
          }}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          placeholder={t('Type a message…')}
          rows={1}
          disabled={isLoading && !isStreaming}
        />
        <div className="input-actions">
         <button
            type="button"
            className={`mute-toggle ${audioEnabled ? '' : 'muted'}`}
            onClick={() => {
              setAudioEnabled((prev) => {
                const next = !prev
                saveAudioEnabled(next)
                if (next) {
                  setRequiresAudioUnlock(false)
                }
                return next
              })
            }}
            disabled={!audioSupported}
            title={audioSupported ? undefined : t('Audio is not supported in this browser')}
            aria-pressed={!audioEnabled}
          >
            {audioEnabled ? t('Audio On') : requiresAudioUnlock ? t('Enable Audio') : t('Audio Off')}
          </button>
          {(micHint || requiresAudioUnlock || !isTouchDevice) && (
            <span className="input-hint">
              {micHint ??
                (requiresAudioUnlock
                  ? t("Enable audio to hear Kurisu's replies")
                  : t('Enter to send · Shift+Enter for newline'))}
            </span>
          )}
          {isStreaming && (
            <button type="button" className="send-button stop-button" onClick={stopCurrentStream}>
              {t('Stop')}
            </button>
          )}
          {/* Continuous-voice controls kept together as one interdependent
              cluster (barge-in needs hands-free, call orchestrates both) so
              they wrap as a unit and don't crowd mute/mic/Send in the primary
              row. Wiring is unchanged — this is purely grouping. */}
          <div
            className="voice-mode-cluster"
            role="group"
            aria-label={t('Continuous voice controls')}
          >
            {handsFreeStatus === 'listening' && (
              <span className="hands-free-listening" role="status">
                <span className="hands-free-listening-dot" aria-hidden="true" />
                {t('Listening…')}
              </span>
            )}
            <button
              type="button"
              className={`hands-free-toggle ${handsFree ? 'active' : ''}`}
              onClick={toggleHandsFree}
              aria-pressed={handsFree}
              disabled={micAvailable !== true || micPermissionDenied}
              title={
                micPermissionDenied
                  ? t('Microphone permission denied — check your browser settings')
                  : micAvailable === false
                    ? micUnavailableReason ?? t('Speech-to-text not installed')
                    : handsFree
                      ? t('Hands-free on — the mic stays open; speak and she answers when you pause')
                      : t('Hands-free: open mic — speak and she answers when you pause')
              }
            >
              {t('Hands-free')}
            </button>
            <button
              type="button"
              className={`hands-free-toggle barge-in-toggle ${autoBargeIn ? 'active' : ''}`}
              onClick={toggleAutoBargeIn}
              aria-pressed={autoBargeIn}
              disabled={!handsFree || micAvailable !== true || micPermissionDenied}
              title={
                !handsFree
                  ? t('Enable hands-free first')
                  : autoBargeIn
                    ? t('Auto barge-in on — interrupt her just by talking (echo-cancelled mic stays open while she speaks)')
                    : t('Auto barge-in: interrupt her just by talking. Quality depends on your device’s echo cancellation — leave off if she interrupts herself.')
              }
            >
              {t('Barge-in')}
            </button>
            <button
              type="button"
              className={`hands-free-toggle call-toggle ${callMode ? 'active' : ''}`}
              onClick={() => (callMode ? endCall() : startCall())}
              aria-pressed={callMode}
              disabled={micAvailable !== true || micPermissionDenied}
              title={
                callMode
                  ? t('End the call')
                  : t('Call mode: continuous conversation — the mic stays open, and you can interrupt just by talking')
              }
            >
              {callMode ? t('End call') : t('📞 Call')}
            </button>
          </div>
          {visionAvailable && (
            <button
              type="button"
              className={`screen-share-button ${pendingImage ? 'active' : ''}`}
              onClick={() => void handleCaptureScreen()}
              disabled={isLoading && !isStreaming}
              aria-pressed={Boolean(pendingImage)}
              title={t('Share a screenshot with the next message (you can also paste or drop an image)')}
            >
              📷
            </button>
          )}
          <button
            type="button"
            className={`mic-button ${micStatus === 'recording' ? 'recording' : ''} ${micStatus === 'transcribing' ? 'transcribing' : ''}`}
            onPointerDown={(event) => {
              event.preventDefault()
              void startRecording()
            }}
            onPointerUp={() => void stopRecordingAndTranscribe()}
            onPointerLeave={() => {
              if (micStatus === 'recording') void stopRecordingAndTranscribe()
            }}
            onPointerCancel={cancelRecording}
            disabled={
              micAvailable !== true ||
              micPermissionDenied ||
              micStatus === 'transcribing' ||
              // Duplex call: the continuous loop owns the mic — a second
              // PTT capture alongside it would double-transcribe.
              duplexActive
            }
            aria-pressed={micStatus === 'recording'}
            title={
              micPermissionDenied
                ? t('Microphone permission denied — check your browser settings')
                : micAvailable === false
                  ? micUnavailableReason ?? t('Speech-to-text not installed')
                  : kurisuSpeaking
                    ? t('Hold to talk — interrupts Kurisu')
                    : t('Hold to talk (or hold Space)')
            }
          >
            {micStatus === 'transcribing' ? (
              <span className="mic-spinner" aria-hidden="true" />
            ) : (
              '🎤'
            )}
          </button>
          <button
            type="button"
            className="send-button"
            onClick={() => sendMessage()}
            disabled={!input.trim() || (isLoading && !isStreaming)}
          >
            {t('Send')}
          </button>
        </div>
      </div>

      {memoryOpen && <MemoryPanel onClose={() => setMemoryOpen(false)} />}

      {settingsOpen && (
        <SettingsPanel
          isOnboarding={settingsIsOnboarding}
          onClose={() => {
            const wasOnboarding = settingsIsOnboarding
            setSettingsOpen(false)
            setSettingsIsOnboarding(false)
            // A save may have hot-swapped providers — re-read runtime truth.
            // probeSettings also refreshes llmIsDummy so the offline-demo
            // banner clears the moment a key is added via normal Settings.
            refreshBackendStatus()
            void probeSettings()
            void maybeRegreetAfterOnboarding(wasOnboarding)
          }}
          onSkipOnboarding={() => {
            const wasOnboarding = settingsIsOnboarding
            setSettingsOpen(false)
            setSettingsIsOnboarding(false)
            void maybeRegreetAfterOnboarding(wasOnboarding)
          }}
        />
      )}
    </div>
  )
}

export default ChatInterface
