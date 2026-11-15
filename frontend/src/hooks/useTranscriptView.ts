/**
 * Transcript-view plumbing extracted from ChatInterface: the message-list
 * scroll container (autoscroll while pinned to the bottom, coalesced to one
 * scroll per animation frame), the textarea autosize effect, transcript
 * persistence to localStorage (skipped while streaming), and the derived
 * per-render views of the message array (group-start flags for the visual
 * message grouping, and the last assistant line for VN mode).
 *
 * Deliberately NOT included: the messages array itself and the input state
 * (both stay on ChatInterface — the send path and JSX own them; this hook
 * receives the values), textareaRef (created at the root because useMic
 * also consumes it), and pinnedToBottomRef (created at the root because the
 * send path re-pins on every send).
 */
import { useEffect, useMemo, useRef } from 'react'
import type { MutableRefObject, RefObject } from 'react'
import { saveTranscript } from '../transcript'
import type { Message } from '../chat/messages'

// Tier 2: a new message-group header (sender + relative context) is shown
// only when the speaker changes or a >5 minute gap separates two messages
// from the same speaker — otherwise consecutive bubbles are visually
// grouped and only reveal their timestamp on hover/focus.
const GROUP_GAP_MS = 5 * 60 * 1000

export interface UseTranscriptViewOptions {
  messages: Message[]
  isLoading: boolean
  isStreaming: boolean
  /** Live textarea value — drives the autosize effect. */
  input: string
  /** Created at the root — useMic also consumes it (hold-Space PTT). */
  textareaRef: RefObject<HTMLTextAreaElement | null>
  /** Created at the root — the send path re-pins to bottom on every send. */
  pinnedToBottomRef: MutableRefObject<boolean>
}

export const useTranscriptView = ({
  messages,
  isLoading,
  isStreaming,
  input,
  textareaRef,
  pinnedToBottomRef,
}: UseTranscriptViewOptions) => {
  const messageListRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const textarea = textareaRef.current
    if (!textarea) return
    textarea.style.height = 'auto'
    textarea.style.height = `${Math.min(200, textarea.scrollHeight)}px`
  }, [input, textareaRef])

  const handleMessagesScroll = () => {
    const container = messageListRef.current
    if (!container) return
    pinnedToBottomRef.current =
      container.scrollHeight - container.scrollTop - container.clientHeight < 48
  }

  // Autoscroll coalesced to at most one scroll per animation frame: during
  // streaming this effect fires on every ~16ms token flush, and a
  // synchronous scrollTo (which reads scrollHeight) forces a full layout
  // each time. If a scroll is already scheduled this frame, skip — the
  // pending callback reads the latest scrollHeight when it fires anyway.
  const scrollRafRef = useRef<number | null>(null)
  useEffect(() => {
    const container = messageListRef.current
    if (!container || !pinnedToBottomRef.current) return
    if (scrollRafRef.current !== null) return
    // Instant during streaming: queued smooth scrolls stutter when tokens
    // arrive faster than the animation completes.
    const behavior: ScrollBehavior = isStreaming ? 'auto' : 'smooth'
    scrollRafRef.current = window.requestAnimationFrame(() => {
      scrollRafRef.current = null
      const target = messageListRef.current
      // Re-check the pin at fire time — the user may have scrolled up in
      // the gap between scheduling and the frame.
      if (!target || !pinnedToBottomRef.current) return
      target.scrollTo({ top: target.scrollHeight, behavior })
    })
  }, [messages, isLoading, isStreaming, pinnedToBottomRef])
  useEffect(
    () => () => {
      // Unmount: drop the pending scroll so it never touches a dead node.
      if (scrollRafRef.current !== null) window.cancelAnimationFrame(scrollRafRef.current)
    },
    [],
  )

  useEffect(() => {
    // Persistence only needs to survive a reload, not mirror every streamed
    // token — a synchronous JSON.stringify + localStorage write per token is
    // the main source of jank during replies. Skip while streaming; when the
    // stream ends (isStreaming flips false) this re-runs and flushes the
    // final transcript.
    if (isStreaming) return
    saveTranscript(messages)
  }, [messages, isStreaming])

  /** A message starts a new visual group (and gets a sender/time header
   * line above it) when the speaker differs from the previous message, or
   * more than GROUP_GAP_MS has elapsed since it — same rule for every
   * role, including agent messages keyed by agentName. */
  const groupStarts = useMemo(() => {
    const starts = new Array<boolean>(messages.length)
    let prevSpeakerKey: string | null = null
    let prevTimestamp = -Infinity
    messages.forEach((message, index) => {
      const speakerKey = message.role === 'agent' ? `agent:${message.agentName ?? ''}` : message.role
      const isNewGroup =
        speakerKey !== prevSpeakerKey || message.timestamp - prevTimestamp > GROUP_GAP_MS
      starts[index] = isNewGroup
      prevSpeakerKey = speakerKey
      prevTimestamp = message.timestamp
    })
    return starts
  }, [messages])

  // VN mode's dialogue box shows only her latest line. Streaming mutates
  // the last assistant message's content in place, so this re-derives (and
  // the box live-updates) for free as tokens arrive — no separate state.
  const lastAssistantText = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i].role === 'assistant') return messages[i].content
    }
    return ''
  }, [messages])

  return {
    messageListRef,
    handleMessagesScroll,
    groupStarts,
    lastAssistantText,
  }
}
