/**
 * The turn-mutating actions extracted from ChatInterface: retry from an
 * error bubble, regenerate (rewind + re-send with the prior reply stashed
 * as a variant chain), swipe between variants (+ best-effort /amend),
 * inline edit of the last user message, and rewind-to-message. Owns the
 * inline-editing state, pendingVariantsRef (the regenerate stash), the
 * turnActionInFlightRef re-entrancy guard + its turnActionBusy state
 * mirror, and the completeTurnMessages / takeRestoredVariantMessage
 * bookkeeping the send path calls at stream-done/POST-completion/terminal-
 * failure time (they read the stash, so they live with their owner here
 * and are injected into useChatTurn).
 *
 * Deliberately NOT owned: the transcript (messages/setMessages and
 * restoredCountRef are injected), the send path itself (sendMessage /
 * stopCurrentStream / cancelDeferredTts come from useChatTurn, along with
 * isLoading/isStreaming and their live ref mirrors), and the input box
 * (rewind refills it via the injected setInput/textareaRef).
 */
import { useRef, useState } from 'react'
import type { Dispatch, MutableRefObject, RefObject, SetStateAction } from 'react'
import { apiFetch } from '../api'
import {
  buildSystemMessage,
  foldPendingVariants,
  rewindTurn,
  stampTurn,
  variantToMessage,
} from '../chat/turnActions'
import type { Message, MessageVariant } from '../chat/messages'
import { resolveUserId } from './useSessions'
import { useStableHandler } from './useStableHandler'

export interface UseTurnActionsOptions {
  messages: Message[]
  setMessages: Dispatch<SetStateAction<Message[]>>
  /** Marks where the restored transcript ends and the live session begins —
   * shrunk when an edit/rewind truncates past it. */
  restoredCountRef: MutableRefObject<number>
  isLoading: boolean
  isStreaming: boolean
  /** Live mirrors — retry fires from a memoized row whose closure may be
   * stale by the time the click lands. */
  isLoadingRef: MutableRefObject<boolean>
  isStreamingRef: MutableRefObject<boolean>
  /** Index of the last non-error assistant message (regenerate only ever
   * applies to it). */
  lastAssistantIndex: number
  sendMessage: (overrideText?: string, opts?: { echoUser?: boolean }) => Promise<void>
  stopCurrentStream: () => void
  cancelDeferredTts: () => void
  /** Rewind refills the input box with the rewound message's text. */
  setInput: Dispatch<SetStateAction<string>>
  textareaRef: RefObject<HTMLTextAreaElement | null>
}

export const useTurnActions = ({
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
}: UseTurnActionsOptions) => {
  // Regenerate (M1): the variant history stashed by regenerateMessage while
  // the replacement reply streams/POSTs in, folded onto the new assistant
  // message once it completes (same place stampTurn runs). Cleared on
  // completion or turn failure so a later unrelated turn never picks it up.
  const pendingVariantsRef = useRef<MessageVariant[] | null>(null)
  // Edit (M1b): the user message currently being edited inline, or null.
  // `text` is the live textarea value (separate from message.content so
  // typing doesn't touch the transcript until Save).
  const [editing, setEditing] = useState<{ target: Message; text: string } | null>(null)
  // Re-entrancy guard for the turn-mutating actions (regenerate/edit): both
  // await a rewind POST before doing anything else, so a double-click/
  // double-Enter before that resolves must not fire two rewinds (or two
  // sends). The ref is checked synchronously (state updates aren't visible
  // until the next render, which a second click before that render could
  // race); the state mirror drives MessageRow's `busy` prop so the buttons
  // visibly disable for the duration. Both are set/cleared together.
  const turnActionInFlightRef = useRef(false)
  const [turnActionBusy, setTurnActionBusy] = useState(false)

  /** Retry from an error bubble: re-sends the original user text through the
   * normal send path (stream-first with POST fallback). Pure error bubbles
   * (network/server — no real content) are removed and replaced by the
   * retried turn; an interrupted-stream bubble keeps its partial content.
   * The user bubble already exists, so the retry doesn't echo it again. */
  const retryMessage = (message: Message) => {
    const text = message.retryText?.trim()
    if (!text) return
    if (isLoadingRef.current && !isStreamingRef.current) return
    if (message.errorKind === 'network' || message.errorKind === 'server') {
      setMessages((prev) => prev.filter((msg) => msg !== message))
    }
    void sendMessage(text, { echoUser: false })
  }

  // Stable identity for the memoized MessageRow: the ref always points at
  // the latest retryMessage closure, the callback never changes.
  const handleRetry = useStableHandler(retryMessage)

  /** Regenerate (M1): asks the server to drop the last completed turn, then
   * re-sends the same prompt. The prior reply (or its whole variant chain)
   * is stashed in pendingVariantsRef so it can be folded back in as history
   * once the new reply lands (see foldPendingVariants, applied at the
   * stream onDone / POST completion sites). Guarded by
   * turnActionInFlightRef so a double-click can't fire two rewinds. */
  const regenerateMessage = async (message: Message) => {
    if (turnActionInFlightRef.current) return
    if (isLoading || isStreaming) return
    if (lastAssistantIndex === -1 || messages[lastAssistantIndex] !== message) return
    if (!message.turnId) return

    let promptText: string | null = null
    for (let i = lastAssistantIndex - 1; i >= 0; i -= 1) {
      if (messages[i].role === 'user') {
        promptText = messages[i].content
        break
      }
    }
    if (!promptText) return

    const stash: MessageVariant[] = message.variants
      ? [...message.variants]
      : [
          {
            content: message.content,
            japanese: message.japanese,
            thought: message.thought,
            timestamp: message.timestamp,
          },
        ]
    pendingVariantsRef.current = stash

    const turnId = message.turnId
    const prompt = promptText
    turnActionInFlightRef.current = true
    setTurnActionBusy(true)
    try {
      const outcome = await rewindTurn(turnId)
      if (outcome !== 'ok') {
        pendingVariantsRef.current = null
        setMessages((prev) => [
          ...prev,
          buildSystemMessage("Couldn't regenerate that reply — the server didn't accept the rewind."),
        ])
        return
      }
      setMessages((prev) => prev.filter((msg) => msg !== message))
      await sendMessage(prompt, { echoUser: false })
    } finally {
      turnActionInFlightRef.current = false
      setTurnActionBusy(false)
    }
  }

  const handleRegenerate = useStableHandler((message: Message) => void regenerateMessage(message))

  /** Swipe through regenerate variants (M1): flips the message's displayed
   * content/japanese/thought to variant `k` without touching the server's
   * durable last-turn text, then fires the /amend PATCH-equivalent so a
   * reload/session-switch shows whichever variant the user last landed on.
   * Fire-and-forget — no TTS replay, this is a silent display swap. */
  const swipeVariant = (message: Message, delta: 1 | -1) => {
    const variants = message.variants
    if (!variants || variants.length === 0) return
    const current = message.activeVariant ?? variants.length - 1
    const k = Math.max(0, Math.min(variants.length - 1, current + delta))
    if (k === current) return
    const variant = variants[k]

    setMessages((prev) =>
      prev.map((msg) =>
        msg === message
          ? {
              ...msg,
              content: variant.content,
              japanese: variant.japanese,
              thought: variant.thought,
              activeVariant: k,
            }
          : msg,
      ),
    )

    if (message.turnId) {
      void apiFetch(
        `/api/sessions/${encodeURIComponent(localStorage.getItem('session_id') || '')}/amend`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            turn_id: message.turnId,
            content: variant.content,
            ja: variant.japanese ?? null,
            user_id: resolveUserId(),
          }),
        },
      ).catch(() => {
        // Best-effort — a failed amend just means a reload shows the
        // pre-swipe text; the live UI already reflects the swipe.
      })
    }
  }

  const handleSwipe = useStableHandler(swipeVariant)

  /** Edit (M1b): opens the inline editor for the last user message. Only
   * that message is editable — canEdit gates on message.turnId === lastTurnId. */
  const startEdit = (message: Message) => {
    if (isLoading || isStreaming) return
    setEditing({ target: message, text: message.content })
  }

  const handleStartEdit = useStableHandler(startEdit)

  const cancelEdit = () => setEditing(null)

  /** Commits an edit: rewinds the server past the turn the edited message
   * started, replaces the user bubble in place, drops the stale assistant
   * (and any immediately-following council summary) bubble, then re-sends
   * the edited text as a fresh turn. Old variants are deliberately
   * discarded — an edited prompt invalidates the old regenerate history. */
  const confirmEdit = async () => {
    if (turnActionInFlightRef.current) return
    if (!editing) return
    const newText = editing.text.trim()
    if (!newText) {
      cancelEdit()
      return
    }
    const target = editing.target
    const turnId = target.turnId
    if (!turnId) {
      cancelEdit()
      return
    }

    turnActionInFlightRef.current = true
    setTurnActionBusy(true)
    try {
      const outcome = await rewindTurn(turnId)
      if (outcome !== 'ok') {
        setMessages((prev) => [
          ...prev,
          buildSystemMessage("Couldn't edit that message — the server didn't accept the rewind."),
        ])
        return
      }

      setMessages((prev) => {
        const targetIndex = prev.indexOf(target)
        if (targetIndex === -1) return prev
        // Drop the assistant reply for this turn (and an immediately
        // following "Council summary:" system bubble, if present) — both are
        // stale once the edited prompt re-sends.
        let dropEnd = targetIndex + 1
        if (dropEnd < prev.length && prev[dropEnd].role === 'assistant' && prev[dropEnd].turnId === turnId) {
          dropEnd += 1
          if (
            dropEnd < prev.length &&
            prev[dropEnd].role === 'system' &&
            prev[dropEnd].content.startsWith('Council summary:')
          ) {
            dropEnd += 1
          }
        }
        const next = [
          ...prev.slice(0, targetIndex),
          { ...target, content: newText, timestamp: Date.now(), turnId: undefined },
          ...prev.slice(dropEnd),
        ]
        if (restoredCountRef.current > next.length) {
          restoredCountRef.current = next.length
        }
        return next
      })

      setEditing(null)
      await sendMessage(newText, { echoUser: false })
    } finally {
      turnActionInFlightRef.current = false
      setTurnActionBusy(false)
    }
  }

  // Stable identities for the inline editor's callbacks, so passing them
  // into the memoized TranscriptList doesn't hand it a fresh closure every
  // render (confirmEdit/cancelEdit are re-created per render; the row-level
  // handlers already go through useStableHandler for the same reason).
  const handleEditTextChange = useStableHandler((text: string) =>
    setEditing((prev) => (prev ? { ...prev, text } : prev)),
  )
  const handleEditSave = useStableHandler(() => void confirmEdit())
  const handleEditCancel = useStableHandler(cancelEdit)

  /** Rewind (M5): truncates the transcript to before `message` (a user
   * message with a turnId), server first. The server truncates the durable
   * transcript at that turn, rebuilds short-term context, and resets the
   * summary; long-term facts already extracted from the erased turns are
   * NOT unwound (they're deletable individually in the Memory panel).
   * Pre-feature messages without a turnId simply show no rewind button. */
  const rewindToMessage = async (message: Message) => {
    if (!window.confirm('Rewind to before this message? Everything after it will be forgotten.')) {
      return
    }
    if (!message.turnId) return

    if (isStreaming) {
      stopCurrentStream()
    }
    cancelDeferredTts()

    const outcome = await rewindTurn(message.turnId)
    if (outcome === 'not_found') {
      setMessages((prev) => [
        ...prev,
        buildSystemMessage(
          "The server no longer knows that turn — it may have already been rewound, or transcripts are disabled on this server.",
        ),
      ])
      return
    }
    if (outcome !== 'ok') {
      setMessages((prev) => [
        ...prev,
        buildSystemMessage('Rewind failed — the server rejected the request, or is unreachable.'),
      ])
      return
    }

    setMessages((prev) => {
      const index = prev.indexOf(message)
      if (index === -1) return prev
      const next = prev.slice(0, index)
      if (restoredCountRef.current > next.length) {
        restoredCountRef.current = next.length
      }
      return next
    })
    setInput(message.content)
    requestAnimationFrame(() => textareaRef.current?.focus())
  }

  const handleRewind = useStableHandler((message: Message) => void rewindToMessage(message))

  /** Turn-completion bookkeeping shared by the stream onDone and the POST
   * completion (called inside a setMessages updater): stamps the turn id
   * onto the just-landed assistant message and, if regenerateMessage
   * stashed a variant chain for this turn, folds it back in — clearing the
   * stash so a later unrelated turn never picks it up. */
  const completeTurnMessages = (
    prev: Message[],
    assistantId: string,
    turnId: string,
  ): Message[] => {
    const stamped = stampTurn(prev, assistantId, turnId)
    const pending = pendingVariantsRef.current
    if (!pending) return stamped
    pendingVariantsRef.current = null
    return foldPendingVariants(stamped, assistantId, pending)
  }

  /** Terminal-failure counterpart, shared by the stream onFailure and the
   * POST catch: the reply being regenerated was removed before this turn
   * started and no successful completion will replace it. Pops the stash
   * and returns the previous good reply rebuilt as a real message (null
   * when nothing was stashed) — each caller decides where it lands
   * relative to its own error bubble. */
  const takeRestoredVariantMessage = (): Message | null => {
    const pending = pendingVariantsRef.current
    pendingVariantsRef.current = null
    if (!pending || pending.length === 0) return null
    return variantToMessage(pending[pending.length - 1], {}, pending)
  }

  return {
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
  }
}
