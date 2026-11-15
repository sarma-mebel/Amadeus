/**
 * Shared helpers for the regenerate/edit/rewind family of turn-mutating
 * actions in ChatInterface. Extracted to de-duplicate the four near-
 * identical rewind-POST-then-classify blocks and the two variant-folding
 * blocks (stream onDone + POST completion).
 */
import { apiFetch } from '../api'
import { resolveUserId } from '../hooks/useSessions'
import type { Message, MessageVariant } from './messages'

/** At most this many variants are kept per message once folded — oldest
 * variants are dropped first so a long regenerate chain doesn't grow the
 * transcript (and localStorage transcript payload) without bound. */
export const MAX_VARIANTS = 5

export type RewindOutcome = 'ok' | 'not_found' | 'error'

/** POSTs /api/sessions/:id/rewind for `turnId`. The backend only ever
 * reports success or a 404 (an 'unavailable' status was never a real
 * server response) — everything else (non-2xx, thrown fetch) is 'error'. */
export const rewindTurn = async (turnId: string): Promise<RewindOutcome> => {
  try {
    const response = await apiFetch(
      `/api/sessions/${encodeURIComponent(localStorage.getItem('session_id') || '')}/rewind`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ turn_id: turnId, user_id: resolveUserId() }),
      },
    )
    if (response.status === 404) return 'not_found'
    if (!response.ok) return 'error'
    return 'ok'
  } catch {
    return 'error'
  }
}

/** Builds a system-role Message ready to append via setMessages. Kept as a
 * pure builder (rather than taking setMessages itself) so call sites stay
 * in control of the single setMessages call and can compose it with other
 * transcript updates in the same update. */
export const buildSystemMessage = (content: string): Message => ({
  role: 'system',
  content,
  timestamp: Date.now(),
})

/** Folds a regenerate's stashed prior variant(s) onto the freshly-completed
 * assistant message identified by `assistantId`, capping the merged list at
 * MAX_VARIANTS (oldest dropped first). No-op (returns `messages` unchanged)
 * when there's nothing pending. Shared by the stream onDone handler and the
 * POST completion handler — both call this right after stampTurn. */
export const foldPendingVariants = (
  messages: Message[],
  assistantId: string | null,
  pending: MessageVariant[] | null,
): Message[] => {
  if (!pending || !assistantId) return messages
  return messages.map((msg) => {
    if (msg.id !== assistantId) return msg
    const merged = [
      ...pending,
      {
        content: msg.content,
        japanese: msg.japanese,
        thought: msg.thought,
        timestamp: msg.timestamp,
      },
    ]
    const capped = merged.slice(Math.max(0, merged.length - MAX_VARIANTS))
    return { ...msg, variants: capped, activeVariant: capped.length - 1 }
  })
}

/** Snapshot of an assistant message's currently-displayed content, used to
 * rebuild a stashed reply as a real Message again after a terminal failure
 * destroys the live one it was regenerating. */
export const variantToMessage = (
  variant: MessageVariant,
  base: Pick<Message, 'turnId' | 'id'>,
  variants: MessageVariant[],
): Message => ({
  role: 'assistant',
  content: variant.content,
  japanese: variant.japanese,
  thought: variant.thought,
  timestamp: variant.timestamp,
  turnId: base.turnId,
  id: base.id,
  variants: variants.length > 1 ? variants : undefined,
  activeVariant: variants.length > 1 ? variants.length - 1 : undefined,
})

/** Stamps a completed turn's server-generated turn_id onto both halves of
 * the exchange: the assistant message (matched by id, falling back to the
 * last assistant message — the WS path always knows assistantId, the POST
 * path doesn't create one) and the nearest preceding user message, which by
 * construction belongs to this turn — always restamped, even if it already
 * carried a (now-stale) turnId from before a regenerate. The scan stops the
 * instant it crosses another assistant message, since anything before that
 * boundary belongs to a different, earlier turn. Regenerate/edit/rewind all
 * key off Message.turnId rather than array position, which shifts on
 * insert/delete. Pure — returns a new array, doesn't mutate. */
export const stampTurn = (prev: Message[], assistantId: string | null, turnId: string): Message[] => {
  let assistantIndex = -1
  if (assistantId) {
    assistantIndex = prev.findIndex((msg) => msg.id === assistantId)
  }
  if (assistantIndex === -1) {
    for (let i = prev.length - 1; i >= 0; i -= 1) {
      if (prev[i].role === 'assistant') {
        assistantIndex = i
        break
      }
    }
  }
  if (assistantIndex === -1) return prev

  let userIndex = -1
  for (let i = assistantIndex - 1; i >= 0; i -= 1) {
    if (prev[i].role === 'assistant') break
    if (prev[i].role === 'user') {
      userIndex = i
      break
    }
  }

  return prev.map((msg, index) => {
    if (index === assistantIndex) return { ...msg, turnId }
    if (index === userIndex) return { ...msg, turnId }
    return msg
  })
}
