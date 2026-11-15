/** Pure choreography helpers for the desktop-pet's voice turn
 * (usePetChat.ts's sendVoiceTurn), extracted the same way chat/voiceTurn.ts
 * and chat/duplexTurn.ts carry the transport-layer logic out of their
 * hooks/components. This module owns none of the transport (that's
 * runVoiceTurn) — only the small pieces of per-turn bookkeeping that don't
 * need React state to compute: the live caption accumulator (sentence
 * frames append into one running string, since the pet has no transcript
 * and the caption must keep pace with the voice instead of only appearing
 * at 'done'), stripping the server's internal " [interrupted]" memory
 * marker before it reaches the caption, minting a fresh acknowledgment
 * turn id per send, and deciding whether the read-together page frame
 * should ride along on this send.
 *
 * Deliberately NOT touching: doze wake, gesture/expression dispatch
 * (already inside runVoiceTurn/frameDemux), busy-state guarding, or
 * anything else that reads/writes a ref or piece of React state directly —
 * those stay in usePetChat.ts, which owns the actual mutable turn state.
 */
import { type ChatImageAttachment } from './payload'

/** Accumulates sentence-frame English text into one running caption string,
 * sentence by sentence, so the pet's caption never trails the voice (which
 * starts on the first audio frame, long before the 'done' frame lands).
 * Mirrors the exact join usePetChat used inline: a leading sentence starts
 * the string, every later one is appended with a single space. */
export const appendLiveReply = (current: string, sentenceText: string): string =>
  current ? `${current} ${sentenceText}` : sentenceText

/** The server's reply text on a cancelled turn carries the internal
 * " [interrupted]" memory marker (see amadeus/web/streaming/turn.py) — the
 * pet caption is the one surface that renders response.reply raw, so this
 * strips it before display. Returns null for an empty/whitespace-only
 * result (matching usePetChat's `reply || null` pattern) so callers can
 * treat "nothing to show" uniformly. */
export const stripInterruptedMarker = (reply: string | null | undefined): string | null => {
  const cleaned = (reply ?? '').replace(/\s*\[interrupted\]\s*$/, '')
  return cleaned || null
}

/** Mints a fresh, unique acknowledgment.ts turn id from a monotonically
 * increasing counter ref value (the caller owns the ref/increment; this
 * just names the id consistently so every call site formats it the same
 * way). `nextCount` is the value AFTER incrementing, matching
 * `++ackTurnRef.current` at the call site. */
export const petAckTurnId = (nextCount: number): string => `pet-${nextCount}`

/** Read-together auto-attach: while the mode is active, the most recent
 * full-res page frame rides along on every turn so "what do you think of
 * this panel?" just works without a manual attach. Deliberately re-read
 * every send rather than cleared after one use — a follow-up question
 * about the SAME page should still see it, and the source ref naturally
 * moves on to the next page once one is detected. Returns undefined (not
 * an empty array) when nothing should attach, matching runVoiceTurn's
 * optional `images` param. */
export const resolveTurnImages = (
  readTogetherActive: boolean,
  lastPageFrame: ChatImageAttachment | null,
): ChatImageAttachment[] | undefined =>
  readTogetherActive && lastPageFrame ? [lastPageFrame] : undefined
