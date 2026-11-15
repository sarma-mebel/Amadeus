/**
 * The chat transcript's message shape, shared by ChatInterface and the
 * hooks extracted from it (useSessions builds Message rows from server
 * transcripts). Moved verbatim out of ChatInterface.tsx.
 */
import type { SendErrorKind } from './errors'

export type MessageRole = 'user' | 'assistant' | 'agent' | 'system'

export interface Message {
  role: MessageRole
  content: string
  timestamp: number
  agentName?: string
  round?: number
  id?: string
  // Marks the synthetic "Error: Could not get response" assistant bubble and
  // the mid-stream "(connection lost…)" marker so they can get a distinct
  // muted-red treatment (Tier 4a). transcript.ts round-trips this fine: it
  // JSON.stringifies Message objects as-is and only type-checks the
  // required fields on load, so unknown/extra properties like this one
  // survive save/restore without any change to transcript.ts.
  isError?: boolean
  // What failed: 'network' (backend unreachable), 'server' (HTTP error
  // status), 'stream' (WebSocket died mid-reply, partial content kept).
  errorKind?: SendErrorKind
  // The original user text of a failed turn, so the error bubble can offer
  // a Retry that re-sends through the normal send path.
  retryText?: string
  // Accumulated Japanese (the text Kurisu actually speaks), shown as a subtitle
  // under the English reply. Streamed per-sentence via onSentence's frame.ja;
  // round-trips through transcript.ts like other extra fields.
  japanese?: string
  // Unspoken [thought] line (web.streaming_inner_thoughts): rendered as a
  // faint ghost line, never voiced; round-trips like `japanese`.
  thought?: string
  // Server-generated turn id (ChatResponse.turn_id / the WS done frame's
  // response.turn_id). Present on an assistant message and the user message
  // that started its turn once stampTurn has run; absent on pre-feature
  // history and while a turn is still in flight. Regenerate/edit/rewind all
  // key off this rather than array position, which shifts on
  // insert/delete.
  turnId?: string
  // Regenerate history for an assistant message: earlier replies to the
  // same prompt, oldest first. The live message's own content/japanese/
  // thought represents the newest variant (mirrored into the array's last
  // slot only when swiping away from it — see swipeVariant in
  // ChatInterface.tsx).
  variants?: MessageVariant[]
  // Index into `variants` the message currently displays. Only meaningful
  // when `variants` is set.
  activeVariant?: number
}

/** One alternative reply captured by regenerate/swipe. Round-trips through
 * transcript.ts like other extra Message fields (plain JSON, no special
 * handling needed). */
export interface MessageVariant {
  content: string
  japanese?: string
  thought?: string
  timestamp: number
}
