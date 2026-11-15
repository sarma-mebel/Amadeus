/** The one request payload shape shared by POST /api/chat and the WS stream. */

export interface ChatImageAttachment {
  data: string
  mime: string
}

export interface ChatPayloadInput {
  message: string
  sessionId: string | null
  userId: string
  audioEnabled: boolean
  /** Vision: at most one user-shared image riding on this turn. */
  images?: ChatImageAttachment[]
}

export const buildChatPayload = (input: ChatPayloadInput) => ({
  message: input.message,
  session_id: input.sessionId,
  user_id: input.userId,
  audio_enabled: input.audioEnabled,
  images: input.images && input.images.length > 0 ? input.images : undefined,
})
