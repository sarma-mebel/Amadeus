/**
 * Transcript continuity: persist recent chat history to localStorage so a
 * reload (or reopening the desktop app) restores the conversation instead
 * of a cold, empty chat every time — the "load save" half of session
 * continuity (the other half, the greeting referencing last time, is
 * already handled server-side via the user profile's last_episode).
 */

export interface StoredMessage {
  role: 'user' | 'assistant' | 'agent' | 'system'
  content: string
  timestamp: number
  agentName?: string
  round?: number
  id?: string
}

const STORAGE_KEY = 'kurisu_transcript_v1'
const MAX_STORED_MESSAGES = 50

export const loadStoredTranscript = (): StoredMessage[] => {
  if (typeof window === 'undefined') return []
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed.filter(
      (entry): entry is StoredMessage =>
        entry &&
        typeof entry === 'object' &&
        typeof entry.content === 'string' &&
        typeof entry.timestamp === 'number' &&
        typeof entry.role === 'string',
    )
  } catch {
    return []
  }
}

export const saveTranscript = (messages: StoredMessage[]): void => {
  if (typeof window === 'undefined') return
  try {
    const trimmed = messages.slice(-MAX_STORED_MESSAGES)
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed))
  } catch {
    // Storage can throw (quota, private mode) — losing transcript
    // persistence isn't worth surfacing an error over.
  }
}

export const clearStoredTranscript = (): void => {
  if (typeof window === 'undefined') return
  window.localStorage.removeItem(STORAGE_KEY)
}
