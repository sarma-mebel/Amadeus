/** Client for the character-switching endpoints (/api/characters). */

import { apiFetch } from './api'

export interface CharacterEntry {
  id: string
  name: string
  full_name: string
  model_id: string | null
  has_voice: boolean
  builtin: boolean
  active: boolean
}

export interface SwitchResult {
  active: string
  character: {
    name: string
    full_name: string
    custom: boolean
    model_id: string | null
  }
  tts_warming: boolean
}

/** Null on any failure (older backend without the endpoint, network error)
 * so callers hide the character picker entirely. */
export const fetchCharacters = async (): Promise<CharacterEntry[] | null> => {
  try {
    const response = await apiFetch('/api/characters')
    if (!response.ok) return null
    const body = (await response.json()) as { characters?: unknown }
    if (!Array.isArray(body.characters)) return null
    return body.characters as CharacterEntry[]
  } catch {
    return null
  }
}

export const switchCharacter = async (id: string): Promise<SwitchResult> => {
  const response = await apiFetch('/api/characters/switch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id }),
  })
  if (!response.ok) {
    let detail = `Switch failed (${response.status})`
    try {
      const body = (await response.json()) as { detail?: string }
      if (body.detail) detail = body.detail
    } catch {
      // keep the generic message
    }
    throw new Error(detail)
  }
  return (await response.json()) as SwitchResult
}

/** Dispatched on window after a successful switch; ChatInterface listens and
 * refreshes identity-dependent state (status, session). */
export const CHARACTER_SWITCH_EVENT = 'kurisu:character-switched'

export const dispatchCharacterSwitched = (): void => {
  if (typeof window === 'undefined') return
  window.dispatchEvent(new CustomEvent(CHARACTER_SWITCH_EVENT))
}
