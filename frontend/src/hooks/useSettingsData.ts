/**
 * Panel-level settings state extracted from SettingsPanel: the GET
 * /api/settings mount load (settings/loading/loadError), the PUT
 * /api/settings write path (`putSettings`), the shared update-response
 * handler (`applyUpdateResponse`, which owns the restart notice), and the
 * persona-version list (`fetchPersonaVersions` + its mount load). Section
 * form state stays with the section components — this hook exposes
 * `onSettingsLoaded` so the panel can seed lifted form state in the same
 * render that opens the settings-loaded gate, exactly as the inline code
 * did.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { apiFetch } from '../api'
import { t } from '../i18n'

interface ProviderStatus {
  llm: string
  llm_is_dummy: boolean
  tts: string
  stt: string
}

interface PersonaSettings {
  tsundere_level: number
  scientific_formality: number
  safe_mode: boolean
}

export interface SettingsResponse {
  is_first_run: boolean
  providers: ProviderStatus
  openai_api_key_set: boolean
  gemini_api_key_set: boolean
  huggingface_api_token_set: boolean
  llm_model: string
  persona: PersonaSettings
  streaming_enabled: boolean
  audio_enable_voice: boolean
  voice_language: string
  memory_enable_long_term: boolean
  web_search_enabled: boolean
  tools_enabled: boolean
  reminders_enabled: boolean
}

export interface SettingsUpdatePayload {
  llm_provider?: string
  tts_provider?: string
  stt_provider?: string
  llm_model?: string
  openai_api_key?: string
  gemini_api_key?: string
  huggingface_api_token?: string
  huggingface_inference_model?: string
  tsundere_level?: number
  scientific_formality?: number
  safe_mode?: boolean
  streaming_enabled?: boolean
  audio_enable_voice?: boolean
  voice_language?: 'japanese' | 'english'
  memory_enable_long_term?: boolean
  web_search_enabled?: boolean
  tools_enabled?: boolean
  reminders_enabled?: boolean
}

export interface SettingsUpdateResponse {
  settings: SettingsResponse
  needs_restart: boolean
  restart_reason: string | null
}

export interface ProviderTestResponse {
  ok: boolean
  message: string
}

export interface PersonaVersionInfo {
  id: string
  name: string
  created_at: string
  pinned: boolean
  fingerprint: string
  is_current: boolean
  diff_vs_current: string | null
}

/** Pull a human-readable message out of a FastAPI error body: either a plain
 * `{detail: "..."}` or pydantic's 422 shape `{detail: [{msg: "Value error, ..."}]}`.
 * Falls back to the raw text so nothing is ever swallowed. */
const extractErrorDetail = (raw: string): string => {
  if (!raw) return ''
  try {
    const parsed = JSON.parse(raw) as { detail?: unknown }
    const detail = parsed.detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail) && detail.length > 0) {
      const msg = (detail[0] as { msg?: unknown }).msg
      if (typeof msg === 'string') return msg.replace(/^Value error, /, '')
    }
  } catch {
    // Not JSON — show as-is.
  }
  return raw
}

export interface UseSettingsDataOptions {
  /** Invoked with the freshly loaded settings, in the same batch as
   * `setSettings`, so lifted form state (persona sliders, TTS provider)
   * is seeded before the settings-loaded gate first renders. */
  onSettingsLoaded?: (data: SettingsResponse) => void
}

export const useSettingsData = ({ onSettingsLoaded }: UseSettingsDataOptions = {}) => {
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [settings, setSettings] = useState<SettingsResponse | null>(null)
  const [restartNotice, setRestartNotice] = useState<string | null>(null)

  // Persona versions (identity continuity: snapshot / restore / pin)
  const [personaVersions, setPersonaVersions] = useState<PersonaVersionInfo[]>([])
  const [versionsCurrentFp, setVersionsCurrentFp] = useState('')

  // Ref so the mount-load effect below can stay once-only without stale-
  // closure risk when the caller passes an inline callback.
  const onLoadedRef = useRef(onSettingsLoaded)
  useEffect(() => {
    onLoadedRef.current = onSettingsLoaded
  })

  useEffect(() => {
    const controller = new AbortController()
    ;(async () => {
      try {
        const response = await apiFetch('/api/settings', { signal: controller.signal })
        if (!response.ok) throw new Error(`Failed to load settings: ${response.status}`)
        const data: SettingsResponse = await response.json()
        setSettings(data)
        onLoadedRef.current?.(data)
      } catch (error) {
        if (!(error instanceof DOMException && error.name === 'AbortError')) {
          setLoadError(t('Could not load settings — is the server reachable?'))
        }
      } finally {
        setLoading(false)
      }
    })()
    return () => controller.abort()
  }, [])

  const applyUpdateResponse = useCallback((data: SettingsUpdateResponse) => {
    setSettings(data.settings)
    if (data.needs_restart) {
      setRestartNotice(data.restart_reason || t('Restart to apply this change.'))
    }
  }, [])

  const putSettings = useCallback(
    async (payload: SettingsUpdatePayload): Promise<SettingsUpdateResponse> => {
      const response = await apiFetch('/api/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!response.ok) {
        const detail = await response.text().catch(() => '')
        throw new Error(extractErrorDetail(detail) || `Save failed: ${response.status}`)
      }
      return response.json()
    },
    [],
  )

  const fetchPersonaVersions = useCallback(async () => {
    try {
      const response = await apiFetch('/api/persona/versions')
      if (!response.ok) return
      const data = await response.json()
      setPersonaVersions(data.versions ?? [])
      setVersionsCurrentFp(data.current_fingerprint ?? '')
    } catch {
      // Version list is best-effort UI; the settings panel still works.
    }
  }, [])

  useEffect(() => {
    void fetchPersonaVersions()
  }, [fetchPersonaVersions])

  return {
    settings,
    loading,
    loadError,
    restartNotice,
    personaVersions,
    versionsCurrentFp,
    applyUpdateResponse,
    putSettings,
    fetchPersonaVersions,
  }
}
