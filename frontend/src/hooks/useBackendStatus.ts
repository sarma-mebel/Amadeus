/**
 * Backend-status state extracted from ChatInterface: the /api/status chip
 * feed (refreshBackendStatus, which also carries the once-per-launch update
 * check), the /healthz voice-warmup poll, the update-available toast state,
 * the character-switch listener (re-read status + start a fresh
 * conversation), and the character-declared avatar auto-apply.
 *
 * Deliberately NOT included: llmIsDummy / the first-run onboarding gate
 * (those live with the send path, see useOnboardingGate) and
 * startNewConversation itself — useSessions owns that; this hook only
 * exposes the same ref-bridge ChatInterface always used
 * (startNewConversationRef), which the component assigns once useSessions
 * has been initialized further down.
 */
import { useEffect, useRef, useState } from 'react'
import { apiFetch } from '../api'
import { fetchBackendStatus } from '../statusApi'
import type { BackendStatus } from '../statusApi'
import { runOnceUpdateCheck } from '../updateCheck'
import { CHARACTER_SWITCH_EVENT } from '../charactersApi'
import {
  dispatchModelChange,
  fetchModelRegistry,
  loadModelId,
  saveModelId,
} from '../models'

/** Last character-declared avatar we auto-applied (see character.model_id
 * in /api/status) — the switch happens once per character, so the user's
 * own outfit picks afterwards are respected. */
const CHARACTER_MODEL_APPLIED_KEY = 'kurisu_character_model_v1'

export const useBackendStatus = () => {
  // GPU build loads the voice model in the background after the app opens;
  // /healthz reports voice: "warming" until it's ready.
  const [voiceWarming, setVoiceWarming] = useState(false)
  // Newer release tag from the launch-time update check, if any.
  const [updateAvailable, setUpdateAvailable] = useState<string | null>(null)
  // Runtime provider truth for the toolbar chip; null hides the chip.
  const [backendStatus, setBackendStatus] = useState<BackendStatus | null>(null)
  const refreshBackendStatus = () => {
    void fetchBackendStatus().then((status) => {
      setBackendStatus(status)
      // Launch-time update check rides the first status response: it
      // carries the running backend version (the thing being updated) and
      // the demo_mode gate. Once-per-launch latch inside; 24h localStorage
      // cache keeps the GitHub API call to about one a day. Failure-silent.
      if (status) {
        void runOnceUpdateCheck(status.version, { demoMode: status.demo_mode }).then(
          (tag) => {
            if (tag) setUpdateAvailable(tag)
          },
        )
      }
    })
  }

  // Character switch (settings panel): the backend identity just changed —
  // re-read status (name, model_id -> avatar swap below) and start a fresh
  // conversation so the new character doesn't inherit mid-thread context
  // addressed to the previous one. Ref-indirect because useSessions (the
  // owner of startNewConversation) is initialized further down in
  // ChatInterface, after this hook.
  const startNewConversationRef = useRef<(() => void | Promise<void>) | null>(null)
  useEffect(() => {
    const onSwitched = () => {
      refreshBackendStatus()
      void startNewConversationRef.current?.()
    }
    window.addEventListener(CHARACTER_SWITCH_EVENT, onSwitched)
    return () => window.removeEventListener(CHARACTER_SWITCH_EVENT, onSwitched)
  }, [])

  // Character packs: when the active character declares its avatar
  // (character.model_id in /api/status), select it automatically — once per
  // character, so a manual outfit choice afterwards still wins.
  const characterModelId = backendStatus?.character?.model_id ?? null
  useEffect(() => {
    if (!characterModelId) return
    try {
      if (window.localStorage.getItem(CHARACTER_MODEL_APPLIED_KEY) === characterModelId) return
    } catch {
      return
    }
    void fetchModelRegistry().then((registry) => {
      if (!registry.some((model) => model.id === characterModelId)) return
      try {
        window.localStorage.setItem(CHARACTER_MODEL_APPLIED_KEY, characterModelId)
      } catch {
        // Marker write failed; still switch, we'll just re-check next load.
      }
      if (loadModelId() !== characterModelId) {
        saveModelId(characterModelId)
        dispatchModelChange(characterModelId)
      }
    })
  }, [characterModelId])

  // Poll /healthz while the GPU build's voice model warms up in the
  // background, so the status line can explain why replies are silent.
  // Stops polling once the voice is ready (or reported unavailable).
  useEffect(() => {
    let cancelled = false
    let timer: number | undefined
    const probe = async () => {
      try {
        const response = await apiFetch('/healthz')
        const data = await response.json()
        if (cancelled) return
        // Keep the status chip in sync with warmup progress: this runs on
        // mount and again on every warming re-poll (amber -> green flip).
        // The launch-time update check rides the status refresh (it needs
        // the backend version + demo_mode from /api/status).
        refreshBackendStatus()
        if (data.voice === 'warming') {
          setVoiceWarming(true)
          timer = window.setTimeout(probe, 10000)
          return
        }
        setVoiceWarming(false)
      } catch {
        if (!cancelled) timer = window.setTimeout(probe, 10000)
      }
    }
    probe()
    return () => {
      cancelled = true
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [])

  return {
    backendStatus,
    voiceWarming,
    updateAvailable,
    setUpdateAvailable,
    refreshBackendStatus,
    startNewConversationRef,
  }
}
