/**
 * First-run/onboarding gate extracted from ChatInterface — the pieces
 * usePresence deliberately declined to absorb (see usePresence.ts): the
 * shared /api/settings probe (llmIsDummy + its live ref mirror), the
 * mount-time first-run check that auto-opens the settings wizard, the
 * settings-modal open/onboarding flags, the post-onboarding regreet, and
 * ensureRealLlmConfigured (the send path's "still on the dummy provider"
 * gate). Its outputs feed usePresence (llmIsDummyRef, waitForFirstRunProbe)
 * and the send path (ensureRealLlmConfigured).
 *
 * Deliberately NOT included: the transcript (setMessages/messages are
 * injected — system notices land in the component-owned array) and the
 * greeting request itself (tryRequestGreeting is usePresence's; injected
 * lazily since usePresence is initialized after this hook).
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import type { Dispatch, SetStateAction } from 'react'
import { apiFetch } from '../api'
import { isOnboardingDismissed } from '../onboarding'
import type { SettingsResponse } from '../components/SettingsPanel'
import type { Message } from '../chat/messages'

export interface UseOnboardingGateOptions {
  /** Component-owned transcript — read to decide whether the skipped
   * greeting truly never landed (regreet path), written for the offline-
   * demo-mode system notice. */
  messages: Message[]
  setMessages: Dispatch<SetStateAction<Message[]>>
  /** usePresence's greeting request — injected lazily (arrow over the
   * later binding) because usePresence is wired up after this hook, using
   * this hook's outputs. */
  tryRequestGreeting: () => void
}

export const useOnboardingGate = ({
  messages,
  setMessages,
  tryRequestGreeting,
}: UseOnboardingGateOptions) => {
  const [llmIsDummy, setLlmIsDummy] = useState(false)
  // Live mirror of llmIsDummy for the intro-mode activation path, which
  // reads this from inside an async closure that may run after the state
  // that captured `llmIsDummy` has gone stale (item 5).
  const llmIsDummyRef = useRef(llmIsDummy)
  useEffect(() => {
    llmIsDummyRef.current = llmIsDummy
  }, [llmIsDummy])
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [settingsIsOnboarding, setSettingsIsOnboarding] = useState(false)
  const onboardingCheckedRef = useRef(false)
  // Resolves once the first /api/settings probe (mount-time) has updated
  // llmIsDummy/llmIsDummyRef — intro-mode activation awaits this so it
  // never judges "is this a real provider" off the state's initial `false`
  // default before the probe has actually answered (item 5).
  const probeDoneRef = useRef<Promise<void> | null>(null)

  /** Shared /api/settings probe: updates llmIsDummy (+ its ref) from the
   * response and returns the resolved dummy state (null if unreachable —
   * callers treat that as "assume real" like the pre-existing best-effort
   * behavior did). Used by the mount-time first-run check and by the
   * onboarding-wizard-close regreet path (item 4), so both read the exact
   * same truth instead of the wizard-close path trusting a stale
   * mount-time value. */
  const probeSettings = useCallback(async (): Promise<
    { isDummy: boolean; data: SettingsResponse } | null
  > => {
    try {
      const response = await apiFetch('/api/settings')
      if (!response.ok) return null
      const data = (await response.json()) as SettingsResponse
      const isDummy = Boolean(data.providers?.llm_is_dummy)
      setLlmIsDummy(isDummy)
      llmIsDummyRef.current = isDummy
      return { isDummy, data }
    } catch {
      return null
    }
  }, [])

  // Wizard handoff: the greeting effect skipped its request while the
  // backend was still on the dummy provider (first-run onboarding). Once
  // Settings closes after that onboarding flow, re-probe /api/settings (the
  // mount-time probe's llmIsDummy may be stale — a save in the wizard can
  // hot-swap providers) and, if the LLM is now real and no assistant
  // message has appeared yet (so the skipped greeting truly never landed),
  // ask again.
  const maybeRegreetAfterOnboarding = useCallback(
    async (wasOnboarding: boolean) => {
      if (!wasOnboarding) return
      const result = await probeSettings()
      if (!result || result.isDummy) return
      const hasAssistantMessage = messages.some((message) => message.role === 'assistant')
      if (hasAssistantMessage) return
      tryRequestGreeting()
    },
    [messages, probeSettings, tryRequestGreeting],
  )

  // First-run flow: peek at /api/settings once on mount and auto-open the
  // settings modal if the backend is still on the offline dummy fallback
  // and the user hasn't dismissed onboarding before. Always runs (even when
  // onboarding was already dismissed) so probeDoneRef reliably resolves —
  // intro-mode activation awaits it before trusting llmIsDummyRef.
  useEffect(() => {
    if (onboardingCheckedRef.current) return
    onboardingCheckedRef.current = true
    probeDoneRef.current = (async () => {
      const result = await probeSettings()
      if (result && result.data?.is_first_run && !isOnboardingDismissed()) {
        setSettingsIsOnboarding(true)
        setSettingsOpen(true)
      }
    })()
  }, [probeSettings])

  /** Resolves once the mount-time first-run probe has updated
   * llmIsDummyRef — usePresence's intro-mode activation awaits this before
   * trusting the ref. */
  const waitForFirstRunProbe = useCallback(
    () => probeDoneRef.current ?? Promise.resolve(),
    [],
  )

  const ensureRealLlmConfigured = async (): Promise<boolean> => {
    try {
      const response = await apiFetch('/api/settings')
      if (!response.ok) return true
      const data = (await response.json()) as SettingsResponse
      const isDummy = Boolean(data.providers?.llm_is_dummy || data.is_first_run)
      setLlmIsDummy(isDummy)
      llmIsDummyRef.current = isDummy
      if (!isDummy) return true
      // An explicitly configured dummy (the wizard's "offline demo" choice)
      // chats with canned replies under the persistent banner — hijacking
      // every send with the setup modal would fight a choice the user just
      // made. Only an unconfigured/broken provider blocks the turn.
      if (data.providers?.llm === 'dummy' && isOnboardingDismissed()) return true
    } catch {
      return true
    }

    setSettingsIsOnboarding(true)
    setSettingsOpen(true)
    setMessages((prev) => {
      const last = prev[prev.length - 1]
      if (last?.role === 'system' && last.content.includes('offline demo mode')) {
        return prev
      }
      return [
        ...prev,
        {
          role: 'system',
          content: 'Kurisu is still in offline demo mode. Add a Gemini or OpenAI API key in settings, then send the message again.',
          timestamp: Date.now(),
        },
      ]
    })
    return false
  }

  return {
    llmIsDummy,
    llmIsDummyRef,
    settingsOpen,
    setSettingsOpen,
    settingsIsOnboarding,
    setSettingsIsOnboarding,
    probeSettings,
    maybeRegreetAfterOnboarding,
    waitForFirstRunProbe,
    ensureRealLlmConfigured,
  }
}
