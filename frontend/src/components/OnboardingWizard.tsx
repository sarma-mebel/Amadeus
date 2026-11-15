/**
 * First-run wizard, rendered inside the settings overlay instead of the
 * normal settings body while `isOnboarding` is true.
 *
 * Three steps:
 *   1. "Choose a brain"   — pick Gemini (recommended) / OpenAI / offline demo.
 *   2. "Connect"          — paste a key, test it (POST /api/settings/test),
 *                           save it (PUT /api/settings). Skipped for the
 *                           offline demo.
 *   3. "Make her yours"   — voice / long-term memory / tools toggles, saved
 *                           in one PUT on finish.
 *
 * Every step keeps a "Skip for now" escape that behaves exactly like the old
 * banner's skip (markOnboardingDismissed + close).
 */

import { useEffect, useState } from 'react'
import { apiFetch } from '../api'
import { t, useLocale } from '../i18n'
import { markOnboardingDismissed } from '../onboarding'
import { isTauri } from '../platform'
import { setAutostartEnabled as applyAutostart } from '../autostart'
import type {
  ProviderTestResponse,
  SettingsResponse,
  SettingsUpdatePayload,
  SettingsUpdateResponse,
} from './SettingsPanel'
import {
  normalizeTtsProvider,
  QWEN_TTS_OPTION,
  TTS_PROVIDER_OPTIONS,
  type TtsProviderChoice,
} from './ttsOptions'
import './OnboardingWizard.css'

type WizardProvider = 'gemini' | 'openai' | 'dummy'

interface ProviderCard {
  value: WizardProvider
  title: string
  tagline: string
  recommended?: boolean
}

const PROVIDER_CARDS: ProviderCard[] = [
  {
    value: 'gemini',
    title: 'Gemini',
    tagline: 'Free API key from Google AI Studio. The sensible choice — even I would pick it.',
    recommended: true,
  },
  {
    value: 'openai',
    title: 'OpenAI',
    tagline: 'GPT models. Bring your own paid API key.',
  },
  {
    value: 'dummy',
    title: 'Offline demo',
    tagline: 'No key, no network — just a hollow shell of me. Fine for a quick look around.',
  },
]

interface OnboardingWizardProps {
  /** Called after the final save succeeds (onboarding already dismissed). */
  onFinished: () => void
  /** The existing skip path: markOnboardingDismissed + close, handled by the caller. */
  onSkip: () => void
}

const putSettings = async (payload: SettingsUpdatePayload): Promise<SettingsUpdateResponse> => {
  const response = await apiFetch('/api/settings', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    const detail = await response.text().catch(() => '')
    throw new Error(detail || `Save failed: ${response.status}`)
  }
  return response.json()
}

const OnboardingWizard = ({ onFinished, onSkip }: OnboardingWizardProps) => {
  useLocale()
  const [step, setStep] = useState<1 | 2 | 3>(1)
  const [provider, setProvider] = useState<WizardProvider>('gemini')

  // Step 2 state
  const [keyInput, setKeyInput] = useState('')
  const [testPending, setTestPending] = useState(false)
  const [testResult, setTestResult] = useState<ProviderTestResponse | null>(null)
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [keySaved, setKeySaved] = useState(false)

  // Step 3 state
  const [voiceEnabled, setVoiceEnabled] = useState(true)
  const [memoryEnabled, setMemoryEnabled] = useState(true)
  const [toolsEnabled, setToolsEnabled] = useState(false)
  const [ttsProvider, setTtsProvider] = useState<TtsProviderChoice>('melotts_onnx')
  const [voiceLanguage, setVoiceLanguage] = useState<'japanese' | 'english'>('japanese')
  // The GPU cloned voice is only offered when the backend already runs it —
  // same rule as SettingsPanel (selecting it in a CPU build breaks the voice).
  const [qwenAvailable, setQwenAvailable] = useState(false)
  const [finishing, setFinishing] = useState(false)
  const [finishError, setFinishError] = useState<string | null>(null)
  const [finishNotice, setFinishNotice] = useState<string | null>(null)
  // "Start with Windows" (Tauri only, default unchecked — autostart is an
  // opt-IN, not a surprise). Applied via set_autostart on finish, alongside
  // the pet-default marker write and the immediate pet-window open below.
  // Failure is a small inline note, never a blocker: autostart is a nicety,
  // not something worth stalling first-run over.
  const [autostartEnabled, setAutostartEnabled] = useState(false)
  const [autostartNotice, setAutostartNotice] = useState<string | null>(null)

  // Prefill the voice step from whatever the backend is actually running, so
  // "Finish" without touching anything never downgrades a working voice.
  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const response = await apiFetch('/api/settings')
        if (!response.ok) return
        const data = (await response.json()) as SettingsResponse
        if (cancelled) return
        const current = normalizeTtsProvider(data.providers?.tts ?? '')
        setTtsProvider(current)
        setQwenAvailable(current === 'qwen3_tts')
        setVoiceLanguage(data.voice_language === 'english' ? 'english' : 'japanese')
        setVoiceEnabled(Boolean(data.audio_enable_voice))
      } catch {
        // Defaults are fine — the backend just won't be prefilled.
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  const ttsOptions = qwenAvailable ? [QWEN_TTS_OPTION, ...TTS_PROVIDER_OPTIONS] : TTS_PROVIDER_OPTIONS

  const trimmedKey = keyInput.trim()
  // Loose sanity hint only — never blocks. Gemini keys start with "AIza",
  // OpenAI keys with "sk-". A mismatch usually means the wrong key was pasted.
  const keyLooksWrong =
    trimmedKey.length >= 8 &&
    ((provider === 'gemini' && !trimmedKey.startsWith('AIza')) ||
      (provider === 'openai' && !trimmedKey.startsWith('sk-')))

  const selectProvider = (value: WizardProvider) => {
    setProvider(value)
    // A different brain invalidates any key test/save done for the old one.
    setKeyInput('')
    setTestResult(null)
    setSaveError(null)
    setKeySaved(false)
  }

  const handleStep1Next = () => {
    setStep(provider === 'dummy' ? 3 : 2)
  }

  const handleTest = async () => {
    setTestPending(true)
    setTestResult(null)
    try {
      const response = await apiFetch('/api/settings/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          kind: 'llm',
          provider,
          api_key: keyInput.trim() || null,
          model: null,
        }),
      })
      const data: ProviderTestResponse = await response.json()
      setTestResult(data)
    } catch {
      setTestResult({ ok: false, message: t('Test request failed — is the server reachable?') })
    } finally {
      setTestPending(false)
    }
  }

  const handleSaveKey = async () => {
    setSaving(true)
    setSaveError(null)
    try {
      const payload: SettingsUpdatePayload = { llm_provider: provider }
      if (provider === 'openai') payload.openai_api_key = keyInput.trim()
      else payload.gemini_api_key = keyInput.trim()
      await putSettings(payload)
      setKeySaved(true)
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : t('Could not save the key.'))
    } finally {
      setSaving(false)
    }
  }

  const handleFinish = async () => {
    setFinishing(true)
    setFinishError(null)
    try {
      const payload: SettingsUpdatePayload = {
        audio_enable_voice: voiceEnabled,
        memory_enable_long_term: memoryEnabled,
        tools_enabled: toolsEnabled,
      }
      if (voiceEnabled) {
        payload.tts_provider = ttsProvider
        payload.voice_language = voiceLanguage
      }
      // The offline path never visited step 2, so persist the choice here.
      if (provider === 'dummy') payload.llm_provider = 'dummy'
      const result = await putSettings(payload)
      // markOnboardingDismissed() writes the pet-default marker (via
      // petDefault.ts) so the NEXT launch boots straight to pet mode —
      // unchanged. The two calls below are additive: show her on the
      // desktop THIS run too, no relaunch needed, and apply the autostart
      // choice. Both are Tauri-only and strictly best-effort (try/catch,
      // non-fatal) — a failure here must never block finishing onboarding,
      // and the tray/Settings toggle remain the recovery path either way.
      markOnboardingDismissed()
      if (isTauri()) {
        try {
          const { invoke } = await import('@tauri-apps/api/core')
          await invoke('toggle_pet_window')
        } catch {
          // Non-fatal — she just won't appear until the next launch or a
          // manual tray click; the marker write above still took effect.
        }
        if (autostartEnabled) {
          const outcome = await applyAutostart(true)
          if (!outcome.ok) {
            // Inline note only — never a reason to stall first-run.
            setAutostartNotice(
              t('Could not enable "Start with Windows" ({error}) — you can try again in Settings.', {
                error: outcome.error,
              }),
            )
          }
        }
      }
      if (result.needs_restart) {
        // Don't vanish silently when the save only takes effect after a
        // restart — the user would sit talking to the old configuration.
        setFinishNotice(
          `${t('Saved. Restart Amadeus to apply')}${result.restart_reason ? `: ${result.restart_reason}` : '.'}`,
        )
        return
      }
      onFinished()
    } catch (error) {
      setFinishError(error instanceof Error ? error.message : t('Could not save preferences.'))
    } finally {
      setFinishing(false)
    }
  }

  return (
    <div className="onboarding-wizard">
      <div className="onboarding-progress" aria-label={t('Step {step} of 3', { step })}>
        {[1, 2, 3].map((n) => (
          <span
            key={n}
            className={`onboarding-progress-dot${n === step ? ' active' : ''}${n < step ? ' done' : ''}`}
          />
        ))}
      </div>

      {step === 1 && (
        <section className="onboarding-step">
          <p className="onboarding-greeting" lang="ja">
            ……ふん、あなたが助手ね。
          </p>
          <h3>{t('Choose a brain')}</h3>
          <p className="onboarding-copy">
            {t(
              "I'm Makise Kurisu — well, an Amadeus reconstruction of her. Pick what I'll think with. It takes a minute, so don't dawdle.",
            )}
          </p>
          <div className="onboarding-cards" role="radiogroup" aria-label={t('LLM provider')}>
            {PROVIDER_CARDS.map((card) => (
              // A div (not a button) because the Gemini card nests a real
              // <a> link — interactive content is invalid inside <button>.
              <div
                key={card.value}
                role="radio"
                aria-checked={provider === card.value}
                tabIndex={0}
                className={`onboarding-card${provider === card.value ? ' selected' : ''}`}
                onClick={() => selectProvider(card.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault()
                    selectProvider(card.value)
                  }
                }}
              >
                <span className="onboarding-card-title">
                  {t(card.title)}
                  {card.recommended && <span className="onboarding-card-badge">{t('Recommended')}</span>}
                </span>
                <span className="onboarding-card-tagline">{t(card.tagline)}</span>
                {card.value === 'gemini' && (
                  <a
                    href="https://aistudio.google.com/app/apikey"
                    target="_blank"
                    rel="noreferrer"
                    className="settings-key-help-link"
                    onClick={(event) => event.stopPropagation()}
                  >
                    {t('Get a free Gemini API key →')}
                  </a>
                )}
              </div>
            ))}
          </div>
          <div className="onboarding-actions">
            <button type="button" className="settings-skip-button" onClick={onSkip}>
              {t('Skip for now')}
            </button>
            <button type="button" className="settings-primary-button" onClick={handleStep1Next}>
              {t('Next')}
            </button>
          </div>
        </section>
      )}

      {step === 2 && (
        <section className="onboarding-step">
          <h3>{t('Connect')}</h3>
          <p className="onboarding-copy">
            {provider === 'openai'
              ? t(
                  "Paste your OpenAI API key. I'll check it works before we go on — I don't run on wishful thinking.",
                )
              : t(
                  "Paste your Gemini API key. I'll check it works before we go on — I don't run on wishful thinking.",
                )}
          </p>
          <div className="settings-field-group">
            <label className="settings-label" htmlFor="onboarding-api-key-input">
              {provider === 'openai' ? t('OpenAI API key') : t('Gemini API key')}
            </label>
            <input
              id="onboarding-api-key-input"
              type="password"
              autoComplete="off"
              placeholder={t('Paste your API key')}
              value={keyInput}
              onChange={(event) => {
                setKeyInput(event.target.value)
                setTestResult(null)
                setKeySaved(false)
              }}
            />
            {provider === 'gemini' && (
              <a
                href="https://aistudio.google.com/app/apikey"
                target="_blank"
                rel="noreferrer"
                className="settings-key-help-link"
              >
                {t('Get a free Gemini API key →')}
              </a>
            )}
            {keyLooksWrong && (
              <div className="onboarding-key-hint" role="note">
                {provider === 'gemini'
                  ? t('That doesn’t look like a Gemini key — they start with "AIza". Sure you copied the right one?')
                  : t('That doesn’t look like an OpenAI key — they start with "sk-". Sure you copied the right one?')}
              </div>
            )}
            <div className="settings-action-row">
              <button
                type="button"
                className="settings-secondary-button"
                onClick={handleTest}
                disabled={testPending || !keyInput.trim()}
              >
                {testPending ? t('Testing…') : t('Test')}
              </button>
              <button
                type="button"
                className="settings-secondary-button"
                onClick={handleSaveKey}
                disabled={saving || !keyInput.trim()}
              >
                {saving ? t('Saving…') : keySaved ? t('Saved ✓') : t('Save')}
              </button>
            </div>
            {testResult && (
              <div className={`settings-test-result ${testResult.ok ? 'ok' : 'fail'}`}>
                {testResult.message}
              </div>
            )}
            {saveError && <div className="settings-status-line error">{saveError}</div>}
          </div>
          <div className="onboarding-actions">
            <button type="button" className="settings-skip-button" onClick={onSkip}>
              {t('Skip for now')}
            </button>
            <div className="onboarding-actions-right">
              <button
                type="button"
                className="settings-secondary-button"
                onClick={() => setStep(1)}
              >
                {t('Back')}
              </button>
              <button
                type="button"
                className="settings-primary-button"
                onClick={() => setStep(3)}
                disabled={!keySaved}
              >
                {t('Next')}
              </button>
            </div>
          </div>
        </section>
      )}

      {step === 3 && (
        <section className="onboarding-step">
          <h3>{t('Make her yours')}</h3>
          <p className="onboarding-copy">
            {t(
              "Last step. Decide what I'm allowed to do — you can change all of this later in Settings.",
            )}
          </p>
          <div className="onboarding-toggles">
            <label className="onboarding-toggle" htmlFor="onboarding-voice-toggle">
              <input
                id="onboarding-voice-toggle"
                type="checkbox"
                checked={voiceEnabled}
                onChange={(event) => setVoiceEnabled(event.target.checked)}
              />
              <span className="onboarding-toggle-text">
                <span className="onboarding-toggle-title">{t('Voice replies')}</span>
                <span className="onboarding-toggle-hint">{t('I speak my answers out loud.')}</span>
              </span>
            </label>
            {voiceEnabled && (
              <div className="onboarding-voice-config">
                <div className="settings-field-group">
                  <label className="settings-label" htmlFor="onboarding-tts-select">
                    {t('Voice engine')}
                  </label>
                  <select
                    id="onboarding-tts-select"
                    value={ttsProvider}
                    onChange={(event) => setTtsProvider(event.target.value as TtsProviderChoice)}
                  >
                    {ttsOptions.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="settings-field-group">
                  <label className="settings-label" htmlFor="onboarding-voice-language">
                    {t('Spoken language')}
                  </label>
                  <select
                    id="onboarding-voice-language"
                    value={voiceLanguage}
                    onChange={(event) =>
                      setVoiceLanguage(event.target.value === 'english' ? 'english' : 'japanese')
                    }
                  >
                    <option value="japanese">{t('Japanese (with English subtitles)')}</option>
                    <option value="english">{t('English')}</option>
                  </select>
                </div>
              </div>
            )}
            <label className="onboarding-toggle" htmlFor="onboarding-memory-toggle">
              <input
                id="onboarding-memory-toggle"
                type="checkbox"
                checked={memoryEnabled}
                onChange={(event) => setMemoryEnabled(event.target.checked)}
              />
              <span className="onboarding-toggle-text">
                <span className="onboarding-toggle-title">{t('Long-term memory')}</span>
                <span className="onboarding-toggle-hint">
                  {t('I remember facts about you across sessions (stored locally).')}
                </span>
              </span>
            </label>
            <label className="onboarding-toggle" htmlFor="onboarding-tools-toggle">
              <input
                id="onboarding-tools-toggle"
                type="checkbox"
                checked={toolsEnabled}
                onChange={(event) => setToolsEnabled(event.target.checked)}
              />
              <span className="onboarding-toggle-text">
                <span className="onboarding-toggle-title">{t('Tools')}</span>
                <span className="onboarding-toggle-hint">
                  {t('Calculator, code interpreter, paper search — sandboxed.')}
                </span>
              </span>
            </label>
            {isTauri() && (
              <label className="onboarding-toggle" htmlFor="onboarding-autostart-toggle">
                <input
                  id="onboarding-autostart-toggle"
                  type="checkbox"
                  checked={autostartEnabled}
                  onChange={(event) => setAutostartEnabled(event.target.checked)}
                />
                <span className="onboarding-toggle-text">
                  <span className="onboarding-toggle-title">{t('Start with Windows')}</span>
                  <span className="onboarding-toggle-hint">
                    {t('She launches automatically when you log in. Off by default — turn it on any time in Settings.')}
                  </span>
                </span>
              </label>
            )}
          </div>
          {finishError && <div className="settings-status-line error">{finishError}</div>}
          {finishNotice && <div className="settings-status-line">{finishNotice}</div>}
          {autostartNotice && <div className="settings-status-line error">{autostartNotice}</div>}
          <div className="onboarding-actions">
            <button type="button" className="settings-skip-button" onClick={onSkip}>
              {t('Skip for now')}
            </button>
            <div className="onboarding-actions-right">
              <button
                type="button"
                className="settings-secondary-button"
                onClick={() => setStep(provider === 'dummy' ? 1 : 2)}
              >
                {t('Back')}
              </button>
              {finishNotice ? (
                <button type="button" className="settings-primary-button" onClick={onFinished}>
                  {t('Close')}
                </button>
              ) : (
                <button
                  type="button"
                  className="settings-primary-button"
                  onClick={handleFinish}
                  disabled={finishing}
                >
                  {finishing ? t('Saving…') : t('Finish')}
                </button>
              )}
            </div>
          </div>
        </section>
      )}
    </div>
  )
}

export default OnboardingWizard
