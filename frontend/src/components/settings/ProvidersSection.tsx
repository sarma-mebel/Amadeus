/**
 * "Providers" settings section: LLM provider/model/key, TTS provider +
 * voice language + key, connection tests, and the combined save. The TTS
 * provider choice is lifted to SettingsPanel because a persona-version
 * restore writes it back; everything else (LLM choice, key inputs, test
 * results, save state) is local form state seeded from the loaded settings.
 */
import { useMemo, useState } from 'react'
import { apiFetch } from '../../api'
import { t } from '../../i18n'
import { markOnboardingDismissed } from '../../onboarding'
import {
  QWEN_TTS_OPTION,
  TTS_PROVIDER_OPTIONS,
  type TtsProviderChoice,
} from '../ttsOptions'
import type {
  ProviderTestResponse,
  SettingsResponse,
  SettingsUpdatePayload,
  SettingsUpdateResponse,
} from '../../hooks/useSettingsData'

type LlmProviderChoice = 'openai' | 'gemini' | 'dummy'

const LLM_PROVIDER_OPTIONS: { value: LlmProviderChoice; label: string }[] = [
  { value: 'openai', label: 'OpenAI' },
  { value: 'gemini', label: 'Gemini' },
  { value: 'dummy', label: 'Use offline (no API key)' },
]

const normalizeLlmProvider = (value: string): LlmProviderChoice => {
  if (value === 'openai' || value === 'gemini') return value
  return 'dummy'
}

const providerNeedsKey = (provider: LlmProviderChoice | TtsProviderChoice) =>
  provider === 'openai' || provider === 'gemini' || provider === 'huggingface_inference'

type TestState = { pending: boolean; result: ProviderTestResponse | null }

interface ProvidersSectionProps {
  settings: SettingsResponse
  ttsProvider: TtsProviderChoice
  setTtsProvider: (value: TtsProviderChoice) => void
  putSettings: (payload: SettingsUpdatePayload) => Promise<SettingsUpdateResponse>
  applyUpdateResponse: (data: SettingsUpdateResponse) => void
}

const ProvidersSection = ({
  settings,
  ttsProvider,
  setTtsProvider,
  putSettings,
  applyUpdateResponse,
}: ProvidersSectionProps) => {
  // Provider form state
  const [llmProvider, setLlmProvider] = useState<LlmProviderChoice>(() =>
    normalizeLlmProvider(settings.providers.llm),
  )
  // True when the backend loaded with the Qwen3 cloned voice (GPU build):
  // keeps the option in the dropdown so the user can switch back to it.
  const [qwenVoiceAvailable] = useState(() => settings.providers.tts === 'qwen3_tts')
  const [voiceLanguage, setVoiceLanguage] = useState<'japanese' | 'english'>(() =>
    settings.voice_language === 'english' ? 'english' : 'japanese',
  )
  const [llmModel, setLlmModel] = useState(() => settings.llm_model ?? '')
  const [openaiKeyInput, setOpenaiKeyInput] = useState('')
  const [geminiKeyInput, setGeminiKeyInput] = useState('')
  const [hfTokenInput, setHfTokenInput] = useState('')

  const [llmTest, setLlmTest] = useState<TestState>({ pending: false, result: null })
  const [ttsTest, setTtsTest] = useState<TestState>({ pending: false, result: null })
  const [providerSaving, setProviderSaving] = useState(false)
  const [providerSaveError, setProviderSaveError] = useState<string | null>(null)

  const currentKeyInput = (provider: LlmProviderChoice | TtsProviderChoice) => {
    if (provider === 'openai') return openaiKeyInput
    if (provider === 'gemini') return geminiKeyInput
    if (provider === 'huggingface_inference') return hfTokenInput
    return ''
  }

  const handleTestProvider = async (kind: 'llm' | 'tts') => {
    const provider = kind === 'llm' ? llmProvider : ttsProvider
    const setTestState = kind === 'llm' ? setLlmTest : setTtsTest
    setTestState({ pending: true, result: null })
    try {
      const response = await apiFetch('/api/settings/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          kind,
          provider,
          api_key: currentKeyInput(provider) || null,
          model: kind === 'llm' && llmModel ? llmModel : null,
        }),
      })
      const data: ProviderTestResponse = await response.json()
      setTestState({ pending: false, result: data })
    } catch {
      setTestState({ pending: false, result: { ok: false, message: t('Test request failed — is the server reachable?') } })
    }
  }

  const handleSaveProviders = async () => {
    setProviderSaving(true)
    setProviderSaveError(null)
    try {
      const payload: SettingsUpdatePayload = {
        llm_provider: llmProvider,
        tts_provider: ttsProvider,
        voice_language: voiceLanguage,
      }
      if (llmModel.trim()) payload.llm_model = llmModel.trim()
      if (openaiKeyInput.trim()) payload.openai_api_key = openaiKeyInput.trim()
      if (geminiKeyInput.trim()) payload.gemini_api_key = geminiKeyInput.trim()
      if (hfTokenInput.trim()) payload.huggingface_api_token = hfTokenInput.trim()

      const data = await putSettings(payload)
      applyUpdateResponse(data)

      // Only dismiss onboarding once a real provider is actually configured
      // (a key was just saved, or one was already on file) — merely having
      // "OpenAI" selected in the dropdown with no key entered yet must not
      // silently suppress the first-run nudge.
      const configuredRealKey =
        (llmProvider === 'openai' && (Boolean(openaiKeyInput.trim()) || data.settings.openai_api_key_set)) ||
        (llmProvider === 'gemini' && (Boolean(geminiKeyInput.trim()) || data.settings.gemini_api_key_set)) ||
        (ttsProvider === 'huggingface_inference' &&
          (Boolean(hfTokenInput.trim()) || data.settings.huggingface_api_token_set))
      if (configuredRealKey) {
        markOnboardingDismissed()
      }

      setOpenaiKeyInput('')
      setGeminiKeyInput('')
      setHfTokenInput('')
    } catch (error) {
      setProviderSaveError(error instanceof Error ? error.message : t('Could not save provider settings.'))
    } finally {
      setProviderSaving(false)
    }
  }

  const alreadyHasKey = useMemo(() => {
    if (!settings) return false
    if (llmProvider === 'openai') return settings.openai_api_key_set
    if (llmProvider === 'gemini') return settings.gemini_api_key_set
    return false
  }, [settings, llmProvider])

  const ttsAlreadyHasKey = useMemo(() => {
    if (!settings) return false
    if (ttsProvider === 'huggingface_inference') return settings.huggingface_api_token_set
    return false
  }, [settings, ttsProvider])

  return (
    <section className="settings-section">
      <h3>{t('Providers')}</h3>

      <div className="settings-field-group">
        <label className="settings-label" htmlFor="llm-provider-select">
          {t('LLM provider')}
        </label>
        <select
          id="llm-provider-select"
          value={llmProvider}
          onChange={(event) => {
            setLlmProvider(event.target.value as LlmProviderChoice)
            setLlmTest({ pending: false, result: null })
          }}
        >
          {LLM_PROVIDER_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {t(option.label)}
            </option>
          ))}
        </select>

        {providerNeedsKey(llmProvider) && (
          <>
            <label className="settings-label" htmlFor="llm-api-key-input">
              {llmProvider === 'openai' ? t('OpenAI API key') : t('Gemini API key')}
              {alreadyHasKey && <span className="settings-key-badge">{t('already configured')}</span>}
            </label>
            <input
              id="llm-api-key-input"
              type="password"
              autoComplete="off"
              placeholder={alreadyHasKey ? t('Leave blank to keep current key') : t('Paste your API key')}
              value={llmProvider === 'openai' ? openaiKeyInput : geminiKeyInput}
              onChange={(event) =>
                llmProvider === 'openai'
                  ? setOpenaiKeyInput(event.target.value)
                  : setGeminiKeyInput(event.target.value)
              }
            />
            {llmProvider === 'gemini' && !alreadyHasKey && (
              <a
                href="https://aistudio.google.com/app/apikey"
                target="_blank"
                rel="noreferrer"
                className="settings-key-help-link"
              >
                {t('Get a free Gemini API key →')}
              </a>
            )}
            <label className="settings-label" htmlFor="llm-model-input">
              {t('Model')} <span className="settings-optional">{t('(optional)')}</span>
            </label>
            <input
              id="llm-model-input"
              type="text"
              placeholder={t('e.g. gpt-4o-mini')}
              value={llmModel}
              onChange={(event) => setLlmModel(event.target.value)}
            />
          </>
        )}

        <div className="settings-action-row">
          <button
            type="button"
            className="settings-secondary-button"
            onClick={() => handleTestProvider('llm')}
            disabled={llmTest.pending}
          >
            {llmTest.pending ? t('Testing…') : t('Test connection')}
          </button>
        </div>
        {llmTest.result && (
          <div className={`settings-test-result ${llmTest.result.ok ? 'ok' : 'fail'}`}>
            {llmTest.result.message}
          </div>
        )}
      </div>

      <div className="settings-field-group">
        <label className="settings-label" htmlFor="tts-provider-select">
          {t('Voice provider')}
        </label>
        <select
          id="tts-provider-select"
          value={ttsProvider}
          onChange={(event) => {
            setTtsProvider(event.target.value as TtsProviderChoice)
            setTtsTest({ pending: false, result: null })
          }}
        >
          {(qwenVoiceAvailable
            ? [QWEN_TTS_OPTION, ...TTS_PROVIDER_OPTIONS]
            : TTS_PROVIDER_OPTIONS
          ).map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>

        <label className="settings-label" htmlFor="voice-language-select">
          {t('Voice language')}
        </label>
        <select
          id="voice-language-select"
          value={voiceLanguage}
          onChange={(event) =>
            setVoiceLanguage(event.target.value as 'japanese' | 'english')
          }
        >
          <option value="japanese">{t('Japanese (anime-dub style)')}</option>
          <option value="english">{t('English (speaks the reply text)')}</option>
        </select>
        <div className="settings-hint">
          {t('Which language she speaks aloud. The chat text stays English either way.')}
        </div>

        {providerNeedsKey(ttsProvider) && ttsProvider === 'huggingface_inference' && (
          <>
            <label className="settings-label" htmlFor="hf-token-input">
              {t('Hugging Face API token')}
              {ttsAlreadyHasKey && <span className="settings-key-badge">{t('already configured')}</span>}
            </label>
            <input
              id="hf-token-input"
              type="password"
              autoComplete="off"
              placeholder={ttsAlreadyHasKey ? t('Leave blank to keep current key') : t('Paste your API token')}
              value={hfTokenInput}
              onChange={(event) => setHfTokenInput(event.target.value)}
            />
          </>
        )}
        {ttsProvider === 'openai' && (
          <div className="settings-hint">{t('Uses the OpenAI API key configured above.')}</div>
        )}

        <div className="settings-action-row">
          <button
            type="button"
            className="settings-secondary-button"
            onClick={() => handleTestProvider('tts')}
            disabled={ttsTest.pending}
          >
            {ttsTest.pending ? t('Testing…') : t('Test connection')}
          </button>
        </div>
        {ttsTest.result && (
          <div className={`settings-test-result ${ttsTest.result.ok ? 'ok' : 'fail'}`}>
            {ttsTest.result.message}
          </div>
        )}
      </div>

      {providerSaveError && <div className="settings-status-line error">{providerSaveError}</div>}
      <div className="settings-action-row settings-save-row">
        <button
          type="button"
          className="settings-primary-button"
          onClick={handleSaveProviders}
          disabled={providerSaving}
        >
          {providerSaving ? t('Saving…') : t('Save providers')}
        </button>
      </div>
    </section>
  )
}

export default ProvidersSection
