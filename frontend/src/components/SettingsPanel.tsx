/**
 * The settings modal: composition shell over the section components in
 * ./settings/ and the shared data hook (useSettingsData). The panel itself
 * keeps only what crosses section boundaries — the persona dials + TTS
 * provider (written back by a persona-version restore), the character-pack
 * list, and the outfit registry (both fetched on panel mount, before the
 * settings-loaded gate opens).
 */
import { useCallback, useEffect, useState } from 'react'
import { t, useLocale } from '../i18n'
import { markOnboardingDismissed } from '../onboarding'
import { fetchCharacters, type CharacterEntry } from '../charactersApi'
import OnboardingWizard from './OnboardingWizard'
import UpdateNudge from './UpdateNudge'
import { normalizeTtsProvider, type TtsProviderChoice } from './ttsOptions'
import { DEFAULT_MODEL, fetchModelRegistry, type ModelDefinition } from '../models'
import { loadPostFxEnabledPref, savePostFxEnabledPref } from '../vrm/postfx'
import {
  useSettingsData,
  type SettingsUpdateResponse,
} from '../hooks/useSettingsData'
import AccessSection from './settings/AccessSection'
import ProvidersSection from './settings/ProvidersSection'
import CharacterSection from './settings/CharacterSection'
import PersonaVersionsSection from './settings/PersonaVersionsSection'
import ChatSection from './settings/ChatSection'
import PresenceSection from './settings/PresenceSection'
import FeaturesSection from './settings/FeaturesSection'
import RhythmSection from './settings/RhythmSection'
import EnvironmentSection from './settings/EnvironmentSection'
import OutfitSection from './settings/OutfitSection'
import AudioSection from './settings/AudioSection'
import DiagnosticsSection from './settings/DiagnosticsSection'
import './SettingsPanel.css'

// Re-exported so existing importers (OnboardingWizard, ChatInterface,
// tests) keep working; the definitions moved to hooks/useSettingsData.
export type {
  PersonaVersionInfo,
  ProviderTestResponse,
  SettingsResponse,
  SettingsUpdatePayload,
  SettingsUpdateResponse,
} from '../hooks/useSettingsData'

interface SettingsPanelProps {
  onClose: () => void
  isOnboarding: boolean
  onSkipOnboarding: () => void
}

const SettingsPanel = ({ onClose, isOnboarding, onSkipOnboarding }: SettingsPanelProps) => {
  // Subscribe the whole panel (header included) to locale changes; the
  // Access section reads the value for its select via its own useLocale.
  useLocale()

  // Lifted Character/Providers form state: a persona-version restore
  // (PersonaVersionsSection) writes these back across section boundaries.
  const [ttsProvider, setTtsProvider] = useState<TtsProviderChoice>('pyttsx3')
  const [tsundereLevel, setTsundereLevel] = useState(0.5)
  const [scientificFormality, setScientificFormality] = useState(0.5)
  const [safeMode, setSafeMode] = useState(true)

  const {
    settings,
    loading,
    loadError,
    restartNotice,
    personaVersions,
    versionsCurrentFp,
    applyUpdateResponse,
    putSettings,
    fetchPersonaVersions,
  } = useSettingsData({
    onSettingsLoaded: (data) => {
      setTtsProvider(normalizeTtsProvider(data.providers.tts))
      setTsundereLevel(data.persona.tsundere_level)
      setScientificFormality(data.persona.scientific_formality)
      setSafeMode(data.persona.safe_mode)
    },
  })

  // Reflect the restored dials in the local slider state.
  const applyRestoredIdentity = (data: SettingsUpdateResponse) => {
    setTsundereLevel(data.settings.persona.tsundere_level)
    setScientificFormality(data.settings.persona.scientific_formality)
    setSafeMode(data.settings.persona.safe_mode)
    setTtsProvider(normalizeTtsProvider(data.settings.providers.tts))
  }

  // Character packs (who she IS — hidden when the backend predates the
  // endpoint or only the built-in exists)
  const [characterList, setCharacterList] = useState<CharacterEntry[]>([])

  const refreshCharacterList = useCallback(() => {
    void fetchCharacters().then((list) => setCharacterList(list ?? []))
  }, [])

  useEffect(() => {
    refreshCharacterList()
  }, [refreshCharacterList])

  // Outfit registry: discovered at runtime from an optional static
  // manifest (models.ts) — defaults to just [DEFAULT_MODEL] until a
  // second .vrm is registered, which is what keeps the picker hidden.
  // Best-effort, mirrors fetchPersonaVersions — a failed/missing manifest
  // just means fetchModelRegistry resolves to [DEFAULT_MODEL].
  const [modelRegistry, setModelRegistry] = useState<ModelDefinition[]>([DEFAULT_MODEL])
  useEffect(() => {
    void fetchModelRegistry().then(setModelRegistry)
  }, [])

  // Post-processing (bloom + vignette, see ../vrm/postfx): client-side and
  // read once at scene init, like kurisu_wind/kurisu_talk_body — flipping
  // it here applies on the next avatar reload, not live.
  const [postFxEnabled, setPostFxEnabled] = useState(loadPostFxEnabledPref)

  const handleSkip = () => {
    markOnboardingDismissed()
    onSkipOnboarding()
  }

  return (
    <div className="settings-overlay" role="dialog" aria-modal="true" aria-label={t('Settings')}>
      <div className="settings-modal">
        <header className="settings-modal-header">
          <div>
            <h2>{t('Settings')}</h2>
            <span className="settings-modal-subtitle">{t('Providers, character, and chat behavior')}</span>
          </div>
          <button type="button" className="settings-close" onClick={onClose} aria-label={t('Close settings')}>
            ×
          </button>
        </header>

        {/* First run: the wizard replaces the normal settings body entirely.
            After onboarding, the panel renders exactly as before. */}
        {isOnboarding && <OnboardingWizard onFinished={onClose} onSkip={handleSkip} />}

        {!isOnboarding && loading && <div className="settings-status-line">{t('Loading settings…')}</div>}
        {!isOnboarding && loadError && <div className="settings-status-line error">{loadError}</div>}

        {/* The Access section is deliberately outside the settings-loaded
            gate: if the token is wrong, /api/settings itself 401s — the
            field must stay reachable so the user can fix it. */}
        {!isOnboarding && !loading && (
          <div className="settings-body">
          {/* Renders nothing unless a newer GitHub release exists (24h-cached
              check; hidden in demo mode / on failure). Nudge only — links to
              the releases page, never downloads anything. */}
          <UpdateNudge />
          <AccessSection />

          {!loadError && settings && (
          <>
            {restartNotice && (
              <div className="settings-restart-notice">
                {t('Restart to apply:')} {restartNotice}
              </div>
            )}

            <ProvidersSection
              settings={settings}
              ttsProvider={ttsProvider}
              setTtsProvider={setTtsProvider}
              putSettings={putSettings}
              applyUpdateResponse={applyUpdateResponse}
            />

            <CharacterSection
              characterList={characterList}
              refreshCharacterList={refreshCharacterList}
              tsundereLevel={tsundereLevel}
              setTsundereLevel={setTsundereLevel}
              scientificFormality={scientificFormality}
              setScientificFormality={setScientificFormality}
              safeMode={safeMode}
              setSafeMode={setSafeMode}
              putSettings={putSettings}
              applyUpdateResponse={applyUpdateResponse}
              fetchPersonaVersions={fetchPersonaVersions}
            />

            <PersonaVersionsSection
              personaVersions={personaVersions}
              versionsCurrentFp={versionsCurrentFp}
              fetchPersonaVersions={fetchPersonaVersions}
              applyUpdateResponse={applyUpdateResponse}
              applyRestoredIdentity={applyRestoredIdentity}
            />

            <ChatSection
              settings={settings}
              putSettings={putSettings}
              applyUpdateResponse={applyUpdateResponse}
            />

            <PresenceSection />

            <FeaturesSection
              settings={settings}
              putSettings={putSettings}
              applyUpdateResponse={applyUpdateResponse}
            />

            <RhythmSection />

            <EnvironmentSection />

            <OutfitSection modelRegistry={modelRegistry} />

            <section className="settings-section">
              <h3>{t('Rendering')}</h3>
              <div className="settings-field-group">
                <label className="settings-toggle-row" htmlFor="postfx-toggle">
                  <input
                    id="postfx-toggle"
                    type="checkbox"
                    checked={postFxEnabled}
                    onChange={(event) => {
                      setPostFxEnabled(event.target.checked)
                      savePostFxEnabledPref(event.target.checked)
                    }}
                  />
                  <span>{t('Bloom & vignette')}</span>
                </label>
                <div className="settings-hint">
                  {t('A subtle glow on highlights and a soft frame around the edges. Applies on the next avatar reload.')}
                </div>
              </div>
            </section>

            <AudioSection />

            <DiagnosticsSection />
          </>
          )}
          </div>
        )}
      </div>
    </div>
  )
}

export default SettingsPanel
