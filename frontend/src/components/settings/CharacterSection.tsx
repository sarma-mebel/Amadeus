/**
 * "Character" settings section: character-pack switching plus the persona
 * dials (tsundere/formality/safe mode). The dial state is lifted to
 * SettingsPanel because a persona-version restore writes it back; the pack
 * list is owned by the panel too (fetched on panel mount, before this
 * section exists). Saving may auto-snapshot the previous identity, so it
 * refreshes the version list via `fetchPersonaVersions`.
 */
import { useState } from 'react'
import { t } from '../../i18n'
import {
  dispatchCharacterSwitched,
  switchCharacter,
  type CharacterEntry,
} from '../../charactersApi'
import type {
  SettingsUpdatePayload,
  SettingsUpdateResponse,
} from '../../hooks/useSettingsData'

interface CharacterSectionProps {
  characterList: CharacterEntry[]
  refreshCharacterList: () => void
  tsundereLevel: number
  setTsundereLevel: (value: number) => void
  scientificFormality: number
  setScientificFormality: (value: number) => void
  safeMode: boolean
  setSafeMode: (value: boolean) => void
  putSettings: (payload: SettingsUpdatePayload) => Promise<SettingsUpdateResponse>
  applyUpdateResponse: (data: SettingsUpdateResponse) => void
  fetchPersonaVersions: () => Promise<void>
}

const CharacterSection = ({
  characterList,
  refreshCharacterList,
  tsundereLevel,
  setTsundereLevel,
  scientificFormality,
  setScientificFormality,
  safeMode,
  setSafeMode,
  putSettings,
  applyUpdateResponse,
  fetchPersonaVersions,
}: CharacterSectionProps) => {
  const [characterSaving, setCharacterSaving] = useState(false)

  // Character packs (who she IS — hidden when the backend predates the
  // endpoint or only the built-in exists)
  const [characterSwitching, setCharacterSwitching] = useState<string | null>(null)
  const [characterSwitchError, setCharacterSwitchError] = useState<string | null>(null)

  const handleSwitchCharacter = async (id: string) => {
    setCharacterSwitching(id)
    setCharacterSwitchError(null)
    try {
      await switchCharacter(id)
      refreshCharacterList()
      dispatchCharacterSwitched()
    } catch (error) {
      setCharacterSwitchError(error instanceof Error ? error.message : String(error))
    } finally {
      setCharacterSwitching(null)
    }
  }

  const handleSaveCharacter = async () => {
    setCharacterSaving(true)
    try {
      const data = await putSettings({
        tsundere_level: tsundereLevel,
        scientific_formality: scientificFormality,
        safe_mode: safeMode,
      })
      applyUpdateResponse(data)
      // A character save may have auto-snapshotted the previous identity.
      await fetchPersonaVersions()
    } catch {
      // Non-fatal; the sliders keep their local state either way.
    } finally {
      setCharacterSaving(false)
    }
  }

  return (
    <section className="settings-section">
      <h3>{t('Character')}</h3>
      {characterList.length > 1 && (
        <div className="settings-field-group">
          <span className="settings-label">{t('Who she is')}</span>
          <div className="settings-character-list">
            {characterList.map((entry) => (
              <button
                key={entry.id}
                type="button"
                className={`settings-character-button${entry.active ? ' active' : ''}`}
                disabled={entry.active || characterSwitching !== null}
                onClick={() => void handleSwitchCharacter(entry.id)}
                title={entry.has_voice ? undefined : t('No voice clip: keeps the default voice')}
              >
                {characterSwitching === entry.id
                  ? t('Switching…')
                  : entry.full_name || entry.name}
                {entry.active ? ' ✓' : ''}
                {!entry.has_voice && !entry.builtin ? ' 🔇' : ''}
              </button>
            ))}
          </div>
          {characterSwitchError && (
            <p className="settings-hint settings-error">{characterSwitchError}</p>
          )}
          <p className="settings-hint">
            {t('Switching starts a new conversation with her persona, voice, and avatar.')}
          </p>
        </div>
      )}
      <div className="settings-field-group">
        <label className="settings-label" htmlFor="tsundere-slider">
          {t('Tsundere level')} <span className="settings-slider-value">{tsundereLevel.toFixed(2)}</span>
        </label>
        <input
          id="tsundere-slider"
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={tsundereLevel}
          onChange={(event) => setTsundereLevel(Number(event.target.value))}
        />

        <label className="settings-label" htmlFor="formality-slider">
          {t('Scientific formality')}{' '}
          <span className="settings-slider-value">{scientificFormality.toFixed(2)}</span>
        </label>
        <input
          id="formality-slider"
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={scientificFormality}
          onChange={(event) => setScientificFormality(Number(event.target.value))}
        />

        <label className="settings-toggle-row" htmlFor="safe-mode-toggle">
          <input
            id="safe-mode-toggle"
            type="checkbox"
            checked={safeMode}
            onChange={(event) => setSafeMode(event.target.checked)}
          />
          <span>{t('Safe mode')}</span>
        </label>
      </div>
      <div className="settings-action-row settings-save-row">
        <button
          type="button"
          className="settings-primary-button"
          onClick={handleSaveCharacter}
          disabled={characterSaving}
        >
          {characterSaving ? t('Saving…') : t('Save character')}
        </button>
      </div>
    </section>
  )
}

export default CharacterSection
