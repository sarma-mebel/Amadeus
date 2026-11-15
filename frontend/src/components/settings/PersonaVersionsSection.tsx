/**
 * "Persona versions" settings section: snapshot / pin / delete / restore of
 * identity versions. The version list itself lives in useSettingsData
 * (shared with the Character section, whose saves auto-snapshot); a restore
 * also writes the restored dials back into the panel-lifted Character/TTS
 * form state via `applyRestoredIdentity`.
 */
import { useState } from 'react'
import { apiFetch } from '../../api'
import { t } from '../../i18n'
import type {
  PersonaVersionInfo,
  SettingsUpdateResponse,
} from '../../hooks/useSettingsData'

interface PersonaVersionsSectionProps {
  personaVersions: PersonaVersionInfo[]
  versionsCurrentFp: string
  fetchPersonaVersions: () => Promise<void>
  applyUpdateResponse: (data: SettingsUpdateResponse) => void
  /** Reflect the restored dials in the panel's lifted slider/TTS state. */
  applyRestoredIdentity: (data: SettingsUpdateResponse) => void
}

const PersonaVersionsSection = ({
  personaVersions,
  versionsCurrentFp,
  fetchPersonaVersions,
  applyUpdateResponse,
  applyRestoredIdentity,
}: PersonaVersionsSectionProps) => {
  const [versionName, setVersionName] = useState('')
  const [versionBusy, setVersionBusy] = useState(false)
  const [versionError, setVersionError] = useState<string | null>(null)

  const handleSaveVersion = async () => {
    setVersionBusy(true)
    setVersionError(null)
    try {
      const response = await apiFetch('/api/persona/versions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: versionName.trim() || null }),
      })
      if (!response.ok) throw new Error(`Save failed: ${response.status}`)
      setVersionName('')
      await fetchPersonaVersions()
    } catch (error) {
      setVersionError(error instanceof Error ? error.message : t('Save failed'))
    } finally {
      setVersionBusy(false)
    }
  }

  const handleTogglePin = async (version: PersonaVersionInfo) => {
    try {
      await apiFetch(`/api/persona/versions/${version.id}/pin`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pinned: !version.pinned }),
      })
      await fetchPersonaVersions()
    } catch {
      // Best-effort; the list refresh reflects reality either way.
    }
  }

  const handleDeleteVersion = async (version: PersonaVersionInfo) => {
    setVersionError(null)
    const response = await apiFetch(`/api/persona/versions/${version.id}`, {
      method: 'DELETE',
    }).catch(() => null)
    if (response && response.status === 409) {
      setVersionError(t('Unpin this version before deleting it.'))
      return
    }
    await fetchPersonaVersions()
  }

  const handleRestoreVersion = async (version: PersonaVersionInfo) => {
    const summary = version.diff_vs_current
      ? t('This will change: {diff}.', { diff: version.diff_vs_current })
      : t('This restores an earlier persona/voice.')
    if (!window.confirm(`${t('Restore "{name}"?', { name: version.name })} ${summary}`)) return
    setVersionBusy(true)
    setVersionError(null)
    try {
      const response = await apiFetch(`/api/persona/versions/${version.id}/rollback`, {
        method: 'POST',
      })
      if (!response.ok) throw new Error(`Restore failed: ${response.status}`)
      const data: SettingsUpdateResponse = await response.json()
      applyUpdateResponse(data)
      // Reflect the restored dials in the local slider state.
      applyRestoredIdentity(data)
      await fetchPersonaVersions()
    } catch (error) {
      setVersionError(error instanceof Error ? error.message : t('Restore failed'))
    } finally {
      setVersionBusy(false)
    }
  }

  return (
    <section className="settings-section">
      <h3>{t('Persona versions')}</h3>
      <p className="settings-hint">
        {t(
          'Persona and voice changes are snapshotted automatically so you can always restore how she used to sound and act. She mentions changes once, the next time she greets you.',
        )}
      </p>
      <div className="settings-field-group">
        <div className="settings-action-row">
          <input
            type="text"
            placeholder={t('Name this version (optional)')}
            maxLength={60}
            value={versionName}
            onChange={(event) => setVersionName(event.target.value)}
          />
          <button
            type="button"
            className="settings-secondary-button"
            onClick={handleSaveVersion}
            disabled={versionBusy}
          >
            {t('Save current')}
          </button>
        </div>
        {versionError && (
          <p className="settings-status-line error">{versionError}</p>
        )}
        {personaVersions.length === 0 ? (
          <p className="settings-hint">{t('No saved versions yet.')}</p>
        ) : (
          <ul className="persona-version-list">
            {[...personaVersions].reverse().map((version) => (
              <li key={version.id} className="persona-version-row">
                <div className="persona-version-info">
                  <span className="persona-version-name">
                    {version.pinned ? '📌 ' : ''}
                    {version.name}
                    {version.fingerprint === versionsCurrentFp && (
                      <span className="persona-version-current"> · {t('current')}</span>
                    )}
                  </span>
                  <span className="persona-version-meta">
                    {version.created_at.replace('T', ' ')}
                    {version.diff_vs_current ? ` — ${version.diff_vs_current}` : ''}
                  </span>
                </div>
                <div className="persona-version-actions">
                  {version.fingerprint !== versionsCurrentFp && (
                    <button
                      type="button"
                      className="settings-secondary-button"
                      onClick={() => handleRestoreVersion(version)}
                      disabled={versionBusy}
                    >
                      {t('Restore')}
                    </button>
                  )}
                  <button
                    type="button"
                    className="settings-ghost-button"
                    onClick={() => handleTogglePin(version)}
                    title={version.pinned ? t('Unpin') : t('Pin (never auto-pruned)')}
                  >
                    {version.pinned ? t('Unpin') : t('Pin')}
                  </button>
                  {!version.pinned && (
                    <button
                      type="button"
                      className="settings-ghost-button"
                      onClick={() => handleDeleteVersion(version)}
                      title={t('Delete this version')}
                    >
                      {t('Delete')}
                    </button>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  )
}

export default PersonaVersionsSection
