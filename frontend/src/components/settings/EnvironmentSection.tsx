/**
 * "Environment" settings section: the VRM backdrop scene picker — a
 * local-only preference (localStorage), no backend round trip, so it
 * applies immediately rather than needing a "Save" click.
 */
import { useState } from 'react'
import { t } from '../../i18n'
import {
  dispatchEnvironmentChange,
  ENVIRONMENTS,
  loadEnvironmentId,
  saveEnvironmentId,
  type EnvironmentId,
} from '../../environments'

const EnvironmentSection = () => {
  const [environmentId, setEnvironmentId] = useState<EnvironmentId>(() => loadEnvironmentId())

  const handleEnvironmentChange = (id: EnvironmentId) => {
    setEnvironmentId(id)
    saveEnvironmentId(id)
    dispatchEnvironmentChange(id)
  }

  return (
    <section className="settings-section">
      <h3>{t('Environment')}</h3>
      <div className="settings-field-group">
        <label className="settings-label" htmlFor="environment-select">
          {t('Scene')}
        </label>
        <select
          id="environment-select"
          value={environmentId}
          onChange={(event) => handleEnvironmentChange(event.target.value as EnvironmentId)}
        >
          {Object.values(ENVIRONMENTS).map((env) => (
            <option key={env.id} value={env.id}>
              {env.label}
            </option>
          ))}
        </select>
        <div className="settings-hint">{t('Switches immediately — no restart needed.')}</div>
      </div>
    </section>
  )
}

export default EnvironmentSection
