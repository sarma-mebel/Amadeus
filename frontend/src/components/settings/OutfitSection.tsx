/**
 * "Outfit" settings section: the VRM model picker. The registry is owned
 * by SettingsPanel (fetched on panel mount, in parallel with the settings
 * load); this section renders nothing until a second .vrm is registered —
 * which is what keeps the picker hidden by default. The selection itself
 * is a local-only preference (localStorage), applied immediately.
 */
import { useState } from 'react'
import { t } from '../../i18n'
import {
  dispatchModelChange,
  loadModelId,
  saveModelId,
  type ModelDefinition,
} from '../../models'

interface OutfitSectionProps {
  modelRegistry: ModelDefinition[]
}

const OutfitSection = ({ modelRegistry }: OutfitSectionProps) => {
  const [modelId, setModelId] = useState<string>(() => loadModelId())

  const handleModelChange = (id: string) => {
    setModelId(id)
    saveModelId(id)
    dispatchModelChange(id)
  }

  if (modelRegistry.length <= 1) return null

  return (
    <section className="settings-section">
      <h3>{t('Outfit')}</h3>
      <div className="settings-field-group">
        <label className="settings-label" htmlFor="model-select">
          {t('Model')}
        </label>
        <select
          id="model-select"
          value={modelId}
          onChange={(event) => handleModelChange(event.target.value)}
        >
          {modelRegistry.map((model) => (
            <option key={model.id} value={model.id}>
              {model.label}
            </option>
          ))}
        </select>
        <div className="settings-hint">
          {t(
            'Switches immediately — no restart needed. Add .vrm files to static/models and list them in models.json for more outfits.',
          )}
        </div>
      </div>
    </section>
  )
}

export default OutfitSection
