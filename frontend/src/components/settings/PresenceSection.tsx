/**
 * "Presence" settings section: one intensity dial (quiet / balanced /
 * lively) over the client-side presence features, plus the individual
 * fine-tune toggles the dial maps onto (thinking sounds, listening
 * reactions, desktop roaming — moved here from FeaturesSection). Picking
 * a level applies its preset; the toggles stay the per-feature truth, so
 * hand-tuning one just marks the dial "customized" instead of fighting
 * it. Screen watching and notifications stay separate opt-ins by design:
 * a preset must never switch on screen capture or fire a permission
 * prompt (see presenceLevel.ts).
 */
import { useState } from 'react'
import { t } from '../../i18n'
import { isTauri } from '../../platform'
import {
  applyPresenceLevel,
  loadPresenceLevel,
  PRESENCE_PRESETS,
  presenceLevelCustomized,
  type PresenceLevel,
} from '../../presenceLevel'
import { loadFillersEnabled, saveFillersEnabled } from '../../audio/fillerPlayer'
import {
  loadListeningReactionsEnabled,
  saveListeningReactionsEnabled,
} from '../../audio/listeningReactions'
import { isPetRoamEnabled, setPetRoamEnabled } from '../../vrm/petRoam'

const PresenceSection = () => {
  const [level, setLevel] = useState<PresenceLevel>(loadPresenceLevel)
  const [customized, setCustomized] = useState(presenceLevelCustomized)
  const [fillersEnabled, setFillersEnabled] = useState(loadFillersEnabled)
  const [listeningReactionsEnabled, setListeningReactionsEnabled] = useState(
    loadListeningReactionsEnabled,
  )
  const [petRoamEnabled, setPetRoamEnabledState] = useState(isPetRoamEnabled)

  const pickLevel = (next: PresenceLevel) => {
    applyPresenceLevel(next)
    const preset = PRESENCE_PRESETS[next]
    setLevel(next)
    setFillersEnabled(preset.fillers)
    setListeningReactionsEnabled(preset.listeningReactions)
    setPetRoamEnabledState(isTauri() ? preset.petRoam : false)
    setCustomized(false)
  }

  const levelOptions: { value: PresenceLevel; label: string; hint: string }[] = [
    {
      value: 'quiet',
      label: t('Quiet'),
      hint: t(
        'She speaks when spoken to: no unprompted chatter, no thinking sounds, no visible reactions while you talk, no wandering.',
      ),
    },
    {
      value: 'balanced',
      label: t('Balanced'),
      hint: t(
        'The default. An occasional unprompted remark after a long quiet stretch, and she visibly listens while you speak.',
      ),
    },
    {
      value: 'lively',
      label: t('Lively'),
      hint: t(
        'Chattier: she checks in more often, hums while composing replies, and on desktop may stroll along the screen edges.',
      ),
    },
  ]

  return (
    <section className="settings-section">
      <h3>{t('Presence')}</h3>
      <div className="settings-field-group">
        <div className="settings-hint">
          {t(
            'How much unprompted life she has. One choice sets the toggles below; fine-tune them afterwards if you like.',
          )}
        </div>
        {levelOptions.map((option) => (
          <div key={option.value}>
            <label
              className="settings-toggle-row"
              htmlFor={`presence-level-${option.value}`}
            >
              <input
                id={`presence-level-${option.value}`}
                type="radio"
                name="presence-level"
                checked={level === option.value}
                onChange={() => pickLevel(option.value)}
                // A click on the ALREADY-selected level fires no change
                // event, but must still reapply the preset — that's the
                // documented recovery from a fine-tuned ("customized")
                // state. pickLevel is idempotent, so both firing is fine.
                onClick={() => pickLevel(option.value)}
              />
              <span>{option.label}</span>
            </label>
            <div className="settings-hint">{option.hint}</div>
          </div>
        ))}
        {customized && (
          <div className="settings-hint">
            {t(
              'Fine-tuned: a toggle below differs from this preset. Picking a level again reapplies it.',
            )}
          </div>
        )}

        <label className="settings-toggle-row" htmlFor="fillers-toggle">
          <input
            id="fillers-toggle"
            type="checkbox"
            checked={fillersEnabled}
            onChange={(event) => {
              setFillersEnabled(event.target.checked)
              saveFillersEnabled(event.target.checked)
              setCustomized(presenceLevelCustomized())
            }}
          />
          <span>{t('Thinking sounds')}</span>
        </label>
        <div className="settings-hint">
          {t(
            'A short hum in her voice while she composes a reply. Applies immediately; needs the generated filler clips (scripts/generate_fillers.py).',
          )}
        </div>

        <label className="settings-toggle-row" htmlFor="listening-reactions-toggle">
          <input
            id="listening-reactions-toggle"
            type="checkbox"
            checked={listeningReactionsEnabled}
            onChange={(event) => {
              setListeningReactionsEnabled(event.target.checked)
              saveListeningReactionsEnabled(event.target.checked)
              setCustomized(presenceLevelCustomized())
            }}
          />
          <span>{t('Listening reactions')}</span>
        </label>
        <div className="settings-hint">
          {t(
            'While you speak into the mic she keeps eye contact, nods along, and may voice a soft "un". Applies immediately; the voiced part needs the generated clips (scripts/generate_fillers.py).',
          )}
        </div>

        {isTauri() && (
          <>
            <label className="settings-toggle-row" htmlFor="pet-roam-toggle">
              <input
                id="pet-roam-toggle"
                type="checkbox"
                checked={petRoamEnabled}
                onChange={(event) => {
                  setPetRoamEnabledState(event.target.checked)
                  setPetRoamEnabled(event.target.checked)
                  setCustomized(presenceLevelCustomized())
                }}
              />
              <span>{t('Let her wander the desktop')}</span>
            </label>
            <div className="settings-hint">
              {t(
                'Every few minutes the pet may take a slow stroll along the screen edge to a new spot. Never while she is speaking, listening, dozing, quiet-hours muted, or within a minute of you interacting. Applies immediately.',
              )}
            </div>
          </>
        )}
      </div>
    </section>
  )
}

export default PresenceSection
