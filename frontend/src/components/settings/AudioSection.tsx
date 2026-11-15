/**
 * "Audio" settings section: per-bus volume sliders + the lab-ambience
 * toggle. All local-only preferences applied immediately through the
 * soundscape singleton — no backend round trip.
 */
import { useState } from 'react'
import { t } from '../../i18n'
import {
  loadAmbienceEnabled,
  loadVolumePrefs,
  saveAmbienceEnabled,
  soundscape,
  type VolumePrefs,
} from '../../soundscape'

const AudioSection = () => {
  const [volumePrefs, setVolumePrefs] = useState<VolumePrefs>(() => loadVolumePrefs())
  const [ambienceEnabled, setAmbienceEnabled] = useState(() => loadAmbienceEnabled())

  const handleVolumeChange = (bus: keyof VolumePrefs, value: number) => {
    soundscape.setVolume(bus, value)
    setVolumePrefs(soundscape.getVolumePrefs())
  }

  const handleAmbienceToggle = (enabled: boolean) => {
    setAmbienceEnabled(enabled)
    saveAmbienceEnabled(enabled)
    if (enabled) {
      soundscape.startAmbience()
    } else {
      soundscape.stopAmbience()
    }
  }

  return (
    <section className="settings-section">
      <h3>{t('Audio')}</h3>
      <div className="settings-field-group">
        <label className="settings-label" htmlFor="voice-volume-slider">
          {t('Voice')}{' '}
          <span className="settings-slider-value">{Math.round(volumePrefs.voice * 100)}%</span>
        </label>
        <input
          id="voice-volume-slider"
          type="range"
          min={0}
          max={100}
          step={1}
          value={Math.round(volumePrefs.voice * 100)}
          onChange={(event) => handleVolumeChange('voice', Number(event.target.value) / 100)}
        />

        <label className="settings-label" htmlFor="fx-volume-slider">
          {t('UI sounds')}{' '}
          <span className="settings-slider-value">{Math.round(volumePrefs.fx * 100)}%</span>
        </label>
        <input
          id="fx-volume-slider"
          type="range"
          min={0}
          max={100}
          step={1}
          value={Math.round(volumePrefs.fx * 100)}
          onChange={(event) => handleVolumeChange('fx', Number(event.target.value) / 100)}
          onPointerUp={() => soundscape.play('receive')}
        />

        <label className="settings-label" htmlFor="ambience-volume-slider">
          {t('Ambience')}{' '}
          <span className="settings-slider-value">{Math.round(volumePrefs.ambience * 100)}%</span>
        </label>
        <input
          id="ambience-volume-slider"
          type="range"
          min={0}
          max={100}
          step={1}
          value={Math.round(volumePrefs.ambience * 100)}
          onChange={(event) => handleVolumeChange('ambience', Number(event.target.value) / 100)}
        />

        <label className="settings-toggle-row" htmlFor="ambience-enabled-toggle">
          <input
            id="ambience-enabled-toggle"
            type="checkbox"
            checked={ambienceEnabled}
            onChange={(event) => handleAmbienceToggle(event.target.checked)}
          />
          <span>{t('Lab ambience')}</span>
        </label>
        <div className="settings-hint">
          {t('A quiet, synthesized lab hum that plays in the background while chatting.')}
        </div>
      </div>
    </section>
  )
}

export default AudioSection
