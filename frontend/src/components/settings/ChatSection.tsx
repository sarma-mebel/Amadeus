/**
 * "Chat" settings section: the streaming toggle + its save. Form state is
 * local, seeded once from the loaded settings (the section only mounts
 * after the settings gate opens).
 */
import { useState } from 'react'
import { t } from '../../i18n'
import type {
  SettingsResponse,
  SettingsUpdatePayload,
  SettingsUpdateResponse,
} from '../../hooks/useSettingsData'

interface ChatSectionProps {
  settings: SettingsResponse
  putSettings: (payload: SettingsUpdatePayload) => Promise<SettingsUpdateResponse>
  applyUpdateResponse: (data: SettingsUpdateResponse) => void
}

const ChatSection = ({ settings, putSettings, applyUpdateResponse }: ChatSectionProps) => {
  const [streamingEnabled, setStreamingEnabled] = useState(() => settings.streaming_enabled)
  const [chatSaving, setChatSaving] = useState(false)

  const handleSaveChat = async () => {
    setChatSaving(true)
    try {
      const data = await putSettings({
        streaming_enabled: streamingEnabled,
      })
      applyUpdateResponse(data)
    } catch {
      // Non-fatal; toggles keep their local state either way.
    } finally {
      setChatSaving(false)
    }
  }

  return (
    <section className="settings-section">
      <h3>{t('Chat')}</h3>
      <div className="settings-field-group">
        <label className="settings-toggle-row" htmlFor="streaming-toggle-settings">
          <input
            id="streaming-toggle-settings"
            type="checkbox"
            checked={streamingEnabled}
            onChange={(event) => setStreamingEnabled(event.target.checked)}
          />
          <span>
            {t('Streaming responses')} <span className="settings-badge-experimental">{t('Advanced / experimental')}</span>
          </span>
        </label>
        <div className="settings-hint">
          {t('Streams replies over a WebSocket for lower latency. May be less stable than the standard mode.')}
        </div>

      </div>
      <div className="settings-action-row settings-save-row">
        <button
          type="button"
          className="settings-primary-button"
          onClick={handleSaveChat}
          disabled={chatSaving}
        >
          {chatSaving ? t('Saving…') : t('Save chat settings')}
        </button>
      </div>
    </section>
  )
}

export default ChatSection
