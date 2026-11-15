/**
 * "Features" settings section: the server-side toggles (long-term memory,
 * tools, web search, reminders, STT provider) with their combined save,
 * plus the instant client-side preferences (notifications, talk-body,
 * hands-free, duplex, pet default, autostart). The presence-flavored
 * client toggles (thinking sounds, listening reactions, roaming) live in
 * PresenceSection under the intensity dial. Server-toggle form state is
 * local, seeded once from the loaded settings; client preferences
 * read/write localStorage directly. Autostart is the odd one out: it has
 * no localStorage mirror at all (see autostart.ts) — it's queried live
 * from the OS-level registration on mount and every toggle round-trips
 * through the same live call.
 */
import { useEffect, useState } from 'react'
import { t } from '../../i18n'
import { isTauri } from '../../platform'
import {
  ensurePermission,
  loadNotificationsEnabled,
  saveNotificationsEnabled,
} from '../../notifications'
import {
  HANDS_FREE_EVENT,
  loadHandsFreeEnabled,
  saveHandsFreeEnabled,
} from '../../audio/handsFree'
import { loadTalkBodyEnabled, saveTalkBodyEnabled } from '../../vrm/animationScheduler'
import { loadPetDefaultEnabled, savePetDefaultEnabled } from '../../petDefault'
import { getAutostartEnabled, setAutostartEnabled as applyAutostart } from '../../autostart'
import { loadDuplexEnabled, saveDuplexEnabled } from '../../chat/duplexTurn'
import { loadWakeWordEnabled, saveWakeWordEnabled } from '../../audio/wakeWord'
import type {
  SettingsResponse,
  SettingsUpdatePayload,
  SettingsUpdateResponse,
} from '../../hooks/useSettingsData'

interface FeaturesSectionProps {
  settings: SettingsResponse
  putSettings: (payload: SettingsUpdatePayload) => Promise<SettingsUpdateResponse>
  applyUpdateResponse: (data: SettingsUpdateResponse) => void
}

const FeaturesSection = ({ settings, putSettings, applyUpdateResponse }: FeaturesSectionProps) => {
  // Features (long-term memory + tools)
  const [sttProvider, setSttProvider] = useState(() => settings.providers.stt)
  const [memoryLongTerm, setMemoryLongTerm] = useState(
    () => settings.memory_enable_long_term ?? true,
  )
  const [toolsEnabled, setToolsEnabled] = useState(() => settings.tools_enabled ?? false)
  const [webSearchEnabled, setWebSearchEnabled] = useState(
    () => settings.web_search_enabled ?? false,
  )
  const [remindersEnabled, setRemindersEnabled] = useState(
    () => settings.reminders_enabled ?? false,
  )
  // Desktop notifications are a client-side preference (localStorage), not a
  // server setting: they apply instantly, and the permission prompt fires
  // only from this toggle — never unprompted.
  const [notificationsEnabled, setNotificationsEnabled] = useState(loadNotificationsEnabled())
  // Body language while she talks (talk-loop + mid-speech gesture clips).
  // Client-side, read live by the animation scheduler — default off
  // (mouth-only talking), see TALK_BODY_STORAGE_KEY's doc comment.
  const [talkBodyEnabled, setTalkBodyEnabled] = useState(loadTalkBodyEnabled())
  // Hands-free conversation: client-side and instant. saveHandsFreeEnabled
  // broadcasts a window event that the active useHandsFree instance
  // (ChatInterface or the pet) reacts to by acquiring/releasing the mic;
  // listening to the same event keeps this checkbox in sync when the mode
  // is toggled elsewhere (composer button, auto-disable on STT loss).
  const [handsFreeEnabled, setHandsFreeEnabled] = useState(loadHandsFreeEnabled())
  useEffect(() => {
    const sync = () => setHandsFreeEnabled(loadHandsFreeEnabled())
    window.addEventListener(HANDS_FREE_EVENT, sync)
    return () => window.removeEventListener(HANDS_FREE_EVENT, sync)
  }, [])
  // Full-duplex voice loop: PROTOTYPE, client-side and instant like the
  // other opt-in toggles above. Needs the server's web.duplex_enabled flag
  // on too (off by default) — the socket answers a clear error frame and
  // the caller falls back if it isn't, so this toggle stays visible either
  // way rather than depending on backend capability plumbing.
  const [duplexEnabled, setDuplexEnabled] = useState(loadDuplexEnabled())
  // Wake word: same capability posture as duplex above (toggle always
  // visible; the /api/wake socket answers a clear error frame when the
  // server has wake_word.enabled off or no model fetched). Deliberately
  // NOT in the presence presets — it's a mic privacy opt-in, same category
  // as screen watching (see presenceLevel.ts).
  const [wakeWordEnabled, setWakeWordEnabled] = useState(loadWakeWordEnabled())
  // Desktop only: boot straight to the pet overlay (marker file read by the
  // Rust shell at startup; this state mirrors it for the toggle).
  const [petDefaultEnabled, setPetDefaultEnabled] = useState(loadPetDefaultEnabled())
  // "Start with Windows": unlike petDefaultEnabled there's no localStorage
  // mirror to seed from (autostart.ts talks to the OS-level registration
  // directly), so this starts undefined ("unknown yet") and the mount
  // effect below resolves it to true/false/null(unavailable).
  const [autostartEnabled, setAutostartEnabledState] = useState<boolean | null>(null)
  const [autostartLoading, setAutostartLoading] = useState(true)
  const [autostartError, setAutostartError] = useState<string | null>(null)
  useEffect(() => {
    if (!isTauri()) {
      setAutostartLoading(false)
      return
    }
    let cancelled = false
    void (async () => {
      const current = await getAutostartEnabled()
      if (!cancelled) {
        setAutostartEnabledState(current)
        setAutostartLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])
  const handleAutostartToggle = async (next: boolean) => {
    const previous = autostartEnabled
    // Optimistic flip so the checkbox feels instant; reverted below on
    // rejection so the visible state never lies about what's actually
    // registered with the OS.
    setAutostartEnabledState(next)
    setAutostartError(null)
    const outcome = await applyAutostart(next)
    if (!outcome.ok) {
      setAutostartEnabledState(previous)
      setAutostartError(outcome.error)
    }
  }
  const [notificationsHint, setNotificationsHint] = useState<string | null>(null)

  const handleNotificationsToggle = async (next: boolean) => {
    if (!next) {
      saveNotificationsEnabled(false)
      setNotificationsEnabled(false)
      setNotificationsHint(null)
      return
    }
    const permission = await ensurePermission()
    if (permission === 'unsupported') {
      setNotificationsHint(t('Notifications are not supported in this window.'))
      return
    }
    if (permission === 'denied') {
      // Desktop uses native toasts (tauri-plugin-notification) — a denial
      // there means the OS blocked the app's notifications.
      setNotificationsHint(
        isTauri()
          ? t('Notifications are blocked for this app in your system settings.')
          : t('The browser blocked notifications — allow them in site settings, then retry.'),
      )
      return
    }
    saveNotificationsEnabled(true)
    setNotificationsEnabled(true)
    setNotificationsHint(null)
  }
  const [featuresSaving, setFeaturesSaving] = useState(false)

  const handleSaveFeatures = async () => {
    setFeaturesSaving(true)
    try {
      const data = await putSettings({
        memory_enable_long_term: memoryLongTerm,
        tools_enabled: toolsEnabled,
        web_search_enabled: webSearchEnabled,
        reminders_enabled: remindersEnabled,
        stt_provider: sttProvider,
      })
      applyUpdateResponse(data)
    } catch {
      // Non-fatal; toggles keep their local state either way.
    } finally {
      setFeaturesSaving(false)
    }
  }

  return (
    <section className="settings-section">
      <h3>{t('Features')}</h3>
      <div className="settings-field-group">
        <label className="settings-toggle-row" htmlFor="memory-long-term-toggle">
          <input
            id="memory-long-term-toggle"
            type="checkbox"
            checked={memoryLongTerm}
            onChange={(event) => setMemoryLongTerm(event.target.checked)}
          />
          <span>{t('Long-term memory')}</span>
        </label>
        <div className="settings-hint">
          {t('Remembers facts about you across sessions (stored locally).')}
        </div>

        <label className="settings-toggle-row" htmlFor="tools-enabled-toggle">
          <input
            id="tools-enabled-toggle"
            type="checkbox"
            checked={toolsEnabled}
            onChange={(event) => setToolsEnabled(event.target.checked)}
          />
          <span>{t('Tools (calculator, code, paper search)')}</span>
        </label>

        <label className="settings-toggle-row" htmlFor="web-search-toggle">
          <input
            id="web-search-toggle"
            type="checkbox"
            checked={webSearchEnabled}
            onChange={(event) => setWebSearchEnabled(event.target.checked)}
          />
          <span>{t('Web search')}</span>
        </label>
        <div className="settings-hint">
          {t('Uses Tavily when an API key is configured, otherwise keyless DuckDuckGo.')}
        </div>

        <label className="settings-toggle-row" htmlFor="reminders-toggle">
          <input
            id="reminders-toggle"
            type="checkbox"
            checked={remindersEnabled}
            onChange={(event) => setRemindersEnabled(event.target.checked)}
          />
          <span>{t('Reminders & timers')}</span>
        </label>

        <label className="settings-toggle-row" htmlFor="notifications-toggle">
          <input
            id="notifications-toggle"
            type="checkbox"
            checked={notificationsEnabled}
            onChange={(event) => void handleNotificationsToggle(event.target.checked)}
          />
          <span>{t('Desktop notifications')}</span>
        </label>
        <div className="settings-hint">
          {t(
            'Only while the app is hidden: due reminders, plus at most one daily nudge each for the morning brief and her diary. Applies immediately (no save needed). Never used to pull you back in.',
          )}
          {notificationsHint && <> {notificationsHint}</>}
        </div>

        <label className="settings-toggle-row" htmlFor="talk-body-toggle">
          <input
            id="talk-body-toggle"
            type="checkbox"
            checked={talkBodyEnabled}
            onChange={(event) => {
              setTalkBodyEnabled(event.target.checked)
              saveTalkBodyEnabled(event.target.checked)
            }}
          />
          <span>{t('Body animation while talking')}</span>
        </label>
        <div className="settings-hint">
          {t(
            'Off (default): talking moves only her mouth, eyes and natural idle sway. On: she also plays talking-gesture clips while speaking — these loop, which can look repetitive on long replies. Applies to the next reply.',
          )}
        </div>

        <label className="settings-toggle-row" htmlFor="hands-free-toggle">
          <input
            id="hands-free-toggle"
            type="checkbox"
            checked={handsFreeEnabled}
            onChange={(event) => {
              setHandsFreeEnabled(event.target.checked)
              saveHandsFreeEnabled(event.target.checked)
            }}
          />
          <span>{t('Hands-free conversation')}</span>
        </label>
        <div className="settings-hint">
          {t(
            'The mic stays open (echo-cancelled): just speak, and she answers when you pause — no push-to-talk needed. Push-to-talk still works and pauses hands-free while held. Applies immediately; needs speech-to-text.',
          )}
        </div>

        <label className="settings-toggle-row" htmlFor="duplex-toggle">
          <input
            id="duplex-toggle"
            type="checkbox"
            checked={duplexEnabled}
            onChange={(event) => {
              setDuplexEnabled(event.target.checked)
              saveDuplexEnabled(event.target.checked)
            }}
          />
          <span>{t('Full-duplex calls')}</span>
        </label>
        <div className="settings-hint">
          {t(
            'With this on, the 📞 Call button runs a true full-duplex conversation: the mic streams continuously, she answers when you pause, and talking over her mid-reply interrupts her — like a phone call. Falls back to the classic call mode automatically if the server has it off (web.duplex_enabled) or the mic is unavailable.',
          )}
        </div>

        <label className="settings-toggle-row" htmlFor="wake-word-toggle">
          <input
            id="wake-word-toggle"
            type="checkbox"
            checked={wakeWordEnabled}
            onChange={(event) => {
              setWakeWordEnabled(event.target.checked)
              saveWakeWordEnabled(event.target.checked)
            }}
          />
          <span>{t('Wake word')}</span>
        </label>
        <div className="settings-hint">
          {t(
            'Say the wake phrase (default "hey Jarvis") and she starts a call — no button. The mic stays open while idle, but detection runs entirely on your own backend (a small local model; audio is never stored and never leaves the machine). Needs the server\'s wake_word.enabled flag plus a one-time model download (scripts/fetch_wake_word.py).',
          )}
        </div>

        {isTauri() && (
          <>
            <label className="settings-toggle-row" htmlFor="pet-default-toggle">
              <input
                id="pet-default-toggle"
                type="checkbox"
                checked={petDefaultEnabled}
                onChange={(event) => {
                  setPetDefaultEnabled(event.target.checked)
                  savePetDefaultEnabled(event.target.checked)
                }}
              />
              <span>{t('Start as desktop pet')}</span>
            </label>
            <div className="settings-hint">
              {t(
                'On the next launch she appears as the desktop pet; the chat window stays one click away (tray or the pet’s chat button). Closing the chat window while the pet is up hides it instead of quitting.',
              )}
            </div>

            {/* Autostart has no client-side source of truth — get_autostart
                is queried live, and a failed query means the toggle is
                unavailable (older shell without the command, or the OS API
                rejected) rather than silently reading as "off". */}
            {!autostartLoading && autostartEnabled !== null && (
              <>
                <label className="settings-toggle-row" htmlFor="autostart-toggle">
                  <input
                    id="autostart-toggle"
                    type="checkbox"
                    checked={autostartEnabled}
                    onChange={(event) => void handleAutostartToggle(event.target.checked)}
                  />
                  <span>{t('Start with Windows')}</span>
                </label>
                <div className="settings-hint">
                  {t('She launches automatically when you log in.')}
                  {autostartError && <> {t('Could not change this: {error}', { error: autostartError })}</>}
                </div>
              </>
            )}
            {!autostartLoading && autostartEnabled === null && (
              <div
                className="settings-toggle-row settings-toggle-row-disabled"
                title={t('Start with Windows is unavailable in this build.')}
              >
                <input type="checkbox" checked={false} disabled aria-disabled="true" />
                <span>{t('Start with Windows')}</span>
              </div>
            )}
          </>
        )}

        <label className="settings-label" htmlFor="stt-provider-select">
          {t('Voice input')}
        </label>
        <select
          id="stt-provider-select"
          value={sttProvider}
          onChange={(event) => setSttProvider(event.target.value)}
        >
          <option value="faster_whisper">{t('Push-to-talk (Whisper, local)')}</option>
          <option value="dummy">{t('Disabled')}</option>
        </select>
      </div>
      <div className="settings-action-row settings-save-row">
        <button
          type="button"
          className="settings-primary-button"
          onClick={handleSaveFeatures}
          disabled={featuresSaving}
        >
          {featuresSaving ? t('Saving…') : t('Save features')}
        </button>
      </div>
    </section>
  )
}

export default FeaturesSection
