/**
 * "Access" settings section: locale, API token, backend URL. All three are
 * purely local preferences (localStorage) applied immediately — no backend
 * round trip. Rendered OUTSIDE the settings-loaded gate on purpose: if the
 * token is wrong, /api/settings itself 401s — these fields must stay
 * reachable so the user can fix it.
 */
import { useState } from 'react'
import { loadApiToken, saveApiToken } from '../../api'
import { loadBackendUrl, saveBackendUrl } from '../../config'
import { saveLocale, t, useLocale, type Locale } from '../../i18n'
import { isCapacitorNative } from '../../platform'

const AccessSection = () => {
  const locale = useLocale()

  // API token — a purely local preference (localStorage), applied
  // immediately: every /api fetch attaches it as a Bearer header and the
  // streaming WebSocket as ?token=. Empty = no auth sent.
  const [apiToken, setApiToken] = useState(() => loadApiToken())
  const handleApiTokenChange = (value: string) => {
    setApiToken(value)
    saveApiToken(value)
  }

  // Backend URL — also local-only (localStorage), read per request, so it
  // applies to new requests; anything already in flight (open WebSocket,
  // pending fetch) keeps the old origin until retried/reloaded.
  const [backendUrl, setBackendUrl] = useState(() => loadBackendUrl())
  const handleBackendUrlChange = (value: string) => {
    setBackendUrl(value)
    saveBackendUrl(value)
  }

  return (
    <section className="settings-section">
      <h3>{t('Access')}</h3>
      <div className="settings-field-group">
        <label className="settings-label" htmlFor="locale-select">
          Language / 言語
        </label>
        <select
          id="locale-select"
          value={locale}
          onChange={(event) => saveLocale(event.target.value as Locale)}
        >
          <option value="en">English</option>
          <option value="ja">日本語</option>
        </select>
        <div className="settings-hint">
          {t('Applies immediately and is stored only on this device.')}
        </div>
      </div>
      <div className="settings-field-group">
        <label className="settings-label" htmlFor="api-token-input">
          {t('API token')} <span className="settings-optional">{t('(optional)')}</span>
        </label>
        <input
          id="api-token-input"
          type="password"
          autoComplete="off"
          placeholder={t("Leave blank if the server doesn't require one")}
          value={apiToken}
          onChange={(event) => handleApiTokenChange(event.target.value)}
        />
        <div className="settings-hint">
          {t(
            'Sent as a Bearer token with every request (and the streaming socket). Stored only on this device; applies immediately — no save needed.',
          )}
        </div>
      </div>
      <div className="settings-field-group">
        <label className="settings-label" htmlFor="backend-url-input">
          {t('Backend URL')} <span className="settings-optional">{t('(optional)')}</span>
        </label>
        <input
          id="backend-url-input"
          type="text"
          autoComplete="off"
          placeholder={t("https://your-server.example.com — blank uses this app's default")}
          value={backendUrl}
          onChange={(event) => handleBackendUrlChange(event.target.value)}
        />
        <div className="settings-hint">
          {t(
            'Point at a remote/hosted backend (include http:// or https://); pairs with the API token above. Stored only on this device and applies to new requests — reload the page to reconnect the streaming socket and refresh loaded data.',
          )}
        </div>
        {isCapacitorNative() && (
          <div className="settings-action-row">
            <button
              type="button"
              className="settings-secondary-button"
              onClick={() => window.location.reload()}
            >
              {t('Reconnect now')}
            </button>
          </div>
        )}
      </div>
    </section>
  )
}

export default AccessSection
