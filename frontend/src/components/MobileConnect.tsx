import { useState } from 'react'
import { saveBackendUrl } from '../config'
import { t, useLocale } from '../i18n'
import './DesktopBackendWait.css'
import './MobileConnect.css'

/**
 * First-run gate for the Capacitor (Android) app: unlike the desktop shell
 * there is no bundled backend, so nothing works until the user points the
 * app at their self-hosted server. Reuses the SG0 terminal look of
 * DesktopBackendWait, plus a URL field and a /healthz probe. On success the
 * URL is saved as the kurisu_backend_url override (which every API, WS, and
 * asset path honors) and the app reloads onto it.
 */

const normalizeUrl = (raw: string): string => {
  let url = raw.trim().replace(/\/+$/, '')
  if (!url) return ''
  if (!/^https?:\/\//i.test(url)) url = `http://${url}`
  return url
}

const probeHealthz = async (base: string): Promise<boolean> => {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), 6000)
  try {
    const response = await fetch(`${base}/healthz`, { signal: controller.signal })
    return response.ok
  } catch {
    return false
  } finally {
    window.clearTimeout(timer)
  }
}

const MobileConnect = () => {
  useLocale()
  const [input, setInput] = useState('')
  const [probing, setProbing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const connect = async () => {
    const base = normalizeUrl(input)
    if (!base || probing) return
    setProbing(true)
    setError(null)
    const ok = await probeHealthz(base)
    if (!ok) {
      setProbing(false)
      setError(
        t(
          'No lab network at that address. Check that the server is running with ' +
            '--host 0.0.0.0 and that your phone is on the same network.',
        ),
      )
      return
    }
    saveBackendUrl(base)
    // Full reload so the API base, WebSocket, and asset URLs all pick up
    // the override from the very first request.
    window.location.reload()
  }

  return (
    <div className="desktop-wait mobile-connect" role="dialog" aria-label={t('Connect to your server')}>
      <div className="desktop-wait-scanlines" aria-hidden="true" />

      <div className="desktop-wait-topbar" aria-hidden="true">
        <span className="desktop-wait-topbar-lab">
          Future Gadget Lab — No. 004
          <br />
          <span className="muted">Field Terminal</span>
        </span>
        <span className="desktop-wait-topbar-net">LAB NETWORK: UNLINKED</span>
      </div>

      <div className="desktop-wait-center">
        <h1 className="desktop-wait-wordmark">AMADEUS</h1>
        <div className="desktop-wait-subtitle">COGNITIVE MEMORY INTERFACE</div>

        <p className="desktop-wait-detail mobile-connect-copy">
          {t(
            "This app is a window to your own Amadeus server. Enter the address of the machine running it — usually your PC's local IP, like http://192.168.1.20:8000.",
          )}
        </p>

        <form
          className="mobile-connect-form"
          // No native URL validation: scheme-less input like
          // "192.168.1.20:8000" is expected and normalized on submit.
          noValidate
          onSubmit={(event) => {
            event.preventDefault()
            void connect()
          }}
        >
          <input
            type="text"
            inputMode="url"
            autoCapitalize="off"
            autoCorrect="off"
            spellCheck={false}
            placeholder="http://192.168.1.20:8000"
            value={input}
            onChange={(event) => {
              setInput(event.target.value)
              setError(null)
            }}
            aria-label={t('Backend server URL')}
          />
          <button type="submit" disabled={probing || !input.trim()}>
            {probing ? t('LINKING…') : t('ESTABLISH LINK')}
          </button>
        </form>

        {error && <p className="desktop-wait-detail mobile-connect-error">{error}</p>}

        <p className="desktop-wait-detail mobile-connect-hint">
          {t('On the server:')} <code>uvicorn amadeus.web.app:app --host 0.0.0.0</code>
          &nbsp;{t('— you can change this address later in Settings → Access.')}
        </p>
      </div>
    </div>
  )
}

export default MobileConnect
