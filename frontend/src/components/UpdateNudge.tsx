import { useEffect, useState } from 'react'
import { t } from '../i18n'
import { fetchBackendStatus } from '../statusApi'
import { checkForUpdate, dismissUpdate, openReleasesPage } from '../updateCheck'
import {
  checkAppUpdate,
  installAppUpdate,
  type AppUpdateProgress,
} from '../appUpdater'

/** Compact dismissible "new release" row for the settings panel.
 *
 * Two tiers. On a Tauri INSTALLED build (not portable), the signed
 * auto-updater is asked first: an available update renders an
 * "Update & restart" button that downloads with progress, swaps the app,
 * and relaunches — no browser, no manual install. Everywhere else
 * (browser, portable/GPU builds, updater unreachable) it falls back to the
 * original GitHub-releases nudge: read the running backend version (and
 * the demo_mode gate) from /api/status, ask the 24h-cached releases check,
 * and link out. Renders nothing when there's no update anywhere, and
 * everything is failure-silent — the nudge simply doesn't appear.
 * Dismissing remembers the exact tag (localStorage), so the same version
 * never nudges again while a newer one still does.
 */
const UpdateNudge = () => {
  const [tag, setTag] = useState<string | null>(null)
  const [autoVersion, setAutoVersion] = useState<string | null>(null)
  const [progress, setProgress] = useState<AppUpdateProgress | null>(null)
  const [installError, setInstallError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    void checkAppUpdate().then((status) => {
      if (cancelled) return
      if (status?.available && !status.portable && status.version) {
        // Auto-update path wins; the releases nudge stays quiet.
        setAutoVersion(status.version)
        return
      }
      void fetchBackendStatus().then((backend) => {
        if (!backend || cancelled) return
        void checkForUpdate(backend.version, {
          demoMode: backend.demo_mode,
        }).then((next) => {
          if (!cancelled && next) setTag(next)
        })
      })
    })
    return () => {
      cancelled = true
    }
  }, [])

  if (autoVersion) {
    const installing = progress !== null && !installError
    const percent =
      progress && progress.total
        ? Math.min(100, Math.round((progress.downloaded / progress.total) * 100))
        : null
    return (
      <div className="settings-update-nudge" role="status">
        <span className="settings-update-nudge-label">
          {installing
            ? percent === null
              ? t('Downloading update…')
              : t('Downloading update… {percent}%', { percent: String(percent) })
            : installError
              ? t('Update failed — try the releases page')
              : t('{version} is ready to install', { version: autoVersion })}
        </span>
        {!installing && !installError && (
          <button
            type="button"
            className="settings-update-nudge-link"
            onClick={() => {
              setInstallError(null)
              setProgress({ downloaded: 0, total: null, done: false })
              void installAppUpdate(setProgress).then((result) => {
                // ok:true normally never lands here — the process exits
                // into the installer. Anything else is a real failure.
                if (!result.ok) {
                  setProgress(null)
                  setInstallError(result.error ?? 'unknown')
                }
              })
            }}
          >
            {t('Update & restart')}
          </button>
        )}
        {installError && (
          <button
            type="button"
            className="settings-update-nudge-link"
            onClick={openReleasesPage}
          >
            {t('Releases')} ↗
          </button>
        )}
        {!installing && (
          <button
            type="button"
            className="settings-update-nudge-dismiss"
            aria-label={t('Dismiss update notice')}
            onClick={() => setAutoVersion(null)}
          >
            ×
          </button>
        )}
      </div>
    )
  }

  if (!tag) return null
  return (
    <div className="settings-update-nudge" role="status">
      <span className="settings-update-nudge-label">
        {t('{version} is available', { version: tag })}
      </span>
      <button
        type="button"
        className="settings-update-nudge-link"
        onClick={openReleasesPage}
      >
        {t('Releases')} ↗
      </button>
      <button
        type="button"
        className="settings-update-nudge-dismiss"
        aria-label={t('Dismiss update notice')}
        onClick={() => {
          dismissUpdate(tag)
          setTag(null)
        }}
      >
        ×
      </button>
    </div>
  )
}

export default UpdateNudge
