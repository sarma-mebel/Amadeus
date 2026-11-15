/** Signed in-app auto-update (Tauri desktop only).
 *
 * Thin JS face over the Rust `check_app_update` / `install_app_update`
 * commands (tauri-plugin-updater behind our own commands — the autostart
 * convention, no plugin guest bindings). The GitHub-releases nudge in
 * updateCheck.ts stays the fallback for browsers, portable/GPU builds
 * (`portable: true` — the packaging step drops a portable.marker next to
 * the exe), and any check failure: auto-update is an upgrade to the nudge,
 * never a replacement.
 *
 * Failure-silent like updateCheck.ts: offline, unreachable endpoint, or a
 * dev build simply reports null — at most one console.debug per launch.
 */

import { isTauri } from './platform'

export interface AppUpdateStatus {
  available: boolean
  version: string | null
  notes: string | null
  portable: boolean
}

export interface AppUpdateProgress {
  downloaded: number
  /** Content-length when the server sent one; null = indeterminate bar. */
  total: number | null
  done: boolean
}

let failureLogged = false
const logFailureOnce = (reason: unknown): void => {
  if (failureLogged) return
  failureLogged = true
  console.debug('app update check skipped:', reason)
}

/** Checks the signed-update endpoint. Resolves null anywhere it can't
 * answer (browser, offline, updater unconfigured) — callers fall back to
 * the releases nudge. */
export const checkAppUpdate = async (): Promise<AppUpdateStatus | null> => {
  if (!isTauri()) return null
  try {
    const { invoke } = await import('@tauri-apps/api/core')
    return await invoke<AppUpdateStatus>('check_app_update')
  } catch (error) {
    logFailureOnce(error)
    return null
  }
}

export interface InstallResult {
  ok: boolean
  error?: string
}

/** Download + install + relaunch. Progress arrives via the Rust side's
 * `app-update-progress` events. On success this promise usually never
 * resolves — the process exits into the installer — so treat resolution
 * with `ok: true` as "install handed off". */
export const installAppUpdate = async (
  onProgress: (progress: AppUpdateProgress) => void,
): Promise<InstallResult> => {
  if (!isTauri()) return { ok: false, error: 'not a desktop build' }
  let unlisten: (() => void) | null = null
  try {
    const { listen } = await import('@tauri-apps/api/event')
    const { invoke } = await import('@tauri-apps/api/core')
    unlisten = await listen<AppUpdateProgress>('app-update-progress', (event) =>
      onProgress(event.payload),
    )
    await invoke('install_app_update')
    return { ok: true }
  } catch (error) {
    return { ok: false, error: String(error) }
  } finally {
    unlisten?.()
  }
}

/** Test hook: reset the log-once flag. */
export const __resetAppUpdaterForTests = (): void => {
  failureLogged = false
}
