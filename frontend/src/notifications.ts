/**
 * Opt-in desktop notifications, with the ethics built into the API:
 *
 * - Default OFF. Enabling is an explicit settings toggle, and the browser
 *   permission prompt only appears at that moment.
 * - Only while the app is hidden (minimized/background tab). When she's on screen
 *   she speaks for herself; a notification would be noise.
 * - Hard caps: the morning brief and her diary notify at most once per day.
 *   Reminders notify as they fire because the user explicitly set them.
 * - No streaks, no "she misses you", no re-engagement pings. Ever.
 *
 * See docs/companion-ethics.md for the full design rules.
 */

import { isTauri } from './platform'
import { localDate } from './dates'

export type NotifyKind = 'reminder' | 'brief' | 'diary' | 'reachout'

// Tauri path: the WebView2 host auto-denies the WEB Notification permission
// (verified live), so the desktop app uses tauri-plugin-notification's
// native toasts instead. Desktop platforms report permission granted, but
// the JS API is async while canNotify() is sync — cache the answer.
let tauriGranted: boolean | null = null

const refreshTauriPermission = async (): Promise<boolean> => {
  try {
    const plugin = await import('@tauri-apps/plugin-notification')
    let granted = await plugin.isPermissionGranted()
    if (!granted) {
      granted = (await plugin.requestPermission()) === 'granted'
    }
    tauriGranted = granted
    return granted
  } catch {
    tauriGranted = false
    return false
  }
}

// Warm the cache on load so canNotify() answers correctly after a restart
// with the toggle already on. No permission PROMPT fires on desktop — the
// plugin reports granted without UI on Windows.
if (typeof window !== 'undefined' && isTauri()) {
  try {
    if (localStorage.getItem('kurisu_notifications') === 'true') {
      void refreshTauriPermission()
    }
  } catch {
    // Cache stays cold; the settings toggle will warm it.
  }
}

const ENABLED_KEY = 'kurisu_notifications'

const DAILY_CAP_KEYS: Partial<Record<NotifyKind, string>> = {
  brief: 'kurisu_notified_brief',
  diary: 'kurisu_notified_diary',
  // The server generates at most one reach-out per day, but the cap is
  // enforced client-side too — belt and braces on the ethics rule.
  reachout: 'kurisu_notified_reachout',
}

export const notificationsSupported = (): boolean =>
  typeof window !== 'undefined' && (isTauri() || 'Notification' in window)

export const loadNotificationsEnabled = (): boolean => {
  try {
    return localStorage.getItem(ENABLED_KEY) === 'true'
  } catch {
    return false
  }
}

export const saveNotificationsEnabled = (enabled: boolean): void => {
  try {
    localStorage.setItem(ENABLED_KEY, enabled ? 'true' : 'false')
  } catch {
    // Storage failure just means the toggle doesn't stick.
  }
}

/** Ask the browser for permission (call only from the settings toggle —
 * permission prompts out of nowhere are exactly the pushiness we avoid). */
export const ensurePermission = async (): Promise<'granted' | 'denied' | 'unsupported'> => {
  if (!notificationsSupported()) return 'unsupported'
  if (isTauri()) {
    return (await refreshTauriPermission()) ? 'granted' : 'denied'
  }
  if (Notification.permission === 'granted') return 'granted'
  if (Notification.permission === 'denied') return 'denied'
  try {
    const result = await Notification.requestPermission()
    return result === 'granted' ? 'granted' : 'denied'
  } catch {
    return 'denied'
  }
}

// Local date (see dates.ts): daily caps should roll at local midnight, the
// same boundary the backend's brief/diary dates use.
const today = (): string => localDate()

/** Whether a notification of this kind would fire right now. Checked before
 * doing any network work so capped kinds don't even cost a fetch. */
export const canNotify = (kind: NotifyKind): boolean => {
  if (!loadNotificationsEnabled()) return false
  if (isTauri()) {
    if (tauriGranted !== true) return false
  } else if (!notificationsSupported() || Notification.permission !== 'granted') {
    return false
  }
  // Only when she can't just say it herself.
  if (typeof document === 'undefined' || !document.hidden) return false
  const capKey = DAILY_CAP_KEYS[kind]
  if (capKey) {
    try {
      if (localStorage.getItem(capKey) === today()) return false
    } catch {
      return false
    }
  }
  return true
}

/** Show a notification if the rules allow it. Returns whether it fired. */
export const notify = (kind: NotifyKind, title: string, body: string): boolean => {
  if (!canNotify(kind)) return false
  if (isTauri()) {
    // Fire-and-forget: the cap is stamped optimistically below, which is
    // the safe failure direction (worst case a capped kind stays quiet).
    void import('@tauri-apps/plugin-notification')
      .then((plugin) => plugin.sendNotification({ title, body }))
      .catch(() => undefined)
  } else {
    try {
      new Notification(title, { body, tag: `kurisu-${kind}` })
    } catch {
      return false
    }
  }
  const capKey = DAILY_CAP_KEYS[kind]
  if (capKey) {
    try {
      localStorage.setItem(capKey, today())
    } catch {
      // Cap stamp failed; worst case the kind fires again today.
    }
  }
  return true
}
