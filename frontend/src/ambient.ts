/**
 * Ambient desktop awareness: pure transition detection over the Tauri
 * `ambient_signals` snapshots (focused-window title, battery, now-playing).
 *
 * The design split: the Rust command is a dumb zero-cost snapshot, THIS
 * module decides what's noteworthy (a transition, never a state), and the
 * server (`/api/ambient` → presence.ambient_comment) decides whether she
 * actually says anything (cooldown + SKIP). Continuous cheap awareness,
 * rationed commentary — the inverse of the screenshot-based screen
 * companion, which stays a separate privacy opt-in.
 *
 * Pure and timer-free so the transition rules are unit-testable; the
 * polling/delivery side lives in hooks/useAmbientSignals.ts.
 */

export interface AmbientSnapshot {
  window_title: string | null
  battery_percent: number | null
  battery_charging: boolean | null
  media_title: string | null
  media_artist: string | null
  media_playing: boolean | null
}

export type AmbientKind = 'focus' | 'battery' | 'music'

export interface AmbientEvent {
  kind: AmbientKind
  detail: string
  /** Structured discriminator for client-side consumers (the reflex layer's
   * battery-critical glance routes on this — `detail` is a free-text
   * sentence and matching on its wording would be a silent-death trap).
   * Only battery events carry it; the server POST ignores it. */
  severity?: 'low' | 'critical'
}

/** Same app focused this long, continuously, before she may remark on it —
 * and again after each further stretch of the same length (3h, 4.5h...). */
export const FOCUS_STRETCH_MS = 90 * 60_000
export const BATTERY_LOW_PERCENT = 15
export const BATTERY_CRITICAL_PERCENT = 8
/** Floor between music events. Owner's call (2026-08-03): every track
 * change should reach the server — the old 30-minute gap meant she ignored
 * whole albums. 60s only damps pause/unpause toggling on the same song;
 * any real track change clears it. Pacing now belongs to the server
 * cooldown (ambient.cooldown_seconds) plus her SKIP judgment. */
export const MUSIC_MIN_GAP_MS = 60_000

/** The stable "which app is this" key for a window title: titles mutate per
 * document ("notes.md - project - VS Code"), but the app name conventionally
 * holds the last " - " segment. Empty/blank titles yield null (no signal). */
export const appKeyFromTitle = (title: string | null): string | null => {
  const trimmed = (title || '').trim()
  if (!trimmed) return null
  const parts = trimmed.split(' - ')
  const key = (parts[parts.length - 1] || '').trim()
  return key || null
}

interface TrackerState {
  focusKey: string | null
  focusSinceMs: number
  focusReportedStretches: number
  batteryLowReported: boolean
  batteryCriticalReported: boolean
  trackKey: string | null
  wasPlaying: boolean
  lastMusicEventMs: number
}

export interface AmbientTracker {
  ingest(snapshot: AmbientSnapshot, nowMs: number): AmbientEvent[]
}

const hoursText = (ms: number): string => {
  const hours = ms / 3_600_000
  const rounded = Math.round(hours * 2) / 2
  return rounded === 1 ? '1 hour' : `${rounded} hours`
}

export const createAmbientTracker = (): AmbientTracker => {
  const state: TrackerState = {
    focusKey: null,
    focusSinceMs: 0,
    focusReportedStretches: 0,
    batteryLowReported: false,
    batteryCriticalReported: false,
    trackKey: null,
    wasPlaying: false,
    lastMusicEventMs: -Infinity,
  }

  const ingestFocus = (snapshot: AmbientSnapshot, nowMs: number): AmbientEvent[] => {
    const key = appKeyFromTitle(snapshot.window_title)
    // Her own windows don't count: "you've been staring at me for two
    // hours" is presence's job, not ambient's.
    if (key === null || /amadeus|kurisu/i.test(key)) {
      state.focusKey = null
      return []
    }
    if (key !== state.focusKey) {
      state.focusKey = key
      state.focusSinceMs = nowMs
      state.focusReportedStretches = 0
      return []
    }
    const elapsed = nowMs - state.focusSinceMs
    const stretches = Math.floor(elapsed / FOCUS_STRETCH_MS)
    if (stretches <= state.focusReportedStretches) return []
    state.focusReportedStretches = stretches
    return [
      {
        kind: 'focus',
        detail:
          `The user has had ${key} focused for about ` +
          `${hoursText(elapsed)} straight.`,
      },
    ]
  }

  const ingestBattery = (snapshot: AmbientSnapshot): AmbientEvent[] => {
    const percent = snapshot.battery_percent
    const charging = snapshot.battery_charging
    if (percent === null) return []
    // Plugging in (or recovering well clear of the threshold) re-arms the
    // one-shot flags for the next discharge cycle.
    if (charging === true || percent >= BATTERY_LOW_PERCENT + 10) {
      state.batteryLowReported = false
      state.batteryCriticalReported = false
      return []
    }
    if (charging !== false) return []
    if (percent <= BATTERY_CRITICAL_PERCENT && !state.batteryCriticalReported) {
      state.batteryCriticalReported = true
      state.batteryLowReported = true
      return [
        {
          kind: 'battery',
          detail: `The laptop battery is down to ${percent}% and not charging.`,
          severity: 'critical',
        },
      ]
    }
    if (percent <= BATTERY_LOW_PERCENT && !state.batteryLowReported) {
      state.batteryLowReported = true
      return [
        {
          kind: 'battery',
          detail: `The laptop battery is at ${percent}% and not charging.`,
          severity: 'low',
        },
      ]
    }
    return []
  }

  const ingestMusic = (snapshot: AmbientSnapshot, nowMs: number): AmbientEvent[] => {
    const playing = snapshot.media_playing === true && !!snapshot.media_title
    const key = playing
      ? `${snapshot.media_title}|${snapshot.media_artist || ''}`
      : null
    const started = playing && !state.wasPlaying
    const changed = playing && state.wasPlaying && key !== state.trackKey
    state.wasPlaying = playing
    state.trackKey = key
    if (!started && !changed) return []
    if (nowMs - state.lastMusicEventMs < MUSIC_MIN_GAP_MS) return []
    state.lastMusicEventMs = nowMs
    const artist = snapshot.media_artist ? ` by ${snapshot.media_artist}` : ''
    return [
      {
        kind: 'music',
        detail: `The user is listening to "${snapshot.media_title}"${artist}.`,
      },
    ]
  }

  return {
    ingest(snapshot: AmbientSnapshot, nowMs: number): AmbientEvent[] {
      return [
        ...ingestFocus(snapshot, nowMs),
        ...ingestBattery(snapshot),
        ...ingestMusic(snapshot, nowMs),
      ]
    },
  }
}
