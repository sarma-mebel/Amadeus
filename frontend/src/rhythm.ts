/**
 * Rhythm aggregation: the shape of the user's week, derived CLIENT-SIDE
 * from the same ambient snapshots `ambient.ts` reads, and reduced to
 * coarse per-day counters before anything crosses the network.
 *
 * Why the aggregation lives here and not on the server. `ambient_signals`
 * runs in the same process as this UI, so the raw window title
 * ("salary-negotiation-notes.md - Obsidian") is already in the renderer.
 * By bucketing here, that string is provably never in an HTTP body, a
 * server log line, a crash dump, or FastAPI's 422 echo path. The privacy
 * property becomes structural rather than a policy someone has to keep
 * honoring — which matters because Amadeus is also deployable as a hosted
 * web app, where "it's all local anyway" stops being true.
 *
 * What crosses the wire is only ever the counters in `RhythmDelta`:
 * how MUCH and roughly WHEN, plus which coarse CATEGORY of app. Never a
 * window title, never a document name, never a track name or artist,
 * never a per-event timestamp.
 *
 * The app CATEGORY (editor/browser/terminal/…) is a deliberate scope
 * decision by the owner (2026-08-07) — it is what makes "you've been deep
 * in the editor all week" possible, and it is also the most identifying
 * thing stored here. It is bucketed to a fixed six-value vocabulary and an
 * unrecognized app becomes 'other', so the store never accumulates a list
 * of the user's software. docs/companion-ethics.md describes exactly this.
 */

import { appKeyFromTitle, type AmbientSnapshot } from './ambient'
import { localDate } from './dates'

/** The fixed vocabulary. An app that matches nothing becomes 'other' —
 * the store must never accumulate free-text app names. */
export type AppBucket = 'editor' | 'browser' | 'terminal' | 'chat' | 'media' | 'other'

export const APP_BUCKETS: readonly AppBucket[] = [
  'editor',
  'browser',
  'terminal',
  'chat',
  'media',
  'other',
]

/** Local hours counted as "late night" for the rhythm tally. Matches the
 * sleep-schedule default's spirit (she is asleep 1->8) without importing
 * it: this is about the USER being up, not about her. */
export const LATE_NIGHT_START_HOUR = 0
export const LATE_NIGHT_END_HOUR = 5

const BUCKET_PATTERNS: ReadonlyArray<[AppBucket, RegExp]> = [
  ['editor', /code|vim|emacs|sublime|intellij|pycharm|webstorm|xcode|notepad|obsidian|word|docs/i],
  ['browser', /chrome|firefox|edge|safari|brave|opera|vivaldi|arc\b/i],
  ['terminal', /terminal|iterm|powershell|cmd\.exe|console|wsl|bash|zsh|tmux/i],
  ['chat', /slack|discord|teams|zoom|telegram|signal|whatsapp|outlook|mail/i],
  ['media', /spotify|youtube|vlc|netflix|music|player|twitch|premiere|resolve/i],
]

/** Map a window-title app key onto the fixed vocabulary. Exported for
 * tests: the guarantee that nothing outside APP_BUCKETS can be stored is
 * the whole privacy argument for shipping app categories at all. */
export const bucketForApp = (appKey: string | null): AppBucket | null => {
  const key = (appKey || '').trim()
  if (!key) return null
  for (const [bucket, pattern] of BUCKET_PATTERNS) {
    if (pattern.test(key)) return bucket
  }
  return 'other'
}

/** One local day's tally. This is the ENTIRE durable artifact — if a field
 * is not here, it is not stored, and new fields need an ethics-doc update. */
export interface RhythmDelta {
  /** ISO local date. */
  date: string
  /** 30-minute blocks with activity between LATE_NIGHT_START/END_HOUR. */
  lateNightBlocks: number
  /** How many FOCUS_STRETCH_MS boundaries were crossed. */
  longFocusStretches: number
  /** Longest single continuous stretch today, in hours, rounded to 0.5. */
  longestFocusHours: number
  /** Sorted unique local hours (0-23) with any activity. ~24 bits/day. */
  activeHourBuckets: number[]
  /** Counts only — never titles or artists. */
  musicEvents: number
  batteryLowEvents: number
  /** Coarse category -> number of 30-minute blocks focused there. */
  appBuckets: Partial<Record<AppBucket, number>>
}

const emptyDelta = (date: string): RhythmDelta => ({
  date,
  lateNightBlocks: 0,
  longFocusStretches: 0,
  longestFocusHours: 0,
  activeHourBuckets: [],
  musicEvents: 0,
  batteryLowEvents: 0,
  appBuckets: {},
})

const BLOCK_MS = 30 * 60_000

export interface RhythmTracker {
  /** Fold one ambient snapshot into today's tally. Returns the current
   * day's delta, or null when nothing has been recorded yet today. */
  ingest(snapshot: AmbientSnapshot, nowMs: number): RhythmDelta | null
  /** The current day's tally without recording anything. */
  current(): RhythmDelta | null
}

interface TrackerState {
  delta: RhythmDelta | null
  focusKey: string | null
  focusSinceMs: number
  focusStretchesCounted: number
  lastBlockStamp: number | null
  wasPlaying: boolean
  batteryLowLatched: boolean
}

export const createRhythmTracker = (): RhythmTracker => {
  const state: TrackerState = {
    delta: null,
    focusKey: null,
    focusSinceMs: 0,
    focusStretchesCounted: 0,
    lastBlockStamp: null,
    wasPlaying: false,
    batteryLowLatched: false,
  }

  const rollDay = (nowMs: number): RhythmDelta => {
    const date = localDate(nowMs)
    if (state.delta === null || state.delta.date !== date) {
      state.delta = emptyDelta(date)
      state.focusStretchesCounted = 0
      state.lastBlockStamp = null
      // Re-anchor the focus stretch at the day boundary: a stretch spanning
      // midnight otherwise gets re-credited WHOLE to the new day (yesterday's
      // stretch count added again, longestFocusHours starting at hours the
      // old day already recorded). The new day only counts time actually in
      // it; focusKey survives so the app is still "the same focus".
      state.focusSinceMs = nowMs
    }
    return state.delta
  }

  return {
    current: () => state.delta,

    ingest(snapshot: AmbientSnapshot, nowMs: number): RhythmDelta | null {
      const delta = rollDay(nowMs)
      const now = new Date(nowMs)
      const hour = now.getHours()

      // One 30-minute block is the unit of "activity" — polling is every
      // 15s, so counting raw ticks would make the numbers meaningless and
      // also much more identifying.
      const blockStamp = Math.floor(nowMs / BLOCK_MS)
      const isNewBlock = state.lastBlockStamp !== blockStamp
      state.lastBlockStamp = blockStamp

      if (!delta.activeHourBuckets.includes(hour)) {
        delta.activeHourBuckets = [...delta.activeHourBuckets, hour].sort((a, b) => a - b)
      }
      if (isNewBlock && hour >= LATE_NIGHT_START_HOUR && hour < LATE_NIGHT_END_HOUR) {
        delta.lateNightBlocks += 1
      }

      // Focus: stretch counting mirrors ambient.ts's rule so the two agree
      // on what "a long stretch" means, but only the CATEGORY is kept.
      const appKey = appKeyFromTitle(snapshot.window_title)
      const selfWindow = appKey !== null && /amadeus|kurisu/i.test(appKey)
      if (appKey === null || selfWindow) {
        state.focusKey = null
      } else {
        if (appKey !== state.focusKey) {
          state.focusKey = appKey
          state.focusSinceMs = nowMs
          state.focusStretchesCounted = 0
        }
        const bucket = bucketForApp(appKey)
        if (bucket && isNewBlock) {
          delta.appBuckets[bucket] = (delta.appBuckets[bucket] || 0) + 1
        }
        const elapsedHours = (nowMs - state.focusSinceMs) / 3_600_000
        const rounded = Math.round(elapsedHours * 2) / 2
        if (rounded > delta.longestFocusHours) delta.longestFocusHours = rounded
        const stretches = Math.floor((nowMs - state.focusSinceMs) / (90 * 60_000))
        if (stretches > state.focusStretchesCounted) {
          delta.longFocusStretches += stretches - state.focusStretchesCounted
          state.focusStretchesCounted = stretches
        }
      }

      const playing = snapshot.media_playing === true && !!snapshot.media_title
      if (playing && !state.wasPlaying) delta.musicEvents += 1
      state.wasPlaying = playing

      const percent = snapshot.battery_percent
      if (percent !== null && snapshot.battery_charging === false && percent <= 15) {
        if (!state.batteryLowLatched) {
          delta.batteryLowEvents += 1
          state.batteryLowLatched = true
        }
      } else if (percent !== null && (snapshot.battery_charging === true || percent > 25)) {
        state.batteryLowLatched = false
      }

      return delta
    },
  }
}

/** Wire shape for POST /api/rhythm — snake_case to match the backend. */
export const toWirePayload = (delta: RhythmDelta) => ({
  date: delta.date,
  late_night_blocks: delta.lateNightBlocks,
  long_focus_stretches: delta.longFocusStretches,
  longest_focus_hours: delta.longestFocusHours,
  active_hour_buckets: delta.activeHourBuckets,
  music_events: delta.musicEvents,
  battery_low_events: delta.batteryLowEvents,
  app_buckets: delta.appBuckets,
})
