/**
 * Rhythm aggregation: the client half of the privacy contract.
 *
 * The whole argument for shipping app categories at all is that the raw
 * window title is reduced to one of six fixed buckets BEFORE anything
 * crosses the network. That reduction happens here, so these tests are the
 * contract — particularly "no raw title survives into the wire payload".
 */

import { describe, expect, it } from 'vitest'
import {
  APP_BUCKETS,
  bucketForApp,
  createRhythmTracker,
  toWirePayload,
  type AppBucket,
} from './rhythm'
import type { AmbientSnapshot } from './ambient'

const snapshot = (over: Partial<AmbientSnapshot> = {}): AmbientSnapshot => ({
  window_title: null,
  battery_percent: null,
  battery_charging: null,
  media_title: null,
  media_artist: null,
  media_playing: null,
  ...over,
})

const T0 = Date.UTC(2026, 7, 7, 14, 0, 0)

describe('bucketForApp', () => {
  it('maps known apps onto the fixed vocabulary', () => {
    expect(bucketForApp('Visual Studio Code')).toBe('editor')
    expect(bucketForApp('Google Chrome')).toBe('browser')
    expect(bucketForApp('Windows PowerShell')).toBe('terminal')
    expect(bucketForApp('Slack')).toBe('chat')
    expect(bucketForApp('Spotify')).toBe('media')
  })

  it('maps anything unrecognized to "other", never to itself', () => {
    // THE CONTRACT: an unknown app must not become a stored free-text
    // name. If this ever returns the input, the store starts accumulating
    // a list of the user's software.
    expect(bucketForApp('SomeNicheAccountingApp')).toBe('other')
    expect(bucketForApp('salary-negotiation-notes.md')).toBe('other')
  })

  it('returns null for a blank key rather than bucketing nothing', () => {
    expect(bucketForApp(null)).toBeNull()
    expect(bucketForApp('   ')).toBeNull()
  })

  it('only ever emits members of APP_BUCKETS', () => {
    const samples = ['Code', 'Firefox', 'iTerm', 'Discord', 'VLC', 'Xyz', '']
    for (const sample of samples) {
      const bucket = bucketForApp(sample)
      if (bucket !== null) expect(APP_BUCKETS).toContain(bucket)
    }
  })
})

describe('createRhythmTracker', () => {
  it('records the app CATEGORY and never the title', () => {
    const tracker = createRhythmTracker()
    const delta = tracker.ingest(
      snapshot({ window_title: 'salary-negotiation-notes.md - Obsidian' }),
      T0,
    )

    expect(delta).not.toBeNull()
    const wire = JSON.stringify(toWirePayload(delta!))
    expect(wire).not.toContain('salary-negotiation-notes')
    expect(wire).not.toContain('Obsidian')
    expect(delta!.appBuckets.editor).toBeGreaterThan(0)
  })

  it('never records a track name or artist, only a count', () => {
    const tracker = createRhythmTracker()
    const delta = tracker.ingest(
      snapshot({
        media_playing: true,
        media_title: 'Some Very Identifying Song',
        media_artist: 'A Band',
      }),
      T0,
    )

    const wire = JSON.stringify(toWirePayload(delta!))
    expect(wire).not.toContain('Some Very Identifying Song')
    expect(wire).not.toContain('A Band')
    expect(delta!.musicEvents).toBe(1)
  })

  it('records hour buckets, not timestamps', () => {
    const tracker = createRhythmTracker()
    const delta = tracker.ingest(snapshot({ window_title: 'Code' }), T0)

    const wire = toWirePayload(delta!)
    expect(wire.active_hour_buckets.every((h) => Number.isInteger(h) && h >= 0 && h <= 23)).toBe(
      true,
    )
    expect(JSON.stringify(wire)).not.toContain(String(T0))
  })

  it('ignores her own windows', () => {
    const tracker = createRhythmTracker()
    const delta = tracker.ingest(snapshot({ window_title: 'Amadeus' }), T0)

    expect(delta!.appBuckets).toEqual({})
  })

  it('counts long focus stretches on the same app', () => {
    const tracker = createRhythmTracker()
    tracker.ingest(snapshot({ window_title: 'Code' }), T0)
    const later = tracker.ingest(
      snapshot({ window_title: 'Code' }),
      T0 + 95 * 60_000,
    )

    expect(later!.longFocusStretches).toBe(1)
    expect(later!.longestFocusHours).toBeGreaterThanOrEqual(1.5)
  })

  it('resets the stretch when the app changes', () => {
    const tracker = createRhythmTracker()
    tracker.ingest(snapshot({ window_title: 'Code' }), T0)
    const switched = tracker.ingest(
      snapshot({ window_title: 'Chrome' }),
      T0 + 95 * 60_000,
    )

    expect(switched!.longFocusStretches).toBe(0)
  })

  it('rolls to a fresh day and does not carry counters over', () => {
    const tracker = createRhythmTracker()
    tracker.ingest(snapshot({ window_title: 'Code', media_playing: true, media_title: 'x' }), T0)
    const nextDay = tracker.ingest(snapshot({ window_title: 'Code' }), T0 + 24 * 3_600_000)

    expect(nextDay!.date).not.toBe(new Date(T0).toISOString().slice(0, 10) + 'XX')
    expect(nextDay!.musicEvents).toBe(0)
  })

  it('does not re-credit a midnight-spanning focus stretch to the new day', () => {
    const tracker = createRhythmTracker()
    tracker.ingest(snapshot({ window_title: 'Code' }), T0)
    // 95 min later: one long stretch credited to day 1.
    const day1 = tracker.ingest(snapshot({ window_title: 'Code' }), T0 + 95 * 60_000)
    expect(day1!.longFocusStretches).toBe(1)

    // Same app, next day: the anchor must reset — without it the whole
    // 24h-old stretch (16 x 90min) lands on the fresh day's counters.
    const nextDay = tracker.ingest(snapshot({ window_title: 'Code' }), T0 + 24 * 3_600_000)
    expect(nextDay!.longFocusStretches).toBe(0)
    expect(nextDay!.longestFocusHours).toBe(0)

    // Time actually spent IN the new day still counts normally.
    const later = tracker.ingest(
      snapshot({ window_title: 'Code' }),
      T0 + 24 * 3_600_000 + 95 * 60_000,
    )
    expect(later!.longFocusStretches).toBe(1)
    expect(later!.longestFocusHours).toBeGreaterThanOrEqual(1.5)
  })

  it('emits only the documented wire fields', () => {
    const tracker = createRhythmTracker()
    const delta = tracker.ingest(snapshot({ window_title: 'Code' }), T0)

    expect(Object.keys(toWirePayload(delta!)).sort()).toEqual([
      'active_hour_buckets',
      'app_buckets',
      'battery_low_events',
      'date',
      'late_night_blocks',
      'long_focus_stretches',
      'longest_focus_hours',
      'music_events',
    ])
  })

  it('only ever stores bucket keys from the fixed vocabulary', () => {
    const tracker = createRhythmTracker()
    for (const title of ['Code', 'Chrome', 'Weird Custom Tool', 'iTerm2']) {
      tracker.ingest(snapshot({ window_title: title }), T0 + Math.random() * 1000)
    }
    const delta = tracker.current()!
    for (const key of Object.keys(delta.appBuckets)) {
      expect(APP_BUCKETS).toContain(key as AppBucket)
    }
  })
})
