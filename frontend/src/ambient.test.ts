/**
 * Ambient transition tracker: events fire on TRANSITIONS (long focus
 * crossing a stretch boundary, battery crossing a threshold while
 * discharging, music starting/changing), never on steady state — the
 * whole point vs. naively reporting every poll.
 */
import { describe, expect, it } from 'vitest'
import {
  appKeyFromTitle,
  BATTERY_CRITICAL_PERCENT,
  BATTERY_LOW_PERCENT,
  createAmbientTracker,
  FOCUS_STRETCH_MS,
  MUSIC_MIN_GAP_MS,
  type AmbientSnapshot,
} from './ambient'

const snap = (over: Partial<AmbientSnapshot>): AmbientSnapshot => ({
  window_title: null,
  battery_percent: null,
  battery_charging: null,
  media_title: null,
  media_artist: null,
  media_playing: null,
  ...over,
})

describe('appKeyFromTitle', () => {
  it('takes the conventional trailing app segment', () => {
    expect(appKeyFromTitle('notes.md - project - Visual Studio Code')).toBe(
      'Visual Studio Code',
    )
    expect(appKeyFromTitle('Spotify')).toBe('Spotify')
    expect(appKeyFromTitle('   ')).toBeNull()
    expect(appKeyFromTitle(null)).toBeNull()
  })
})

describe('focus stretches', () => {
  it('fires once per stretch boundary of continuous same-app focus', () => {
    const tracker = createAmbientTracker()
    const title = 'main.rs - amadeus-shell - Visual Studio Code'
    expect(tracker.ingest(snap({ window_title: title }), 0)).toEqual([])
    expect(
      tracker.ingest(snap({ window_title: title }), FOCUS_STRETCH_MS - 1000),
    ).toEqual([])
    const events = tracker.ingest(
      snap({ window_title: title }),
      FOCUS_STRETCH_MS + 1000,
    )
    expect(events).toHaveLength(1)
    expect(events[0].kind).toBe('focus')
    expect(events[0].detail).toContain('Visual Studio Code')
    // Steady state after the report: quiet until the NEXT boundary.
    expect(
      tracker.ingest(snap({ window_title: title }), FOCUS_STRETCH_MS + 60_000),
    ).toEqual([])
    expect(
      tracker.ingest(snap({ window_title: title }), 2 * FOCUS_STRETCH_MS + 1000),
    ).toHaveLength(1)
  })

  it('an app switch resets the stretch', () => {
    const tracker = createAmbientTracker()
    tracker.ingest(snap({ window_title: 'a - AppOne' }), 0)
    tracker.ingest(snap({ window_title: 'b - AppTwo' }), FOCUS_STRETCH_MS)
    expect(
      tracker.ingest(snap({ window_title: 'b - AppTwo' }), FOCUS_STRETCH_MS + 1000),
    ).toEqual([])
  })

  it('never reports her own window', () => {
    const tracker = createAmbientTracker()
    tracker.ingest(snap({ window_title: 'Amadeus' }), 0)
    expect(
      tracker.ingest(snap({ window_title: 'Amadeus' }), FOCUS_STRETCH_MS * 2),
    ).toEqual([])
  })
})

describe('battery', () => {
  it('fires once per discharge cycle at low, once more at critical', () => {
    const tracker = createAmbientTracker()
    const low = snap({
      battery_percent: BATTERY_LOW_PERCENT - 1,
      battery_charging: false,
    })
    const lowEvents = tracker.ingest(low, 0)
    expect(lowEvents).toHaveLength(1)
    // Structured discriminator the reflex layer routes on (the detail text
    // is a free sentence and must stay unmatched-against).
    expect(lowEvents[0].severity).toBe('low')
    expect(tracker.ingest(low, 1)).toEqual([]) // steady state: quiet
    const critical = snap({
      battery_percent: BATTERY_CRITICAL_PERCENT - 1,
      battery_charging: false,
    })
    const criticalEvents = tracker.ingest(critical, 2)
    expect(criticalEvents).toHaveLength(1)
    expect(criticalEvents[0].severity).toBe('critical')
    expect(tracker.ingest(critical, 3)).toEqual([])
  })

  it('charging re-arms the flags for the next cycle', () => {
    const tracker = createAmbientTracker()
    const low = snap({ battery_percent: 10, battery_charging: false })
    expect(tracker.ingest(low, 0)).toHaveLength(1)
    tracker.ingest(snap({ battery_percent: 40, battery_charging: true }), 1)
    expect(tracker.ingest(low, 2)).toHaveLength(1)
  })

  it('never fires while plugged in or percent unknown', () => {
    const tracker = createAmbientTracker()
    expect(
      tracker.ingest(snap({ battery_percent: 5, battery_charging: true }), 0),
    ).toEqual([])
    expect(tracker.ingest(snap({ battery_charging: false }), 1)).toEqual([])
  })
})

describe('music', () => {
  it('fires when music starts and on every real track change', () => {
    const tracker = createAmbientTracker()
    const playing = snap({
      media_title: 'Hacking to the Gate',
      media_artist: 'Kanako Itou',
      media_playing: true,
    })
    const events = tracker.ingest(playing, 0)
    expect(events).toHaveLength(1)
    expect(events[0].detail).toContain('Hacking to the Gate')
    expect(events[0].detail).toContain('Kanako Itou')
    // Change inside the 60s floor (pause-toggle damping): no event.
    const next = snap({ media_title: 'Sky Clad Observer', media_playing: true })
    expect(tracker.ingest(next, MUSIC_MIN_GAP_MS - 1000)).toEqual([])
    // Every real track change past the floor is its own event — pacing is
    // the server cooldown's job now, not the client's.
    const third = snap({ media_title: 'Fatima', media_playing: true })
    expect(tracker.ingest(third, MUSIC_MIN_GAP_MS + 1000)).toHaveLength(1)
    const fourth = snap({ media_title: 'Star-Crossed', media_playing: true })
    expect(tracker.ingest(fourth, MUSIC_MIN_GAP_MS * 2 + 2000)).toHaveLength(1)
  })

  it('steady playback and pauses stay quiet', () => {
    const tracker = createAmbientTracker()
    const playing = snap({ media_title: 'Song', media_playing: true })
    tracker.ingest(playing, 0)
    expect(tracker.ingest(playing, 1000)).toEqual([])
    expect(tracker.ingest(snap({ media_playing: false }), 2000)).toEqual([])
  })
})
