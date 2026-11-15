// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import {
  clearSeenMilestones,
  filterUnseen,
  markSeen,
  milestoneClip,
  milestoneKey,
  type MilestoneEvent,
} from './milestones'

describe('milestones', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })
  afterEach(() => {
    window.localStorage.clear()
  })

  describe('milestoneKey', () => {
    it('keys a sessions event by value', () => {
      expect(milestoneKey({ kind: 'sessions', value: 10, label: '10 sessions' })).toBe(
        'sessions:10',
      )
    })

    it('keys an anniversary event by date when present', () => {
      expect(
        milestoneKey({ kind: 'anniversary', date: '2026-08-01', months: 1, label: '1 month' }),
      ).toBe('anniversary:2026-08-01')
    })

    it('keys an anniversary event by months when date is absent', () => {
      expect(milestoneKey({ kind: 'anniversary', months: 6, label: '6 months' })).toBe(
        'anniversary:6',
      )
    })
  })

  describe('milestoneClip', () => {
    it.each([
      [10, 'Joy'],
      [25, 'Cheering'],
      [50, 'Excitement 2'],
      [100, 'Pride'],
      [250, 'Joy 2'],
    ])('maps session milestone %i to %s', (value, expected) => {
      expect(milestoneClip({ kind: 'sessions', value, label: '' })).toBe(expected)
    })

    it('falls back to Joy for an unmapped session value', () => {
      expect(milestoneClip({ kind: 'sessions', value: 17, label: '' })).toBe('Joy')
    })

    it.each([
      [1, 'Gratitude'],
      [6, 'Pride'],
    ])('maps anniversary month %i to %s', (months, expected) => {
      expect(milestoneClip({ kind: 'anniversary', months, label: '' })).toBe(expected)
    })

    it('maps 12+ months to Joy 3', () => {
      expect(milestoneClip({ kind: 'anniversary', months: 12, label: '' })).toBe('Joy 3')
      expect(milestoneClip({ kind: 'anniversary', months: 24, label: '' })).toBe('Joy 3')
    })

    it('falls back to Gratitude for an unmapped anniversary month', () => {
      expect(milestoneClip({ kind: 'anniversary', months: 3, label: '' })).toBe('Gratitude')
    })
  })

  describe('filterUnseen / markSeen', () => {
    const events: MilestoneEvent[] = [
      { kind: 'sessions', value: 10, label: '10 sessions' },
      { kind: 'anniversary', months: 1, label: '1 month' },
    ]

    it('returns all events when nothing has been seen', () => {
      expect(filterUnseen(events)).toEqual(events)
    })

    it('excludes events already marked seen', () => {
      markSeen([events[0]])
      expect(filterUnseen(events)).toEqual([events[1]])
    })

    it('caps the seen list at 50 entries', () => {
      const many = Array.from({ length: 60 }, (_, i) => ({
        kind: 'sessions' as const,
        value: i,
        label: `${i}`,
      }))
      markSeen(many)
      const stored = JSON.parse(window.localStorage.getItem('kurisu_milestones_seen_v1') ?? '[]')
      expect(stored.length).toBe(50)
      // The most recent (highest value) entries survive the cap.
      expect(stored).toContain('sessions:59')
      expect(stored).not.toContain('sessions:0')
    })

    it('never throws when localStorage is unavailable', () => {
      const original = window.localStorage.setItem
      window.localStorage.setItem = () => {
        throw new Error('quota')
      }
      expect(() => markSeen(events)).not.toThrow()
      window.localStorage.setItem = original
    })
  })

  describe('per-user namespacing (item 9)', () => {
    const event: MilestoneEvent = { kind: 'sessions', value: 10, label: '10 sessions' }

    it('marking seen for one user does not affect another user with the same event', () => {
      markSeen([event], 'user-a')
      expect(filterUnseen([event], 'user-a')).toEqual([])
      expect(filterUnseen([event], 'user-b')).toEqual([event])
    })

    it('stores under a namespaced key, not the bare v1 key', () => {
      markSeen([event], 'user-a')
      expect(window.localStorage.getItem('kurisu_milestones_seen_v1')).toBeNull()
      expect(window.localStorage.getItem('kurisu_milestones_seen_v1:user-a')).toContain('sessions:10')
    })

    it('falls back to the bare key when no userId is supplied', () => {
      markSeen([event])
      expect(window.localStorage.getItem('kurisu_milestones_seen_v1')).toContain('sessions:10')
    })

    it('clearSeenMilestones re-arms a previously-seen milestone for that user', () => {
      markSeen([event], 'user-a')
      expect(filterUnseen([event], 'user-a')).toEqual([])

      clearSeenMilestones('user-a')

      expect(filterUnseen([event], 'user-a')).toEqual([event])
    })

    it('clearSeenMilestones never throws when localStorage is unavailable', () => {
      const original = window.localStorage.removeItem
      window.localStorage.removeItem = () => {
        throw new Error('quota')
      }
      expect(() => clearSeenMilestones('user-a')).not.toThrow()
      window.localStorage.removeItem = original
    })
  })
})
