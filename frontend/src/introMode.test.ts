// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { diffNewFactKeys, isIntroDone, markIntroDone, resetIntroDone } from './introMode'

describe('introMode', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })
  afterEach(() => {
    window.localStorage.clear()
  })

  describe('isIntroDone / markIntroDone', () => {
    it('is not done by default', () => {
      expect(isIntroDone()).toBe(false)
    })

    it('is done after markIntroDone', () => {
      markIntroDone()
      expect(isIntroDone()).toBe(true)
    })
  })

  describe('per-user namespacing (item 9)', () => {
    it('marking done for one user does not affect another user', () => {
      markIntroDone('user-a')
      expect(isIntroDone('user-a')).toBe(true)
      expect(isIntroDone('user-b')).toBe(false)
    })

    it('stores under a namespaced key, not the bare key', () => {
      markIntroDone('user-a')
      expect(window.localStorage.getItem('kurisu_intro_done')).toBeNull()
      expect(window.localStorage.getItem('kurisu_intro_done:user-a')).toBe('true')
    })

    it('falls back to the bare key when no userId is supplied', () => {
      markIntroDone()
      expect(window.localStorage.getItem('kurisu_intro_done')).toBe('true')
    })

    it('resetIntroDone re-arms conversational onboarding for that user', () => {
      markIntroDone('user-a')
      expect(isIntroDone('user-a')).toBe(true)

      resetIntroDone('user-a')

      expect(isIntroDone('user-a')).toBe(false)
    })

    it('resetIntroDone never throws when localStorage is unavailable', () => {
      const original = window.localStorage.removeItem
      window.localStorage.removeItem = () => {
        throw new Error('quota')
      }
      expect(() => resetIntroDone('user-a')).not.toThrow()
      window.localStorage.removeItem = original
    })
  })

  describe('diffNewFactKeys', () => {
    it('returns keys present in next but not prev', () => {
      expect(diffNewFactKeys({ name: 'Alex' }, { name: 'Alex', occupation: 'researcher' })).toEqual([
        'occupation',
      ])
    })

    it('returns an empty array when nothing new was learned', () => {
      expect(diffNewFactKeys({ name: 'Alex' }, { name: 'Alex' })).toEqual([])
    })

    it('returns all keys when prev is empty', () => {
      expect(diffNewFactKeys({}, { name: 'Alex' })).toEqual(['name'])
    })

    it('does not report a changed value for an existing key as new', () => {
      expect(diffNewFactKeys({ name: 'Alex' }, { name: 'Someone Else' })).toEqual([])
    })
  })
})
