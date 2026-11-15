// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { LOCALE_EVENT, LOCALE_STORAGE_KEY, loadLocale, saveLocale, t } from './i18n'

const stubLanguage = (value: string) => {
  Object.defineProperty(window.navigator, 'language', {
    value,
    configurable: true,
  })
}

const originalLanguage = window.navigator.language

beforeEach(() => {
  window.localStorage.clear()
})

afterEach(() => {
  stubLanguage(originalLanguage)
  window.localStorage.clear()
})

describe('loadLocale', () => {
  it('defaults to en for a non-Japanese browser language', () => {
    stubLanguage('en-US')
    expect(loadLocale()).toBe('en')
  })

  it('defaults to ja when the browser language is Japanese', () => {
    stubLanguage('ja-JP')
    expect(loadLocale()).toBe('ja')
  })

  it('prefers the stored locale over the browser language', () => {
    stubLanguage('ja-JP')
    window.localStorage.setItem(LOCALE_STORAGE_KEY, 'en')
    expect(loadLocale()).toBe('en')

    stubLanguage('en-US')
    window.localStorage.setItem(LOCALE_STORAGE_KEY, 'ja')
    expect(loadLocale()).toBe('ja')
  })

  it('ignores garbage stored values', () => {
    stubLanguage('en-US')
    window.localStorage.setItem(LOCALE_STORAGE_KEY, 'fr')
    expect(loadLocale()).toBe('en')
  })
})

describe('t', () => {
  it('returns English under the en locale', () => {
    window.localStorage.setItem(LOCALE_STORAGE_KEY, 'en')
    expect(t('Send')).toBe('Send')
    expect(t('Settings')).toBe('Settings')
  })

  it('returns Japanese under the ja locale', () => {
    window.localStorage.setItem(LOCALE_STORAGE_KEY, 'ja')
    expect(t('Send')).toBe('送信')
    expect(t('Settings')).toBe('設定')
  })

  it('falls back to English for keys with no ja translation', () => {
    window.localStorage.setItem(LOCALE_STORAGE_KEY, 'ja')
    // A key that has no STRINGS entry at all degrades to the English text.
    expect(t('Some untranslated string')).toBe('Some untranslated string')
  })

  it('interpolates {name}-style params', () => {
    window.localStorage.setItem(LOCALE_STORAGE_KEY, 'en')
    expect(t('LOADING {percent}%', { percent: 42 })).toBe('LOADING 42%')
    window.localStorage.setItem(LOCALE_STORAGE_KEY, 'ja')
    expect(t('LOADING {percent}%', { percent: 42 })).toBe('読み込み中 42%')
  })
})

describe('saveLocale', () => {
  it('persists the locale and dispatches the locale event', () => {
    const listener = vi.fn()
    window.addEventListener(LOCALE_EVENT, listener)
    try {
      saveLocale('ja')
      expect(window.localStorage.getItem(LOCALE_STORAGE_KEY)).toBe('ja')
      expect(listener).toHaveBeenCalledTimes(1)
      expect(loadLocale()).toBe('ja')
    } finally {
      window.removeEventListener(LOCALE_EVENT, listener)
    }
  })
})
