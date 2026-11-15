// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { DEFAULT_UI_MODE, loadUiMode, saveUiMode } from './uiMode'

describe('uiMode', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  afterEach(() => {
    window.localStorage.clear()
  })

  it('falls back to the default when nothing is stored', () => {
    expect(loadUiMode()).toBe(DEFAULT_UI_MODE)
    expect(DEFAULT_UI_MODE).toBe('vn')
  })

  it('round-trips a saved value', () => {
    saveUiMode('vn')
    expect(loadUiMode()).toBe('vn')

    saveUiMode('full')
    expect(loadUiMode()).toBe('full')
  })

  it('falls back to the default when the stored value is garbage', () => {
    window.localStorage.setItem('kurisu_ui_mode_v1', 'not-a-real-mode')
    expect(loadUiMode()).toBe('vn')
  })
})
