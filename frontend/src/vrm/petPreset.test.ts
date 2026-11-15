// @vitest-environment jsdom
import { describe, expect, it } from 'vitest'
import { loadPetPreset, PET_PRESET_STORAGE_KEY, petPresetToFramingMode } from './petPreset'

describe('loadPetPreset', () => {
  it('defaults to full when the key is absent', () => {
    window.localStorage.removeItem(PET_PRESET_STORAGE_KEY)
    expect(loadPetPreset()).toBe('full')
  })

  it('reads compact when explicitly set', () => {
    window.localStorage.setItem(PET_PRESET_STORAGE_KEY, 'compact')
    expect(loadPetPreset()).toBe('compact')
    window.localStorage.removeItem(PET_PRESET_STORAGE_KEY)
  })

  it('falls back to full for any unrecognized value', () => {
    window.localStorage.setItem(PET_PRESET_STORAGE_KEY, 'huge')
    expect(loadPetPreset()).toBe('full')
    window.localStorage.removeItem(PET_PRESET_STORAGE_KEY)
  })
})

describe('petPresetToFramingMode', () => {
  it('maps full -> medium and compact -> compact', () => {
    expect(petPresetToFramingMode('full')).toBe('medium')
    expect(petPresetToFramingMode('compact')).toBe('compact')
  })
})
