// @vitest-environment jsdom
/**
 * Presence intensity dial: preset application, drift ("customized")
 * detection, and the idle-chatter cadence contract the presence hooks
 * read at schedule/fire time.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import {
  applyPresenceLevel,
  idleChatterDelayMultiplier,
  loadPresenceLevel,
  PRESENCE_LEVEL_KEY,
  presenceLevelCustomized,
} from './presenceLevel'
import { FILLERS_STORAGE_KEY, loadFillersEnabled } from './audio/fillerPlayer'
import {
  LISTENING_REACTIONS_STORAGE_KEY,
  loadListeningReactionsEnabled,
} from './audio/listeningReactions'
import { isPetRoamEnabled, PET_ROAM_KEY } from './vrm/petRoam'

declare global {
  interface Window {
    __TAURI_INTERNALS__?: unknown
  }
}

describe('presenceLevel', () => {
  beforeEach(() => {
    window.localStorage.clear()
    delete window.__TAURI_INTERNALS__
  })
  afterEach(() => {
    window.localStorage.clear()
    delete window.__TAURI_INTERNALS__
  })

  it('defaults to balanced (matching the shipped toggle defaults)', () => {
    expect(loadPresenceLevel()).toBe('balanced')
    // An untouched install already satisfies the balanced preset, so the
    // dial starts clean rather than claiming the user customized things.
    expect(presenceLevelCustomized()).toBe(false)
  })

  it('ignores junk stored values', () => {
    window.localStorage.setItem(PRESENCE_LEVEL_KEY, 'shouty')
    expect(loadPresenceLevel()).toBe('balanced')
  })

  it('quiet turns every mapped feature off', () => {
    applyPresenceLevel('quiet')
    expect(loadPresenceLevel()).toBe('quiet')
    expect(loadFillersEnabled()).toBe(false)
    expect(loadListeningReactionsEnabled()).toBe(false)
    expect(isPetRoamEnabled()).toBe(false)
    expect(presenceLevelCustomized()).toBe(false)
  })

  it('lively enables fillers and reactions, but roam only under Tauri', () => {
    applyPresenceLevel('lively')
    expect(loadFillersEnabled()).toBe(true)
    expect(loadListeningReactionsEnabled()).toBe(true)
    // Browser build: roaming can't run, so the preset must not write true.
    expect(window.localStorage.getItem(PET_ROAM_KEY)).not.toBe('true')

    window.__TAURI_INTERNALS__ = {}
    applyPresenceLevel('lively')
    expect(window.localStorage.getItem(PET_ROAM_KEY)).toBe('true')
    expect(presenceLevelCustomized()).toBe(false)
  })

  it('detects a hand-tuned toggle as customized', () => {
    applyPresenceLevel('quiet')
    window.localStorage.setItem(FILLERS_STORAGE_KEY, 'true')
    expect(presenceLevelCustomized()).toBe(true)
  })

  it('ignores roam drift outside Tauri (the toggle is not even shown)', () => {
    applyPresenceLevel('balanced')
    window.localStorage.setItem(PET_ROAM_KEY, 'true')
    expect(presenceLevelCustomized()).toBe(false)
    window.localStorage.setItem(LISTENING_REACTIONS_STORAGE_KEY, 'false')
    expect(presenceLevelCustomized()).toBe(true)
  })

  it('maps levels to idle-chatter cadence: quiet suppresses, lively speeds up', () => {
    applyPresenceLevel('quiet')
    expect(idleChatterDelayMultiplier()).toBeNull()
    applyPresenceLevel('balanced')
    expect(idleChatterDelayMultiplier()).toBe(1)
    applyPresenceLevel('lively')
    expect(idleChatterDelayMultiplier()).toBeLessThan(1)
  })
})
