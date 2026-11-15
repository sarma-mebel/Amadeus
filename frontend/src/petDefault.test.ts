// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest'

const { invokeMock, tauriFlag } = vi.hoisted(() => ({
  invokeMock: vi.fn(async () => undefined),
  tauriFlag: { value: true },
}))
vi.mock('@tauri-apps/api/core', () => ({ invoke: invokeMock }))
vi.mock('./platform', () => ({ isTauri: () => tauriFlag.value }))

import {
  PET_DEFAULT_KEY,
  enablePetDefaultAfterOnboarding,
  loadPetDefaultEnabled,
  savePetDefaultEnabled,
} from './petDefault'

const flushAsync = async () => {
  // The save path does a dynamic import before invoking; give it a real
  // macrotask (plus a microtask drain) to settle.
  await new Promise((resolve) => setTimeout(resolve, 0))
  await Promise.resolve()
}

describe('petDefault', () => {
  afterEach(() => {
    window.localStorage.clear()
    invokeMock.mockClear()
    tauriFlag.value = true
  })

  it('mirrors the preference and pushes it to the Rust flag', async () => {
    expect(loadPetDefaultEnabled()).toBe(false)
    savePetDefaultEnabled(true)
    expect(loadPetDefaultEnabled()).toBe(true)
    await flushAsync()
    expect(invokeMock).toHaveBeenCalledWith('set_pet_default', { enabled: true })
  })

  it('onboarding enables it once but never overrides an explicit choice', async () => {
    enablePetDefaultAfterOnboarding()
    await flushAsync()
    expect(loadPetDefaultEnabled()).toBe(true)

    // User turns it off; a (hypothetical) re-onboarding must not re-enable.
    savePetDefaultEnabled(false)
    await flushAsync() // let the off-push land before clearing the spy
    invokeMock.mockClear()
    enablePetDefaultAfterOnboarding()
    await flushAsync()
    expect(loadPetDefaultEnabled()).toBe(false)
    expect(invokeMock).not.toHaveBeenCalled()
  })

  it('does nothing in browsers', async () => {
    tauriFlag.value = false
    enablePetDefaultAfterOnboarding()
    await flushAsync()
    expect(window.localStorage.getItem(PET_DEFAULT_KEY)).toBeNull()
    savePetDefaultEnabled(true)
    await flushAsync()
    // localStorage mirror still updates (harmless), but no invoke happens.
    expect(invokeMock).not.toHaveBeenCalled()
  })
})
