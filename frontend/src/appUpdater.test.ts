// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest'

const { invokeMock, listenMock, unlistenMock, tauriFlag } = vi.hoisted(() => ({
  invokeMock: vi.fn(async (): Promise<unknown> => undefined),
  listenMock: vi.fn(),
  unlistenMock: vi.fn(),
  tauriFlag: { value: true },
}))
vi.mock('@tauri-apps/api/core', () => ({ invoke: invokeMock }))
vi.mock('@tauri-apps/api/event', () => ({ listen: listenMock }))
vi.mock('./platform', () => ({ isTauri: () => tauriFlag.value }))

import {
  __resetAppUpdaterForTests,
  checkAppUpdate,
  installAppUpdate,
  type AppUpdateProgress,
} from './appUpdater'

describe('appUpdater', () => {
  afterEach(() => {
    invokeMock.mockReset()
    invokeMock.mockImplementation(async () => undefined)
    listenMock.mockReset()
    unlistenMock.mockClear()
    tauriFlag.value = true
    __resetAppUpdaterForTests()
  })

  it('checkAppUpdate returns the Rust status verbatim', async () => {
    const status = {
      available: true,
      version: '0.14.0',
      notes: 'notes',
      portable: false,
    }
    invokeMock.mockResolvedValueOnce(status)
    expect(await checkAppUpdate()).toEqual(status)
    expect(invokeMock).toHaveBeenCalledWith('check_app_update')
  })

  it('checkAppUpdate is null in the browser and never invokes', async () => {
    tauriFlag.value = false
    expect(await checkAppUpdate()).toBeNull()
    expect(invokeMock).not.toHaveBeenCalled()
  })

  it('checkAppUpdate swallows failures as null (offline, unconfigured)', async () => {
    invokeMock.mockRejectedValueOnce(new Error('endpoint unreachable'))
    expect(await checkAppUpdate()).toBeNull()
  })

  it('installAppUpdate forwards progress events and unlistens after', async () => {
    const seen: AppUpdateProgress[] = []
    let handler: ((event: { payload: AppUpdateProgress }) => void) | null = null
    listenMock.mockImplementation(async (_name: string, cb: typeof handler) => {
      handler = cb
      return unlistenMock
    })
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === 'install_app_update') {
        handler?.({ payload: { downloaded: 10, total: 100, done: false } })
        handler?.({ payload: { downloaded: 100, total: 100, done: true } })
      }
      return undefined
    })

    const result = await installAppUpdate((p) => seen.push(p))
    expect(result.ok).toBe(true)
    expect(seen).toEqual([
      { downloaded: 10, total: 100, done: false },
      { downloaded: 100, total: 100, done: true },
    ])
    expect(listenMock).toHaveBeenCalledWith(
      'app-update-progress',
      expect.any(Function),
    )
    expect(unlistenMock).toHaveBeenCalled()
  })

  it('installAppUpdate surfaces install failures and still unlistens', async () => {
    listenMock.mockResolvedValue(unlistenMock)
    invokeMock.mockRejectedValueOnce(new Error('signature mismatch'))
    const result = await installAppUpdate(() => {})
    expect(result.ok).toBe(false)
    expect(result.error).toContain('signature mismatch')
    expect(unlistenMock).toHaveBeenCalled()
  })

  it('installAppUpdate refuses outside the desktop shell', async () => {
    tauriFlag.value = false
    const result = await installAppUpdate(() => {})
    expect(result.ok).toBe(false)
    expect(invokeMock).not.toHaveBeenCalled()
  })
})
