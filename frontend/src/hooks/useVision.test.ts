// @vitest-environment jsdom
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, renderHook } from '@testing-library/react'
import type { ClipboardEvent } from 'react'
import { useVision } from './useVision'
import { captureScreen, fileToPendingImage, type PendingImage } from '../images'
import { diagnostics } from '../diagnostics'

const { apiFetchMock } = vi.hoisted(() => ({ apiFetchMock: vi.fn() }))
vi.mock('../api', () => ({ apiFetch: apiFetchMock }))
vi.mock('../diagnostics', () => ({ diagnostics: { increment: vi.fn() } }))
vi.mock('../images', () => ({
  captureScreen: vi.fn(),
  fileToPendingImage: vi.fn(),
}))

const pending = (data = 'abc'): PendingImage => ({
  data,
  mime: 'image/jpeg',
  width: 10,
  height: 10,
  previewUrl: `data:image/jpeg;base64,${data}`,
})

const statusResponse = (payload: unknown, ok = true) => ({
  ok,
  json: async () => payload,
})

const pasteEvent = (files: File[]) =>
  ({
    clipboardData: { files },
    preventDefault: vi.fn(),
  }) as unknown as ClipboardEvent

describe('useVision', () => {
  const mount = () => renderHook(() => useVision())
  const flush = () => act(async () => {})

  beforeEach(() => {
    vi.clearAllMocks()
    apiFetchMock.mockResolvedValue(statusResponse({ available: true }))
  })

  afterEach(() => {
    cleanup()
  })

  it('an available status probe enables the vision button', async () => {
    const { result } = mount()
    expect(result.current.visionAvailable).toBe(false)
    await flush()
    expect(apiFetchMock).toHaveBeenCalledWith('/api/vision/status')
    expect(result.current.visionAvailable).toBe(true)
  })

  it('a non-ok status probe leaves vision disabled', async () => {
    apiFetchMock.mockResolvedValue(statusResponse({}, false))
    const { result } = mount()
    await flush()
    expect(result.current.visionAvailable).toBe(false)
  })

  it('a failed status probe degrades quietly to disabled, and counts', async () => {
    apiFetchMock.mockRejectedValue(new Error('offline'))
    const { result } = mount()
    await flush()
    expect(result.current.visionAvailable).toBe(false)
    // Degrade-loudly: a failed probe must be distinguishable from
    // vision.enabled=false in Diagnostics.
    expect(diagnostics.increment).toHaveBeenCalledWith(
      'vision.status_probe_failed',
      expect.stringContaining('offline'),
    )
  })

  it('server limits from the probe thread into captureScreen', async () => {
    apiFetchMock.mockResolvedValue(
      statusResponse({ available: true, max_image_px: 1000, jpeg_quality: 70 }),
    )
    vi.mocked(captureScreen).mockResolvedValue(pending())

    const { result } = mount()
    await flush()
    await act(async () => {
      await result.current.handleCaptureScreen()
    })

    expect(captureScreen).toHaveBeenCalledWith(1000, 0.7)
  })

  it('a successful capture attaches the pending image (state + ref)', async () => {
    const image = pending('captured')
    vi.mocked(captureScreen).mockResolvedValue(image)

    const { result } = mount()
    await flush()
    await act(async () => {
      await result.current.handleCaptureScreen()
    })

    // Defaults apply when the probe carried no limit overrides.
    expect(captureScreen).toHaveBeenCalledWith(1568, 0.85)
    expect(result.current.pendingImage).toEqual(image)
    expect(result.current.pendingImageRef.current).toEqual(image)
  })

  it('a failed or cancelled capture leaves no pending image and never throws', async () => {
    vi.mocked(captureScreen).mockRejectedValue(new Error('picker dismissed'))

    const { result } = mount()
    await flush()
    await act(async () => {
      await result.current.handleCaptureScreen()
    })
    expect(result.current.pendingImage).toBeNull()

    // A null capture result (Tauri path returning nothing) also attaches nothing.
    vi.mocked(captureScreen).mockResolvedValue(null)
    await act(async () => {
      await result.current.handleCaptureScreen()
    })
    expect(result.current.pendingImage).toBeNull()
  })

  it('paste is ignored while vision is unavailable', async () => {
    apiFetchMock.mockResolvedValue(statusResponse({ available: false }))
    const { result } = mount()
    await flush()

    const event = pasteEvent([new File(['x'], 'shot.png', { type: 'image/png' })])
    await act(async () => {
      await result.current.handlePaste(event)
    })

    expect(fileToPendingImage).not.toHaveBeenCalled()
    expect(event.preventDefault).not.toHaveBeenCalled()
  })

  it('pasting an image file attaches it when vision is available', async () => {
    const image = pending('pasted')
    vi.mocked(fileToPendingImage).mockResolvedValue(image)
    const { result } = mount()
    await flush()

    const file = new File(['x'], 'shot.png', { type: 'image/png' })
    const event = pasteEvent([new File(['y'], 'note.txt', { type: 'text/plain' }), file])
    await act(async () => {
      await result.current.handlePaste(event)
    })

    expect(event.preventDefault).toHaveBeenCalled()
    expect(fileToPendingImage).toHaveBeenCalledWith(file, 1568, 0.85)
    expect(result.current.pendingImage).toEqual(image)
  })
})
