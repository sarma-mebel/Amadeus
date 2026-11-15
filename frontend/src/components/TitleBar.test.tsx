// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom/vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

// Hoisted so the mock factories (evaluated before the imports below run)
// can close over them — same idiom as appUpdater.test.ts / LoadingScreen's
// tauriFlag convention in this repo.
const {
  tauriFlag,
  capacitorFlag,
  labelValue,
  minimizeMock,
  toggleMaximizeMock,
  closeMock,
  isMaximizedMock,
  onResizedMock,
  incrementMock,
} = vi.hoisted(() => ({
  tauriFlag: { value: true },
  capacitorFlag: { value: false },
  labelValue: { value: 'main' },
  minimizeMock: vi.fn(async () => undefined),
  toggleMaximizeMock: vi.fn(async () => undefined),
  closeMock: vi.fn(async () => undefined),
  isMaximizedMock: vi.fn(async () => false),
  // Typed via the generic (not an impl arg) so the callback param exists for
  // mockImplementation callers without tripping no-unused-vars; every test
  // gets its real default from beforeEach's mockResolvedValue.
  onResizedMock: vi.fn<(cb: () => void | Promise<void>) => Promise<() => undefined>>(),
  incrementMock: vi.fn(),
}))

vi.mock('../platform', () => ({
  isTauri: () => tauriFlag.value,
  isCapacitorNative: () => capacitorFlag.value,
}))

vi.mock('../diagnostics', () => ({
  diagnostics: { increment: incrementMock },
}))

vi.mock('@tauri-apps/api/window', () => ({
  getCurrentWindow: () => ({
    get label() {
      return labelValue.value
    },
    minimize: minimizeMock,
    toggleMaximize: toggleMaximizeMock,
    close: closeMock,
    isMaximized: isMaximizedMock,
    onResized: onResizedMock,
  }),
}))

import TitleBar from './TitleBar'

describe('TitleBar', () => {
  beforeEach(() => {
    tauriFlag.value = true
    capacitorFlag.value = false
    labelValue.value = 'main'
    minimizeMock.mockClear()
    toggleMaximizeMock.mockClear()
    closeMock.mockClear()
    isMaximizedMock.mockClear()
    isMaximizedMock.mockResolvedValue(false)
    onResizedMock.mockClear()
    onResizedMock.mockResolvedValue(() => undefined)
    incrementMock.mockClear()
    window.history.pushState({}, '', '/')
  })

  afterEach(() => {
    cleanup()
  })

  it('renders nothing in the browser (isTauri false)', () => {
    tauriFlag.value = false
    const { container } = render(<TitleBar />)
    expect(container).toBeEmptyDOMElement()
  })

  it('renders nothing in the Android/Capacitor app even if isTauri were true', () => {
    capacitorFlag.value = true
    const { container } = render(<TitleBar />)
    expect(container).toBeEmptyDOMElement()
  })

  it('renders nothing in the pet view (?view=pet)', () => {
    window.history.pushState({}, '', '/?view=pet')
    const { container } = render(<TitleBar />)
    expect(container).toBeEmptyDOMElement()
  })

  it('renders nothing in the animlab view (?view=animlab)', () => {
    window.history.pushState({}, '', '/?view=animlab')
    const { container } = render(<TitleBar />)
    expect(container).toBeEmptyDOMElement()
  })

  it('renders nothing when the current window label is not "main"', async () => {
    labelValue.value = 'pet'
    const { container } = render(<TitleBar />)
    await waitFor(() => expect(container).toBeEmptyDOMElement())
  })

  it('renders the title and controls under a mocked tauri main window', async () => {
    render(<TitleBar />)
    await waitFor(() => expect(screen.getByText('Amadeus')).toBeInTheDocument())
    expect(screen.getByRole('button', { name: 'Minimize' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Maximize' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Close' })).toBeInTheDocument()
  })

  it('the bar and title carry the drag-region attribute; buttons do not', async () => {
    const { container } = render(<TitleBar />)
    await waitFor(() => expect(screen.getByText('Amadeus')).toBeInTheDocument())
    const bar = container.querySelector('.titlebar')
    expect(bar).toHaveAttribute('data-tauri-drag-region')
    expect(screen.getByText('Amadeus')).toHaveAttribute('data-tauri-drag-region')
    for (const btn of screen.getAllByRole('button')) {
      expect(btn).not.toHaveAttribute('data-tauri-drag-region')
    }
  })

  it('minimize button calls the mocked window API', async () => {
    const user = userEvent.setup()
    render(<TitleBar />)
    await waitFor(() => expect(screen.getByText('Amadeus')).toBeInTheDocument())
    await user.click(screen.getByRole('button', { name: 'Minimize' }))
    await waitFor(() => expect(minimizeMock).toHaveBeenCalledTimes(1))
  })

  it('maximize button calls toggleMaximize', async () => {
    const user = userEvent.setup()
    render(<TitleBar />)
    await waitFor(() => expect(screen.getByText('Amadeus')).toBeInTheDocument())
    await user.click(screen.getByRole('button', { name: 'Maximize' }))
    await waitFor(() => expect(toggleMaximizeMock).toHaveBeenCalledTimes(1))
  })

  it('close button calls the normal close() flow, never a bypass', async () => {
    const user = userEvent.setup()
    render(<TitleBar />)
    await waitFor(() => expect(screen.getByText('Amadeus')).toBeInTheDocument())
    await user.click(screen.getByRole('button', { name: 'Close' }))
    await waitFor(() => expect(closeMock).toHaveBeenCalledTimes(1))
  })

  it('maximize icon/label flips to Restore when isMaximized() resolves true', async () => {
    isMaximizedMock.mockResolvedValue(true)
    render(<TitleBar />)
    await waitFor(() => expect(screen.getByRole('button', { name: 'Restore' })).toBeInTheDocument())
    expect(screen.queryByRole('button', { name: 'Maximize' })).not.toBeInTheDocument()
  })

  it('maximize state updates when onResized fires', async () => {
    let resizeHandler: (() => void | Promise<void>) | undefined
    onResizedMock.mockImplementation(async (cb: () => void | Promise<void>) => {
      resizeHandler = cb
      return () => undefined
    })
    render(<TitleBar />)
    await waitFor(() => expect(screen.getByRole('button', { name: 'Maximize' })).toBeInTheDocument())

    isMaximizedMock.mockResolvedValue(true)
    // The subscription lands after a dynamic import + isMaximized() await —
    // strictly later than the button render the waitFor above gated on.
    await waitFor(() => expect(resizeHandler).toBeDefined())
    await resizeHandler?.()

    await waitFor(() => expect(screen.getByRole('button', { name: 'Restore' })).toBeInTheDocument())
  })

  it('degrades loudly: a rejected window API call is counted via diagnostics, not thrown', async () => {
    minimizeMock.mockRejectedValueOnce(new Error('boom'))
    const user = userEvent.setup()
    render(<TitleBar />)
    await waitFor(() => expect(screen.getByText('Amadeus')).toBeInTheDocument())
    await user.click(screen.getByRole('button', { name: 'Minimize' }))
    await waitFor(() =>
      expect(incrementMock).toHaveBeenCalledWith('titlebar.window_api_failed', expect.stringContaining('minimize')),
    )
  })
})
