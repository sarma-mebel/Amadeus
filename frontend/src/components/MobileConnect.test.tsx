// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom/vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import MobileConnect from './MobileConnect'
import { BACKEND_URL_STORAGE_KEY } from '../config'

describe('MobileConnect', () => {
  let fetchMock: ReturnType<typeof vi.fn>
  let reloadMock: ReturnType<typeof vi.fn>

  beforeEach(() => {
    window.localStorage.clear()
    fetchMock = vi.fn(async () => new Response('ok', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    reloadMock = vi.fn()
    Object.defineProperty(window, 'location', {
      value: { ...window.location, reload: reloadMock },
      configurable: true,
      writable: true,
    })
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it('probes /healthz, saves the URL, and reloads on success', async () => {
    const user = userEvent.setup()
    render(<MobileConnect />)

    await user.type(screen.getByLabelText(/backend server url/i), '192.168.1.20:8000')
    await user.click(screen.getByRole('button', { name: /establish link/i }))

    await waitFor(() => expect(reloadMock).toHaveBeenCalled())
    // Scheme-less input gets http:// prepended before probing/saving.
    expect(String(fetchMock.mock.calls[0][0])).toBe('http://192.168.1.20:8000/healthz')
    expect(window.localStorage.getItem(BACKEND_URL_STORAGE_KEY)).toBe(
      'http://192.168.1.20:8000',
    )
  })

  it('shows an error and saves nothing when the probe fails', async () => {
    fetchMock.mockImplementation(async () => new Response('nope', { status: 502 }))
    const user = userEvent.setup()
    render(<MobileConnect />)

    await user.type(screen.getByLabelText(/backend server url/i), 'http://10.0.0.9:8000')
    await user.click(screen.getByRole('button', { name: /establish link/i }))

    expect(await screen.findByText(/no lab network/i)).toBeInTheDocument()
    expect(window.localStorage.getItem(BACKEND_URL_STORAGE_KEY)).toBeNull()
    expect(reloadMock).not.toHaveBeenCalled()
  })

  it('treats a network error like an unreachable server', async () => {
    fetchMock.mockImplementation(async () => {
      throw new TypeError('Failed to fetch')
    })
    const user = userEvent.setup()
    render(<MobileConnect />)

    await user.type(screen.getByLabelText(/backend server url/i), 'example.com')
    await user.click(screen.getByRole('button', { name: /establish link/i }))

    expect(await screen.findByText(/no lab network/i)).toBeInTheDocument()
    expect(reloadMock).not.toHaveBeenCalled()
  })
})
