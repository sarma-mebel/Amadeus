// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import {
  API_BASE_URL,
  ASSET_BASE_URL,
  BACKEND_URL_STORAGE_KEY,
  loadBackendUrl,
  resolveApiPath,
  resolveAssetPath,
  resolveWsPath,
  saveBackendUrl,
} from './config'

// API_BASE_URL comes from .env / .env.local at build time, so tests assert
// relative to it rather than hardcoding a machine-specific origin.
const defaultApiUrl = (path: string) => (API_BASE_URL ? `${API_BASE_URL}${path}` : path)

const defaultWsUrl = (path: string) => {
  if (API_BASE_URL) {
    return `${API_BASE_URL.replace(/^https:\/\//, 'wss://').replace(/^http:\/\//, 'ws://')}${path}`
  }
  const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${scheme}//${window.location.host}${path}`
}

beforeEach(() => {
  window.localStorage.clear()
})

afterEach(() => {
  window.localStorage.clear()
})

describe('backend URL storage', () => {
  it('round-trips a URL and strips trailing slashes/whitespace', () => {
    saveBackendUrl('  https://kurisu.example.com/  ')
    expect(loadBackendUrl()).toBe('https://kurisu.example.com')
    expect(window.localStorage.getItem(BACKEND_URL_STORAGE_KEY)).toBe('https://kurisu.example.com')
  })

  it('clears storage when saving an empty/whitespace URL', () => {
    saveBackendUrl('https://kurisu.example.com')
    saveBackendUrl('   ')
    expect(loadBackendUrl()).toBe('')
    expect(window.localStorage.getItem(BACKEND_URL_STORAGE_KEY)).toBeNull()
  })
})

describe('resolveApiPath', () => {
  it('matches the default behavior byte-for-byte when no override is set', () => {
    expect(resolveApiPath('/api/chat')).toBe(defaultApiUrl('/api/chat'))
  })

  it('prefixes the override origin when set', () => {
    saveBackendUrl('https://kurisu.example.com')
    expect(resolveApiPath('/api/chat')).toBe('https://kurisu.example.com/api/chat')
  })

  it('adds a leading slash to bare paths under the override', () => {
    saveBackendUrl('http://192.168.1.20:8080')
    expect(resolveApiPath('api/status')).toBe('http://192.168.1.20:8080/api/status')
  })

  it('reverts to the default once the override is cleared', () => {
    saveBackendUrl('https://kurisu.example.com')
    saveBackendUrl('')
    expect(resolveApiPath('/api/chat')).toBe(defaultApiUrl('/api/chat'))
  })
})

describe('resolveAssetPath', () => {
  it('matches the default behavior when no override is set', () => {
    const expected = ASSET_BASE_URL
      ? `${ASSET_BASE_URL}/static/models/Kurisu.vrm`
      : '/static/models/Kurisu.vrm'
    expect(resolveAssetPath('/static/models/Kurisu.vrm')).toBe(expected)
  })

  it('honors the backend override (native builds have no bundled /static)', () => {
    saveBackendUrl('http://192.168.1.20:8000')
    expect(resolveAssetPath('/static/models/Kurisu.vrm')).toBe(
      'http://192.168.1.20:8000/static/models/Kurisu.vrm',
    )
  })
})

describe('resolveWsPath', () => {
  it('matches the default behavior when no override is set', () => {
    expect(resolveWsPath('/api/chat/stream')).toBe(defaultWsUrl('/api/chat/stream'))
  })

  it('derives wss:// from an https override', () => {
    saveBackendUrl('https://kurisu.example.com')
    expect(resolveWsPath('/api/chat/stream')).toBe('wss://kurisu.example.com/api/chat/stream')
  })

  it('derives ws:// from an http override', () => {
    saveBackendUrl('http://192.168.1.20:8080')
    expect(resolveWsPath('/api/chat/stream')).toBe('ws://192.168.1.20:8080/api/chat/stream')
  })
})
