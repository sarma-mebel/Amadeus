/**
 * Shared /api fetch helper.
 *
 * Every backend call in the app goes through `apiFetch` so the optional
 * API token (Settings → API token, stored locally) rides along as an
 * `Authorization: Bearer <token>` header on each request — and never when
 * unset, so tokenless servers see byte-identical requests to before.
 *
 * A 401 anywhere fires a single window-level `kurisu:auth-error` event
 * (ChatInterface listens and surfaces one in-theme transcript message
 * pointing at the settings field) instead of each call site inventing its
 * own copy.
 *
 * The WebSocket path can't carry headers, so `resolveWsUrl` appends the
 * token as a `?token=` query param per the backend contract.
 */

import { resolveApiPath, resolveWsPath } from './config'

export const API_TOKEN_STORAGE_KEY = 'kurisu_api_token'
export const AUTH_ERROR_EVENT = 'kurisu:auth-error'

export const loadApiToken = (): string => {
  if (typeof window === 'undefined') return ''
  try {
    return window.localStorage.getItem(API_TOKEN_STORAGE_KEY)?.trim() ?? ''
  } catch {
    return ''
  }
}

export const saveApiToken = (token: string): void => {
  if (typeof window === 'undefined') return
  try {
    const trimmed = token.trim()
    if (trimmed) {
      window.localStorage.setItem(API_TOKEN_STORAGE_KEY, trimmed)
    } else {
      window.localStorage.removeItem(API_TOKEN_STORAGE_KEY)
    }
  } catch {
    // Storage can throw (quota, private mode); the token just won't persist.
  }
}

/** fetch() against an /api path, with the Bearer header attached when a
 * token is configured. Signals 401s via the AUTH_ERROR_EVENT window event
 * and returns the response unchanged (callers keep their own !ok handling). */
export const apiFetch = async (path: string, init?: RequestInit): Promise<Response> => {
  const token = loadApiToken()
  let request: RequestInit | undefined = init
  if (token) {
    const headers = new Headers(init?.headers)
    headers.set('Authorization', `Bearer ${token}`)
    request = { ...init, headers }
  }
  const response = await fetch(resolveApiPath(path), request)
  if (response.status === 401 && typeof window !== 'undefined') {
    try {
      window.dispatchEvent(new Event(AUTH_ERROR_EVENT))
    } catch {
      // Never let auth signaling break the actual request path.
    }
  }
  return response
}

/** WebSocket URL for an /api path, with `?token=` appended when a token is
 * configured (WS handshakes can't carry an Authorization header). */
export const resolveWsUrl = (path: string): string => {
  const base = resolveWsPath(path)
  const token = loadApiToken()
  if (!token) return base
  const separator = base.includes('?') ? '&' : '?'
  return `${base}${separator}token=${encodeURIComponent(token)}`
}
