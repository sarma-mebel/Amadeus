const normalizePath = (value: string | undefined | null) => {
  if (!value) return ''
  return value.trim().replace(/\/+$/, '')
}

const joinWithBase = (base: string, path: string) => {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  return base ? `${base}${normalizedPath}` : normalizedPath
}

const apiBaseEnv = import.meta.env.VITE_API_BASE_URL as string | undefined
const assetBaseEnv = import.meta.env.VITE_ASSET_BASE_URL as string | undefined

export const API_BASE_URL = normalizePath(apiBaseEnv)
export const ASSET_BASE_URL = normalizePath(assetBaseEnv ?? apiBaseEnv)

// Optional user-set backend origin (Settings → Access → Backend URL). Stored
// locally and read per-call so a change applies to every *new* request
// without a rebuild; empty keeps the env-derived behavior byte-for-byte.
export const BACKEND_URL_STORAGE_KEY = 'kurisu_backend_url'

export const loadBackendUrl = (): string => {
  if (typeof window === 'undefined') return ''
  try {
    return normalizePath(window.localStorage.getItem(BACKEND_URL_STORAGE_KEY))
  } catch {
    return ''
  }
}

export const saveBackendUrl = (url: string): void => {
  if (typeof window === 'undefined') return
  try {
    const normalized = normalizePath(url)
    if (normalized) {
      window.localStorage.setItem(BACKEND_URL_STORAGE_KEY, normalized)
    } else {
      window.localStorage.removeItem(BACKEND_URL_STORAGE_KEY)
    }
  } catch {
    // Storage can throw (quota, private mode); the override just won't persist.
  }
}

export const resolveApiPath = (path: string) => joinWithBase(loadBackendUrl() || API_BASE_URL, path)
// Assets (/static VRM models, animations, environments) are served by the
// backend, so the user's backend override must apply here too — on a phone
// the app bundle has no /static and a relative URL would 404 the avatar.
export const resolveAssetPath = (path: string) =>
  joinWithBase(loadBackendUrl() || ASSET_BASE_URL, path)

const toWsBase = (httpBase: string): string => {
  if (httpBase.startsWith('https://')) return `wss://${httpBase.slice('https://'.length)}`
  if (httpBase.startsWith('http://')) return `ws://${httpBase.slice('http://'.length)}`
  return httpBase
}

// Capacitor's native webview origin (capacitor://, file://) has no usable
// http(s) scheme to flip, so WS URLs must be absolute against the API base;
// only fall back to window.location when no API base is configured at all
// (typical browser dev setup where the app and API share an origin).
export const resolveWsPath = (path: string) => {
  const base = loadBackendUrl() || API_BASE_URL
  if (base) {
    return joinWithBase(toWsBase(base), path)
  }
  if (typeof window !== 'undefined' && window.location) {
    const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    return joinWithBase(`${scheme}//${window.location.host}`, path)
  }
  return joinWithBase('', path)
}
