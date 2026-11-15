/**
 * "Start with Windows" (Tauri only): thin wrappers over the Rust
 * `set_autostart`/`get_autostart` commands (ACL'd for the main window).
 *
 * Unlike petDefault.ts there is no localStorage mirror here — autostart is
 * an OS-level registration (registry / login item / systemd, depending on
 * platform), so the Rust side is the only source of truth and every reader
 * re-queries it live rather than trusting a locally cached guess. Both
 * commands return a Promise and reject with an error string on failure
 * (missing permission, platform API failure, …); every call site here is
 * try/catch so a rejection degrades to "treat as unavailable/unchanged"
 * instead of throwing into a React event handler.
 */
import { isTauri } from './platform'

/** Reads the current autostart registration. Returns null when the query
 * itself failed or we're not in Tauri — callers treat null as "unknown /
 * unavailable" (hide or disable the toggle), never as false. */
export const getAutostartEnabled = async (): Promise<boolean | null> => {
  if (!isTauri()) return null
  try {
    const { invoke } = await import('@tauri-apps/api/core')
    return await invoke<boolean>('get_autostart')
  } catch {
    return null
  }
}

/** Applies the autostart choice. Resolves true on success; on rejection
 * returns the error string so the caller can show it inline and revert its
 * optimistic UI state, rather than throwing. */
export const setAutostartEnabled = async (
  enabled: boolean,
): Promise<{ ok: true } | { ok: false; error: string }> => {
  if (!isTauri()) return { ok: false, error: 'not running in the desktop shell' }
  try {
    const { invoke } = await import('@tauri-apps/api/core')
    await invoke('set_autostart', { enabled })
    return { ok: true }
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) }
  }
}
