/**
 * The single source of truth for every `window.*` global in the app.
 *
 * ChatInterface and VRMViewer now talk through the typed singleton in
 * `src/vrm/kurisuBridge.ts` instead of mutable window globals. Two avatar
 * globals are kept as thin aliases of bridge slots for pre-existing
 * consumers: `window.kurisuLipSync` (LoadingScreen's voiced-greeting hook)
 * and `window.vrmAnimations` (console/debug use and the anim lab's polling
 * effect). `VrmClipInfo`/`VrmAnimationControls` are defined in
 * kurisuBridge.ts (the canonical accessor) and re-exported here only for the
 * `Window` interface augmentation. Runtime helpers (e.g. `isTauri`) live in
 * `src/platform.ts`.
 */

import type { KurisuLipSync, VrmAnimationControls } from '../vrm/kurisuBridge'

export type { VrmClipInfo, VrmAnimationControls } from '../vrm/kurisuBridge'

declare global {
  interface Window {
    // Tauri runtime markers (see platform.ts isTauri()).
    __TAURI_INTERNALS__?: unknown
    __TAURI__?: unknown
    // Debug switches for animation discovery (VRMViewer).
    __kurisuDiscoverAnimations?: boolean
    __kurisuAnimationManifestPath?: string | null
    // Legacy WebKit constructor used as an AudioContext fallback.
    webkitAudioContext?: typeof AudioContext
    // Thin alias of kurisuBridge.vrmAnimations kept for console/debug use
    // and the anim lab (see module docstring).
    vrmAnimations?: VrmAnimationControls
    // Thin alias of kurisuBridge.lipSync kept for LoadingScreen.
    kurisuLipSync?: KurisuLipSync
  }
}
