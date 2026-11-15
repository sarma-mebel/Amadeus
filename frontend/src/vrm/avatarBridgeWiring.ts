/**
 * kurisuBridge object construction, moved verbatim out of VRMViewer's
 * VRM-load callback. Once the animation scheduler, lip-sync controller, and
 * expression driver exist, this builds the three window-facing control
 * surfaces (vrmAnimations, lipSync, expressions) that ChatInterface/
 * LoadingScreen/AnimLab call through kurisuBridge (see ../vrm/kurisuBridge
 * for the singleton and the window-alias back-compat rationale). Pure
 * wiring — every dependency is passed in explicitly; the component still
 * owns assigning the results onto kurisuBridge/window and clearing them on
 * teardown.
 *
 * Also carries deriveAutomaticPoolNames — the idle/action pool fallback
 * VRMViewer runs right after loadAnimationClips resolves, feeding straight
 * into the bridge above (idleNames drives describe()'s `looping` field).
 */

import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'
import type { AnimationFile, ClipMeta } from './clipLoader'
import type { KurisuLipSync, KurisuExpressions, VrmAnimationControls, VrmClipInfo } from './kurisuBridge'
import { VRM_EXPRESSION_NAMES } from './expressionDriver'

/** `disabled` keeps a clip loaded (manual play still works, e.g. from the
 * anim lab) but out of every automatic pool — including the looping-idle
 * rotation. Falls back to every loaded clip (idle) / every loaded clip with
 * "vrma" in its name (action) when the manifest's own idle/action sets
 * resolve to nothing loaded. */
export const deriveAutomaticPoolNames = (
  idleFiles: AnimationFile[],
  actionFiles: AnimationFile[],
  loadedClips: Record<string, THREE.AnimationClip>,
  clipMeta: Record<string, ClipMeta>
): { idleNames: string[]; actionNames: string[] } => {
  const availableNames = Object.keys(loadedClips)
  let idleNames = idleFiles
    .map((file) => file.name)
    .filter((name) => loadedClips[name] && !clipMeta[name]?.disabled)
  let actionNames = actionFiles.map((file) => file.name).filter((name) => loadedClips[name])
  if (idleNames.length === 0) {
    idleNames = availableNames
  }
  if (actionNames.length === 0) {
    actionNames = availableNames.filter((name) => name.toLowerCase().includes('vrma'))
  }
  return { idleNames, actionNames }
}

export interface BuildVrmAnimationsBridgeDeps {
  loadedClips: Record<string, THREE.AnimationClip>
  mixer: THREE.AnimationMixer
  allFiles: AnimationFile[]
  idleNames: string[]
  playByName: (name: string, options?: { fadeDuration?: number; loop?: boolean; additive?: boolean }) => void
  getCurrentClipName: () => string | null
}

/** Structured clip inventory for the anim-lab view (?view=animlab). Lazy
 * library clips are listed before they're loaded (duration 0, loaded
 * false); playing one loads it on demand. */
export const describeVrmClips = (
  allFiles: AnimationFile[],
  loadedClips: Record<string, THREE.AnimationClip>,
  idleNames: string[]
): VrmClipInfo[] =>
  allFiles
    .filter((file) => loadedClips[file.name] || file.lazy)
    .map((file) => ({
      name: file.name,
      filename: file.filename,
      directory: file.directory ?? null,
      role: file.role ?? 'gesture',
      tags: file.tags ?? [],
      disabled: Boolean(file.disabled),
      additive: Boolean(file.additive),
      isFbx: Boolean(file.isFbx),
      lazy: Boolean(file.lazy),
      loaded: Boolean(loadedClips[file.name]),
      duration: loadedClips[file.name]?.duration ?? 0,
      looping: idleNames.includes(file.name),
    }))

export const buildVrmAnimationsBridge = (deps: BuildVrmAnimationsBridgeDeps): VrmAnimationControls => ({
  clips: deps.loadedClips,
  mixer: deps.mixer,
  play: (name, options) => deps.playByName(name, options),
  list: () => {
    console.log('Available animations:')
    deps.allFiles.forEach((file) => {
      if (deps.loadedClips[file.name]) {
        console.log(`  • ${file.name} (${file.filename})`)
      }
    })
  },
  describe: () => describeVrmClips(deps.allFiles, deps.loadedClips, deps.idleNames),
  current: () => deps.getCurrentClipName(),
})

export interface BuildLipSyncBridgeDeps {
  startLipSync: (base64Audio: string, mimeType: string) => Promise<void> | void
  fadeOutAndStopLipSync: () => void
  beginStream: () => Promise<void> | void
  enqueueLipSyncChunk: (base64Audio: string, seq?: number, emotion?: string, beat?: string) => Promise<void>
  endLipSyncStream: () => void
  isLipSyncAudible: () => boolean
}

export const buildLipSyncBridge = (deps: BuildLipSyncBridgeDeps): KurisuLipSync => ({
  play: ({ base64Audio, mimeType = 'audio/wav' }: { base64Audio: string; mimeType?: string }) =>
    deps.startLipSync(base64Audio, mimeType),
  stop: () => deps.fadeOutAndStopLipSync(),
  beginStream: () => {
    void deps.beginStream()
  },
  enqueue: ({
    base64Audio,
    seq,
    emotion,
    beat,
  }: {
    base64Audio: string
    mimeType?: string
    seq?: number
    emotion?: string
    beat?: string
  }) => deps.enqueueLipSyncChunk(base64Audio, seq, emotion, beat),
  endStream: () => deps.endLipSyncStream(),
  // isAudible, not `streaming`: the one-shot play() path (greeting/
  // presence/reminder lines, fillers) never sets `streaming`, so reading it
  // here let filler/presence guards fire while a clip was audibly playing —
  // two voices at once.
  isSpeaking: () => deps.isLipSyncAudible(),
})

export interface BuildExpressionsBridgeDeps {
  setExpression: (expression: string, intensity?: number, duration?: number) => void
  vrm: VRM
}

export const buildExpressionsBridge = (deps: BuildExpressionsBridgeDeps): KurisuExpressions => ({
  set: deps.setExpression,
  available: VRM_EXPRESSION_NAMES,
  // The MODEL's real expression roster (not the driver's canonical six) —
  // VRM0 packs carry custom names, VRoid models more — so listing the
  // hardcoded array was lying about what this VRM has.
  list: () =>
    console.log(
      'Available VRM expressions:',
      deps.vrm.expressionManager ? Object.keys(deps.vrm.expressionManager.expressionMap) : [],
    ),
})
