/**
 * Animation/clip loading machinery for the VRM avatar, extracted from
 * VRMViewer's scene effect so it can live as a plain (React-free) module:
 * manifest loading, file-name mapping, HTTP existence probing/discovery,
 * and FBX/VRMA clip loading + retargeting. Every function takes its
 * dependencies (vrm instance, manifest path, debug logger) explicitly —
 * no React imports, no window globals.
 */
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { VRM, type VRMHumanBoneName } from '@pixiv/three-vrm'
import { VRMAnimationLoaderPlugin, createVRMAnimationClip } from '@pixiv/three-vrm-animation'
import { resolveAssetPath } from '../config'

// 'subtle' = idle-variety one-shot pool (item 4): played through the
// one-shot ('actual') path on the long unattended-idle timer, never looped.
// Distinct from 'gesture', which is reserved for user/reply-triggered
// reactions (emotion map, poke, manual trigger).
export type AnimationRole = 'idle' | 'gesture' | 'talk' | 'subtle'

export type AnimationDescriptor = {
  filename: string
  label?: string
  directory?: string
  role?: AnimationRole
  tags?: string[]
  additive?: boolean
  disabled?: boolean
  lazy?: boolean
}

export type AnimationFile = {
  name: string
  filename: string
  url: string
  directory?: string
  relativePath: string
  role?: AnimationRole
  tags?: string[]
  // Mixamo sourcing pipeline (docs/animations.md): manifest entries ending
  // in .fbx are loaded via mixamoRetarget.ts instead of the VRMA loader.
  isFbx?: boolean
  // VRM-named BVH packs (docs/animations.md): .bvh entries load via
  // bvhRetarget.ts.
  isBvh?: boolean
  // Item 3: play this clip as an additive layer (THREE.AnimationUtils.
  // makeClipAdditive) over the running base idle instead of crossfading to
  // replace it. Only safe for clips whose motion is naturally additive
  // (localized upper-body/hand motion); full-body clips can double-transform
  // and look broken, so this must be explicitly opted into per clip via the
  // manifest and visually vetted.
  additive?: boolean
  // Excludes a clip from all automatic pools (emotion map, subtle idle
  // variety) after visual vetting found it broken; still loaded/listed but
  // never auto-triggered.
  disabled?: boolean
  // Library clip: listed in describe()/the anim lab but not fetched or
  // parsed until first played (kurisuBridge.vrmAnimations.play / the lab). Keeps
  // big clip libraries from slowing startup. Lazy clips never join
  // automatic pools (idle rotation, talk, emotion map) — those need
  // eager-loaded, vetted clips.
  lazy?: boolean
}

/** Per-clip flags carried from the manifest, keyed by resolved clip name. */
export type ClipMeta = { additive?: boolean; disabled?: boolean; role?: AnimationRole }

export type DebugLog = (...args: unknown[]) => void

export const ACTION_NAME_OVERRIDES: Record<string, string> = {
  'VRMA_01.vrma': 'Show Full Body',
  'VRMA_02.vrma': 'Greeting',
  'VRMA_03.vrma': 'Peace Sign',
  'VRMA_04.vrma': 'Shoot',
  'VRMA_05.vrma': 'Spin',
  'VRMA_06.vrma': 'Model Pose',
  'VRMA_07.vrma': 'Squat',
  'VRMA_08.vrma': 'Look Around',
  'VRMA_09.vrma': 'Action 09',
}

// Animation files may still be loaded and available through
// kurisuBridge.vrmAnimations (window.vrmAnimations alias), but unattended
// limb-heavy gestures are intentionally disabled here. The sourced action
// pack includes clips that look fine in isolation but can make arms/hands
// twist when fired automatically over the conversation UI.

const HEAD_STABILIZED_BONES: VRMHumanBoneName[] = ['head', 'neck']
const IDLE_STABILIZED_BONES: VRMHumanBoneName[] = ['hips', 'head', 'neck']
const STABILIZED_TRACK_PROPERTIES = new Set(['quaternion', 'rotation', 'position'])

export const withoutBoneMotionTracks = (
  clip: THREE.AnimationClip,
  vrm: VRM,
  boneNames: VRMHumanBoneName[],
): THREE.AnimationClip => {
  const lockedNames = new Set<string>(boneNames)
  boneNames.forEach((boneName) => {
    const node = vrm.humanoid?.getNormalizedBoneNode(boneName)
    if (node?.name) lockedNames.add(node.name)
  })

  const tracks = clip.tracks.filter((track) => {
    const [target, property] = track.name.split('.')
    if (!STABILIZED_TRACK_PROPERTIES.has(property)) return true
    return !lockedNames.has(target)
  })

  if (tracks.length === clip.tracks.length) return clip
  return new THREE.AnimationClip(clip.name, clip.duration, tracks)
}

export const DEFAULT_IDLE_PRESETS: AnimationDescriptor[] = [
  { filename: 'idle_animation03.vrma', label: 'Standing Idle', directory: 'idle' },
]

export const DEFAULT_ACTION_PRESETS: AnimationDescriptor[] = [
  { filename: 'VRMA_02.vrma', label: 'Greeting', directory: 'action' },
]

export const normalizeStaticPath = (input?: string | null) => {
  if (!input) return null
  if (/^(https?:|capacitor:|file:)/i.test(input)) {
    return input
  }
  return resolveAssetPath(input)
}

const sanitizeDirectory = (directory?: string | null): string | null => {
  if (!directory) return null
  const trimmed = directory.trim().replace(/^\/+|\/+$/g, '')
  return trimmed.length > 0 ? trimmed : null
}

export const toAnimationFile = (
  filename: string,
  friendlyName?: string,
  directoryHint?: string,
  role?: AnimationRole,
  tags?: string[],
  additive?: boolean,
  disabled?: boolean,
  lazy?: boolean
): AnimationFile => {
  let normalizedFilename = filename.trim()
  let explicitDirectory: string | null = null

  if (normalizedFilename.includes('/')) {
    const segments = normalizedFilename.split('/')
    explicitDirectory = segments.slice(0, -1).join('/')
    normalizedFilename = segments[segments.length - 1]
  }

  const isFbx = /\.fbx$/i.test(normalizedFilename)
  const isBvh = /\.bvh$/i.test(normalizedFilename)
  if (!isFbx && !isBvh && !normalizedFilename.endsWith('.vrma')) {
    normalizedFilename = `${normalizedFilename}.vrma`
  }

  const directory = sanitizeDirectory(explicitDirectory ?? directoryHint)
  const relativePath = directory ? `${directory}/${normalizedFilename}` : normalizedFilename

  const defaultName = normalizedFilename
    .replace(/\.(vrma|fbx)$/i, '')
    .replace(/_/g, ' ')
    .replace(/\b([a-z])/g, (char) => char.toUpperCase())

  return {
    filename: normalizedFilename,
    name: friendlyName ?? defaultName,
    directory: directory ?? undefined,
    relativePath,
    url: resolveAssetPath(`/static/animations/${relativePath}`),
    role,
    tags,
    isFbx,
    isBvh,
    additive,
    disabled,
    lazy,
  }
}

export const checkFileExists = async (url: string): Promise<boolean> => {
  try {
    const headResponse = await fetch(url, { method: 'HEAD', cache: 'no-store' })
    if (headResponse.ok) return true
    if ([301, 302, 307, 308].includes(headResponse.status)) {
      return true
    }
    if ([403, 405, 501].includes(headResponse.status)) {
      const getResponse = await fetch(url, { method: 'GET', cache: 'no-store' })
      if (getResponse.body?.cancel) {
        try {
          getResponse.body.cancel()
        } catch {
          /* ignore cancellation errors */
        }
      }
      return getResponse.ok
    }
    if (headResponse.status === 404) {
      const getResponse = await fetch(url, { method: 'GET', cache: 'no-store' })
      if (getResponse.body?.cancel) {
        try {
          getResponse.body.cancel()
        } catch {
          /* ignore cancellation errors */
        }
      }
      return getResponse.ok
    }
  } catch (error) {
    console.warn('Failed to probe animation asset:', url, error)
  }
  return false
}

export const discoverAnimations = async ({
  prefix,
  suffix = '.vrma',
  labelPrefix,
  overrides,
  maxIndex = 60,
  zeroPad = 2,
  skipIndices,
  directory,
}: {
  prefix: string
  suffix?: string
  labelPrefix?: string
  overrides?: Record<string, string>
  maxIndex?: number
  zeroPad?: number
  skipIndices?: Set<number>
  directory?: string
}): Promise<AnimationFile[]> => {
  const discovered: AnimationFile[] = []
  const seen = new Set<string>()

  for (let i = 1; i <= maxIndex; i += 1) {
    if (skipIndices?.has(i)) {
      continue
    }

    const padded = i.toString().padStart(zeroPad, '0')
    const suffixVariants = [suffix, suffix.toLowerCase(), suffix.toUpperCase()]
    const candidateVariants = [
      { stem: `${prefix}${padded}`, numberLabel: padded },
      { stem: `${prefix}${i}`, numberLabel: i.toString() },
      { stem: `${prefix}${padded.toLowerCase()}`, numberLabel: padded },
      { stem: `${prefix}${padded}`.toLowerCase(), numberLabel: padded },
    ]

    let foundForIndex = false
    for (const variant of candidateVariants) {
      for (const suffixVariant of suffixVariants) {
        const filename = `${variant.stem}${suffixVariant}`
        const animationFile = toAnimationFile(filename, undefined, directory)
        if (seen.has(animationFile.relativePath)) {
          foundForIndex = true
          break
        }
        const exists = await checkFileExists(animationFile.url)
        if (!exists) {
          continue
        }

        const overrideName = overrides?.[filename] ?? overrides?.[animationFile.filename]
        const friendlyName =
          overrideName ||
          (labelPrefix ? `${labelPrefix} ${variant.numberLabel}` : filename.replace(suffixVariant, ''))

        animationFile.name = friendlyName
        discovered.push(animationFile)
        seen.add(animationFile.relativePath)
        foundForIndex = true
        break
      }
      if (foundForIndex) {
        break
      }
    }

    if (!foundForIndex) {
      const rawVariants = [
        `${prefix}${i}`,
        `${prefix}${padded}`,
        `${prefix}${padded.toLowerCase()}`,
      ]

      let anyRawExists = false
      for (const raw of rawVariants) {
        for (const suffixVariant of suffixVariants) {
          const filename = `${raw}${suffixVariant}`
          const animationFile = toAnimationFile(filename, undefined, directory)
          if (seen.has(animationFile.relativePath)) {
            anyRawExists = true
            break
          }
          if (await checkFileExists(animationFile.url)) {
            seen.add(animationFile.relativePath)
            anyRawExists = true
            break
          }
        }
        if (anyRawExists) break
      }

      if (!anyRawExists) {
        break
      }
    }
  }

  return discovered
}

export const parseIndexFromFilename = (filename: string, prefix: string): number | null => {
  const baseFilename = filename.split('/').pop() ?? filename
  const lowerFilename = baseFilename.toLowerCase()
  const lowerPrefix = prefix.toLowerCase()
  if (!lowerFilename.startsWith(lowerPrefix)) return null
  const remainder = lowerFilename.slice(lowerPrefix.length)
  const match = remainder.match(/^(\d+)/)
  if (!match) return null
  return Number.parseInt(match[1], 10)
}

export const resolvePresetAnimations = async (
  presets: AnimationDescriptor[],
  prefix: string,
  verify: boolean,
  directoryHint?: string
): Promise<{ files: AnimationFile[]; indices: Set<number> }> => {
  const resolvedEntries = verify
    ? (
        await Promise.all(
          presets.map(async (preset) => {
            const file = toAnimationFile(
              preset.filename,
              preset.label,
              preset.directory ?? directoryHint,
              preset.role,
              preset.tags,
              preset.additive,
              preset.disabled,
              preset.lazy
            )
            // .fbx entries are retargeted at load time (mixamoRetarget),
            // not fetched as a VRMA over HTTP HEAD/GET here — the
            // existence probe below is only meaningful for the
            // /static/animations/{idle,action}/*.vrma convention.
            if (file.isFbx) {
              return { preset, file }
            }
            const exists = await checkFileExists(file.url)
            if (!exists) {
              console.warn(`🛈 Animation asset missing: ${file.relativePath}`)
              return null
            }
            return { preset, file }
          })
        )
      ).filter(
        (item): item is { preset: AnimationDescriptor; file: AnimationFile } => Boolean(item)
      )
    : presets.map((preset) => ({
        preset,
        file: toAnimationFile(
          preset.filename,
          preset.label,
          preset.directory ?? directoryHint,
          preset.role,
          preset.tags,
          preset.additive,
          preset.disabled,
          preset.lazy
        ),
      }))

  const files: AnimationFile[] = []
  const indices = new Set<number>()

  resolvedEntries.forEach(({ preset, file }) => {
    files.push(file)
    const index = parseIndexFromFilename(preset.filename, prefix)
    if (index !== null) {
      indices.add(index)
    }
  })

  return { files, indices }
}

export const mergeAnimationFiles = (primary: AnimationFile[], secondary: AnimationFile[]): AnimationFile[] => {
  const map = new Map<string, AnimationFile>()
  const pushFiles = (list: AnimationFile[]) => {
    list.forEach((item) => {
      const key = item.relativePath
      if (!map.has(key)) {
        map.set(key, item)
      }
    })
  }
  pushFiles(primary)
  pushFiles(secondary)
  return Array.from(map.values())
}

const unionNumberSets = (target: Set<number>, source: Set<number>) => {
  source.forEach((value) => target.add(value))
}

type AnimationManifestEntry =
  | string
  | {
      file?: string
      path?: string
      name?: string
      label?: string
      directory?: string
      role?: AnimationRole
      tags?: string[]
      additive?: boolean
      disabled?: boolean
      lazy?: boolean
    }
type AnimationManifest = {
  idle?: AnimationManifestEntry[]
  action?: AnimationManifestEntry[]
}

const normalizeManifestEntry = (entry: AnimationManifestEntry): AnimationDescriptor | null => {
  if (typeof entry === 'string') {
    return { filename: entry }
  }
  if (!entry) {
    return null
  }
  const filename = entry.path ?? entry.file
  if (!filename) {
    return null
  }
  const descriptor: AnimationDescriptor = {
    filename,
    label: entry.name ?? entry.label,
    role: entry.role,
    tags: entry.tags,
    additive: entry.additive,
    disabled: entry.disabled,
    lazy: entry.lazy,
  }
  if (entry.directory) {
    descriptor.directory = entry.directory
  }
  return descriptor
}

const fetchManifestData = async (path: string): Promise<AnimationManifest | null> => {
  try {
    const response = await fetch(path, { cache: 'no-store' })
    if (!response.ok) {
      return null
    }
    return (await response.json()) as AnimationManifest
  } catch {
    return null
  }
}

const manifestEntriesToDescriptors = (
  entries: AnimationManifestEntry[] | undefined
): AnimationDescriptor[] =>
  (Array.isArray(entries) ? entries : [])
    .map((entry) => normalizeManifestEntry(entry))
    .filter((item): item is AnimationDescriptor => Boolean(item))

/**
 * Overlay local descriptors onto the committed set. Same file (by
 * directory/filename) → the local entry replaces the committed one IN
 * PLACE, so a manifest.local.json can re-role or disable a built-in clip
 * without editing the committed manifest; unknown files append at the end.
 */
const mergeDescriptorLists = (
  base: AnimationDescriptor[],
  overlay: AnimationDescriptor[],
  directoryHint: string
): AnimationDescriptor[] => {
  if (overlay.length === 0) return base
  const keyOf = (descriptor: AnimationDescriptor) =>
    `${(descriptor.directory ?? directoryHint).toLowerCase()}/${descriptor.filename.toLowerCase()}`
  const merged = [...base]
  const indexByKey = new Map(merged.map((descriptor, index) => [keyOf(descriptor), index]))
  overlay.forEach((descriptor) => {
    const existingIndex = indexByKey.get(keyOf(descriptor))
    if (existingIndex !== undefined) {
      merged[existingIndex] = descriptor
    } else {
      indexByKey.set(keyOf(descriptor), merged.length)
      merged.push(descriptor)
    }
  })
  return merged
}

export const loadAnimationManifest = async ({
  manifestPath,
  enableDiscovery,
}: {
  manifestPath: string | null
  enableDiscovery: boolean
}): Promise<
  | {
      idle: AnimationFile[]
      idleIndices: Set<number>
      action: AnimationFile[]
      actionIndices: Set<number>
    }
  | null
> => {
  if (!manifestPath) {
    return null
  }
  try {
    const data = await fetchManifestData(manifestPath)
    // Optional uncommitted overlay next to the main manifest — the local
    // sourcing workflow (docs/animations.md): personal clips (e.g. Mixamo
    // conversions that can't be redistributed) live in gitignored dirs and
    // are wired through this gitignored manifest instead of the committed
    // one.
    const localData = /manifest\.json$/.test(manifestPath)
      ? await fetchManifestData(manifestPath.replace(/manifest\.json$/, 'manifest.local.json'))
      : null
    if (!data && !localData) {
      return null
    }

    const idleDescriptors = mergeDescriptorLists(
      manifestEntriesToDescriptors(data?.idle),
      manifestEntriesToDescriptors(localData?.idle),
      'idle'
    )
    const actionDescriptors = mergeDescriptorLists(
      manifestEntriesToDescriptors(data?.action),
      manifestEntriesToDescriptors(localData?.action),
      'action'
    )

    const idleResult = await resolvePresetAnimations(
      idleDescriptors,
      'idle_animation',
      enableDiscovery,
      'idle'
    )
    const actionResult = await resolvePresetAnimations(
      actionDescriptors,
      'vrma_',
      enableDiscovery,
      'action'
    )

    return {
      idle: idleResult.files,
      idleIndices: idleResult.indices,
      action: actionResult.files,
      actionIndices: actionResult.indices,
    }
  } catch (error) {
    console.warn('Animation manifest unavailable', error)
    return null
  }
}

/**
 * Resolves the full idle/action AnimationFile lists for a session: manifest
 * first, built-in presets as the fallback when the manifest is missing or
 * empty, plus optional HTTP-probe discovery of numbered clip files.
 */
export const resolveAnimationFiles = async ({
  manifestPath,
  enableDiscovery,
  debugLog,
}: {
  manifestPath: string | null
  enableDiscovery: boolean
  debugLog: DebugLog
}): Promise<{ idleFiles: AnimationFile[]; actionFiles: AnimationFile[] }> => {
  const manifest = await loadAnimationManifest({ manifestPath, enableDiscovery })

  let idleFiles: AnimationFile[] = []
  let actionFiles: AnimationFile[] = []
  let idleIndices = new Set<number>()
  let actionIndices = new Set<number>()

  if (manifest) {
    idleFiles = manifest.idle
    actionFiles = manifest.action
    idleIndices = new Set(manifest.idleIndices)
    actionIndices = new Set(manifest.actionIndices)
    debugLog('📄 Loaded animation manifest with', idleFiles.length, 'idle and', actionFiles.length, 'action clips')
  }

  if (!manifest || idleFiles.length === 0) {
    const idlePreset = await resolvePresetAnimations(
      DEFAULT_IDLE_PRESETS,
      'idle_animation',
      enableDiscovery,
      'idle'
    )
    if (idleFiles.length === 0) {
      idleFiles = idlePreset.files
    } else {
      idleFiles = mergeAnimationFiles(idleFiles, idlePreset.files)
    }
    unionNumberSets(idleIndices, idlePreset.indices)
  }

  if (!manifest || actionFiles.length === 0) {
    const actionPreset = await resolvePresetAnimations(
      DEFAULT_ACTION_PRESETS,
      'vrma_',
      enableDiscovery,
      'action'
    )
    if (actionFiles.length === 0) {
      actionFiles = actionPreset.files
    } else {
      actionFiles = mergeAnimationFiles(actionFiles, actionPreset.files)
    }
    unionNumberSets(actionIndices, actionPreset.indices)
  }

  if (enableDiscovery) {
    const discoveredIdle = await discoverAnimations({
      prefix: 'idle_animation',
      labelPrefix: 'Idle',
      skipIndices: idleIndices,
      directory: 'idle',
    })
    idleFiles = mergeAnimationFiles(idleFiles, discoveredIdle)

    const discoveredAction = await discoverAnimations({
      prefix: 'VRMA_',
      labelPrefix: 'VRMA',
      overrides: ACTION_NAME_OVERRIDES,
      skipIndices: actionIndices,
      directory: 'action',
    })
    actionFiles = mergeAnimationFiles(actionFiles, discoveredAction)
  }

  return { idleFiles, actionFiles }
}

export const loadVRMAAnimation = async (file: AnimationFile, vrm: VRM): Promise<THREE.AnimationClip> => {
  const vrmaLoader = new GLTFLoader()
  vrmaLoader.register((parser) => new VRMAnimationLoaderPlugin(parser))

  const vrmaGltf = await vrmaLoader.loadAsync(file.url)
  const vrmAnimations = vrmaGltf.userData.vrmAnimations

  if (!vrmAnimations || vrmAnimations.length === 0) {
    throw new Error(`No VRM animations found in ${file.url}`)
  }

  return createVRMAnimationClip(vrmAnimations[0], vrm)
}

// Lazily imported: FBXLoader (and the retargeting math in
// mixamoRetarget.ts) only get pulled into a loaded chunk when a
// manifest actually references an .fbx clip, so the default bundle
// (VRMA-only setups, i.e. everyone who hasn't dropped in Mixamo
// clips) doesn't grow.
export const loadFbxAnimation = async (file: AnimationFile, vrm: VRM): Promise<THREE.AnimationClip> => {
  const { loadMixamoAnimation } = await import('../mixamoRetarget')
  return loadMixamoAnimation(file.url, vrm)
}

// Same lazy-import treatment for BVHLoader (VRM-named BVH packs).
export const loadBvhAnimationFile = async (
  file: AnimationFile,
  vrm: VRM
): Promise<THREE.AnimationClip> => {
  const { loadBvhAnimation } = await import('../bvhRetarget')
  return loadBvhAnimation(file.url, vrm)
}

export interface LoadedClipLibrary {
  loadedClips: Record<string, THREE.AnimationClip>
  // Item 3 (additive layering) + item 4 (idle-variety pool)/vetting:
  // per-clip flags carried from the manifest, keyed by the same
  // resolved clip name used in loadedClips.
  clipMeta: Record<string, ClipMeta>
  clipAliases: Map<string, string>
  // Manifest entries tagged role: "talk" (docs/animations.md). The
  // scheduler rotates among them per speaking stretch; empty = no talk
  // clip, speech leaves the body animation alone.
  talkClipNames: string[]
  allFiles: AnimationFile[]
  /** Fetch+parse a lazy library clip on demand; resolves true once the
   * clip is available in loadedClips. Safe to call for eager clips too. */
  ensureClipLoaded: (name: string) => Promise<boolean>
}

/**
 * Loads (and retargets, for .fbx) every resolved clip against the given VRM:
 * the first idle first (so the base loop is available as soon as possible),
 * then the rest in parallel. Individual clip failures are logged and
 * skipped, matching the original inline behavior.
 */
export const loadAnimationClips = async ({
  idleFiles,
  actionFiles,
  vrm,
  debugLog,
}: {
  idleFiles: AnimationFile[]
  actionFiles: AnimationFile[]
  vrm: VRM
  debugLog: DebugLog
}): Promise<LoadedClipLibrary> => {
  const allFiles = [...idleFiles, ...actionFiles]

  const loadedClips: Record<string, THREE.AnimationClip> = {}
  const clipMeta: Record<string, ClipMeta> = {}
  const clipAliases = new Map<string, string>()

  const registerAliases = (aliases: string[], target: string) => {
    aliases.forEach((alias) => {
      if (!alias) return
      clipAliases.set(alias, target)
      clipAliases.set(alias.toLowerCase(), target)
    })
  }

  const loadedPaths = new Set<string>()
  const talkClipNames: string[] = []

  const loadAnimationFile = async (file: AnimationFile) => {
    if (loadedPaths.has(file.relativePath)) {
      return
    }
    try {
      const rawClip = file.isFbx
        ? await loadFbxAnimation(file, vrm)
        : file.isBvh
          ? await loadBvhAnimationFile(file, vrm)
          : await loadVRMAAnimation(file, vrm)
      const clip = withoutBoneMotionTracks(
        rawClip,
        vrm,
        file.role === 'idle' ? IDLE_STABILIZED_BONES : HEAD_STABILIZED_BONES,
      )
      loadedClips[file.name] = clip
      clipMeta[file.name] = { additive: file.additive, disabled: file.disabled, role: file.role }
      const baseName = file.filename.replace(/\.(vrma|fbx)$/i, '')
      registerAliases(
        [file.name, file.filename, baseName, baseName.replace(/_/g, ' ')],
        file.name
      )
      if (file.role === 'talk' && !talkClipNames.includes(file.name)) {
        talkClipNames.push(file.name)
      }
      debugLog(`✅ Loaded ${file.isFbx ? 'FBX' : 'VRMA'}: ${file.name} (Duration: ${clip.duration.toFixed(2)}s)`)
      loadedPaths.add(file.relativePath)
    } catch (error) {
      console.error(`❌ Failed to load animation ${file.name}:`, error)
    }
  }

  // Lazy library clips: register meta + aliases so they resolve and list,
  // but defer the fetch/parse until first played (ensureClipLoaded below).
  const lazyByName = new Map<string, AnimationFile>()
  allFiles
    .filter((file) => file.lazy)
    .forEach((file) => {
      lazyByName.set(file.name, file)
      clipMeta[file.name] = { additive: file.additive, disabled: file.disabled, role: file.role }
      const baseName = file.filename.replace(/\.(vrma|fbx)$/i, '')
      registerAliases([file.name, file.filename, baseName, baseName.replace(/_/g, ' ')], file.name)
    })

  // Load all eager animations
  try {
    const eagerFiles = allFiles.filter((file) => !file.lazy)
    let initialIdleName: string | null = null
    const firstEagerIdle = idleFiles.find((file) => !file.lazy)
    if (firstEagerIdle) {
      await loadAnimationFile(firstEagerIdle)
      initialIdleName = firstEagerIdle.name
    }

    const remainingFiles = eagerFiles.filter((file) => file.name !== initialIdleName)
    await Promise.all(remainingFiles.map((file) => loadAnimationFile(file)))
  } catch (error) {
    console.error('❌ Failed to load one or more VRM animations:', error)
  }

  const pendingLoads = new Map<string, Promise<boolean>>()
  const ensureClipLoaded = async (name: string): Promise<boolean> => {
    if (loadedClips[name]) return true
    const file = lazyByName.get(name)
    if (!file) return false
    let pending = pendingLoads.get(name)
    if (!pending) {
      pending = loadAnimationFile(file).then(() => Boolean(loadedClips[name]))
      pendingLoads.set(name, pending)
    }
    return pending
  }

  return { loadedClips, clipMeta, clipAliases, talkClipNames, allFiles, ensureClipLoaded }
}
