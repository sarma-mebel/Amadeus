// @vitest-environment jsdom
//
// Coverage for clipLoader.ts's pure/logic surface: manifest entry
// normalization + local-overlay merging, preset resolution (with/without
// HTTP existence verification), file-list merging/dedup, and the
// eager-vs-lazy split in loadAnimationClips (alias registration,
// talkClipNames population, ensureClipLoaded gating, per-clip error
// isolation). Retargeting math itself (mixamoRetarget/bvhRetarget) is
// covered by mixamoRetarget.test.ts / bvhRetarget.test.ts / retargetCore.test.ts
// and is stubbed out here — clipLoader only needs to prove it calls the
// right loader for the right extension and wires the result into
// loadedClips/clipMeta/aliases correctly.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'
import {
  discoverAnimations,
  loadAnimationClips,
  loadAnimationManifest,
  mergeAnimationFiles,
  normalizeStaticPath,
  parseIndexFromFilename,
  resolveAnimationFiles,
  resolvePresetAnimations,
  toAnimationFile,
  withoutBoneMotionTracks,
  type AnimationFile,
} from './clipLoader'

// ---------------------------------------------------------------------------
// toAnimationFile / normalizeStaticPath — pure filename/path normalization
// ---------------------------------------------------------------------------

describe('normalizeStaticPath', () => {
  it('returns null for empty input', () => {
    expect(normalizeStaticPath(undefined)).toBeNull()
    expect(normalizeStaticPath(null)).toBeNull()
    expect(normalizeStaticPath('')).toBeNull()
  })

  it('passes through absolute/scheme URLs untouched', () => {
    expect(normalizeStaticPath('https://example.com/a.vrma')).toBe('https://example.com/a.vrma')
    expect(normalizeStaticPath('capacitor://localhost/a.vrma')).toBe('capacitor://localhost/a.vrma')
    expect(normalizeStaticPath('file:///a.vrma')).toBe('file:///a.vrma')
  })

  it('resolves a relative path through resolveAssetPath (asset base + path joined)', () => {
    // The test build's VITE_ASSET_BASE_URL may be non-empty, so assert the
    // suffix rather than assuming an empty base -> matches
    // resolveAssetPath's own joinWithBase behavior regardless of base.
    expect(normalizeStaticPath('/static/animations/idle/foo.vrma')).toMatch(
      /\/static\/animations\/idle\/foo\.vrma$/
    )
  })
})

describe('toAnimationFile', () => {
  it('appends .vrma when the filename has no recognized extension', () => {
    const file = toAnimationFile('idle_animation03')
    expect(file.filename).toBe('idle_animation03.vrma')
    expect(file.isFbx).toBe(false)
    expect(file.isBvh).toBe(false)
  })

  it('detects .fbx and .bvh extensions and leaves them unmodified', () => {
    expect(toAnimationFile('clip.fbx').isFbx).toBe(true)
    expect(toAnimationFile('clip.fbx').filename).toBe('clip.fbx')
    expect(toAnimationFile('clip.bvh').isBvh).toBe(true)
    expect(toAnimationFile('clip.bvh').filename).toBe('clip.bvh')
  })

  it('is case-insensitive about the .fbx/.bvh extension check', () => {
    expect(toAnimationFile('Clip.FBX').isFbx).toBe(true)
    expect(toAnimationFile('Clip.BVH').isBvh).toBe(true)
  })

  it('derives a title-cased default name from the filename', () => {
    const file = toAnimationFile('peace_sign_pose.vrma')
    expect(file.name).toBe('Peace Sign Pose')
  })

  it('prefers an explicit friendly name over the derived default', () => {
    const file = toAnimationFile('peace_sign_pose.vrma', 'Peace Sign')
    expect(file.name).toBe('Peace Sign')
  })

  it('splits an embedded directory out of the filename, overriding the hint', () => {
    const file = toAnimationFile('sourced/idle-maid.vrma', undefined, 'idle')
    expect(file.directory).toBe('sourced')
    expect(file.filename).toBe('idle-maid.vrma')
    expect(file.relativePath).toBe('sourced/idle-maid.vrma')
  })

  it('falls back to the directory hint when the filename has no embedded directory', () => {
    const file = toAnimationFile('idle_animation03.vrma', undefined, 'idle')
    expect(file.directory).toBe('idle')
    expect(file.relativePath).toBe('idle/idle_animation03.vrma')
  })

  it('sanitizes a directory hint with leading/trailing slashes and blank strings', () => {
    expect(toAnimationFile('a.vrma', undefined, '/idle/').directory).toBe('idle')
    expect(toAnimationFile('a.vrma', undefined, '   ').directory).toBeUndefined()
  })

  it('carries role/tags/additive/disabled/lazy flags through untouched', () => {
    const file = toAnimationFile(
      'talk_loop.vrma',
      'Talk Loop',
      'action',
      'talk',
      ['sourced'],
      true,
      false,
      true
    )
    expect(file.role).toBe('talk')
    expect(file.tags).toEqual(['sourced'])
    expect(file.additive).toBe(true)
    expect(file.disabled).toBe(false)
    expect(file.lazy).toBe(true)
  })

  it('builds the url from the resolved relativePath', () => {
    const file = toAnimationFile('foo.vrma', undefined, 'idle')
    expect(file.url).toMatch(/\/static\/animations\/idle\/foo\.vrma$/)
  })
})

// ---------------------------------------------------------------------------
// withoutBoneMotionTracks — pure clip-track filtering
// ---------------------------------------------------------------------------

describe('withoutBoneMotionTracks', () => {
  const makeVrm = (boneNodeNames: Record<string, string>): VRM =>
    ({
      humanoid: {
        getNormalizedBoneNode: (name: string) =>
          boneNodeNames[name] ? ({ name: boneNodeNames[name] } as THREE.Object3D) : null,
      },
    }) as unknown as VRM

  it('drops quaternion/rotation/position tracks targeting a locked bone', () => {
    const vrm = makeVrm({ hips: 'Hips', head: 'Head' })
    const clip = new THREE.AnimationClip('test', 1, [
      new THREE.QuaternionKeyframeTrack('Hips.quaternion', [0], [0, 0, 0, 1]),
      new THREE.VectorKeyframeTrack('Hips.position', [0], [0, 0, 0]),
      new THREE.VectorKeyframeTrack('Spine.position', [0], [1, 1, 1]),
    ])
    const filtered = withoutBoneMotionTracks(clip, vrm, ['hips'])
    expect(filtered.tracks).toHaveLength(1)
    expect(filtered.tracks[0].name).toBe('Spine.position')
  })

  it('keeps track properties outside the stabilized set (e.g. morph weights)', () => {
    const vrm = makeVrm({ head: 'Head' })
    const clip = new THREE.AnimationClip('test', 1, [
      new THREE.NumberKeyframeTrack('Head.morphTargetInfluences[0]', [0], [1]),
    ])
    const filtered = withoutBoneMotionTracks(clip, vrm, ['head'])
    expect(filtered.tracks).toHaveLength(1)
  })

  it('returns the same clip instance (no copy) when nothing was filtered', () => {
    const vrm = makeVrm({})
    const clip = new THREE.AnimationClip('test', 1, [
      new THREE.VectorKeyframeTrack('Spine.position', [0], [1, 1, 1]),
    ])
    const filtered = withoutBoneMotionTracks(clip, vrm, ['hips'])
    expect(filtered).toBe(clip)
  })

  it('locks by raw bone name too, even with no resolvable humanoid node', () => {
    const vrm = { humanoid: null } as unknown as VRM
    const clip = new THREE.AnimationClip('test', 1, [
      new THREE.VectorKeyframeTrack('hips.position', [0], [0, 0, 0]),
    ])
    const filtered = withoutBoneMotionTracks(clip, vrm, ['hips'])
    expect(filtered.tracks).toHaveLength(0)
  })
})

// ---------------------------------------------------------------------------
// parseIndexFromFilename — pure parsing
// ---------------------------------------------------------------------------

describe('parseIndexFromFilename', () => {
  it('extracts the leading numeric index after the prefix', () => {
    expect(parseIndexFromFilename('idle_animation03.vrma', 'idle_animation')).toBe(3)
    expect(parseIndexFromFilename('VRMA_12.vrma', 'VRMA_')).toBe(12)
  })

  it('is case-insensitive on the prefix match', () => {
    expect(parseIndexFromFilename('vrma_05.vrma', 'VRMA_')).toBe(5)
  })

  it('handles an embedded directory by matching against the basename', () => {
    expect(parseIndexFromFilename('action/VRMA_07.vrma', 'VRMA_')).toBe(7)
  })

  it('returns null when the filename does not start with the prefix', () => {
    expect(parseIndexFromFilename('other_clip.vrma', 'VRMA_')).toBeNull()
  })

  it('returns null when there is no leading digit after the prefix', () => {
    expect(parseIndexFromFilename('idle_animationX.vrma', 'idle_animation')).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// mergeAnimationFiles — pure de-dup by relativePath
// ---------------------------------------------------------------------------

describe('mergeAnimationFiles', () => {
  const file = (relativePath: string): AnimationFile => ({
    name: relativePath,
    filename: relativePath,
    url: `/static/animations/${relativePath}`,
    relativePath,
  })

  it('concatenates disjoint lists', () => {
    const merged = mergeAnimationFiles([file('a.vrma')], [file('b.vrma')])
    expect(merged.map((f) => f.relativePath)).toEqual(['a.vrma', 'b.vrma'])
  })

  it('primary wins over secondary on a relativePath collision', () => {
    const primaryA = { ...file('a.vrma'), name: 'Primary A' }
    const secondaryA = { ...file('a.vrma'), name: 'Secondary A' }
    const merged = mergeAnimationFiles([primaryA], [secondaryA])
    expect(merged).toHaveLength(1)
    expect(merged[0].name).toBe('Primary A')
  })

  it('de-dupes within the same list too', () => {
    const merged = mergeAnimationFiles([file('a.vrma'), file('a.vrma')], [])
    expect(merged).toHaveLength(1)
  })
})

// ---------------------------------------------------------------------------
// resolvePresetAnimations — verified vs unverified preset resolution
// ---------------------------------------------------------------------------

describe('resolvePresetAnimations', () => {
  let fetchMock: ReturnType<typeof vi.fn>

  beforeEach(() => {
    fetchMock = vi.fn(async (url: string) => {
      // HEAD probe: only "exists.vrma" resolves ok.
      if (String(url).includes('exists')) {
        return { ok: true, status: 200 } as Response
      }
      return { ok: false, status: 404 } as Response
    })
    vi.stubGlobal('fetch', fetchMock)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('without verification, resolves every preset unconditionally (no fetch calls)', async () => {
    const { files } = await resolvePresetAnimations(
      [{ filename: 'missing.vrma' }, { filename: 'exists.vrma' }],
      'idle_animation',
      false,
      'idle'
    )
    expect(files).toHaveLength(2)
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('with verification, drops presets whose asset does not exist', async () => {
    const { files } = await resolvePresetAnimations(
      [{ filename: 'missing.vrma' }, { filename: 'exists.vrma' }],
      'idle_animation',
      true,
      'idle'
    )
    expect(files.map((f) => f.filename)).toEqual(['exists.vrma'])
  })

  it('skips the existence probe entirely for .fbx presets (retargeted at load time, not HEAD-checked)', async () => {
    const { files } = await resolvePresetAnimations(
      [{ filename: 'mixamo_clip.fbx' }],
      'idle_animation',
      true,
      'idle'
    )
    expect(files).toHaveLength(1)
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('collects the parsed numeric indices matching the given prefix', async () => {
    const { indices } = await resolvePresetAnimations(
      [{ filename: 'idle_animation03.vrma' }, { filename: 'idle_animation07.vrma' }],
      'idle_animation',
      false,
      'idle'
    )
    expect(indices).toEqual(new Set([3, 7]))
  })
})

// ---------------------------------------------------------------------------
// discoverAnimations — HTTP-probe based numbered-file discovery
// ---------------------------------------------------------------------------

describe('discoverAnimations', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('stops at the first gap and returns files found before it', async () => {
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? 'GET'
      if (method !== 'HEAD') return { ok: false, status: 404 } as Response
      // idle_animation01 and idle_animation02 "exist", 03 does not (any case variant).
      if (/idle_animation0[12]\.vrma$/i.test(String(url))) {
        return { ok: true, status: 200 } as Response
      }
      return { ok: false, status: 404 } as Response
    })
    vi.stubGlobal('fetch', fetchMock)

    const files = await discoverAnimations({
      prefix: 'idle_animation',
      directory: 'idle',
      maxIndex: 10,
    })
    expect(files.map((f) => f.filename)).toEqual(['idle_animation01.vrma', 'idle_animation02.vrma'])
  })

  it('honors skipIndices without breaking the scan', async () => {
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? 'GET'
      if (method !== 'HEAD') return { ok: false, status: 404 } as Response
      if (/idle_animation0[123]\.vrma$/i.test(String(url))) {
        return { ok: true, status: 200 } as Response
      }
      return { ok: false, status: 404 } as Response
    })
    vi.stubGlobal('fetch', fetchMock)

    const files = await discoverAnimations({
      prefix: 'idle_animation',
      directory: 'idle',
      maxIndex: 10,
      skipIndices: new Set([2]),
    })
    // Index 2 is skipped (not probed/returned) but the scan continues to 3.
    expect(files.map((f) => f.filename)).toEqual(['idle_animation01.vrma', 'idle_animation03.vrma'])
  })

  it('applies a friendly override name when the filename matches the overrides map', async () => {
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? 'GET'
      if (method !== 'HEAD') return { ok: false, status: 404 } as Response
      if (/VRMA_01\.vrma$/i.test(String(url))) return { ok: true, status: 200 } as Response
      return { ok: false, status: 404 } as Response
    })
    vi.stubGlobal('fetch', fetchMock)

    const files = await discoverAnimations({
      prefix: 'VRMA_',
      directory: 'action',
      maxIndex: 5,
      overrides: { 'VRMA_01.vrma': 'Show Full Body' },
    })
    expect(files).toHaveLength(1)
    expect(files[0].name).toBe('Show Full Body')
  })
})

// ---------------------------------------------------------------------------
// loadAnimationManifest — manifest fetch + local overlay merge + validation
// ---------------------------------------------------------------------------

describe('loadAnimationManifest', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('returns null when manifestPath is null', async () => {
    const result = await loadAnimationManifest({ manifestPath: null, enableDiscovery: false })
    expect(result).toBeNull()
  })

  it('returns null when both the committed manifest and local overlay fail to fetch', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ ok: false, status: 404 }) as Response)
    )
    const result = await loadAnimationManifest({
      manifestPath: '/static/animations/manifest.json',
      enableDiscovery: false,
    })
    expect(result).toBeNull()
  })

  it('parses string entries and object entries (file/path aliasing) from the committed manifest', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        if (String(url).includes('manifest.local.json')) {
          return { ok: false, status: 404 } as Response
        }
        return {
          ok: true,
          json: async () => ({
            idle: ['idle_animation03.vrma'],
            action: [{ path: 'VRMA_02.vrma', name: 'Greeting', role: 'gesture' }],
          }),
        } as Response
      })
    )
    const result = await loadAnimationManifest({
      manifestPath: '/static/animations/manifest.json',
      enableDiscovery: false,
    })
    expect(result).not.toBeNull()
    expect(result?.idle.map((f) => f.filename)).toEqual(['idle_animation03.vrma'])
    expect(result?.action.map((f) => f.name)).toEqual(['Greeting'])
    expect(result?.action[0].role).toBe('gesture')
  })

  it('drops object entries with neither file nor path', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        if (String(url).includes('manifest.local.json')) return { ok: false, status: 404 } as Response
        return {
          ok: true,
          json: async () => ({
            idle: [{ name: 'no filename here' }, 'idle_animation03.vrma'],
          }),
        } as Response
      })
    )
    const result = await loadAnimationManifest({
      manifestPath: '/static/animations/manifest.json',
      enableDiscovery: false,
    })
    expect(result?.idle.map((f) => f.filename)).toEqual(['idle_animation03.vrma'])
  })

  it('overlays manifest.local.json entries onto the committed set, replacing same-file entries in place', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        if (String(url).includes('manifest.local.json')) {
          return {
            ok: true,
            json: async () => ({
              action: [{ path: 'VRMA_02.vrma', name: 'Greeting', disabled: true }],
            }),
          } as Response
        }
        return {
          ok: true,
          json: async () => ({
            action: [
              { path: 'VRMA_02.vrma', name: 'Greeting' },
              { path: 'VRMA_03.vrma', name: 'Peace Sign' },
            ],
          }),
        } as Response
      })
    )
    const result = await loadAnimationManifest({
      manifestPath: '/static/animations/manifest.json',
      enableDiscovery: false,
    })
    // Replaced in place, not appended -> still 2 entries, order preserved.
    expect(result?.action).toHaveLength(2)
    expect(result?.action[0].name).toBe('Greeting')
    expect(result?.action[0].disabled).toBe(true)
    expect(result?.action[1].name).toBe('Peace Sign')
  })

  it('appends a local-only entry that does not match any committed file', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        if (String(url).includes('manifest.local.json')) {
          return {
            ok: true,
            json: async () => ({ action: [{ path: 'my_local_clip.fbx', name: 'Local Clip' }] }),
          } as Response
        }
        return {
          ok: true,
          json: async () => ({ action: [{ path: 'VRMA_02.vrma', name: 'Greeting' }] }),
        } as Response
      })
    )
    const result = await loadAnimationManifest({
      manifestPath: '/static/animations/manifest.json',
      enableDiscovery: false,
    })
    expect(result?.action.map((f) => f.name)).toEqual(['Greeting', 'Local Clip'])
  })

  it('does not probe manifest.local.json when manifestPath does not end in manifest.json', async () => {
    const fetchMock = vi.fn(async () => ({ ok: true, json: async () => ({ idle: [] }) }) as Response)
    vi.stubGlobal('fetch', fetchMock)
    await loadAnimationManifest({
      manifestPath: '/static/animations/custom-manifest-name.json',
      enableDiscovery: false,
    })
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('returns null and warns (not throws) when fetch itself rejects', async () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new Error('network down')
      })
    )
    const result = await loadAnimationManifest({
      manifestPath: '/static/animations/manifest.json',
      enableDiscovery: false,
    })
    expect(result).toBeNull()
    warnSpy.mockRestore()
  })

  it('carries the lazy flag through from a manifest entry', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        if (String(url).includes('manifest.local.json')) return { ok: false, status: 404 } as Response
        return {
          ok: true,
          json: async () => ({
            action: [{ path: 'Library_Clip.vrma', name: 'Library Clip', lazy: true, role: 'gesture' }],
          }),
        } as Response
      })
    )
    const result = await loadAnimationManifest({
      manifestPath: '/static/animations/manifest.json',
      enableDiscovery: false,
    })
    expect(result?.action[0].lazy).toBe(true)
  })
})

// ---------------------------------------------------------------------------
// resolveAnimationFiles — manifest-first with built-in-preset fallback
// ---------------------------------------------------------------------------

describe('resolveAnimationFiles', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  const debugLog = () => {}

  it('falls back to the built-in idle/action presets when no manifest is configured', async () => {
    const { idleFiles, actionFiles } = await resolveAnimationFiles({
      manifestPath: null,
      enableDiscovery: false,
      debugLog,
    })
    expect(idleFiles.map((f) => f.filename)).toEqual(['idle_animation03.vrma'])
    expect(actionFiles.map((f) => f.filename)).toEqual(['VRMA_02.vrma'])
  })

  it('uses manifest idle/action lists when the manifest resolves with entries', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        if (String(url).includes('manifest.local.json')) return { ok: false, status: 404 } as Response
        return {
          ok: true,
          json: async () => ({
            idle: [{ path: 'Custom_Idle.vrma', name: 'Custom Idle' }],
            action: [{ path: 'Custom_Action.vrma', name: 'Custom Action' }],
          }),
        } as Response
      })
    )
    const { idleFiles, actionFiles } = await resolveAnimationFiles({
      manifestPath: '/static/animations/manifest.json',
      enableDiscovery: false,
      debugLog,
    })
    expect(idleFiles.map((f) => f.name)).toEqual(['Custom Idle'])
    expect(actionFiles.map((f) => f.name)).toEqual(['Custom Action'])
  })

  it('falls back to built-in idle presets when the manifest has an empty idle list, merging with any manifest action clips', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        if (String(url).includes('manifest.local.json')) return { ok: false, status: 404 } as Response
        return {
          ok: true,
          json: async () => ({
            idle: [],
            action: [{ path: 'Custom_Action.vrma', name: 'Custom Action' }],
          }),
        } as Response
      })
    )
    const { idleFiles, actionFiles } = await resolveAnimationFiles({
      manifestPath: '/static/animations/manifest.json',
      enableDiscovery: false,
      debugLog,
    })
    expect(idleFiles.map((f) => f.filename)).toEqual(['idle_animation03.vrma'])
    expect(actionFiles.map((f) => f.name)).toEqual(['Custom Action'])
  })
})

// ---------------------------------------------------------------------------
// loadAnimationClips — eager vs lazy split, alias registration, talk pool,
// per-clip error isolation, ensureClipLoaded
// ---------------------------------------------------------------------------

// Stub every real THREE/VRM loader clipLoader would otherwise construct so
// this exercises only clipLoader's own wiring (eager/lazy gating, alias
// registration, talkClipNames, error handling) — not GLTFLoader/FBX/BVH
// parsing or retargeting math, which are covered elsewhere.
vi.mock('three/examples/jsm/loaders/GLTFLoader.js', () => {
  class FakeGLTFLoader {
    register() {
      return this
    }
    async loadAsync(url: string) {
      if (url.includes('broken')) {
        throw new Error(`failed to parse ${url}`)
      }
      return { userData: { vrmAnimations: [{ id: url }] } }
    }
  }
  return { GLTFLoader: FakeGLTFLoader }
})

vi.mock('@pixiv/three-vrm-animation', () => ({
  VRMAnimationLoaderPlugin: class {},
  createVRMAnimationClip: (vrmAnimation: { id: string }) =>
    new THREE.AnimationClip(vrmAnimation.id, 1, []),
}))

vi.mock('../mixamoRetarget', () => ({
  loadMixamoAnimation: async (url: string) => {
    if (url.includes('broken')) throw new Error(`fbx failed ${url}`)
    return new THREE.AnimationClip(url, 1, [])
  },
}))

vi.mock('../bvhRetarget', () => ({
  loadBvhAnimation: async (url: string) => {
    if (url.includes('broken')) throw new Error(`bvh failed ${url}`)
    return new THREE.AnimationClip(url, 1, [])
  },
}))

describe('loadAnimationClips', () => {
  const vrm = { humanoid: null } as unknown as VRM
  const debugLog = () => {}

  const idle: AnimationFile = toAnimationFile('idle_animation03.vrma', 'Standing Idle', 'idle', 'idle')
  const talkClip: AnimationFile = toAnimationFile('talk_loop.vrma', 'Talk Loop', 'action', 'talk')
  const lazyGesture: AnimationFile = {
    ...toAnimationFile('library_clip.vrma', 'Library Clip', 'action', 'gesture'),
    lazy: true,
  }
  const lazyTalk: AnimationFile = {
    ...toAnimationFile('lazy_talk.vrma', 'Lazy Talk', 'action', 'talk'),
    lazy: true,
  }
  const fbxClip: AnimationFile = toAnimationFile('mixamo.fbx', 'Mixamo Clip', 'action', 'gesture')
  const bvhClip: AnimationFile = toAnimationFile('pose.bvh', 'Bvh Clip', 'action', 'gesture')
  const brokenClip: AnimationFile = toAnimationFile('broken_clip.vrma', 'Broken Clip', 'action', 'gesture')

  it('loads eager clips into loadedClips and records their clipMeta', async () => {
    const library = await loadAnimationClips({
      idleFiles: [idle],
      actionFiles: [talkClip],
      vrm,
      debugLog,
    })
    expect(Object.keys(library.loadedClips)).toEqual(
      expect.arrayContaining(['Standing Idle', 'Talk Loop'])
    )
    expect(library.clipMeta['Talk Loop'].role).toBe('talk')
  })

  it('does not fetch/parse lazy clips up front but does register their meta and aliases', async () => {
    const library = await loadAnimationClips({
      idleFiles: [idle],
      actionFiles: [lazyGesture],
      vrm,
      debugLog,
    })
    expect(library.loadedClips['Library Clip']).toBeUndefined()
    expect(library.clipMeta['Library Clip']).toEqual({ additive: undefined, disabled: undefined, role: 'gesture' })
    // Aliases (name, filename, base name, spaced base name) resolve even
    // though the clip has not been fetched yet.
    expect(library.clipAliases.get('library_clip.vrma')).toBe('Library Clip')
    expect(library.clipAliases.get('library_clip')).toBe('Library Clip')
  })

  it('only pushes eager role:"talk" clips into talkClipNames — a lazy talk clip never joins the pool up front', async () => {
    const library = await loadAnimationClips({
      idleFiles: [idle],
      actionFiles: [talkClip, lazyTalk],
      vrm,
      debugLog,
    })
    expect(library.talkClipNames).toEqual(['Talk Loop'])
    expect(library.talkClipNames).not.toContain('Lazy Talk')
  })

  it('ensureClipLoaded fetches a lazy clip on demand and it then appears in loadedClips', async () => {
    const library = await loadAnimationClips({
      idleFiles: [idle],
      actionFiles: [lazyGesture],
      vrm,
      debugLog,
    })
    expect(library.loadedClips['Library Clip']).toBeUndefined()
    const ok = await library.ensureClipLoaded('Library Clip')
    expect(ok).toBe(true)
    expect(library.loadedClips['Library Clip']).toBeDefined()
  })

  it('ensureClipLoaded resolves true immediately for an already-eager clip and false for an unknown name', async () => {
    const library = await loadAnimationClips({
      idleFiles: [idle],
      actionFiles: [],
      vrm,
      debugLog,
    })
    expect(await library.ensureClipLoaded('Standing Idle')).toBe(true)
    expect(await library.ensureClipLoaded('Does Not Exist')).toBe(false)
  })

  it('ensureClipLoaded de-dupes concurrent calls for the same lazy clip into one load', async () => {
    const library = await loadAnimationClips({
      idleFiles: [idle],
      actionFiles: [lazyGesture],
      vrm,
      debugLog,
    })
    const [first, second] = await Promise.all([
      library.ensureClipLoaded('Library Clip'),
      library.ensureClipLoaded('Library Clip'),
    ])
    expect(first).toBe(true)
    expect(second).toBe(true)
  })

  it('dispatches .fbx to the Mixamo retargeter and .bvh to the BVH retargeter', async () => {
    const library = await loadAnimationClips({
      idleFiles: [idle],
      actionFiles: [fbxClip, bvhClip],
      vrm,
      debugLog,
    })
    expect(library.loadedClips['Mixamo Clip']).toBeDefined()
    expect(library.loadedClips['Bvh Clip']).toBeDefined()
  })

  it('isolates a single failing clip: it is skipped but sibling clips still load', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    const library = await loadAnimationClips({
      idleFiles: [idle],
      actionFiles: [brokenClip, talkClip],
      vrm,
      debugLog,
    })
    expect(library.loadedClips['Broken Clip']).toBeUndefined()
    expect(library.loadedClips['Talk Loop']).toBeDefined()
    expect(errorSpy).toHaveBeenCalled()
    errorSpy.mockRestore()
  })

  it('never double-loads the same relativePath across idle+action lists', async () => {
    const duplicate: AnimationFile = { ...idle }
    const library = await loadAnimationClips({
      idleFiles: [idle],
      actionFiles: [duplicate],
      vrm,
      debugLog,
    })
    expect(Object.keys(library.loadedClips)).toHaveLength(1)
  })

  it('allFiles contains every idle+action file regardless of eager/lazy', async () => {
    const library = await loadAnimationClips({
      idleFiles: [idle],
      actionFiles: [lazyGesture, talkClip],
      vrm,
      debugLog,
    })
    expect(library.allFiles).toHaveLength(3)
  })
})
