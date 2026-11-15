// Pure decision-logic coverage for avatarBridgeWiring.ts: the clip
// inventory transform (describeVrmClips) and the idle/action automatic
// pool fallback (deriveAutomaticPoolNames). buildVrmAnimationsBridge/
// buildLipSyncBridge/buildExpressionsBridge are thin wiring, covered by a
// light call-through check each (do they forward to the right
// collaborator with the right arguments) rather than re-testing three.js.
import { describe, expect, it, vi } from 'vitest'
import * as THREE from 'three'
import {
  buildExpressionsBridge,
  buildLipSyncBridge,
  buildVrmAnimationsBridge,
  deriveAutomaticPoolNames,
  describeVrmClips,
} from './avatarBridgeWiring'
import type { AnimationFile, ClipMeta } from './clipLoader'

const animFile = (overrides: Partial<AnimationFile> = {}): AnimationFile => ({
  name: 'Idle A',
  filename: 'idle_a.vrma',
  url: '/static/animations/idle/idle_a.vrma',
  relativePath: 'idle/idle_a.vrma',
  ...overrides,
})

describe('describeVrmClips', () => {
  it('includes loaded clips and lazy-not-yet-loaded clips, excludes neither loaded nor lazy', () => {
    const files = [
      animFile({ name: 'Loaded', lazy: false }),
      animFile({ name: 'LazyUnloaded', lazy: true }),
      animFile({ name: 'NeverLoadedNonLazy', lazy: false }),
    ]
    const loadedClips = {
      Loaded: new THREE.AnimationClip('Loaded', 1, []),
    }
    const result = describeVrmClips(files, loadedClips, [])
    const names = result.map((c) => c.name)
    expect(names).toContain('Loaded')
    expect(names).toContain('LazyUnloaded')
    expect(names).not.toContain('NeverLoadedNonLazy')
  })

  it('reports duration 0 and loaded:false for a lazy clip that has not been played yet', () => {
    const files = [animFile({ name: 'LibraryClip', lazy: true })]
    const result = describeVrmClips(files, {}, [])
    expect(result[0].loaded).toBe(false)
    expect(result[0].duration).toBe(0)
  })

  it('reports the real duration and loaded:true once a lazy clip has been loaded', () => {
    const files = [animFile({ name: 'LibraryClip', lazy: true })]
    const loadedClips = { LibraryClip: new THREE.AnimationClip('LibraryClip', 3.5, []) }
    const result = describeVrmClips(files, loadedClips, [])
    expect(result[0].loaded).toBe(true)
    expect(result[0].duration).toBe(3.5)
  })

  it('marks looping true only when the clip is in the idle pool', () => {
    const files = [animFile({ name: 'A' }), animFile({ name: 'B' })]
    const loadedClips = {
      A: new THREE.AnimationClip('A', 1, []),
      B: new THREE.AnimationClip('B', 1, []),
    }
    const result = describeVrmClips(files, loadedClips, ['A'])
    expect(result.find((c) => c.name === 'A')?.looping).toBe(true)
    expect(result.find((c) => c.name === 'B')?.looping).toBe(false)
  })

  it('defaults role to "gesture" and normalizes optional flags to booleans/empty arrays', () => {
    const files = [animFile({ name: 'A', role: undefined, tags: undefined, directory: undefined })]
    const loadedClips = { A: new THREE.AnimationClip('A', 1, []) }
    const result = describeVrmClips(files, loadedClips, [])
    expect(result[0].role).toBe('gesture')
    expect(result[0].tags).toEqual([])
    expect(result[0].directory).toBeNull()
    expect(result[0].disabled).toBe(false)
    expect(result[0].additive).toBe(false)
    expect(result[0].isFbx).toBe(false)
  })
})

describe('deriveAutomaticPoolNames', () => {
  const clip = (name: string) => new THREE.AnimationClip(name, 1, [])

  it('uses the manifest idle/action sets when they resolve to loaded clips', () => {
    const idleFiles = [animFile({ name: 'IdleA' })]
    const actionFiles = [animFile({ name: 'ActionA' })]
    const loadedClips = { IdleA: clip('IdleA'), ActionA: clip('ActionA') }
    const result = deriveAutomaticPoolNames(idleFiles, actionFiles, loadedClips, {})
    expect(result).toEqual({ idleNames: ['IdleA'], actionNames: ['ActionA'] })
  })

  it('excludes a disabled clip from the idle pool', () => {
    const idleFiles = [animFile({ name: 'IdleA' }), animFile({ name: 'IdleDisabled' })]
    const loadedClips = { IdleA: clip('IdleA'), IdleDisabled: clip('IdleDisabled') }
    const clipMeta: Record<string, ClipMeta> = { IdleDisabled: { disabled: true } }
    const result = deriveAutomaticPoolNames(idleFiles, [], loadedClips, clipMeta)
    expect(result.idleNames).toEqual(['IdleA'])
  })

  it('falls back to every loaded clip for idle when the manifest idle set resolves to nothing', () => {
    const idleFiles = [animFile({ name: 'NeverLoaded' })]
    const loadedClips = { SomeOtherClip: clip('SomeOtherClip') }
    const result = deriveAutomaticPoolNames(idleFiles, [], loadedClips, {})
    expect(result.idleNames).toEqual(['SomeOtherClip'])
  })

  it('falls back to vrma-named clips for action when the manifest action set resolves to nothing', () => {
    const loadedClips = {
      'Something.vrma': clip('Something.vrma'),
      OtherClip: clip('OtherClip'),
    }
    const result = deriveAutomaticPoolNames([], [], loadedClips, {})
    expect(result.actionNames).toEqual(['Something.vrma'])
  })
})

describe('buildVrmAnimationsBridge', () => {
  it('play() forwards to playByName with the given options', () => {
    const playByName = vi.fn()
    const bridge = buildVrmAnimationsBridge({
      loadedClips: {},
      mixer: new THREE.AnimationMixer(new THREE.Object3D()),
      allFiles: [],
      idleNames: [],
      playByName,
      getCurrentClipName: () => 'CurrentClip',
    })
    bridge.play('Greeting', { loop: true })
    expect(playByName).toHaveBeenCalledWith('Greeting', { loop: true })
  })

  it('current() forwards to getCurrentClipName', () => {
    const bridge = buildVrmAnimationsBridge({
      loadedClips: {},
      mixer: new THREE.AnimationMixer(new THREE.Object3D()),
      allFiles: [],
      idleNames: [],
      playByName: vi.fn(),
      getCurrentClipName: () => 'CurrentClip',
    })
    expect(bridge.current()).toBe('CurrentClip')
  })
})

describe('buildLipSyncBridge', () => {
  it('play() forwards base64Audio and defaults mimeType to audio/wav', () => {
    const startLipSync = vi.fn()
    const bridge = buildLipSyncBridge({
      startLipSync,
      fadeOutAndStopLipSync: vi.fn(),
      beginStream: vi.fn(),
      enqueueLipSyncChunk: vi.fn().mockResolvedValue(undefined),
      endLipSyncStream: vi.fn(),
      isLipSyncAudible: () => false,
    })
    bridge.play({ base64Audio: 'abc' })
    expect(startLipSync).toHaveBeenCalledWith('abc', 'audio/wav')
  })

  it('isSpeaking() forwards to isLipSyncAudible, not a `streaming` flag', () => {
    const bridge = buildLipSyncBridge({
      startLipSync: vi.fn(),
      fadeOutAndStopLipSync: vi.fn(),
      beginStream: vi.fn(),
      enqueueLipSyncChunk: vi.fn().mockResolvedValue(undefined),
      endLipSyncStream: vi.fn(),
      isLipSyncAudible: () => true,
    })
    expect(bridge.isSpeaking()).toBe(true)
  })
})

describe('buildExpressionsBridge', () => {
  it('list() reads the live VRM expression map, not a hardcoded roster', () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {})
    const vrm = {
      expressionManager: { expressionMap: { custom_smile: {}, custom_sad: {} } },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any
    const bridge = buildExpressionsBridge({ setExpression: vi.fn(), vrm })
    bridge.list()
    expect(logSpy).toHaveBeenCalledWith('Available VRM expressions:', ['custom_smile', 'custom_sad'])
    logSpy.mockRestore()
  })

  it('list() reports an empty array when the VRM has no expressionManager', () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {})
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const vrm = { expressionManager: null } as any
    const bridge = buildExpressionsBridge({ setExpression: vi.fn(), vrm })
    bridge.list()
    expect(logSpy).toHaveBeenCalledWith('Available VRM expressions:', [])
    logSpy.mockRestore()
  })
})
