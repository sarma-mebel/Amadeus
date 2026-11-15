// @vitest-environment jsdom
//
// Pure decision-logic coverage for modelLoadFlow.ts: registry-miss
// resolution, progress-percent guarding against a missing Content-Length,
// the load-failure retry-once gate, and the pet-preset reframe's early-out
// guards (the imperative body reuses computeDefaultFraming, already
// covered by cameraFraming.test.ts). placeAndFrameLoadedVrm is exercised
// with real (cheap) THREE objects rather than mocks, following
// animationScheduler.test.ts's precedent of using a real THREE.Object3D/
// AnimationMixer instead of stubbing out the whole library. jsdom is
// needed only because placeAndFrameLoadedVrm writes window.kurisuDebugVrm
// (the headless-harness bridge) — same rationale as
// animationScheduler.test.ts's own jsdom pragma.
import { describe, expect, it, vi } from 'vitest'
import * as THREE from 'three'
import {
  placeAndFrameLoadedVrm,
  progressPercent,
  reframeForPetPresetChange,
  resolveRequestedModel,
  shouldRetryWithDefaultModel,
  type PlaceAndFrameLoadedVrmDeps,
} from './modelLoadFlow'
import { DEFAULT_MODEL, type ModelDefinition } from '../models'
import { ENVIRONMENTS } from '../environments'
import { createGazeRuntimeState } from './gazeDirector'

describe('resolveRequestedModel', () => {
  const registry: ModelDefinition[] = [
    { id: 'outfit_a', label: 'A', file: '/static/models/A.vrm' },
  ]
  const resolveModelFile = (id: string, reg: ModelDefinition[]): string =>
    reg.find((m) => m.id === id)?.file ?? DEFAULT_MODEL.file

  it('resolves a registered id without flagging a registry miss', () => {
    const result = resolveRequestedModel('outfit_a', registry, resolveModelFile)
    expect(result).toEqual({ file: '/static/models/A.vrm', isRegistryMiss: false })
  })

  it('flags an unregistered non-default id as a registry miss and falls back to the default file', () => {
    const result = resolveRequestedModel('stale_id', registry, resolveModelFile)
    expect(result.file).toBe(DEFAULT_MODEL.file)
    expect(result.isRegistryMiss).toBe(true)
  })

  it('never flags the default id as a registry miss, even if the registry is empty', () => {
    const result = resolveRequestedModel(DEFAULT_MODEL.id, [], resolveModelFile)
    expect(result.isRegistryMiss).toBe(false)
  })
})

describe('progressPercent', () => {
  it('computes a percentage when total is known', () => {
    expect(progressPercent(50, 200)).toBe(25)
  })

  it('returns 0 (not NaN/Infinity) when total is 0 — no Content-Length header', () => {
    expect(progressPercent(50, 0)).toBe(0)
  })
})

describe('shouldRetryWithDefaultModel', () => {
  it('retries once for a non-default model that has not been retried yet', () => {
    expect(shouldRetryWithDefaultModel('some_outfit', false)).toBe(true)
  })

  it('does not retry a second time', () => {
    expect(shouldRetryWithDefaultModel('some_outfit', true)).toBe(false)
  })

  it('never retries a failure on the default model itself', () => {
    expect(shouldRetryWithDefaultModel(DEFAULT_MODEL.id, false)).toBe(false)
  })
})

describe('reframeForPetPresetChange', () => {
  const cameraDirector = () => ({
    adoptDefaultFraming: vi.fn(),
    startReframeTween: vi.fn(),
  })

  it('is a no-op when there is no VRM loaded yet', () => {
    const director = cameraDirector()
    reframeForPetPresetChange({
      vrm: null,
      canvas: { clientWidth: 100, clientHeight: 100 } as HTMLCanvasElement,
      cameraFov: 45,
      cameraDirector: director,
    })
    expect(director.adoptDefaultFraming).not.toHaveBeenCalled()
  })

  it('is a no-op when there is no canvas', () => {
    const director = cameraDirector()
    reframeForPetPresetChange({
      vrm: {} as Parameters<typeof reframeForPetPresetChange>[0]['vrm'],
      canvas: null,
      cameraFov: 45,
      cameraDirector: director,
    })
    expect(director.adoptDefaultFraming).not.toHaveBeenCalled()
  })
})

/** Minimal duck-typed VRM matching computeDefaultFraming's needs (see
 * cameraFraming.test.ts's buildFakeVrm) plus the top-level fields
 * placeAndFrameLoadedVrm itself touches (scene, lookAt, expressionManager
 * left undefined -> "no lookAt bones" branch). */
const buildFakeVrm = () => {
  const root = new THREE.Object3D()
  const bones: Record<string, THREE.Object3D> = {
    head: new THREE.Object3D(),
    hips: new THREE.Object3D(),
    leftLowerLeg: new THREE.Object3D(),
    leftUpperArm: new THREE.Object3D(),
    rightUpperArm: new THREE.Object3D(),
  }
  bones.head.position.set(0, 1.6, 0)
  bones.hips.position.set(0, 0.9, 0)
  bones.leftLowerLeg.position.set(0.1, 0.5, 0)
  bones.leftUpperArm.position.set(-0.2, 1.4, 0)
  bones.rightUpperArm.position.set(0.2, 1.4, 0)
  Object.values(bones).forEach((bone) => root.add(bone))

  return {
    scene: root,
    humanoid: {
      getNormalizedBoneNode: (name: string) => bones[name] ?? null,
    },
    lookAt: null,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  } as any
}

describe('placeAndFrameLoadedVrm', () => {
  const buildDeps = (
    overrides: Partial<PlaceAndFrameLoadedVrmDeps> = {}
  ): PlaceAndFrameLoadedVrmDeps => {
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100)
    // compileAsync isn't implemented by the plain (non-WebGL) renderer
    // stub three.js exposes outside a browser; a resolved-promise stand-in
    // is enough since placeAndFrameLoadedVrm only fires it and ignores the
    // result (fire-and-forget, see the module doc comment).
    const renderer = {
      compileAsync: vi.fn().mockResolvedValue(undefined),
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any
    return {
      scene,
      camera,
      renderer,
      cameraDirector: { adoptDefaultFraming: vi.fn() },
      gazeStateRef: { current: createGazeRuntimeState() },
      getActiveEnvironment: () => ENVIRONMENTS.time_machine,
      getCanvas: () => ({ clientWidth: 100, clientHeight: 100 }) as HTMLCanvasElement,
      mainLightColor: new THREE.Color(0xffffff),
      fillLightColor: new THREE.Color(0xffffff),
      variant: 'full',
      fileToLoad: '/static/models/Test.vrm',
      debugLog: () => {},
      ...overrides,
    }
  }

  it('adds the VRM scene and a gaze proxy to the scene graph', () => {
    const vrm = buildFakeVrm()
    const deps = buildDeps()
    const before = deps.scene.children.length
    placeAndFrameLoadedVrm(vrm, deps)
    // vrm.scene (the fake root) + the gaze proxy Object3D.
    expect(deps.scene.children.length).toBe(before + 2)
  })

  it('returns the gaze proxy and sets it as gazeStateRef.current.proxy', () => {
    const vrm = buildFakeVrm()
    const deps = buildDeps()
    const proxy = placeAndFrameLoadedVrm(vrm, deps)
    expect(deps.gazeStateRef.current.proxy).toBe(proxy)
    expect(proxy).toBeInstanceOf(THREE.Object3D)
  })

  it('calls adoptDefaultFraming when the bones resolve a valid framing', () => {
    const vrm = buildFakeVrm()
    const deps = buildDeps()
    placeAndFrameLoadedVrm(vrm, deps)
    expect(deps.cameraDirector.adoptDefaultFraming).toHaveBeenCalledTimes(1)
  })

  it('fires the fire-and-forget shader precompile', () => {
    const vrm = buildFakeVrm()
    const deps = buildDeps()
    placeAndFrameLoadedVrm(vrm, deps)
    expect(deps.renderer.compileAsync).toHaveBeenCalledWith(deps.scene, deps.camera)
  })
})
