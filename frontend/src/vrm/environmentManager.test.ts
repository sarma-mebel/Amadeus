// @vitest-environment jsdom
import { describe, expect, it, vi } from 'vitest'
import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'
import { ENVIRONMENTS, type EnvironmentDefinition } from '../environments'
import {
  disposeEnvironmentGroup,
  loadEnvironmentGroup,
  repositionForEnvironment,
  type EnvironmentGlbLoader,
} from './environmentManager'

const makeEnvironment = (overrides: Partial<EnvironmentDefinition> = {}): EnvironmentDefinition => ({
  id: 'lab',
  label: 'Future Gadget Lab',
  file: '/static/environment/test.glb',
  anchor: { position: [1, 2, 3], rotationY: 0.5 },
  ...overrides,
})

const makeVrm = (metaVersion: '0' | '1'): VRM =>
  ({
    scene: new THREE.Group(),
    meta: { metaVersion },
  }) as unknown as VRM

describe('repositionForEnvironment', () => {
  it('refits light position/target/frustum and the catcher around the shadow center', () => {
    const mainLight = new THREE.DirectionalLight()
    const shadowCatcher = new THREE.Mesh(new THREE.PlaneGeometry(6, 6), new THREE.ShadowMaterial())
    const environment = makeEnvironment({ shadowCenter: [10, 20, 30] })

    repositionForEnvironment({ mainLight, shadowCatcher, vrm: null }, environment)

    // Same offsets the scene rig applies at construction time.
    expect(mainLight.position.toArray()).toEqual([12, 23, 32])
    expect(mainLight.target.position.toArray()).toEqual([10, 20, 30])
    expect(mainLight.shadow.camera.left).toBe(10 - 3)
    expect(mainLight.shadow.camera.right).toBe(10 + 7)
    expect(mainLight.shadow.camera.top).toBe(20 + 6)
    expect(mainLight.shadow.camera.bottom).toBe(20 - 3)
    // Tiny lift above the anchor's y to avoid z-fighting with the set floor.
    expect(shadowCatcher.position.toArray()).toEqual([10, 20.001, 30])
  })

  it('falls back to the anchor position when no explicit shadowCenter is set', () => {
    const mainLight = new THREE.DirectionalLight()
    const environment = makeEnvironment() // anchor [1, 2, 3], no shadowCenter
    repositionForEnvironment({ mainLight, shadowCatcher: null, vrm: null }, environment)
    expect(mainLight.target.position.toArray()).toEqual([1, 2, 3])
  })

  it('repositions an already-loaded VRM at the new anchor', () => {
    const vrm = makeVrm('1')
    repositionForEnvironment({ mainLight: null, shadowCatcher: null, vrm }, makeEnvironment())
    expect(vrm.scene.position.toArray()).toEqual([1, 2, 3])
    expect(vrm.scene.rotation.y).toBeCloseTo(0.5)
  })

  it('adds the 180° VRM0 yaw flip when repositioning a VRM0 model', () => {
    const vrm = makeVrm('0')
    repositionForEnvironment({ mainLight: null, shadowCatcher: null, vrm }, makeEnvironment())
    expect(vrm.scene.rotation.y).toBeCloseTo(0.5 + Math.PI)
  })

  it('is a safe no-op with nothing to reposition', () => {
    expect(() =>
      repositionForEnvironment({ mainLight: null, shadowCatcher: null, vrm: null }, makeEnvironment())
    ).not.toThrow()
  })

  it('accepts the real registry entries', () => {
    const mainLight = new THREE.DirectionalLight()
    repositionForEnvironment({ mainLight, shadowCatcher: null, vrm: null }, ENVIRONMENTS.time_machine)
    expect(mainLight.target.position.toArray()).toEqual([-5, 0, 0])
  })
})

describe('disposeEnvironmentGroup', () => {
  it('removes the group from the scene and disposes geometries + materials', () => {
    const scene = new THREE.Scene()
    const group = new THREE.Group()
    const geometry = new THREE.BoxGeometry()
    const material = new THREE.MeshBasicMaterial()
    const mesh = new THREE.Mesh(geometry, material)
    group.add(mesh)
    scene.add(group)

    const geometryDispose = vi.spyOn(geometry, 'dispose')
    const materialDispose = vi.spyOn(material, 'dispose')

    disposeEnvironmentGroup(scene, group)

    expect(scene.children).not.toContain(group)
    expect(geometryDispose).toHaveBeenCalledTimes(1)
    expect(materialDispose).toHaveBeenCalledTimes(1)
  })

  it('disposes every entry of a multi-material mesh', () => {
    const scene = new THREE.Scene()
    const group = new THREE.Group()
    const materials = [new THREE.MeshBasicMaterial(), new THREE.MeshStandardMaterial()]
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(), materials)
    group.add(mesh)
    scene.add(group)

    const spies = materials.map((mat) => vi.spyOn(mat, 'dispose'))
    disposeEnvironmentGroup(scene, group)
    spies.forEach((spy) => expect(spy).toHaveBeenCalledTimes(1))
  })

  it('ignores non-mesh children (lights, plain groups)', () => {
    const scene = new THREE.Scene()
    const group = new THREE.Group()
    group.add(new THREE.Object3D())
    group.add(new THREE.PointLight())
    scene.add(group)
    expect(() => disposeEnvironmentGroup(scene, group)).not.toThrow()
    expect(scene.children).not.toContain(group)
  })
})

describe('loadEnvironmentGroup', () => {
  const makeFakeLoader = () => {
    const calls: { url: string; onLoad: (gltf: { scene: THREE.Group }) => void }[] = []
    const loader: EnvironmentGlbLoader = {
      load: (url, onLoad) => {
        calls.push({ url, onLoad })
      },
    }
    return { calls, loader }
  }

  it('adds the loaded group to the scene with receiveShadow on and reports it', () => {
    const scene = new THREE.Scene()
    const { calls, loader } = makeFakeLoader()
    const onLoaded = vi.fn()

    loadEnvironmentGroup({
      scene,
      environment: makeEnvironment(),
      debugLog: () => {},
      onLoaded,
      createLoader: () => loader,
    })

    expect(calls).toHaveLength(1)
    expect(calls[0].url).toContain('/static/environment/test.glb')

    const glbScene = new THREE.Group()
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial())
    glbScene.add(mesh)
    calls[0].onLoad({ scene: glbScene })

    expect(scene.children).toContain(glbScene)
    expect(mesh.receiveShadow).toBe(true)
    expect(onLoaded).toHaveBeenCalledWith(glbScene)
  })

  it('never touches the scene when cancelled before the load resolves', () => {
    const scene = new THREE.Scene()
    const { calls, loader } = makeFakeLoader()
    const onLoaded = vi.fn()

    const handle = loadEnvironmentGroup({
      scene,
      environment: makeEnvironment(),
      debugLog: () => {},
      onLoaded,
      createLoader: () => loader,
    })

    handle.cancel()
    calls[0].onLoad({ scene: new THREE.Group() })

    expect(scene.children).toHaveLength(0)
    expect(onLoaded).not.toHaveBeenCalled()
  })

  it('supports repeated swaps without leaking the previous set', () => {
    // The soak concern: swap A -> B -> A and make sure each outgoing
    // group is removed and disposed before the next lands.
    const scene = new THREE.Scene()
    const { calls, loader } = makeFakeLoader()
    let current: THREE.Group | null = null

    const swap = (environment: EnvironmentDefinition) => {
      if (current) {
        disposeEnvironmentGroup(scene, current)
        current = null
      }
      loadEnvironmentGroup({
        scene,
        environment,
        debugLog: () => {},
        onLoaded: (group) => {
          current = group
        },
        createLoader: () => loader,
      })
    }

    const resolveWith = (call: { onLoad: (gltf: { scene: THREE.Group }) => void }) => {
      const glbScene = new THREE.Group()
      const geometry = new THREE.BoxGeometry()
      const material = new THREE.MeshBasicMaterial()
      glbScene.add(new THREE.Mesh(geometry, material))
      call.onLoad({ scene: glbScene })
      return { glbScene, geometry, material }
    }

    swap(makeEnvironment({ id: 'time_machine' as EnvironmentDefinition['id'] }))
    const first = resolveWith(calls[0])
    const firstGeometryDispose = vi.spyOn(first.geometry, 'dispose')
    const firstMaterialDispose = vi.spyOn(first.material, 'dispose')

    swap(makeEnvironment({ id: 'lab' as EnvironmentDefinition['id'] }))
    expect(scene.children).not.toContain(first.glbScene)
    expect(firstGeometryDispose).toHaveBeenCalledTimes(1)
    expect(firstMaterialDispose).toHaveBeenCalledTimes(1)

    const second = resolveWith(calls[1])
    expect(scene.children).toContain(second.glbScene)
    // Exactly one environment group in the scene after every swap.
    expect(scene.children.filter((child) => child === first.glbScene || child === second.glbScene)).toHaveLength(1)
  })
})
