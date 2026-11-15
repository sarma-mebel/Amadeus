// @vitest-environment jsdom
import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest'
import {
  BLOOM_RADIUS,
  BLOOM_STRENGTH,
  BLOOM_THRESHOLD,
  loadPostFxEnabledPref,
  savePostFxEnabledPref,
  VIGNETTE_DARKEN,
} from './postfx'

describe('postfx localStorage preference', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })
  afterEach(() => {
    window.localStorage.clear()
  })

  it('defaults to enabled when unset', () => {
    expect(loadPostFxEnabledPref()).toBe(true)
  })

  it('reads disabled only for the explicit "0" sentinel', () => {
    window.localStorage.setItem('kurisu_postfx', '0')
    expect(loadPostFxEnabledPref()).toBe(false)
  })

  it('treats any other stored value as enabled', () => {
    window.localStorage.setItem('kurisu_postfx', '1')
    expect(loadPostFxEnabledPref()).toBe(true)
    window.localStorage.setItem('kurisu_postfx', 'garbage')
    expect(loadPostFxEnabledPref()).toBe(true)
  })

  it('savePostFxEnabledPref round-trips through loadPostFxEnabledPref', () => {
    savePostFxEnabledPref(false)
    expect(loadPostFxEnabledPref()).toBe(false)
    savePostFxEnabledPref(true)
    expect(loadPostFxEnabledPref()).toBe(true)
  })

  it('degrades to the default when localStorage throws', () => {
    const spy = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('blocked')
    })
    expect(loadPostFxEnabledPref()).toBe(true)
    spy.mockRestore()
  })
})

describe('postfx tuning constants stay subtle', () => {
  // Guards against an accidental "make it dramatic" edit — the task spec
  // is explicit that this should catch hair/eye highlights, not glow the
  // whole scene, and the vignette should read as a gentle frame.
  it('bloom strength/radius/threshold stay in the subtle range', () => {
    expect(BLOOM_STRENGTH).toBeGreaterThan(0)
    expect(BLOOM_STRENGTH).toBeLessThanOrEqual(0.3)
    expect(BLOOM_RADIUS).toBeLessThanOrEqual(0.5)
    expect(BLOOM_THRESHOLD).toBeGreaterThanOrEqual(0.8)
  })

  it('vignette darkening stays gentle', () => {
    expect(VIGNETTE_DARKEN).toBeGreaterThan(0)
    expect(VIGNETTE_DARKEN).toBeLessThanOrEqual(0.35)
  })
})

describe('createDisabledPostFx-shaped fallback contract', () => {
  // createPostFx itself needs a WebGLRenderer (jsdom has no WebGL context),
  // so construction success/failure isn't exercised here — but the
  // "disabled" shape it falls back to must be safe to call unconditionally,
  // which is a pure contract we CAN pin without a real renderer.
  it('a disabled-shaped postfx object never throws when driven like the render loop drives it', async () => {
    // Import lazily so a WebGL-dependent import failure (none expected,
    // but defensive) doesn't blow up the whole suite file.
    const { createPostFx } = await import('./postfx')
    // Passing an object that merely LOOKS like a WebGLRenderer/Scene/Camera
    // (no real GL context) is expected to throw inside EffectComposer
    // construction — createPostFx must catch that and hand back the inert
    // shape rather than propagating.
    const fakeRenderer = {
      getSize: () => ({ x: 1, y: 1 }),
      getContext: () => null,
      getPixelRatio: () => 1,
      capabilities: {},
      render: () => {},
    } as unknown as import('three').WebGLRenderer
    const fakeScene = {} as unknown as import('three').Scene
    const fakeCamera = {} as unknown as import('three').PerspectiveCamera

    const postfx = createPostFx(fakeRenderer, fakeScene, fakeCamera)
    expect(() => postfx.render(0.016)).not.toThrow()
    expect(() => postfx.setSize(800, 600, 1)).not.toThrow()
    expect(() => postfx.setEnabled(true)).not.toThrow()
    expect(() => postfx.dispose()).not.toThrow()
    // Whatever happened during construction, `enabled` must be a boolean
    // the caller can safely branch render-path selection on.
    expect(typeof postfx.enabled).toBe('boolean')
  })
})

describe('createPostFxForViewer pet bypass + preference gating', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })
  afterEach(() => {
    window.localStorage.clear()
  })

  const fakeArgs = () => {
    const fakeRenderer = {
      getSize: () => ({ x: 1, y: 1 }),
      getContext: () => null,
      getPixelRatio: () => 1,
      capabilities: {},
      render: () => {},
    } as unknown as import('three').WebGLRenderer
    const fakeScene = {} as unknown as import('three').Scene
    const fakeCamera = {} as unknown as import('three').PerspectiveCamera
    return [fakeRenderer, fakeScene, fakeCamera] as const
  }

  it('the pet variant is always permanently disabled, regardless of the preference', async () => {
    const { createPostFxForViewer } = await import('./postfx')
    window.localStorage.setItem('kurisu_postfx', '1')
    const postfx = createPostFxForViewer(...fakeArgs(), 'pet')
    expect(postfx.enabled).toBe(false)
    // The disabled shape's setEnabled must not resurrect it.
    postfx.setEnabled(true)
    expect(postfx.enabled).toBe(false)
  })

  it('the full variant is disabled when the localStorage preference is off', async () => {
    const { createPostFxForViewer } = await import('./postfx')
    window.localStorage.setItem('kurisu_postfx', '0')
    const postfx = createPostFxForViewer(...fakeArgs(), 'full')
    expect(postfx.enabled).toBe(false)
  })

  it('an explicit override skips localStorage entirely (test seam)', async () => {
    const { createPostFxForViewer } = await import('./postfx')
    window.localStorage.setItem('kurisu_postfx', '1')
    const postfx = createPostFxForViewer(...fakeArgs(), 'full', false)
    expect(postfx.enabled).toBe(false)
  })
})
