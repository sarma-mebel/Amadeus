/**
 * Subtle post-processing stack: EffectComposer(RenderPass -> UnrealBloomPass
 * -> vignette ShaderPass -> OutputPass). Deliberately restrained — this is a
 * "nicer highlights + gentle frame" pass, not a stylized bloom look. DoF/
 * BokehPass was considered and skipped (perf cost vs. payoff not worth it
 * for a mostly-static portrait shot); future work if ever revisited.
 *
 * OutputPass MUST be last in an r180 composer: without it the composer's
 * final buffer skips the renderer's tone-mapping/color-space conversion
 * (ACESFilmicToneMapping, set in sceneSetup.ts) and everything looks flat/
 * washed relative to the non-composer render path.
 *
 * Not created at all for the pet variant — see VRMViewer.tsx: the pet
 * window is a transparent overlay compositing onto the desktop, and both
 * bloom and vignette read as visible halos/dark corners against alpha
 * there, on top of the perf cost mattering more for an always-on-top
 * overlay that sits on screen all day.
 */

import * as THREE from 'three'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'
import { diagnostics } from '../diagnostics'

/** localStorage `kurisu_postfx`: '0' disables, anything else (including
 * unset) is ON. Read once at scene init — mirrors kurisu_wind/kurisu_talk_body's
 * "read at construction, not reactive" contract; SettingsPanel's toggle
 * applies on the next avatar reload (see its hint text). */
const POSTFX_STORAGE_KEY = 'kurisu_postfx'

export const loadPostFxEnabledPref = (): boolean => {
  if (typeof window === 'undefined') return true
  try {
    return window.localStorage.getItem(POSTFX_STORAGE_KEY) !== '0'
  } catch {
    return true
  }
}

export const savePostFxEnabledPref = (enabled: boolean): void => {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(POSTFX_STORAGE_KEY, enabled ? '1' : '0')
  } catch {
    // Non-fatal — losing the preference isn't worth surfacing an error.
  }
}

/** Bloom tuning: subtle enough to catch hair/eye specular highlights and
 * emissive bits without a scene-wide glow. Values found by inspection
 * against the flagship model — strength/radius/threshold are the three
 * UnrealBloomPass knobs that matter, tuned toward "barely there". */
export const BLOOM_STRENGTH = 0.2
export const BLOOM_RADIUS = 0.4
export const BLOOM_THRESHOLD = 0.85

/** Vignette darkening at the corners, 0-1. Gentle — a mood frame, not a
 * spotlight. Pure function of the fragment's distance from center so it's
 * resolution-independent (no uniform needed besides the darkening amount). */
export const VIGNETTE_DARKEN = 0.25

const VignetteShader = {
  uniforms: {
    tDiffuse: { value: null as THREE.Texture | null },
    darkness: { value: VIGNETTE_DARKEN },
  },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse;
    uniform float darkness;
    varying vec2 vUv;
    void main() {
      vec4 texel = texture2D(tDiffuse, vUv);
      // Distance from center in [0, ~0.71] (corner) — smoothstep gives a
      // soft falloff instead of a hard ring.
      vec2 centered = vUv - vec2(0.5);
      float dist = length(centered);
      float vignette = smoothstep(0.35, 0.85, dist);
      texel.rgb *= 1.0 - vignette * darkness;
      gl_FragColor = texel;
    }
  `,
}

export interface PostFx {
  /** True only once construction succeeded AND the caller hasn't disabled
   * it (either via setEnabled(false) or a permanent construction-failure
   * fallback). Informational for callers — render() branches internally. */
  readonly enabled: boolean
  /** Renders the frame: through the composer's pass chain when enabled,
   * plain renderer.render(scene, camera) otherwise. The animate loop makes
   * this one unconditional call — the fallback branch living here means a
   * caller can't forget it. */
  render: (deltaTime: number) => void
  setSize: (width: number, height: number, pixelRatio: number) => void
  /** Live toggle — disposes/recreates internal render targets are NOT
   * needed for a simple on/off (the composer just stops being invoked);
   * this only flips the `enabled` flag the caller branches on. */
  setEnabled: (value: boolean) => void
  dispose: () => void
}

/** A composer-shaped fallback: `enabled` is permanently false, render()
 * draws the plain renderer path, setSize()/setEnabled() are inert. Returned
 * when construction fails (exotic GPU/context loss) or for the pet variant,
 * so callers never need a null check or their own render branch. */
const createDisabledPostFx = (
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  camera: THREE.PerspectiveCamera
): PostFx => ({
  enabled: false,
  render: () => renderer.render(scene, camera),
  setSize: () => {},
  setEnabled: () => {},
  dispose: () => {},
})

/** Build the post-processing composer for the main (non-pet) window. Never
 * throws — a construction failure (exotic GPU/context loss) counts
 * `vrm.postfx_failed` and returns a permanently-disabled PostFx instead, so
 * the caller's fallback to plain renderer.render() is unconditional and
 * doesn't need its own try/catch. */
export const createPostFx = (
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  camera: THREE.PerspectiveCamera
): PostFx => {
  try {
    const composer = new EffectComposer(renderer)

    const renderPass = new RenderPass(scene, camera)
    composer.addPass(renderPass)

    const size = renderer.getSize(new THREE.Vector2())
    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(Math.max(1, size.x), Math.max(1, size.y)),
      BLOOM_STRENGTH,
      BLOOM_RADIUS,
      BLOOM_THRESHOLD
    )
    composer.addPass(bloomPass)

    const vignettePass = new ShaderPass(VignetteShader)
    composer.addPass(vignettePass)

    // Last pass: tone mapping + color space conversion (see module doc).
    const outputPass = new OutputPass()
    composer.addPass(outputPass)

    let enabled = true
    let disposed = false
    // Construction succeeding doesn't prove the composer can render (targets
    // are built lazily; a lost/limited GL context surfaces on first use), so
    // runtime failures latch the same permanent fallback as construction ones.
    let failed = false

    const latchFailure = (error: unknown) => {
      failed = true
      diagnostics.increment(
        'vrm.postfx_failed',
        error instanceof Error ? error.message : String(error)
      )
    }

    return {
      get enabled() {
        return enabled && !disposed && !failed
      },
      render: (deltaTime: number) => {
        if (disposed) return
        if (!enabled || failed) {
          renderer.render(scene, camera)
          return
        }
        try {
          composer.render(deltaTime)
        } catch (error) {
          latchFailure(error)
          // Cover this frame so the failure isn't a black flash; the latch
          // routes every later frame down the plain path above.
          try {
            renderer.render(scene, camera)
          } catch {
            // Renderer itself is broken — nothing more to do this frame.
          }
        }
      },
      setSize: (width: number, height: number, pixelRatio: number) => {
        if (disposed || failed) return
        try {
          composer.setPixelRatio(pixelRatio)
          composer.setSize(width, height)
        } catch (error) {
          latchFailure(error)
        }
      },
      setEnabled: (value: boolean) => {
        if (disposed) return
        enabled = value
      },
      dispose: () => {
        if (disposed) return
        disposed = true
        composer.dispose()
      },
    }
  } catch (error) {
    diagnostics.increment(
      'vrm.postfx_failed',
      error instanceof Error ? error.message : String(error)
    )
    return createDisabledPostFx(renderer, scene, camera)
  }
}

/** VRMViewer's one call site: bundles the pet-bypass (transparent overlay,
 * see module doc) and the localStorage preference check so the component
 * doesn't need its own branching — it always gets back a safe-to-call
 * PostFx, permanently disabled for the pet variant or when the user has
 * turned the setting off. `initialEnabledOverride` is a test seam (skips
 * localStorage entirely when provided). */
export const createPostFxForViewer = (
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  camera: THREE.PerspectiveCamera,
  variant: 'full' | 'pet',
  initialEnabledOverride?: boolean
): PostFx => {
  if (variant === 'pet') return createDisabledPostFx(renderer, scene, camera)
  const wantsPostFx = initialEnabledOverride ?? loadPostFxEnabledPref()
  if (!wantsPostFx) return createDisabledPostFx(renderer, scene, camera)
  return createPostFx(renderer, scene, camera)
}
