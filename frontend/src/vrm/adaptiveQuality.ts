/**
 * Adaptive quality (dynamic resolution), moved verbatim out of VRMViewer's
 * render loop: hold frame rate by stepping rendered resolution down under
 * sustained load and back up when there's headroom — the same trick every
 * console/mobile game uses. This matters here specifically because the GPU
 * build shares one laptop GPU with Qwen3-TTS: synthesis bursts steal GPU
 * time exactly while she's speaking, and dropping to 0.8x resolution for
 * those seconds is invisible; a stuttering face is not. Thresholds: degrade
 * after ~1.5s net time spent above 26ms/frame (<38fps), recover after ~12s
 * of solid sub-18ms frames (upgrade bar must sit above a 60Hz vsync
 * interval or it would never accumulate), 3s cooldown between steps against
 * oscillation. Frame samples are wall-clock (not the clamped animation
 * delta) capped at 100ms so a tab-restore doesn't register as load.
 */

export const QUALITY_SCALES = [1, 0.8, 0.62, 0.5]

/** Tier index at/above which post-processing (bloom/vignette composer) is
 * disabled — the composer is a second full-scene render pass, so it's the
 * first thing dropped under sustained load, before resolution even steps
 * down once. Re-enabled the instant the tier recovers to 0. */
const POSTFX_DISABLE_TIER = 1

export interface AdaptiveQuality {
  /** Push the current tier's scale to the renderer (also called on resize,
   * since setSize resets nothing but the canvas backing store). */
  applyPixelRatio: () => void
  /** Feed one frame's wall-clock timestamp; may step the tier. */
  update: (frameNow: number) => void
  /** Forget the last frame timestamp (hidden time is not a slow frame). */
  resetFrameClock: () => void
  /** Idle entry: restore full resolution (see implementation comment). */
  notifyIdle: () => void
}

export const createAdaptiveQuality = ({
  setPixelRatioScale,
  debugLog,
  setPostFxEnabled,
}: {
  /** Applies a 0-1 resolution scale on top of the device pixel ratio cap —
   * VRMViewer passes renderer.setPixelRatio(min(devicePixelRatio, 2) * scale). */
  setPixelRatioScale: (scale: number) => void
  debugLog: (...args: unknown[]) => void
  /** Optional hook: called with false the moment the tier first drops to
   * POSTFX_DISABLE_TIER or below, and true the moment it recovers back to
   * tier 0. Never called redundantly (only on the tier crossing that
   * threshold, not on every step change) — VRMViewer wires this to
   * postfx.setEnabled so a GPU-contention burst sheds the composer's extra
   * render pass before resolution needs to drop at all. Optional so
   * existing callers/tests that don't care about postfx are unaffected. */
  setPostFxEnabled?: (enabled: boolean) => void
}): AdaptiveQuality => {
  const quality = { tier: 0, overBudgetS: 0, underBudgetS: 0, lastChangeAt: 0, lastFrameAt: 0 }
  const applyPixelRatio = () => {
    setPixelRatioScale(QUALITY_SCALES[quality.tier])
  }
  const notifyPostFxForTier = (previousTier: number) => {
    if (!setPostFxEnabled) return
    const wasAbove = previousTier >= POSTFX_DISABLE_TIER
    const isAbove = quality.tier >= POSTFX_DISABLE_TIER
    if (wasAbove === isAbove) return
    setPostFxEnabled(!isAbove)
  }
  const update = (frameNow: number) => {
    if (quality.lastFrameAt === 0) {
      quality.lastFrameAt = frameNow
      return
    }
    const frameMs = Math.min(frameNow - quality.lastFrameAt, 100)
    quality.lastFrameAt = frameNow
    if (frameMs > 26) {
      quality.overBudgetS += frameMs / 1000
      quality.underBudgetS = 0
    } else {
      // Decay rather than reset, so alternating good/bad frames (thermal
      // throttling, TTS bursts) still eventually trip the downgrade.
      quality.overBudgetS = Math.max(0, quality.overBudgetS - frameMs / 2000)
      if (frameMs < 18) {
        quality.underBudgetS += frameMs / 1000
      } else {
        quality.underBudgetS = 0
      }
    }
    const cooledDown = frameNow - quality.lastChangeAt > 3000
    if (quality.overBudgetS > 1.5 && quality.tier < QUALITY_SCALES.length - 1 && cooledDown) {
      const previousTier = quality.tier
      quality.tier += 1
      quality.overBudgetS = 0
      quality.underBudgetS = 0
      quality.lastChangeAt = frameNow
      applyPixelRatio()
      notifyPostFxForTier(previousTier)
      debugLog(`Adaptive quality: resolution scale down to ${QUALITY_SCALES[quality.tier]}x`)
    } else if (quality.underBudgetS > 12 && quality.tier > 0 && cooledDown) {
      const previousTier = quality.tier
      quality.tier -= 1
      quality.overBudgetS = 0
      quality.underBudgetS = 0
      quality.lastChangeAt = frameNow
      applyPixelRatio()
      notifyPostFxForTier(previousTier)
      debugLog(`Adaptive quality: resolution scale up to ${QUALITY_SCALES[quality.tier]}x`)
    }
  }
  return {
    applyPixelRatio,
    update,
    resetFrameClock: () => {
      quality.lastFrameAt = 0
    },
    // Idle entry restores full resolution. Degradation exists to survive
    // ACTIVE-load bursts (TTS stealing the GPU while she speaks); an idle
    // avatar renders a tiny throttled pipeline, so a degraded tier held
    // through idle just parks the pet at 0.5x — visibly grainy —
    // indefinitely, because idle frames never feed update() and so no
    // recovery path exists (live finding, 2026-08-03 shakedown).
    // Idempotent: called every throttled idle frame.
    notifyIdle: () => {
      if (quality.tier === 0) return
      const previousTier = quality.tier
      quality.tier = 0
      quality.overBudgetS = 0
      quality.underBudgetS = 0
      applyPixelRatio()
      notifyPostFxForTier(previousTier)
      debugLog('Adaptive quality: idle — restoring full resolution')
    },
  }
}
