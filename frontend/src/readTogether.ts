/**
 * "Read together" mode: the client watches the screen for page turns (a
 * book/manga/document the user is reading) and hands one screenshot per
 * settled page to the caller, which posts it to /api/read_along so Kurisu
 * can keep quiet notes and occasionally comment aloud.
 *
 * Framework-free by design (no React) so the page-turn state machine and
 * the pixel-diff math can be unit tested without a DOM or a MediaStream —
 * see readTogether.test.ts. The thin capture wrapper at the bottom of this
 * file is the only part that touches the platform (Tauri invoke / browser
 * getDisplayMedia + canvas) and is intentionally not unit tested here.
 *
 * Privacy shape mirrors captureScreen/screen-companion: nothing is kept
 * beyond the current in-memory frame the caller chooses to hold onto (see
 * usePetChat's lastPageFrameRef), and the browser path uses a SINGLE
 * getDisplayMedia() picker call for the whole session — the user explicitly
 * chooses which window/tab is shared once, rather than this module ever
 * being able to silently re-prompt for (or widen) access.
 *
 * The platform capture session (Tauri invoke / browser getDisplayMedia +
 * <video>) is factored out as `startFrameCaptureSession` below so
 * watchTogether.ts (periodic screen frames for "watch together" mode) can
 * reuse the exact same plumbing instead of re-implementing the
 * getDisplayMedia-once / Tauri-per-call split — see that file for how it's
 * composed into a different polling cadence and posting rule.
 */
import { diagnostics } from './diagnostics'
import { downscaleImage } from './images'
import { isTauri } from './platform'

// ---------------------------------------------------------------------------
// Pure page-turn detection (DOM-free, unit tested directly)
// ---------------------------------------------------------------------------

/** Mean absolute luma difference between two equal-length grayscale probe
 * frames, normalized to [0, 1] (0 = identical, 1 = maximally different —
 * e.g. all-black vs all-white). Callers downscale to a small probe (~64px)
 * before calling this, so the cost is trivial regardless of poll rate. */
export const frameDiff = (a: Uint8ClampedArray, b: Uint8ClampedArray): number => {
  const length = Math.min(a.length, b.length)
  if (length === 0) return 0
  let total = 0
  for (let i = 0; i < length; i++) {
    total += Math.abs(a[i] - b[i])
  }
  return total / length / 255
}

/** Diff above this fraction marks a CANDIDATE page turn (item moved enough
 * that it's probably not just cursor blink / minor rendering noise). */
export const PAGE_TURN_DIFF_THRESHOLD = 0.1
/** Minimum time between two FIRED page events — a settled page can't turn
 * again faster than a human actually flips pages, so this also acts as a
 * backstop against a flaky diff re-triggering on the same page. */
export const MIN_PAGE_GAP_MS = 5_000

type DetectorState = 'idle' | 'changed'

/**
 * Pure page-turn state machine: idle -> changed -> (fire | back to idle).
 *
 * Every ~2.5s the caller feeds one downscaled grayscale probe frame via
 * `observe(frame, now)`. The machine:
 *   - idle: diffs against the last-seen probe. Above threshold -> 'changed'
 *     (a candidate turn), but does NOT fire yet.
 *   - changed: diffs the NEW frame against the CANDIDATE frame (not the
 *     pre-turn frame). Below threshold -> the page has settled -> fire and
 *     return to idle. Still above threshold -> still mid-turn/scrolling;
 *     stay in 'changed' with the candidate replaced by this newer frame
 *     (tracks a slow scroll/animation instead of comparing against a stale
 *     mid-motion frame forever).
 *
 * This "require the NEXT probe to be stable" rule is what skips mid-scroll
 * and page-flip animation frames: a single-frame blip back toward the old
 * page (rare, but page-turn animations can pass through intermediate
 * states) never fires because settling is checked against the LATEST
 * candidate, not the original pre-turn frame.
 *
 * The min-gap is enforced here too (not just by the caller) so the state
 * machine's own fire() contract is self-consistent for tests: firing twice
 * within MIN_PAGE_GAP_MS never happens even if observe() is called on a
 * tighter cadence than the real capture loop uses.
 */
export class PageTurnDetector {
  private state: DetectorState = 'idle'
  private lastFrame: Uint8ClampedArray | null = null
  private lastFireAt = -Infinity
  private readonly diffThreshold: number
  private readonly minGapMs: number

  // Plain assignment (not TS parameter-property shorthand) — this repo
  // builds with erasableSyntaxOnly, which rejects the `constructor(private
  // readonly x: T)` shorthand.
  constructor(diffThreshold: number = PAGE_TURN_DIFF_THRESHOLD, minGapMs: number = MIN_PAGE_GAP_MS) {
    this.diffThreshold = diffThreshold
    this.minGapMs = minGapMs
  }

  /** Feed one probe frame. Returns true iff this call fires a page-turn
   * event (the caller should capture+send the full-res frame now). */
  observe(frame: Uint8ClampedArray, now: number): boolean {
    if (!this.lastFrame) {
      this.lastFrame = frame
      return false
    }
    const diff = frameDiff(this.lastFrame, frame)
    if (this.state === 'idle') {
      if (diff > this.diffThreshold) {
        this.state = 'changed'
      }
      this.lastFrame = frame
      return false
    }
    // state === 'changed': diff was measured against the previous
    // candidate frame, so "below threshold" means settled relative to the
    // last thing we saw — i.e. two consecutive stable probes.
    this.lastFrame = frame
    if (diff > this.diffThreshold) {
      // Still moving (scroll/animation) — keep tracking, don't fire.
      return false
    }
    this.state = 'idle'
    if (now - this.lastFireAt < this.minGapMs) {
      // Settled too soon after the last fire (shouldn't normally happen
      // given minGapMs is usually « the time a human spends per page, but
      // keeps the contract airtight for callers/tests driving observe()
      // on an arbitrary cadence).
      return false
    }
    this.lastFireAt = now
    return true
  }
}

// ---------------------------------------------------------------------------
// Capture plumbing (Tauri / browser) — thin, not unit tested directly
// ---------------------------------------------------------------------------

const PROBE_INTERVAL_MS = 2_500
const PROBE_PX = 64
const PAGE_MAX_PX = 1280
const PAGE_JPEG_QUALITY = 0.8

export interface ReadTogetherHandle {
  stop: () => void
}

interface StartReadTogetherOpts {
  onPage: (image: { mime: string; data: string }) => void
  /** Fired when the source of frames goes away on its own — e.g. the user
   * ends screen-sharing from the browser's own chrome (the shared-tab
   * "Stop sharing" bar) rather than through our toggle. Lets the caller
   * flip its UI state back off instead of silently capturing nothing. */
  onEnded: () => void
}

/** Draws `source` at PROBE_PX (longest edge) into a reusable canvas and
 * returns its grayscale luma as a Uint8ClampedArray — the cheap probe frame
 * frameDiff/PageTurnDetector (and watchTogether's diff-based skip) operate
 * on. Kept tiny (≤64px) so a poll is effectively free next to the LLM/TTS
 * cost these modes are guarding. */
const grabProbeFrame = (
  source: CanvasImageSource,
  sourceWidth: number,
  sourceHeight: number,
  canvas: HTMLCanvasElement,
): Uint8ClampedArray | null => {
  const scale = Math.min(1, PROBE_PX / Math.max(sourceWidth, sourceHeight))
  const width = Math.max(1, Math.round(sourceWidth * scale))
  const height = Math.max(1, Math.round(sourceHeight * scale))
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) return null
  ctx.drawImage(source, 0, 0, width, height)
  const { data } = ctx.getImageData(0, 0, width, height)
  const luma = new Uint8ClampedArray(width * height)
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    // Rec. 601 luma weights — good enough for a page-turn heuristic, and
    // cheap (no need for perceptual accuracy here).
    luma[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]
  }
  return luma
}

export interface FrameCaptureSession {
  /** Cheap downscaled grayscale probe frame (~64px), or null on a transient
   * capture failure the caller should just skip this tick for. */
  probeOnce: () => Promise<Uint8ClampedArray | null>
  /** Full-resolution frame, downscaled/re-encoded via downscaleImage at the
   * given max-edge/quality. */
  captureFrame: (
    maxPx: number,
    quality: number,
  ) => Promise<{ mime: string; data: string } | null>
  /** Releases the browser MediaStream (Tauri: no-op, nothing is held open). */
  stop: () => void
}

/**
 * Opens a platform frame-capture session shared by read-together (page
 * turns) and watch-together (periodic frames): resolves once the source is
 * ready (browser: after the user picks a window/tab in the getDisplayMedia
 * prompt; Tauri: immediately, since capture_screenshot needs no persistent
 * handle).
 *
 * Tauri path: each probe/frame capture independently invokes
 * `capture_screenshot` (the same silent, repeatable, already-ACL'd command
 * images.ts's captureScreen uses for the main-window screen companion) —
 * there is nothing to hold open between calls.
 *
 * Browser path: ONE getDisplayMedia() call up front. This is deliberate,
 * not a limitation worked around — captureScreen's one-shot picker-per-call
 * pattern is unusable for a polling loop, but more importantly, a single
 * upfront window/tab-scoped grant is the privacy-preferable shape: the user
 * picks exactly the shared window once, and that's the only surface this
 * mode can ever see for the rest of the session (never silently promoted to
 * full-screen access, never re-prompted). The stream is held for the mode's
 * lifetime and frames are pulled by drawing the live <video> element to
 * canvas — no separate captureScreen() call needed. If the user ends
 * sharing from the browser's own "Stop sharing" chrome (the track's 'ended'
 * event), `onEnded` fires so the caller can flip its toggle off instead of
 * polling a dead stream forever.
 */
export const startFrameCaptureSession = async (
  onEnded: () => void,
): Promise<FrameCaptureSession> => {
  const probeCanvas = document.createElement('canvas')
  let stopped = false

  // Browser-only state; unused (stays null) on Tauri.
  let stream: MediaStream | null = null
  let video: HTMLVideoElement | null = null

  const grabTauriShot = async (): Promise<HTMLImageElement> => {
    const { invoke } = await import('@tauri-apps/api/core')
    const shot = (await invoke('capture_screenshot')) as {
      data: string
      width: number
      height: number
    }
    const image = new Image()
    const loaded = new Promise<void>((resolve, reject) => {
      image.onload = () => resolve()
      image.onerror = () => reject(new Error('image decode failed'))
    })
    image.src = `data:image/png;base64,${shot.data}`
    await loaded
    return image
  }

  const captureFrame = async (
    maxPx: number,
    quality: number,
  ): Promise<{ mime: string; data: string } | null> => {
    if (isTauri()) {
      const image = await grabTauriShot()
      const frame = await downscaleImage(
        image,
        image.naturalWidth,
        image.naturalHeight,
        maxPx,
        quality,
      )
      return { mime: frame.mime, data: frame.data }
    }
    if (!video) return null
    const width = video.videoWidth
    const height = video.videoHeight
    if (!width || !height) return null
    const frame = await downscaleImage(video, width, height, maxPx, quality)
    return { mime: frame.mime, data: frame.data }
  }

  const probeOnce = async (): Promise<Uint8ClampedArray | null> => {
    if (isTauri()) {
      const image = await grabTauriShot()
      return grabProbeFrame(image, image.naturalWidth, image.naturalHeight, probeCanvas)
    }
    if (!video) return null
    const width = video.videoWidth
    const height = video.videoHeight
    if (!width || !height) return null
    return grabProbeFrame(video, width, height, probeCanvas)
  }

  if (!isTauri()) {
    stream = await navigator.mediaDevices.getDisplayMedia({ video: true })
    const track = stream.getVideoTracks()[0]
    if (track) {
      track.addEventListener('ended', () => {
        if (stopped) return
        stopped = true
        onEnded()
      })
    }
    video = document.createElement('video')
    video.srcObject = stream
    video.muted = true
    await video.play()
    // One decoded frame before the loop starts, matching captureScreen's
    // own "some browsers need a tick after play()" note.
    await new Promise((resolve) => requestAnimationFrame(resolve))
  }

  return {
    probeOnce,
    captureFrame,
    stop: () => {
      if (stopped) return
      stopped = true
      if (stream) {
        for (const track of stream.getTracks()) track.stop()
      }
      video = null
      stream = null
    },
  }
}

/**
 * Starts read-together capture: polls a cheap probe frame every 2.5s,
 * feeds it through PageTurnDetector, and captures+hands off one full-res
 * page image each time a page turn fires. See startFrameCaptureSession for
 * the platform capture details this builds on.
 */
export const startReadTogether = async (
  opts: StartReadTogetherOpts,
): Promise<ReadTogetherHandle> => {
  const detector = new PageTurnDetector()
  let stopped = false
  let pollTimer = 0

  const session = await startFrameCaptureSession(() => {
    if (stopped) return
    stopped = true
    window.clearTimeout(pollTimer)
    opts.onEnded()
  })

  const tick = async () => {
    if (stopped) return
    try {
      const probe = await session.probeOnce()
      if (probe && !stopped) {
        const fired = detector.observe(probe, Date.now())
        if (fired) {
          const page = await session.captureFrame(PAGE_MAX_PX, PAGE_JPEG_QUALITY)
          if (page && !stopped) opts.onPage(page)
        }
      }
    } catch (error) {
      // A single failed probe (e.g. a transient capture_screenshot error)
      // shouldn't kill the whole mode — just skip this tick and retry on
      // the next one, same best-effort spirit as the screen-companion peek.
      // This can fire once per ~2.5s poll tick, so it's only ever COUNTED
      // here — the warn-once rule in diagnostics.increment already keeps
      // the console quiet after the first occurrence.
      console.warn('[readTogether] probe failed:', error)
      diagnostics.increment('read_together.probe_failed', String(error))
    } finally {
      if (!stopped) pollTimer = window.setTimeout(() => void tick(), PROBE_INTERVAL_MS)
    }
  }

  pollTimer = window.setTimeout(() => void tick(), PROBE_INTERVAL_MS)

  return {
    stop: () => {
      if (stopped) return
      stopped = true
      window.clearTimeout(pollTimer)
      session.stop()
    },
  }
}
