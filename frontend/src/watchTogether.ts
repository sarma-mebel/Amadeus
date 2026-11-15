/**
 * "Watch together" mode: the client samples the screen on a slow timer
 * while the mode is on and hands one frame at a time to the caller, which
 * posts it to /api/watch_together so Kurisu can keep quiet notes and
 * occasionally comment aloud on whatever's on screen (a video, a stream, a
 * long-running build — anything read-together's page-turn heuristic isn't
 * shaped for).
 *
 * Framework-free by design (no React), same pattern as readTogether.ts: the
 * posting decision is pure and DOM-free so it's unit-testable without a
 * MediaStream — see watchTogether.test.ts. The capture session itself is
 * REUSED from readTogether.ts's `startFrameCaptureSession` rather than
 * re-implemented — same Tauri-per-call / single-getDisplayMedia-for-the-
 * session split, same privacy shape (nothing kept beyond the current
 * in-memory frame the caller chooses to hold onto).
 *
 * Unlike read-together (event-driven: a page turn fires a capture), watch-
 * together samples on a fixed slow cadence (~60s, jittered) and only some
 * of those samples are worth posting — a static screen (paused video, idle
 * desktop) adds no information and would just burn an LLM call every
 * minute for nothing. `shouldPostFrame` is the pure "is this probe
 * different enough from the last POSTED probe" decision; the first frame
 * always posts (nothing to compare against yet, and posting establishes the
 * server-side session).
 */
import { diagnostics } from './diagnostics'
import { frameDiff, startFrameCaptureSession } from './readTogether'

// ---------------------------------------------------------------------------
// Pure posting decision (DOM-free, unit tested directly)
// ---------------------------------------------------------------------------

/** Diff below this fraction against the last POSTED probe means "nothing
 * changed enough to be worth a frame" — a static screen (paused video, idle
 * desktop, unattended terminal) never re-posts on every sample tick. This
 * is deliberately much smaller than PAGE_TURN_DIFF_THRESHOLD (0.1): a page
 * turn is a real content change, so it wants an explicit "clearly moved"
 * threshold, but watch-together's whole point is to catch modest ongoing
 * change (video motion, scrolling text, a filling progress bar) that would
 * never fire a page-turn-sized jump — a much lower bar is what makes the
 * "always sample, only sometimes post" shape work. */
export const WATCH_TOGETHER_DIFF_THRESHOLD = 0.03

/**
 * Pure decision: should this newly-sampled probe frame be posted?
 *
 * `lastPosted` is the probe frame from the last frame that WAS posted (null
 * before the first post — always posts then, both because there's nothing
 * to diff against and because the first post is what creates the
 * server-side watch-together session). Returns the probe to remember as the
 * new `lastPosted` alongside the post decision, since the caller has no
 * other clean place to thread that state through without re-deriving it.
 */
export const shouldPostFrame = (
  probe: Uint8ClampedArray,
  lastPosted: Uint8ClampedArray | null,
  diffThreshold: number = WATCH_TOGETHER_DIFF_THRESHOLD,
): boolean => {
  if (!lastPosted) return true
  return frameDiff(lastPosted, probe) >= diffThreshold
}

// ---------------------------------------------------------------------------
// Capture plumbing (Tauri / browser) — thin, not unit tested directly
// ---------------------------------------------------------------------------

const SAMPLE_BASE_MS = 60_000
const SAMPLE_JITTER_MS = 15_000
const FRAME_MAX_PX = 1280
const FRAME_JPEG_QUALITY = 0.8

export interface WatchTogetherHandle {
  stop: () => void
}

interface StartWatchTogetherOpts {
  onFrame: (image: { mime: string; data: string }) => void
  /** Fired when the source of frames goes away on its own — e.g. the user
   * ends screen-sharing from the browser's own "Stop sharing" chrome rather
   * than through our toggle. Mirrors readTogether's onEnded. */
  onEnded: () => void
}

/**
 * Starts watch-together sampling: every ~60s (±15s jitter, so a fleet of
 * sessions doesn't all sample in lockstep) grabs a cheap probe frame,
 * decides via `shouldPostFrame` whether it differs enough from the last
 * POSTED probe to be worth sending, and if so captures+hands off one
 * full-res frame via `onFrame`. Builds on readTogether's
 * `startFrameCaptureSession` for the platform capture details (Tauri
 * invoke per call / single getDisplayMedia session for browsers).
 */
export const startWatchTogether = async (
  opts: StartWatchTogetherOpts,
): Promise<WatchTogetherHandle> => {
  let stopped = false
  let sampleTimer = 0
  let lastPosted: Uint8ClampedArray | null = null

  const session = await startFrameCaptureSession(() => {
    if (stopped) return
    stopped = true
    window.clearTimeout(sampleTimer)
    opts.onEnded()
  })

  const schedule = () => {
    const jitter = (Math.random() * 2 - 1) * SAMPLE_JITTER_MS
    sampleTimer = window.setTimeout(() => void tick(), Math.max(0, SAMPLE_BASE_MS + jitter))
  }

  const tick = async () => {
    if (stopped) return
    try {
      const probe = await session.probeOnce()
      if (probe && !stopped && shouldPostFrame(probe, lastPosted)) {
        const frame = await session.captureFrame(FRAME_MAX_PX, FRAME_JPEG_QUALITY)
        if (frame && !stopped) {
          lastPosted = probe
          opts.onFrame(frame)
        }
      }
    } catch (error) {
      // A single failed sample (e.g. a transient capture_screenshot error)
      // shouldn't kill the whole mode — just skip this tick and retry on
      // the next one, same best-effort spirit as read-together/screen
      // companion. diagnostics.increment warns once, not every ~60s.
      console.warn('[watchTogether] sample failed:', error)
      diagnostics.increment('watch_together.sample_failed', String(error))
    } finally {
      if (!stopped) schedule()
    }
  }

  schedule()

  return {
    stop: () => {
      if (stopped) return
      stopped = true
      window.clearTimeout(sampleTimer)
      session.stop()
    },
  }
}
