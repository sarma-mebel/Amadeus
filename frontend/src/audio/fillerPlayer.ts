/** Filler backchannel: short pre-synthesized "thinking" clips (んー…, そうね…)
 * played in the gap between sending a message and the first real audio frame,
 * so the wait reads as her thinking instead of the app hanging.
 *
 * Clips are generated offline in her own voice by scripts/generate_fillers.py
 * into static/fillers/ (manifest.json + wav files). No manifest (404) means
 * the feature silently stays off — it's an optional asset, not a dependency.
 *
 * Playback rides kurisuBridge.lipSync.play (the presence-line one-shot path),
 * so the mouth moves with the hum, and the real reply's beginStream() already
 * hard-stops any one-shot source — the filler can never talk over her.
 */
import { resolveAssetPath } from '../config'
import { diagnostics } from '../diagnostics'
import { kurisuBridge } from '../vrm/kurisuBridge'

export const FILLERS_STORAGE_KEY = 'kurisu_fillers'

// Opt-in (default OFF) since 2026-07-19: a synthesized "thinking sound"
// right after the user sends text read as unnatural in daily use. The
// Settings toggle still enables it for anyone who liked it.
export const loadFillersEnabled = (): boolean => {
  try {
    return window.localStorage.getItem(FILLERS_STORAGE_KEY) === 'true'
  } catch {
    // Storage blocked: fall back to the default (off) — returning true here
    // (a leftover from the default-ON era) force-enabled fillers with no
    // way to turn them off.
    return false
  }
}

export const saveFillersEnabled = (enabled: boolean): void => {
  try {
    window.localStorage.setItem(FILLERS_STORAGE_KEY, String(enabled))
  } catch {
    // Preference just doesn't persist.
  }
}

export interface FillerClip {
  file: string
  text: string
  /** Base64 WAV, fetched once and cached for the session. */
  base64?: string
}

const parseClipList = (data: unknown, key: string): FillerClip[] => {
  const list = (data as Record<string, unknown>)?.[key]
  if (!Array.isArray(list)) return []
  return list
    .filter(
      (entry): entry is { file: string; text: string } =>
        typeof (entry as { file?: unknown })?.file === 'string' &&
        typeof (entry as { text?: unknown })?.text === 'string',
    )
    .map((entry) => ({ file: entry.file, text: entry.text }))
}

/** Validate the manifest shape defensively — it's a user-generated asset. */
export const parseFillerManifest = (data: unknown): FillerClip[] =>
  parseClipList(data, 'fillers')

/** Soft listening acknowledgements ("うん。", "ええ。") from the same manifest;
 * older manifests without the key just yield an empty pool. */
export const parseBackchannelManifest = (data: unknown): FillerClip[] =>
  parseClipList(data, 'backchannels')

/** Random pick that never repeats the previous clip (when there are ≥2). */
export const pickFillerIndex = (
  count: number,
  lastIndex: number | null,
  random: () => number = Math.random,
): number | null => {
  if (count <= 0) return null
  if (count === 1) return 0
  let index = Math.floor(random() * count)
  if (index >= count) index = count - 1
  if (index === lastIndex) index = (index + 1) % count
  return index
}

const bytesToBase64 = (bytes: Uint8Array): string => {
  let binary = ''
  const chunk = 0x8000
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk))
  }
  return btoa(binary)
}

let manifestPromise: Promise<unknown> | null = null
let clipsPromise: Promise<FillerClip[]> | null = null
let backchannelsPromise: Promise<FillerClip[]> | null = null

/** The manifest fetch happens once per session, kicked off by the first
 * armed turn (or first backchannel) so app startup never pays for an
 * optional asset. */
const ensureManifestLoaded = (): Promise<unknown> => {
  if (!manifestPromise) {
    manifestPromise = (async () => {
      try {
        const response = await fetch(resolveAssetPath('/static/fillers/manifest.json'))
        if (!response.ok) {
          // Missing manifest is an EXPECTED, silent "feature off" state
          // (an optional asset the user's setup may never have generated)
          // — still worth one counted+warned event so it's distinguishable
          // from "generated but broken" in the diagnostics panel.
          diagnostics.increment('audio.fillers_manifest_missing', `HTTP ${response.status}`)
          return null
        }
        return await response.json()
      } catch (error) {
        diagnostics.increment('audio.fillers_manifest_missing', String(error))
        return null
      }
    })()
  }
  return manifestPromise
}

const ensureClipsLoaded = (): Promise<FillerClip[]> => {
  if (!clipsPromise) {
    clipsPromise = ensureManifestLoaded().then((manifest) => {
      const clips = parseFillerManifest(manifest)
      // Prefetch audio in the background; a clip whose fetch hasn't
      // finished by play time just awaits its own fetch then.
      clips.forEach((clip) => {
        void fetchClipAudio(clip)
      })
      return clips
    })
  }
  return clipsPromise
}

const ensureBackchannelsLoaded = (): Promise<FillerClip[]> => {
  if (!backchannelsPromise) {
    backchannelsPromise = ensureManifestLoaded().then((manifest) => {
      const clips = parseBackchannelManifest(manifest)
      clips.forEach((clip) => {
        void fetchClipAudio(clip)
      })
      return clips
    })
  }
  return backchannelsPromise
}

const fetchClipAudio = async (clip: FillerClip): Promise<string | null> => {
  if (clip.base64) return clip.base64
  try {
    const response = await fetch(resolveAssetPath(`/static/fillers/${clip.file}`))
    if (!response.ok) return null
    clip.base64 = bytesToBase64(new Uint8Array(await response.arrayBuffer()))
    return clip.base64
  } catch {
    return null
  }
}

export interface FillerHandle {
  /** Stop a not-yet-fired filler. A clip already playing is left to finish
   * (≤ ~1.5s) or be superseded by the real stream's beginStream(). */
  cancel: () => void
}

const NOOP_HANDLE: FillerHandle = { cancel: () => {} }

let lastFillerIndex: number | null = null

/** Arm one filler for the turn that just started. Fires once after
 * ``delayMs`` unless cancelled first (real audio arrived, turn ended, or the
 * turn failed) — a fast reply never gets stepped on. */
export const armFillerForTurn = (delayMs = 600): FillerHandle => {
  if (!loadFillersEnabled()) return NOOP_HANDLE

  let cancelled = false
  void ensureClipsLoaded()

  const fire = async () => {
    const clips = await ensureClipsLoaded()
    if (cancelled || clips.length === 0) return
    // A presence line (greeting, reminder) may still be playing — humming
    // over her own voice reads as a glitch, so skip this turn's filler.
    if (kurisuBridge.lipSync?.isSpeaking?.() === true) return
    const index = pickFillerIndex(clips.length, lastFillerIndex)
    if (index === null) return
    lastFillerIndex = index
    const base64 = await fetchClipAudio(clips[index])
    if (cancelled || !base64) return
    // Last-chance check, deliberately duplicated right against the play()
    // call rather than trusting the check above alone: fetchClipAudio's
    // await is the widest gap in this function, and cancel() can land in
    // the same microtask-queue window the fetch promise resolves in. This
    // second check costs nothing and closes the window all the way to the
    // actual sound-producing call, so a future refactor that inserts any
    // async step between the two checks can't reopen the race silently.
    if (cancelled) return
    try {
      await kurisuBridge.lipSync?.play?.({ base64Audio: base64, mimeType: 'audio/wav' })
    } catch (error) {
      // Best-effort, like every presence sound — but still counted, so a
      // silently-broken filler path doesn't read as "nothing to say".
      diagnostics.increment('voice.filler_play_failed', String(error))
    }
  }

  const timeoutId = window.setTimeout(() => {
    void fire()
  }, delayMs)

  return {
    cancel: () => {
      cancelled = true
      window.clearTimeout(timeoutId)
    },
  }
}

let lastBackchannelIndex: number | null = null

/** Play one soft listening acknowledgement in her voice, immediately.

 * Called by the listening reactor while the USER is mid-speech, so unlike
 * fillers there is no arming delay. Skips (returning false) when no clips
 * exist or she is already making a sound — an "un" over her own line reads
 * as a glitch. The caller owns rate-limiting; this stays mechanical. */
export const playBackchannel = async (): Promise<boolean> => {
  const clips = await ensureBackchannelsLoaded()
  if (clips.length === 0) return false
  if (kurisuBridge.lipSync?.isSpeaking?.() === true) return false
  const index = pickFillerIndex(clips.length, lastBackchannelIndex)
  if (index === null) return false
  lastBackchannelIndex = index
  const base64 = await fetchClipAudio(clips[index])
  if (!base64) return false
  try {
    await kurisuBridge.lipSync?.play?.({ base64Audio: base64, mimeType: 'audio/wav' })
    return true
  } catch (error) {
    diagnostics.increment('voice.filler_play_failed', String(error))
    return false
  }
}

/** Test seam: forget fetched clips and no-repeat state. */
export const resetFillersForTest = (): void => {
  manifestPromise = null
  clipsPromise = null
  backchannelsPromise = null
  lastFillerIndex = null
  lastBackchannelIndex = null
}
