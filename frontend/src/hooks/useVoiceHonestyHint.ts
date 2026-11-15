/**
 * Voice-honesty chip for the pet window: one transient caption line
 * explaining WHY a reply or presence line arrived without audio, instead of
 * letting a silent pet read as "the audio is broken".
 *
 * Extracted from usePetChat (file-budget split): the mechanism is fully
 * self-contained — probe bookkeeping, hint state, its auto-clear — and the
 * caller only needs the entry points returned here.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { t } from '../i18n'
import { fetchBackendStatus } from '../statusApi'

// Voice-warming re-probe window: a silent line is worth asking /api/status
// about again after this much quiet since the last probe, even with no
// voiced line in between.
const VOICE_WARMING_REPROBE_MS = 5 * 60_000

export const useVoiceHonestyHint = () => {
  // On a GPU install the qwen3 clone takes ~2min to warm on a background
  // thread, so the very first presence line (almost always the open-window
  // greeting) can legitimately have no audio yet. The /api/status probe
  // used to be a strict one-shot for the window's whole lifetime — which
  // meant a synthesis failure or client-side drop AFTER warmup had been
  // ruled out got no explanation, ever, for the rest of the session. Now
  // re-armable: the probe may run again on a later silent line when either
  // enough time has passed (VOICE_WARMING_REPROBE_MS) that warming could
  // plausibly still be the cause, or a voiced line has actually played
  // since the last probe — a NEW silence after real audio is a fresh,
  // distinct event worth explaining again, not a continuation of the same
  // one-shot mute. Still at most one hint per silent event, never nagging.
  const lastProbeRef = useRef(0)
  // Last time real audio actually played (a successful presence line, or a
  // turn's first audio frame) — read by the re-arm check, written by
  // markAudioPlayed whenever playback genuinely succeeds.
  const lastVoicedAtRef = useRef(0)
  const markAudioPlayed = useCallback(() => {
    lastVoicedAtRef.current = Date.now()
  }, [])
  const [voiceWarmingHint, setVoiceWarmingHint] = useState<string | null>(null)
  // Same transient-hint lifetime as useMic's micHint (3.5s) — long enough
  // to read, short enough to not linger as clutter.
  useEffect(() => {
    if (!voiceWarmingHint) return
    const timer = window.setTimeout(() => setVoiceWarmingHint(null), 3500)
    return () => window.clearTimeout(timer)
  }, [voiceWarmingHint])

  // A presence line arrived audio-less while audio is on: maybe ask the
  // backend why. Re-arm gate: never probed yet, OR the re-probe window
  // elapsed, OR a voiced line played since the last probe (a fresh
  // regression, not a continuation of the same already-explained mute).
  const probeSilentLine = useCallback(() => {
    const lastProbe = lastProbeRef.current
    const rearmed =
      lastProbe === 0 ||
      Date.now() - lastProbe >= VOICE_WARMING_REPROBE_MS ||
      lastVoicedAtRef.current > lastProbe
    if (!rearmed) return
    lastProbeRef.current = Date.now()
    void (async () => {
      const status = await fetchBackendStatus()
      if (status?.tts.warming) {
        setVoiceWarmingHint(t('her voice is still waking up — text only for a moment'))
      }
      // status null (unreachable/older backend) or warming false: stay
      // silent. Either the mute is real, or this particular line just
      // legitimately had none — neither deserves a chip.
    })()
  }, [])

  // The done frame reported sentence(s) that failed synthesis server-side:
  // the reply text landed but some of it went out silent — say so.
  const noteTtsFailure = useCallback(() => {
    setVoiceWarmingHint(t('her voice glitched on part of that reply'))
  }, [])

  return { voiceWarmingHint, markAudioPlayed, probeSilentLine, noteTtsFailure }
}
