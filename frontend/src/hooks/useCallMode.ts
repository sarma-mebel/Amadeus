/**
 * Call mode extracted from ChatInterface: one session-scoped flag that runs
 * the conversation like a phone call — a minimal HUD, no typing needed.
 *
 * Two engines, chosen at startCall:
 * - FULL DUPLEX (preferred when `duplexPreferred()` — the kurisu_duplex
 *   opt-in — and a `tryStartDuplex` hook is wired): continuous mic
 *   streaming with server-side endpointing/barge-in (useDuplexCall). The
 *   hands-free/auto-barge-in toggles are NOT touched — the duplex loop owns
 *   the mic. If duplex reports it can't run (server off, mic denied,
 *   socket died), `fallbackToClassic()` downgrades the SAME call in place.
 * - CLASSIC (default): hands-free + auto barge-in forced on, exactly the
 *   original behavior. The user's own toggle prefs are stashed on entry
 *   and restored on hang-up (never persisted by the call itself).
 *
 * Deliberately NOT included: the hands-free / auto barge-in toggles
 * themselves (useHandsFree owns those, including their localStorage
 * persistence), the duplex session loop (useDuplexCall), and the
 * stream-cancel path — this hook receives all of them as inputs and only
 * orchestrates the enter/exit/downgrade choreography.
 */
import { useRef, useState } from 'react'

export interface UseCallModeOptions {
  handsFree: boolean
  toggleHandsFree: () => void
  autoBargeIn: boolean
  toggleAutoBargeIn: () => void
  /** Barge-in/stop path — hang-up silences anything still in flight. */
  stopCurrentStream: () => void
  /** () => the kurisu_duplex opt-in. Read at startCall time so flipping the
   * Settings toggle applies to the very next call. Absent = classic only. */
  duplexPreferred?: () => boolean
  /** useDuplexCall's start/stop — optional so surfaces without a duplex
   * loop (tests, older wiring) keep the classic behavior untouched. */
  tryStartDuplex?: () => void
  stopDuplex?: () => void
}

export const useCallMode = ({
  handsFree,
  toggleHandsFree,
  autoBargeIn,
  toggleAutoBargeIn,
  stopCurrentStream,
  duplexPreferred,
  tryStartDuplex,
  stopDuplex,
}: UseCallModeOptions) => {
  const [callMode, setCallMode] = useState(false)
  const preCallPrefsRef = useRef<{ handsFree: boolean; autoBargeIn: boolean } | null>(null)
  // True only when the classic toggles were actually forced on (classic
  // start, or a duplex→classic downgrade) — a pure-duplex call never
  // touched them, so hang-up must not "restore" them into a flipped state.
  const classicEngagedRef = useRef(false)

  const engageClassic = () => {
    classicEngagedRef.current = true
    if (!handsFree) toggleHandsFree()
    if (!autoBargeIn) toggleAutoBargeIn()
  }

  const startCall = () => {
    if (callMode) return
    preCallPrefsRef.current = { handsFree, autoBargeIn }
    classicEngagedRef.current = false
    if (duplexPreferred?.() && tryStartDuplex) {
      // Duplex-first: the call is live immediately; capability/mic failures
      // arrive async through useDuplexCall's onFallback → fallbackToClassic.
      setCallMode(true)
      tryStartDuplex()
      return
    }
    engageClassic()
    setCallMode(true)
  }

  /** Duplex couldn't run (or died mid-call): downgrade this same call to
   * the classic hands-free loop instead of dropping to dead air. No-op if
   * the call already ended. */
  const fallbackToClassic = () => {
    if (!callMode || classicEngagedRef.current) return
    engageClassic()
  }

  const endCall = () => {
    if (!callMode) return
    setCallMode(false)
    stopDuplex?.()
    stopCurrentStream()
    const stash = preCallPrefsRef.current
    preCallPrefsRef.current = null
    const engaged = classicEngagedRef.current
    classicEngagedRef.current = false
    // Restore only what the call changed; the toggles re-persist their own
    // localStorage keys and tear down the recorder/monitor themselves. A
    // pure-duplex call changed nothing.
    if (engaged && stash && stash.handsFree !== true) toggleHandsFree()
    if (engaged && stash && stash.autoBargeIn !== true) toggleAutoBargeIn()
  }

  return { callMode, startCall, endCall, fallbackToClassic }
}
