/**
 * Consolidated voice-state visualization for the chat header — replaces the
 * lone "speaking dot" with a single indicator that always shows which of
 * five states the voice pipeline is in, so the user knows whether Kurisu
 * can hear them right now:
 *
 *   idle          — no capture, no reply in flight
 *   listening     — mic capturing (push-to-talk held, or hands-free open) —
 *                   bars animate live off the shared mic-level bus
 *   transcribing  — audio sent to STT, waiting on the transcript
 *   thinking      — a reply is loading but no tokens/audio yet
 *   speaking      — TTS audio is playing
 *
 * Bar heights during 'listening' are painted directly onto refs from a
 * requestAnimationFrame loop (no React state) so mic-level updates never
 * trigger a re-render — see src/audio/levelBus.ts.
 */
import { useEffect, useRef } from 'react'
import { readMicLevel } from '../audio/levelBus'
import './VoiceStatusIndicator.css'

export type VoiceState = 'idle' | 'listening' | 'transcribing' | 'thinking' | 'speaking'

const LABELS: Record<VoiceState, string> = {
  idle: 'Voice idle',
  listening: 'Listening',
  transcribing: 'Transcribing',
  thinking: 'Kurisu is thinking',
  speaking: 'Kurisu is speaking',
}

// Per-bar multipliers applied to the current mic level, tallest in the
// middle so the resting silhouette already reads as a little waveform.
const BAR_MULTIPLIERS = [0.6, 1.0, 1.4, 1.0, 0.6]
const BAR_COUNT = BAR_MULTIPLIERS.length

export interface VoiceStatusIndicatorProps {
  state: VoiceState
}

const VoiceStatusIndicator = ({ state }: VoiceStatusIndicatorProps) => {
  const barRefs = useRef<Array<HTMLElement | null>>([])
  const rafRef = useRef<number | null>(null)

  useEffect(() => {
    if (state !== 'listening') return undefined

    const tick = () => {
      const level = readMicLevel()
      const scale = 0.15 + level * 6
      for (let i = 0; i < BAR_COUNT; i += 1) {
        const bar = barRefs.current[i]
        if (!bar) continue
        const value = Math.min(1, scale * BAR_MULTIPLIERS[i])
        bar.style.transform = `scaleY(${value})`
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)

    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
  }, [state])

  // Leaving 'listening' should drop the bars back to their CSS resting
  // height rather than freezing on the last painted scale.
  useEffect(() => {
    if (state === 'listening') return
    for (const bar of barRefs.current) {
      if (bar) bar.style.transform = ''
    }
  }, [state])

  return (
    <span
      className={`voice-status voice-status-${state}`}
      role="status"
      aria-label={LABELS[state]}
      title={LABELS[state]}
    >
      {Array.from({ length: BAR_COUNT }, (_, i) => (
        <i
          key={i}
          ref={(el) => {
            barRefs.current[i] = el
          }}
          aria-hidden="true"
        />
      ))}
    </span>
  )
}

export default VoiceStatusIndicator
