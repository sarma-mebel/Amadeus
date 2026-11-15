/**
 * Mic/voice-input state extracted from ChatInterface: STT availability
 * probing, permission tracking, the transient mic hint line, Kurisu's
 * speaking flag (used to disable the mic while she talks), and the
 * push-to-talk recording lifecycle (useMicRecorder, which owns the
 * capture + /api/stt round trip).
 */
import { useEffect, useRef, useState } from 'react'
import type { RefObject } from 'react'
import { apiFetch } from '../api'
import { useMicRecorder } from '../components/useMicRecorder'
import { kurisuBridge } from '../vrm/kurisuBridge'

export interface UseMicOptions {
  /** True while a reply is streaming — opening the mic then barges in. */
  isStreaming: boolean
  /** Called with the trimmed transcript of a successful recording. */
  onTranscript: (text: string) => void
  /** Barge-in: stop the in-flight stream/audio before the mic opens. */
  stopCurrentStream: () => void
  /** Hold-Space PTT is suppressed while this textarea has focus. */
  textareaRef: RefObject<HTMLTextAreaElement | null>
}

export const useMic = ({ isStreaming, onTranscript, stopCurrentStream, textareaRef }: UseMicOptions) => {
  // Push-to-talk voice input: STT availability + hint UI live here; the
  // recording lifecycle itself is useMicRecorder (wired below).
  const [micAvailable, setMicAvailable] = useState<boolean | null>(null) // null = probing
  const [micUnavailableReason, setMicUnavailableReason] = useState<string | null>(null)
  const [micPermissionDenied, setMicPermissionDenied] = useState(false)
  const [kurisuSpeaking, setKurisuSpeaking] = useState(false)
  const [micHint, setMicHint] = useState<string | null>(null)
  const micHintTimerRef = useRef<number | null>(null)
  const showMicHint = (hint: string) => {
    if (micHintTimerRef.current !== null) window.clearTimeout(micHintTimerRef.current)
    setMicHint(hint)
    micHintTimerRef.current = window.setTimeout(() => setMicHint(null), 3500)
  }

  // Voice input availability: probe once on mount so the mic button can
  // show "Speech-to-text not installed" instead of failing silently on
  // first press.
  useEffect(() => {
    const controller = new AbortController()
    ;(async () => {
      try {
        const response = await apiFetch('/api/stt/status', {
          signal: controller.signal,
        })
        if (!response.ok) {
          setMicAvailable(false)
          setMicUnavailableReason('Speech-to-text is not available on this server.')
          return
        }
        const data = await response.json()
        setMicAvailable(Boolean(data.available))
        if (!data.available) {
          setMicUnavailableReason(
            typeof data.reason === 'string' ? data.reason : 'Speech-to-text not installed',
          )
        }
      } catch {
        if (!controller.signal.aborted) {
          setMicAvailable(false)
          setMicUnavailableReason('Speech-to-text is not available on this server.')
        }
      }
    })()
    return () => controller.abort()
  }, [])

  // Kurisu's speaking state (dispatched by VRMViewer's lip-sync graph) is
  // used to disable the mic while she's talking, avoiding audio bleed into
  // the recording.
  useEffect(() => {
    const handleSpeakingChange = (event: Event) => {
      const detail = (event as CustomEvent<{ speaking: boolean }>).detail
      setKurisuSpeaking(Boolean(detail?.speaking))
    }
    window.addEventListener('kurisu:speaking', handleSpeakingChange)
    return () => window.removeEventListener('kurisu:speaking', handleSpeakingChange)
  }, [])

  const canUseMic = micAvailable === true && !micPermissionDenied

  const {
    micStatus,
    startRecording,
    stopRecordingAndTranscribe,
    cancelRecording,
  } = useMicRecorder({
    canUseMic,
    onTranscript,
    shouldBargeIn: () => kurisuSpeaking || isStreaming,
    onBargeIn: () => {
      stopCurrentStream()
      if (kurisuSpeaking) kurisuBridge.expressions?.set('angry', 0.35, 1600)
    },
    onPermissionDenied: () => setMicPermissionDenied(true),
    onSttUnavailable: () => setMicAvailable(false),
    onTapTooShort: () =>
      showMicHint('Hold the mic (or Space) while you speak — release to send'),
    textareaRef,
  })

  return {
    micAvailable,
    micUnavailableReason,
    micPermissionDenied,
    kurisuSpeaking,
    micHint,
    showMicHint,
    micStatus,
    startRecording,
    stopRecordingAndTranscribe,
    cancelRecording,
  }
}
