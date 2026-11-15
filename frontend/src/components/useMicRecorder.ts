import { useEffect, useRef, useState } from 'react'
import type { RefObject } from 'react'
import { publishMicLevel, resetMicLevel } from '../audio/levelBus'
import { MicRecorder } from '../audio/micRecorder'
import { apiFetch } from '../api'

export type MicStatus = 'idle' | 'recording' | 'transcribing'

export interface UseMicRecorderOptions {
  /** STT available + permission not denied — gates recording start. */
  canUseMic: boolean
  /** Called with the trimmed transcript of a successful recording. */
  onTranscript: (text: string) => void
  /** True when Kurisu is speaking/streaming — recording then barges in. */
  shouldBargeIn: () => boolean
  /** Cut Kurisu off before the mic opens (keeps her tail out of the clip). */
  onBargeIn: () => void
  /** getUserMedia was refused — owner flips its micPermissionDenied state. */
  onPermissionDenied: () => void
  /** Server answered 501 — STT not installed on this backend. */
  onSttUnavailable: () => void
  /** A sub-350ms press is a tap, not push-to-talk — owner shows the hint. */
  onTapTooShort: () => void
  /** Hold-Space PTT is suppressed while this textarea has focus. */
  textareaRef: RefObject<HTMLTextAreaElement | null>
}

/** Push-to-talk recording lifecycle: mic capture, hold-Space binding, and the
 * /api/stt round trip. UI concerns (availability probe, hint text, button
 * markup) stay with the owner. */
export const useMicRecorder = ({
  canUseMic,
  onTranscript,
  shouldBargeIn,
  onBargeIn,
  onPermissionDenied,
  onSttUnavailable,
  onTapTooShort,
  textareaRef,
}: UseMicRecorderOptions) => {
  const [micStatus, setMicStatus] = useState<MicStatus>('idle')
  const micRecorderRef = useRef<MicRecorder | null>(null)
  const micActiveRef = useRef(false)
  const spaceKeyDownRef = useRef(false)
  const micPressStartRef = useRef(0)

  const startRecording = async () => {
    if (micActiveRef.current) return
    if (!canUseMic || micStatus !== 'idle') return
    // Voice barge-in: opening the mic while she's talking cuts her off —
    // the natural conversational move — instead of the mic being locked out
    // until she finishes. Stopping first also keeps the tail of her speech
    // out of the recording (her stop fades in ~140ms; getUserMedia setup
    // takes at least that long).
    if (shouldBargeIn()) {
      onBargeIn()
    }
    micActiveRef.current = true
    try {
      const recorder = new MicRecorder()
      recorder.onLevel = (rms) => publishMicLevel(rms)
      await recorder.start()
      // Released while getUserMedia was still pending (a quick tap):
      // stopRecordingAndTranscribe already ran, found no recorder, and
      // reset to idle — if we kept going, the mic would come up recording
      // with micActiveRef false and BOTH guards would dead-end: stuck
      // red button no press can clear. Discard the capture and teach the
      // hold gesture instead.
      if (!micActiveRef.current) {
        recorder.cancel()
        resetMicLevel()
        onTapTooShort()
        return
      }
      micRecorderRef.current = recorder
      micPressStartRef.current = performance.now()
      setMicStatus('recording')
    } catch (error) {
      micActiveRef.current = false
      micRecorderRef.current = null
      const domError = error as DOMException
      if (domError?.name === 'NotAllowedError' || domError?.name === 'SecurityError') {
        onPermissionDenied()
      }
      console.warn('Failed to start mic recording:', error)
    }
  }

  const stopRecordingAndTranscribe = async () => {
    if (!micActiveRef.current) return
    micActiveRef.current = false
    const recorder = micRecorderRef.current
    micRecorderRef.current = null
    if (!recorder) {
      setMicStatus('idle')
      return
    }
    // A press this short is a tap, not push-to-talk — someone clicking the
    // button expecting it to toggle. There's no usable audio in it anyway;
    // teach the gesture instead of silently discarding the recording.
    if (performance.now() - micPressStartRef.current < 350) {
      recorder.cancel()
      setMicStatus('idle')
      resetMicLevel()
      onTapTooShort()
      return
    }
    setMicStatus('transcribing')
    try {
      const wavBlob = await recorder.stop()
      if (!wavBlob) {
        setMicStatus('idle')
        return
      }
      const formData = new FormData()
      formData.append('audio', wavBlob, 'clip.wav')
      const response = await apiFetch('/api/stt', {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) {
        if (response.status === 501) {
          onSttUnavailable()
        }
        setMicStatus('idle')
        return
      }
      const data = await response.json()
      if (typeof data.text === 'string' && data.text.trim()) {
        onTranscript(data.text.trim())
      }
    } catch (error) {
      console.warn('Transcription failed:', error)
    } finally {
      setMicStatus('idle')
      resetMicLevel()
    }
  }

  const cancelRecording = () => {
    if (!micActiveRef.current) return
    micActiveRef.current = false
    micRecorderRef.current?.cancel()
    micRecorderRef.current = null
    setMicStatus('idle')
    resetMicLevel()
  }

  // Hold-Space push-to-talk, active only while no text input has focus
  // (otherwise Space would hijack normal typing) — not just the main
  // composer textarea: the message-edit textarea (regenerate/edit/rewind)
  // is a second focusable text field on the same page and needs the same
  // exclusion, so this checks for any focused textarea/text input rather
  // than one specific ref.
  useEffect(() => {
    const isTextEntryFocused = () => {
      const active = document.activeElement
      if (!active) return false
      if (active === textareaRef.current) return true
      const tag = active.tagName
      if (tag === 'TEXTAREA') return true
      if (tag === 'INPUT') {
        const type = (active as HTMLInputElement).type
        return type === 'text' || type === 'search' || type === 'password' || type === 'email' || type === 'url'
      }
      return (active as HTMLElement).isContentEditable === true
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.code !== 'Space' || event.repeat) return
      if (isTextEntryFocused()) return
      if (spaceKeyDownRef.current) return
      spaceKeyDownRef.current = true
      event.preventDefault()
      void startRecording()
    }

    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.code !== 'Space') return
      if (!spaceKeyDownRef.current) return
      spaceKeyDownRef.current = false
      event.preventDefault()
      void stopRecordingAndTranscribe()
    }

    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canUseMic, micStatus])

  // Release the mic if the component unmounts mid-recording.
  useEffect(() => {
    return () => {
      micRecorderRef.current?.cancel()
    }
  }, [])

  return { micStatus, startRecording, stopRecordingAndTranscribe, cancelRecording }
}
