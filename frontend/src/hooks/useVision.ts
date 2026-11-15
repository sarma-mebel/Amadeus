/**
 * Vision (screenshot attachment) state extracted from ChatInterface: the
 * GET /api/vision/status probe (gates the UI button + mirrors the server's
 * resize/quality limits), the at-most-one pending image chip, and the three
 * ways an image gets attached (capture button, paste, drop).
 *
 * turnImageRef is intentionally exposed as a plain ref, not wrapped: the
 * send path (sendMessage/sendMessageViaStream/sendMessageViaPost in
 * ChatInterface) writes it directly at the start of a turn (consuming
 * pendingImageRef.current and clearing the chip) and reads it again from
 * inside the WS->POST fallback, so ownership of "when does this turn's
 * image get consumed" has to stay with the send path, not this hook.
 */
import { useEffect, useRef, useState } from 'react'
import { apiFetch } from '../api'
import { diagnostics } from '../diagnostics'
import { captureScreen, fileToPendingImage, type PendingImage } from '../images'
import type { ChatImageAttachment } from '../chat/payload'

export const useVision = () => {
  // At most one pending image attachment for the next turn. The ref carries
  // it through the send (incl. the WS->POST fallback) after the chip state
  // is cleared; visionLimits mirrors /api/vision/status.
  const [pendingImage, setPendingImage] = useState<PendingImage | null>(null)
  const pendingImageRef = useRef<PendingImage | null>(null)
  useEffect(() => {
    pendingImageRef.current = pendingImage
  }, [pendingImage])
  const turnImageRef = useRef<ChatImageAttachment[] | undefined>(undefined)
  const [visionAvailable, setVisionAvailable] = useState(false)
  const visionLimitsRef = useRef<{ maxPx: number; quality: number }>({
    maxPx: 1568,
    quality: 0.85,
  })
  useEffect(() => {
    ;(async () => {
      try {
        const response = await apiFetch('/api/vision/status')
        if (!response.ok) return
        const data = await response.json()
        setVisionAvailable(Boolean(data.available))
        if (typeof data.max_image_px === 'number') {
          visionLimitsRef.current.maxPx = data.max_image_px
        }
        if (typeof data.jpeg_quality === 'number') {
          visionLimitsRef.current.quality = data.jpeg_quality / 100
        }
      } catch (error) {
        // Vision button just stays hidden. Counted (degrade-loudly): a
        // failed status probe otherwise looks identical to vision being
        // disabled in config.
        diagnostics.increment('vision.status_probe_failed', String(error))
      }
    })()
  }, [])

  const attachPendingImage = (image: PendingImage | null) => {
    if (image) setPendingImage(image)
  }

  const handleCaptureScreen = async () => {
    try {
      const image = await captureScreen(
        visionLimitsRef.current.maxPx,
        visionLimitsRef.current.quality,
      )
      attachPendingImage(image)
    } catch {
      // User cancelled the picker / capture failed — nothing to do.
    }
  }

  const handlePaste = async (event: React.ClipboardEvent) => {
    if (!visionAvailable) return
    const file = Array.from(event.clipboardData?.files ?? []).find((f) =>
      f.type.startsWith('image/'),
    )
    if (!file) return
    event.preventDefault()
    try {
      attachPendingImage(
        await fileToPendingImage(
          file,
          visionLimitsRef.current.maxPx,
          visionLimitsRef.current.quality,
        ),
      )
    } catch {
      // Undecodable clipboard image — ignore.
    }
  }

  const handleDrop = async (event: React.DragEvent) => {
    if (!visionAvailable) return
    const file = Array.from(event.dataTransfer?.files ?? []).find((f) =>
      f.type.startsWith('image/'),
    )
    if (!file) return
    event.preventDefault()
    try {
      attachPendingImage(
        await fileToPendingImage(
          file,
          visionLimitsRef.current.maxPx,
          visionLimitsRef.current.quality,
        ),
      )
    } catch {
      // Undecodable dropped file — ignore.
    }
  }

  return {
    pendingImage,
    setPendingImage,
    pendingImageRef,
    turnImageRef,
    visionAvailable,
    handleCaptureScreen,
    handlePaste,
    handleDrop,
  }
}
