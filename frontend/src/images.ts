/**
 * Vision capture helpers: turn a screen capture, pasted image, or dropped
 * file into the normalized base64 attachment the chat payload carries.
 *
 * Privacy invariants: capture happens only on an explicit user action, the
 * image lives in React state + one request body, and it is downscaled/
 * re-encoded client-side (canvas) so nothing larger than needed ever leaves
 * the machine. Never logged, never persisted.
 */

import { isTauri } from './platform'

export interface PendingImage {
  data: string // raw base64, no data: prefix
  mime: string
  width: number
  height: number
  previewUrl: string // data: URL for the thumbnail chip
}

export const DEFAULT_MAX_PX = 1568
export const DEFAULT_JPEG_QUALITY = 0.85

const dataUrlToPending = (dataUrl: string, width: number, height: number): PendingImage => {
  const [head, data] = dataUrl.split(',', 2)
  const mime = head.replace('data:', '').replace(';base64', '')
  return { data, mime, width, height, previewUrl: dataUrl }
}

/** Downscale (longest edge -> maxPx) and re-encode as JPEG via canvas. */
export const downscaleImage = async (
  source: CanvasImageSource & { width?: number; height?: number },
  sourceWidth: number,
  sourceHeight: number,
  maxPx: number = DEFAULT_MAX_PX,
  quality: number = DEFAULT_JPEG_QUALITY,
): Promise<PendingImage> => {
  const scale = Math.min(1, maxPx / Math.max(sourceWidth, sourceHeight))
  const width = Math.max(1, Math.round(sourceWidth * scale))
  const height = Math.max(1, Math.round(sourceHeight * scale))
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('canvas 2d context unavailable')
  ctx.drawImage(source, 0, 0, width, height)
  return dataUrlToPending(canvas.toDataURL('image/jpeg', quality), width, height)
}

const imageFromDataUrl = (url: string): Promise<HTMLImageElement> =>
  new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('image decode failed'))
    image.src = url
  })

/** A pasted/dropped image file -> normalized attachment. */
export const fileToPendingImage = async (
  file: File,
  maxPx: number = DEFAULT_MAX_PX,
  quality: number = DEFAULT_JPEG_QUALITY,
): Promise<PendingImage | null> => {
  if (!file.type.startsWith('image/')) return null
  const url = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(new Error('file read failed'))
    reader.readAsDataURL(file)
  })
  const image = await imageFromDataUrl(url)
  return downscaleImage(image, image.naturalWidth, image.naturalHeight, maxPx, quality)
}

/** One-shot screen capture.
 *
 * Tauri: a native Rust command captures the monitor under the cursor (the
 * webview's getDisplayMedia picker is unreliable there). Browser: a
 * getDisplayMedia stream is opened, one frame is grabbed, and every track
 * is stopped immediately — no ongoing recording.
 */
export const captureScreen = async (
  maxPx: number = DEFAULT_MAX_PX,
  quality: number = DEFAULT_JPEG_QUALITY,
): Promise<PendingImage | null> => {
  if (isTauri()) {
    const { invoke } = await import('@tauri-apps/api/core')
    const shot = (await invoke('capture_screenshot')) as {
      data: string
      width: number
      height: number
    }
    const image = await imageFromDataUrl(`data:image/png;base64,${shot.data}`)
    return downscaleImage(image, image.naturalWidth, image.naturalHeight, maxPx, quality)
  }

  const stream = await navigator.mediaDevices.getDisplayMedia({ video: true })
  try {
    const track = stream.getVideoTracks()[0]
    if (!track) return null
    const video = document.createElement('video')
    video.srcObject = stream
    video.muted = true
    await video.play()
    // One decoded frame is enough; some browsers need a tick after play().
    await new Promise((resolve) => requestAnimationFrame(resolve))
    const width = video.videoWidth
    const height = video.videoHeight
    if (!width || !height) return null
    return await downscaleImage(video, width, height, maxPx, quality)
  } finally {
    for (const track of stream.getTracks()) track.stop()
  }
}
