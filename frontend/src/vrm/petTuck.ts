/**
 * Hide & peek ("tuck away", Desktop Mate lineage): a DND-popover action
 * slides the pet mostly off the nearest side edge, leaving a sliver of
 * hair/shoulder visible. While tucked she is quiet (usePetPresence's
 * requestLine treats tucked like DND — server pops are never burned),
 * roaming stands down, hovering the sliver makes her lean out a little,
 * and a click brings her back fully.
 *
 * Geometry is CACHED at tuck time (inside the localStorage value): once
 * tucked past the left edge the window's top-left corner is off every
 * monitor, so `pet_roam_context` errors by design ("No monitor found") —
 * peek/untuck must not depend on a live query that can't succeed.
 *
 * Motion uses direct `setPosition` steps (not `animate_pet_window_to`,
 * which clamps targets into the work area — correct for strolls, fatal
 * for deliberately-off-screen tucks) and marks itself as programmatic
 * motion so the drag-settle poke never fires on a tuck slide.
 */
import { isTauri } from '../platform'
import { markProgrammaticPetMotion } from './petRoam'

export const PET_TUCKED_KEY = 'kurisu_pet_tucked'
export const PET_TUCK_CHANGE_EVENT = 'kurisu:pet-tuck-change'

/** Physical px of window left visible while tucked / extra while hovered. */
export const TUCK_SLIVER_PX = 64
export const PEEK_EXTRA_PX = 96
const SLIDE_MS = 280
const SLIDE_STEPS = 14

export type TuckSide = 'left' | 'right'

export interface TuckState {
  side: TuckSide
  /** Geometry snapshot from tuck time, physical px. */
  work_x: number
  work_w: number
  win_w: number
  win_y: number
}

export const loadTuckState = (): TuckState | null => {
  try {
    const raw = window.localStorage.getItem(PET_TUCKED_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as TuckState
    if (parsed.side !== 'left' && parsed.side !== 'right') return null
    if (
      typeof parsed.work_x !== 'number' ||
      typeof parsed.work_w !== 'number' ||
      typeof parsed.win_w !== 'number' ||
      typeof parsed.win_y !== 'number'
    ) {
      return null
    }
    return parsed
  } catch {
    return null
  }
}

export const isPetTucked = (): boolean => loadTuckState() !== null

/** X for a given number of visible pixels on the tucked side. Exported for
 * tests — the slide/persistence around it needs a real window. */
export const tuckedX = (state: TuckState, visiblePx: number): number =>
  state.side === 'left'
    ? state.work_x - state.win_w + visiblePx
    : state.work_x + state.work_w - visiblePx

const easeInOut = (t: number): number =>
  t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2

const slideTo = async (x: number, y: number, durationMs = SLIDE_MS): Promise<void> => {
  const { getCurrentWindow, PhysicalPosition } = await import('@tauri-apps/api/window')
  const win = getCurrentWindow()
  const start = await win.outerPosition()
  markProgrammaticPetMotion(durationMs)
  const stepMs = Math.max(10, Math.round(durationMs / SLIDE_STEPS))
  for (let i = 1; i <= SLIDE_STEPS; i += 1) {
    const t = easeInOut(i / SLIDE_STEPS)
    const px = Math.round(start.x + (x - start.x) * t)
    const py = Math.round(start.y + (y - start.y) * t)
    await win.setPosition(new PhysicalPosition(px, py))
    if (i < SLIDE_STEPS) {
      await new Promise((resolve) => window.setTimeout(resolve, stepMs))
    }
  }
}

/** Tucks toward whichever side edge is nearer. Returns false when geometry
 * is unavailable (non-Tauri, window closed, roam-context error). */
export const tuckPetWindow = async (): Promise<boolean> => {
  if (!isTauri()) return false
  try {
    const { invoke } = await import('@tauri-apps/api/core')
    const ctx = (await invoke('pet_roam_context')) as {
      win_x: number
      win_y: number
      win_w: number
      work_x: number
      work_w: number
    }
    const center = ctx.win_x + ctx.win_w / 2
    const side: TuckSide =
      center - ctx.work_x < ctx.work_x + ctx.work_w - center ? 'left' : 'right'
    const state: TuckState = {
      side,
      work_x: ctx.work_x,
      work_w: ctx.work_w,
      win_w: ctx.win_w,
      win_y: ctx.win_y,
    }
    await slideTo(tuckedX(state, TUCK_SLIVER_PX), state.win_y)
    window.localStorage.setItem(PET_TUCKED_KEY, JSON.stringify(state))
    window.dispatchEvent(new CustomEvent(PET_TUCK_CHANGE_EVENT))
    return true
  } catch {
    return false
  }
}

/** Slides back fully onto the screen (flush against the tucked edge, which
 * is exactly where the perch snap wants a side perch) and clears the flag. */
export const untuckPetWindow = async (): Promise<void> => {
  const state = loadTuckState()
  if (!state) return
  try {
    window.localStorage.removeItem(PET_TUCKED_KEY)
    window.dispatchEvent(new CustomEvent(PET_TUCK_CHANGE_EVENT))
    await slideTo(tuckedX(state, state.win_w), state.win_y)
  } catch {
    // Window gone mid-slide — the cleared flag is still correct.
  }
}

/** Lean out a little while the sliver is hovered; settle back on leave. */
export const peekPetWindow = async (hovering: boolean): Promise<void> => {
  const state = loadTuckState()
  if (!state) return
  try {
    const visible = TUCK_SLIVER_PX + (hovering ? PEEK_EXTRA_PX : 0)
    await slideTo(tuckedX(state, visible), state.win_y, 180)
  } catch {
    // Best-effort — a failed peek just leaves her where she is.
  }
}
