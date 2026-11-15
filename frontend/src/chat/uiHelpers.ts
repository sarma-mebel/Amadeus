/**
 * Small pure UI-label helpers extracted from ChatInterface: the time-of-day
 * bucket used for the chat panel's tint, and the tool-activity strip's
 * friendly labels.
 */

export type TimeOfDayBucket = 'dawn' | 'day' | 'evening' | 'night'

/** Same local-hour buckets VRMViewer's getTimeOfDayLighting() uses for the
 * 3D scene, re-derived here so the chat panel's tint can (subtly) agree
 * with the scene without importing anything from VRMViewer. */
export const getTimeOfDayBucket = (hour: number = new Date().getHours()): TimeOfDayBucket => {
  if (hour >= 5 && hour < 8) return 'dawn'
  if (hour >= 8 && hour < 17) return 'day'
  if (hour >= 17 && hour < 20) return 'evening'
  return 'night'
}

// Friendly labels for the tool activity strip fed by 'tool_progress' WS
// frames during tool-augmented turns (Feature: tool activity strip).
const TOOL_ACTIVITY_LABELS: Record<string, string> = {
  web_search: 'Searching the web…',
  literature_search: 'Searching papers…',
  pubmed: 'Searching papers…',
  arxiv: 'Searching papers…',
  code_interpreter: 'Running code…',
  calculator: 'Calculating…',
  desktop_files: 'Reading files…',
  reminders: 'Checking reminders…',
}

export const describeToolActivity = (tool: string) => TOOL_ACTIVITY_LABELS[tool] ?? `Using ${tool}…`
