/**
 * Model (outfit) registry: the set of selectable VRM avatar models. Mirrors
 * the localStorage-preference pattern in environments.ts (load/save helpers,
 * a versioned key, a window event for a live hot-swap) but the registry
 * itself is discovered at runtime from an optional static manifest instead
 * of being a fixed enum — today only static/models/Kurisu.vrm ships, so the
 * picker (SettingsPanel) stays hidden until a second model is registered via
 * static/models/models.json.
 */

import { resolveAssetPath } from './config'

export interface ModelDefinition {
  id: string
  label: string
  file: string
}

export const DEFAULT_MODEL: ModelDefinition = {
  id: 'kurisu_default',
  label: 'Lab coat (default)',
  file: '/static/models/Kurisu.vrm',
}

const MODEL_KEY = 'kurisu_model_v1'

export const loadModelId = (): string => {
  if (typeof window === 'undefined') return DEFAULT_MODEL.id
  try {
    const stored = window.localStorage.getItem(MODEL_KEY)
    if (typeof stored === 'string' && stored.trim().length > 0) return stored
    return DEFAULT_MODEL.id
  } catch {
    return DEFAULT_MODEL.id
  }
}

export const saveModelId = (id: string): void => {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(MODEL_KEY, id)
  } catch {
    // Non-fatal — losing the model preference isn't worth surfacing an error.
  }
}

/** Dispatched on window when the user changes the outfit in SettingsPanel;
 * VRMViewer listens and hot-swaps (full scene remount) without a reload. */
export const MODEL_CHANGE_EVENT = 'kurisu:model-change'

export const dispatchModelChange = (id: string): void => {
  if (typeof window === 'undefined') return
  window.dispatchEvent(new CustomEvent<string>(MODEL_CHANGE_EVENT, { detail: id }))
}

const MODELS_DIR_PREFIX = '/static/models/'

const isNonEmptyString = (value: unknown): value is string =>
  typeof value === 'string' && value.trim().length > 0

/** Validates one raw manifest entry into a ModelDefinition, or null if it's
 * malformed/unsafe. `file` must live under static/models/ — rejects
 * traversal (`../..`) and absolute/external URLs. */
const normalizeModelEntry = (entry: unknown): ModelDefinition | null => {
  if (!entry || typeof entry !== 'object') return null
  const { id, label, file } = entry as Record<string, unknown>
  if (!isNonEmptyString(id) || !isNonEmptyString(label) || !isNonEmptyString(file)) return null
  if (!file.startsWith(MODELS_DIR_PREFIX)) return null
  return { id, label, file }
}

let registryPromise: Promise<ModelDefinition[]> | null = null

const fetchModelRegistryUncached = async (): Promise<ModelDefinition[]> => {
  try {
    const response = await fetch(resolveAssetPath('/static/models/models.json'), { cache: 'no-store' })
    if (!response.ok) return [DEFAULT_MODEL]
    const data = (await response.json()) as { models?: unknown[] }
    const rawEntries = Array.isArray(data?.models) ? data.models : []

    const result: ModelDefinition[] = [DEFAULT_MODEL]
    const seenIds = new Set<string>([DEFAULT_MODEL.id])
    for (const rawEntry of rawEntries) {
      const entry = normalizeModelEntry(rawEntry)
      if (!entry) continue
      if (seenIds.has(entry.id)) continue
      seenIds.add(entry.id)
      result.push(entry)
    }
    return result
  } catch {
    return [DEFAULT_MODEL]
  }
}

/** Cached at module scope so repeated calls (VRMViewer's load effect,
 * SettingsPanel's mount) don't refetch the manifest every time. */
export const fetchModelRegistry = (): Promise<ModelDefinition[]> => {
  if (!registryPromise) {
    registryPromise = fetchModelRegistryUncached()
  }
  return registryPromise
}

/** Test-only: clears the module-level cache so each test gets a fresh fetch. */
export const __resetModelRegistryCache = (): void => {
  registryPromise = null
}

export const resolveModelFile = (id: string, registry: ModelDefinition[]): string =>
  registry.find((model) => model.id === id)?.file ?? DEFAULT_MODEL.file
