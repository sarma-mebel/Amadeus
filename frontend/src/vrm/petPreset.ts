/**
 * Pet preset contract: localStorage key + change event a sibling agent's
 * Settings UI writes/dispatches. This module only owns reading the value —
 * VRMViewer (pet variant) is the sole consumer, reframing the camera via
 * cameraFraming's 'compact'/'medium' framingMode on read and on the event.
 */

export const PET_PRESET_STORAGE_KEY = 'kurisu_pet_preset'
export const PET_PRESET_CHANGE_EVENT = 'kurisu-pet-preset-changed'

export type PetPreset = 'full' | 'compact'

/** Absent/unrecognized value = 'full' (current framing, unchanged default). */
export const loadPetPreset = (): PetPreset => {
  try {
    return window.localStorage.getItem(PET_PRESET_STORAGE_KEY) === 'compact' ? 'compact' : 'full'
  } catch {
    return 'full'
  }
}

/** cameraFraming's computeDefaultFraming takes 'medium' | 'compact' —
 * 'full' preset maps to the existing 'medium' shot (no new name for the
 * same framing). */
export const petPresetToFramingMode = (preset: PetPreset): 'medium' | 'compact' =>
  preset === 'compact' ? 'compact' : 'medium'
