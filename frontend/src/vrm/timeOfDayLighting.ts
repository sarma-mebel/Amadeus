/**
 * Companion-app touch: the scene's color temperature nods to the local hour
 * instead of always looking like flat midday, using the same local clock the
 * desktop shell's greeting/idle-chatter prompts already read server-side
 * (this process and the browser share a machine, so no client->server hour
 * needs to be threaded through the API).
 *
 * ACES tone-mapping note: switching the renderer to ACESFilmicToneMapping
 * (see VRMViewer's renderer setup) rolls midtones off a bit darker than the
 * old NoToneMapping path did, so the ambientIntensity values below carry a
 * small (~10-15%) bump versus their pre-ACES originals to keep the overall
 * scene brightness in parity with before. Verified against day/evening/
 * night screenshots — see job report for before/after.
 *
 * Time-of-day 3D lighting is intentionally disabled: the dim night preset
 * made Kurisu hard to see. The scene now always uses the bright 'day'
 * lighting regardless of local hour. (The subtle ChatInterface UI tint is
 * unaffected.) Kept as a function (not an inlined constant) so re-enabling
 * per-hour lighting later is a one-place change.
 */

export interface TimeOfDayLighting {
  label: string
  ambientColor: number
  ambientIntensity: number
  mainColor: number
  fillColor: number
  // Fog/depth-cueing (item 3): FogExp2 color + density per bucket, lerped
  // alongside the light colors in the same 2s transition. Kills the hard
  // black void past the environment GLB's draw distance at night without
  // washing out the (already dim) day/dawn/evening sets.
  fogColor: number
  fogDensity: number
}

export const getTimeOfDayLighting = (): TimeOfDayLighting => {
  return {
    label: 'day',
    ambientColor: 0xffffff,
    ambientIntensity: 0.45,
    mainColor: 0xffffff,
    fillColor: 0x8888ff,
    fogColor: 0xe8ecf5,
    fogDensity: 0.018,
  }
}
