/**
 * Reflex layer: declarative, instant, LLM-free avatar reactions to discrete
 * events. A generalization of the instant-acknowledgment idea (see
 * acknowledgment.ts) into parallel latency bands — while the LLM path takes
 * seconds and presence lines take minutes, reflexes land the same frame the
 * event does, with zero network involvement. She flinches, glances, and
 * tilts on her own nervous system.
 *
 * Unlike attentionEvents.ts (bespoke per-event handlers mutating frame
 * state, with the anti-spam logic living in each EMITTER), every reflex
 * here is one table row: event name, an optional match filter, a gate
 * list, a per-reflex cooldown, and an action composed from a small injected
 * primitive vocabulary (blink / gaze snap / curious tilt / expression
 * flick). Gating (reduced-motion, speaking, dozing, tucked) and cooldowns
 * are applied centrally by the engine, so a reflex can never fire while
 * she's mid-sentence-and-shouldn't, and a burst of events can't turn her
 * twitchy. A gated or cooling-down event is a silent no-op by design — the
 * whole layer is optional garnish; only a THROWING action is counted
 * (diagnostics `reflex.action_failed`), never raised.
 *
 * New discrete-event reflexes belong HERE as table rows, not as new
 * bespoke window listeners. attentionEvents.ts remains the high-rate
 * continuous machinery (typing, mic levels, per-sentence beats) whose
 * handlers need direct frame-state access; this table is for one-shot
 * "something just happened" moments.
 */

import { diagnostics } from '../diagnostics'

/** Everything a reflex action may do, plus the gate probes, injected by the
 * host (VRMViewer wires these to expressionDriver/gazeDirector/
 * proceduralBody/lip-sync). Framework-free so the engine tests run with
 * plain fakes and a fake clock. */
export interface ReflexPrimitives {
  /** expressionDriver.requestBlink — probability 0..1. */
  requestBlink: (probability: number) => void
  /** Hold eye contact: bump gazeSocial.lockOnUntil ~1600ms out, same move
   * attentionEvents' mic-speaking handler makes. */
  snapGaze: () => void
  /** proceduralBody.queueCuriousTilt — the standalone-amplitude head tilt. */
  queueCuriousTilt: () => void
  /** Brief eased facial expression (expressionDriver.set). */
  expressionFlick: (name: string, weight: number, durationMs: number) => void
  // --- gate probes ---
  isSpeaking: () => boolean
  isDozing: () => boolean
  isTucked: () => boolean
  prefersReducedMotion: () => boolean
  /** Monotonic clock for cooldowns (performance.now in production). */
  now: () => number
}

/** 'reducedMotion' skips the reflex when prefersReducedMotion(); the
 * 'notWhile*' gates skip it while the corresponding probe is true. */
export type ReflexGate = 'reducedMotion' | 'notWhileSpeaking' | 'notWhileDozing' | 'notWhileTucked'

export interface ReflexDef {
  id: string
  /** Window event name this reflex listens for. */
  event: string
  cooldownMs: number
  gates: ReflexGate[]
  /** Optional filter over the concrete event (CustomEvent detail routing)
   * and/or current state via the primitives (e.g. untuck-blink reads
   * isTucked() AFTER the change event). Falsy result = silent no-op. */
  matches?: (ev: Event, primitives: ReflexPrimitives) => boolean
  action: (primitives: ReflexPrimitives) => void
}

export interface ReflexEngine {
  attach: () => void
  detach: () => void
  /** Test/manual seam: fire a reflex by id through the exact same gate +
   * cooldown path a real event takes (the `matches` filter is skipped —
   * there is no event to match against). Unknown ids are silent no-ops. */
  fire: (id: string) => void
}

const ambientDetail = (ev: Event): { kind?: string; severity?: string } => {
  const detail = (ev as CustomEvent<{ kind?: string; severity?: string }>).detail
  return detail ?? {}
}

/** The shipped reflex table. Everything is deliberately SUBTLE — one blink,
 * one gaze snap, one tilt — she reads as alive, not twitchy, and generous
 * cooldowns keep each of these a rare beat rather than a tic. */
export const REFLEX_TABLE: ReflexDef[] = [
  {
    // She was mid-speech and the user talked over her (duplexTurn dispatches
    // this only when playback was actually cut) — a startled catch-blink +
    // renewed eye contact. Deliberately NOT gated on notWhileSpeaking:
    // being interrupted while speaking is the trigger's whole meaning.
    id: 'barge-in-catch',
    event: 'kurisu:barge-in',
    cooldownMs: 8_000,
    gates: ['reducedMotion'],
    action: (p) => {
      p.requestBlink(1)
      p.snapGaze()
    },
  },
  {
    // The user came back to the window: quiet re-engagement, eye contact
    // first. Subtle by design — overlapping with renderLoop's rare
    // >2-min-hidden return Greeting is harmless.
    id: 'window-focus-reengage',
    event: 'focus',
    cooldownMs: 30_000,
    gates: ['reducedMotion', 'notWhileSpeaking', 'notWhileDozing'],
    action: (p) => {
      p.snapGaze()
      p.requestBlink(0.7)
    },
  },
  {
    // Battery hit critical and isn't charging: a concerned glance at the
    // tray, not a lecture — the ambient presence line (if she says one)
    // carries the words. Once per ten minutes at most.
    id: 'battery-critical-glance',
    event: 'kurisu:ambient',
    cooldownMs: 600_000,
    gates: ['reducedMotion', 'notWhileSpeaking', 'notWhileDozing'],
    matches: (ev) => {
      const { kind, severity } = ambientDetail(ev)
      return kind === 'battery' && severity === 'critical'
    },
    action: (p) => {
      p.queueCuriousTilt()
      p.expressionFlick('sad', 0.2, 1_800)
    },
  },
  {
    // Music started (or the track changed): one curious head-tilt, as if
    // catching the tune. Tucked pets don't react — she's out of sight.
    id: 'music-tilt',
    event: 'kurisu:ambient',
    cooldownMs: 300_000,
    gates: ['reducedMotion', 'notWhileSpeaking', 'notWhileDozing', 'notWhileTucked'],
    matches: (ev) => ambientDetail(ev).kind === 'music',
    action: (p) => p.queueCuriousTilt(),
  },
  {
    // The pet slid back out from the screen edge (petTuck.ts fires the
    // change event for both directions; isTucked() reads the state AFTER
    // the change, so false = this was an untuck): a waking-up blink and a
    // look at the user.
    id: 'untuck-blink',
    event: 'kurisu:pet-tuck-change',
    cooldownMs: 60_000,
    gates: ['reducedMotion'],
    matches: (_ev, p) => !p.isTucked(),
    action: (p) => {
      p.requestBlink(1)
      p.snapGaze()
    },
  },
]

export const createReflexEngine = (
  primitives: ReflexPrimitives,
  defs: ReflexDef[] = REFLEX_TABLE,
): ReflexEngine => {
  const lastFiredAt = new Map<string, number>()

  const gateBlocked = (gates: ReflexGate[]): boolean => {
    if (gates.includes('reducedMotion') && primitives.prefersReducedMotion()) return true
    if (gates.includes('notWhileSpeaking') && primitives.isSpeaking()) return true
    if (gates.includes('notWhileDozing') && primitives.isDozing()) return true
    if (gates.includes('notWhileTucked') && primitives.isTucked()) return true
    return false
  }

  // One funnel for real events and fire(): matches (when there is an event)
  // -> gates -> cooldown -> action. The engine itself never throws — a
  // failing action (or a failing host-injected probe) counts and moves on.
  const run = (def: ReflexDef, ev: Event | null): void => {
    try {
      if (ev && def.matches && !def.matches(ev, primitives)) return
      if (gateBlocked(def.gates)) return
      const nowMs = primitives.now()
      const last = lastFiredAt.get(def.id)
      if (last !== undefined && nowMs - last < def.cooldownMs) return
      lastFiredAt.set(def.id, nowMs)
      def.action(primitives)
    } catch (error) {
      diagnostics.increment('reflex.action_failed', `${def.id}: ${String(error)}`)
    }
  }

  const listeners = defs.map((def) => ({
    def,
    handler: (ev: Event) => run(def, ev),
  }))

  const attach = () => {
    for (const { def, handler } of listeners) {
      window.addEventListener(def.event, handler)
    }
  }

  const detach = () => {
    for (const { def, handler } of listeners) {
      window.removeEventListener(def.event, handler)
    }
  }

  const fire = (id: string) => {
    const entry = defs.find((def) => def.id === id)
    if (entry) run(entry, null)
  }

  return { attach, detach, fire }
}
