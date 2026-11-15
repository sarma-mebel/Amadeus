/**
 * Client-side failure visibility.
 *
 * The convention in this codebase has long been "degrade gracefully and
 * silently" — a poke that 404s, a courtesy poll that quietly disables
 * itself after a few failures, a presence fetch that best-effort no-ops.
 * That silence is correct for the USER (nobody wants an error toast every
 * time an optional glance fails) but it has repeatedly let genuinely dead
 * features ship unnoticed: a screen-glance ACL failure once looked
 * identical to "working, nothing to say" for a whole release; drag
 * reactions 422'd invisibly for a week. The backend has had process-lifetime
 * health counters (amadeus/telemetry.py, surfaced on GET /api/status) for a
 * while; this module is the client-side equivalent.
 *
 * In-memory only, page-lifetime — no localStorage, no network calls of its
 * own. It exists so a developer (or the Diagnostics panel in Settings) can
 * SEE a silent path die instead of inferring it from "she seems quiet".
 *
 * Convention for callers: increment at the exact catch/early-return that
 * would otherwise swallow a failure, with a short kebab/dot-separated name
 * (e.g. `pet.poke_failed`) and an optional one-line `detail` (HTTP status,
 * a path, an error message) — enough to grep the counter name straight back
 * to its call site.
 */

export interface DiagnosticEvent {
  name: string
  detail?: string
  at: number
}

export interface DiagnosticsSnapshot {
  /** name -> total increments this page lifetime. */
  counters: Record<string, number>
  /** Most recent detail string per counter name (undefined if none given). */
  lastDetail: Record<string, string | undefined>
  /** Most recent event time per counter name (epoch ms). */
  lastSeen: Record<string, number>
  /** Ring buffer of the most recent events, oldest first. */
  events: DiagnosticEvent[]
}

type Subscriber = (snapshot: DiagnosticsSnapshot) => void

const RING_BUFFER_SIZE = 25

class DiagnosticsStore {
  private counters = new Map<string, number>()
  private lastDetail = new Map<string, string | undefined>()
  private lastSeen = new Map<string, number>()
  private events: DiagnosticEvent[] = []
  private warned = new Set<string>()
  private subscribers = new Set<Subscriber>()

  /** Record one occurrence of `name`. `detail` is a short, human-readable
   * note (HTTP status, path, error message) — kept as the counter's most
   * recent detail and appended to the ring buffer. Warns to the console
   * exactly once per counter NAME per page lifetime, so a feature dying
   * always shows up in a dev console without spamming on every repeat. */
  increment(name: string, detail?: string): void {
    const at = Date.now()
    this.counters.set(name, (this.counters.get(name) ?? 0) + 1)
    this.lastDetail.set(name, detail)
    this.lastSeen.set(name, at)
    this.events.push({ name, detail, at })
    if (this.events.length > RING_BUFFER_SIZE) {
      this.events.splice(0, this.events.length - RING_BUFFER_SIZE)
    }
    if (!this.warned.has(name)) {
      this.warned.add(name)
      console.warn(`[diagnostics] ${name}${detail ? `: ${detail}` : ''}`)
    }
    this.notify()
  }

  /** A point-in-time copy — safe for callers to hold onto without it
   * mutating underneath them. */
  snapshot(): DiagnosticsSnapshot {
    return {
      counters: Object.fromEntries(this.counters),
      lastDetail: Object.fromEntries(this.lastDetail),
      lastSeen: Object.fromEntries(this.lastSeen),
      events: [...this.events],
    }
  }

  /** Subscribe to every future increment; returns an unsubscribe function.
   * Does NOT fire immediately on subscribe — callers that want the current
   * state should call snapshot() themselves first (matches the rest of this
   * codebase's ref/subscribe patterns, e.g. levelBus). */
  subscribe(cb: Subscriber): () => void {
    this.subscribers.add(cb)
    return () => {
      this.subscribers.delete(cb)
    }
  }

  private notify(): void {
    const snapshot = this.snapshot()
    for (const cb of this.subscribers) {
      cb(snapshot)
    }
  }

  /** Test seam: reset all state (counters, ring buffer, warn-once memory,
   * subscribers untouched). Not used by production code. */
  resetForTest(): void {
    this.counters.clear()
    this.lastDetail.clear()
    this.lastSeen.clear()
    this.events = []
    this.warned.clear()
  }
}

export const diagnostics = new DiagnosticsStore()
