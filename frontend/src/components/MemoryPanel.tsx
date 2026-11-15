/**
 * "Her memories" slide-over: a read/write window into the layered memory
 * system via POST /api/memory. Shows the distilled user profile ("About
 * you"), recent long-term facts (pinnable/editable/deletable), recent
 * session summaries ("Recent moments"), her private diary entry, and open
 * threads, plus a small "remember something" input (action:'remember').
 * Mirrors the /remember, /forget and /memory chat commands, just browsable.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import SynapseMap, { RECORD_PREFIX, type MemoryGraph } from './SynapseMap'
import EmptyState from './EmptyState'
import InsightsSection from './InsightsSection'
import { apiFetch } from '../api'
import { localDate } from '../dates'
import { t, useLocale } from '../i18n'
import { memoryRequest } from '../memoryApi'
import { resolveUserId } from '../hooks/useSessions'
import { clearSeenMilestones } from '../milestones'
import { resetIntroDone } from '../introMode'
import './MemoryPanel.css'

interface MemoryProfile {
  facts: Record<string, string>
  openThreads: string[]
  sessionCount: number | null
  firstMet: string | null
  lastEpisode: string | null
}

interface RelationshipArc {
  trustLevel: number
  arcCuriosity: number
  arcLogic: number
  arcPride: number
  toneTrend: string | null
  /** How the LAST conversation felt: {label, subject?}. Already in the
   * /api/memory payload but rendered nowhere until now — the most legible
   * item in the whole relationship block. */
  lastFelt: { label: string; subject: string | null } | null
}

/** A single fact/episode memory. `id` is null for legacy string-only
 * entries from an older backend (no pin/edit affordance possible without a
 * record id — delete still falls back to forget-by-text). */
export interface FactRecord {
  id: string | null
  text: string
  pinned: boolean
}

interface MemoryOverview {
  profile: MemoryProfile
  facts: FactRecord[]
  episodes: FactRecord[]
  relationship: RelationshipArc | null
}

type PanelState = 'loading' | 'ready' | 'unavailable' | 'error'

interface DiaryEntry {
  date: string
  text: string
}

// -1..+1 bipolar values; anything outside that range or non-finite is
// treated as malformed, which folds the whole relationship record to
// absent (older backends simply omit the key, but a half-written one
// shouldn't render a broken bar either).
const bipolar = (value: unknown): number | null => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null
  if (value < -1 || value > 1) return null
  return value
}

const parseRelationship = (raw: unknown): RelationshipArc | null => {
  if (!raw || typeof raw !== 'object') return null
  const record = raw as {
    trust_level?: unknown
    arc?: {
      curiosity_vs_annoyance?: unknown
      logic_vs_emotion?: unknown
      pride_vs_vulnerability?: unknown
    }
    tone_trend?: unknown
    last_felt?: unknown
  }
  const trust = record.trust_level
  if (typeof trust !== 'number' || !Number.isFinite(trust) || trust < 0 || trust > 1) return null
  const arc = record.arc && typeof record.arc === 'object' ? record.arc : {}
  const curiosity = bipolar(arc.curiosity_vs_annoyance)
  const logic = bipolar(arc.logic_vs_emotion)
  const pride = bipolar(arc.pride_vs_vulnerability)
  if (curiosity === null || logic === null || pride === null) return null
  const toneTrend = typeof record.tone_trend === 'string' ? record.tone_trend : null
  // A malformed last_felt folds to null on its own rather than taking the
  // whole relationship record with it — it's decorative, the arc is not.
  let lastFelt: RelationshipArc['lastFelt'] = null
  if (record.last_felt && typeof record.last_felt === 'object') {
    const felt = record.last_felt as { label?: unknown; subject?: unknown }
    if (typeof felt.label === 'string' && felt.label.trim().length > 0) {
      lastFelt = {
        label: felt.label,
        subject: typeof felt.subject === 'string' && felt.subject.trim() ? felt.subject : null,
      }
    }
  }
  return {
    trustLevel: trust,
    arcCuriosity: curiosity,
    arcLogic: logic,
    arcPride: pride,
    toneTrend,
    lastFelt,
  }
}

const strings = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string' && item.trim().length > 0) : []

/** Prefers the new `fact_records`/`episode_records` shape ({id, text,
 * pinned, timestamp}[]); falls back to the legacy plain-string arrays for
 * an older backend, giving each a null id (no pin/edit possible, delete
 * falls back to forget-by-text). Pinned records sort first so they stay
 * visible without scrolling. */
const parseRecords = (records: unknown, legacyStrings: unknown): FactRecord[] => {
  if (Array.isArray(records)) {
    const parsed = records
      .map((item): FactRecord | null => {
        if (!item || typeof item !== 'object') return null
        const entry = item as { id?: unknown; text?: unknown; pinned?: unknown }
        if (typeof entry.text !== 'string' || !entry.text.trim()) return null
        return {
          id: typeof entry.id === 'string' ? entry.id : null,
          text: entry.text,
          pinned: entry.pinned === true,
        }
      })
      .filter((item): item is FactRecord => item !== null)
    return [...parsed].sort((a, b) => Number(b.pinned) - Number(a.pinned))
  }
  return strings(legacyStrings).map((text) => ({ id: null, text, pinned: false }))
}

const parseOverview = (raw: unknown): MemoryOverview => {
  const overview = ((raw ?? {}) as { overview?: unknown }).overview ?? {}
  const record = overview as {
    profile?: {
      facts?: unknown
      open_threads?: unknown
      session_count?: unknown
      first_met?: unknown
      last_episode?: unknown
    }
    facts?: unknown
    episodes?: unknown
    fact_records?: unknown
    episode_records?: unknown
    relationship?: unknown
  }
  const profile = record.profile ?? {}
  const profileFacts: Record<string, string> = {}
  if (profile.facts && typeof profile.facts === 'object') {
    Object.entries(profile.facts as Record<string, unknown>).forEach(([key, value]) => {
      if (typeof value === 'string' && value.trim()) profileFacts[key] = value
    })
  }
  return {
    profile: {
      facts: profileFacts,
      openThreads: strings(profile.open_threads),
      sessionCount: typeof profile.session_count === 'number' ? profile.session_count : null,
      firstMet: typeof profile.first_met === 'string' ? profile.first_met : null,
      lastEpisode: typeof profile.last_episode === 'string' ? profile.last_episode : null,
    },
    facts: parseRecords(record.fact_records, record.facts),
    episodes: parseRecords(record.episode_records, record.episodes),
    relationship: parseRelationship(record.relationship),
  }
}

// Read-only bipolar bar: value in -1..+1, filled from the center marker
// (0) toward whichever pole it leans. Plain divs — no chart lib in this
// project.
const BipolarBar = ({
  value,
  leftLabel,
  rightLabel,
}: {
  value: number
  leftLabel: string
  rightLabel: string
}) => {
  const clamped = Math.max(-1, Math.min(1, value))
  const magnitudePct = (Math.abs(clamped) / 2) * 100
  const fillStyle =
    clamped >= 0
      ? { left: '50%', width: `${magnitudePct}%` }
      : { right: '50%', width: `${magnitudePct}%` }
  return (
    <div className="memory-bipolar" role="meter" aria-valuemin={-1} aria-valuemax={1} aria-valuenow={clamped}>
      <div className="memory-bipolar-labels">
        <span>{leftLabel}</span>
        <span>{rightLabel}</span>
      </div>
      <div className="memory-bipolar-track">
        <div className="memory-bipolar-center" />
        <div className={`memory-bipolar-fill ${clamped >= 0 ? 'positive' : 'negative'}`} style={fillStyle} />
      </div>
    </div>
  )
}

const MemoryPanel = ({ onClose }: { onClose: () => void }) => {
  const locale = useLocale()
  const [state, setState] = useState<PanelState>('loading')
  const [overview, setOverview] = useState<MemoryOverview | null>(null)
  const [view, setView] = useState<'list' | 'map'>('list')
  const [graph, setGraph] = useState<MemoryGraph | null>(null)
  const [graphState, setGraphState] = useState<'idle' | 'loading' | 'error'>('idle')
  const [rememberText, setRememberText] = useState('')
  const [rememberBusy, setRememberBusy] = useState(false)
  const [resetArmed, setResetArmed] = useState(false)
  const [resetBusy, setResetBusy] = useState(false)
  // Inline edit target: the record currently swapped for a text input, and
  // its live draft text. Cleared on save/cancel or when the row disappears.
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editingText, setEditingText] = useState('')
  // Per-record "pinned — unpin it first" message, shown inline instead of
  // removing the row (status 'protected' from a delete attempt).
  const [protectedIds, setProtectedIds] = useState<Record<string, boolean>>({})
  const [diary, setDiary] = useState<DiaryEntry | null>(null)
  const closeButtonRef = useRef<HTMLButtonElement>(null)

  const loadOverview = useCallback(async () => {
    try {
      const response = await memoryRequest('overview')
      if (!response.ok) {
        setState('error')
        return
      }
      const data = await response.json()
      if (data?.status === 'unavailable') {
        setState('unavailable')
        return
      }
      setOverview(parseOverview(data))
      setState('ready')
    } catch {
      setState('error')
    }
  }, [])

  useEffect(() => {
    void loadOverview()
  }, [loadOverview])

  // Her diary (M3): best-effort fetch on open. Hidden outright on a null
  // entry, a malformed payload, or any network/parse failure — this is a
  // small bonus section, not something worth an error state of its own.
  useEffect(() => {
    ;(async () => {
      try {
        const response = await apiFetch(
          `/api/diary?user_id=${encodeURIComponent(resolveUserId())}`,
        )
        if (!response.ok) return
        const data = await response.json()
        const entry = data?.diary
        if (!entry || typeof entry !== 'object') return
        if (typeof entry.date !== 'string' || typeof entry.text !== 'string' || !entry.text.trim()) {
          return
        }
        setDiary({ date: entry.date, text: entry.text })
      } catch {
        // Diary is best-effort; the section just stays hidden.
      }
    })()
  }, [])

  // The graph is fetched lazily, on first switch to the map view (it runs
  // an O(n^2) similarity pass server-side — no point paying it for the
  // default list view).
  const showMap = useCallback(async () => {
    setView('map')
    if (graph || graphState === 'loading') return
    setGraphState('loading')
    try {
      const response = await memoryRequest('graph')
      if (!response.ok) throw new Error(String(response.status))
      const data = await response.json()
      const raw = data?.graph
      setGraph({
        nodes: Array.isArray(raw?.nodes) ? raw.nodes : [],
        edges: Array.isArray(raw?.edges) ? raw.edges : [],
      })
      setGraphState('idle')
    } catch {
      setGraphState('error')
    }
  }, [graph, graphState])

  // Escape closes, like the other overlays; focus starts on the close
  // button so keyboard users aren't stranded behind the panel.
  useEffect(() => {
    closeButtonRef.current?.focus()
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onClose])

  /** Removes a fact/episode row from the given list, by record id when one
   * exists, else by exact legacy text match. Shared by forgetFact and
   * forgetEpisode. */
  const removeRecord = (
    list: 'facts' | 'episodes',
    record: FactRecord,
  ) => {
    setOverview((prev) =>
      prev
        ? {
            ...prev,
            [list]: prev[list].filter((item) =>
              record.id ? item.id !== record.id : item.text !== record.text,
            ),
          }
        : prev,
    )
  }

  /** Deletes a fact or episode record. Uses forget_id when a real record id
   * is present (pin/edit/protected all need one), else falls back to the
   * legacy forget-by-text action for pre-M4 backends. Optimistic, except
   * for the 'protected' response: a pinned record refuses to delete, so the
   * row is restored and an inline message shown instead. */
  const forgetRecord = async (list: 'facts' | 'episodes', record: FactRecord) => {
    removeRecord(list, record)
    try {
      const response = record.id
        ? await memoryRequest('forget_id', { recordId: record.id })
        : await memoryRequest('forget', { text: record.text })
      const data = await response.json().catch(() => null)
      if (data?.status === 'unavailable') {
        setState('unavailable')
        return
      }
      if (data?.status === 'protected') {
        // Restore the row (pinned first, matching the initial sort) and
        // flag it so the row can show the server's message inline.
        setOverview((prev) => {
          if (!prev) return prev
          const restored = [...prev[list], record].sort(
            (a, b) => Number(b.pinned) - Number(a.pinned),
          )
          return { ...prev, [list]: restored }
        })
        if (record.id) setProtectedIds((prev) => ({ ...prev, [record.id as string]: true }))
        return
      }
      if (data?.status === 'not_found') {
        await loadOverview()
      }
    } catch {
      // Optimistic removal stands; the record reappears on next refresh if
      // the delete never landed.
    }
  }

  const forgetFact = (record: FactRecord) => forgetRecord('facts', record)
  const forgetEpisode = (record: FactRecord) => forgetRecord('episodes', record)

  /** Toggles pinned on a fact/episode record; requires a real record id
   * (the pin button is only rendered when one exists). Optimistic. */
  const pinRecord = async (list: 'facts' | 'episodes', recordId: string, pinned: boolean) => {
    setOverview((prev) => {
      if (!prev) return prev
      const updated = prev[list].map((item) =>
        item.id === recordId ? { ...item, pinned } : item,
      )
      const sorted = [...updated].sort((a, b) => Number(b.pinned) - Number(a.pinned))
      return { ...prev, [list]: sorted }
    })
    try {
      const response = await memoryRequest(pinned ? 'pin' : 'unpin', { recordId })
      const data = await response.json().catch(() => null)
      if (data?.status === 'unavailable') setState('unavailable')
      if (data?.status === 'not_found') await loadOverview()
    } catch {
      // Optimistic; reappears/reverts on next refresh if the call never landed.
    }
  }

  /** Edits a fact/episode record's text in place. Optimistic. */
  const editRecord = async (list: 'facts' | 'episodes', recordId: string, text: string) => {
    setOverview((prev) => {
      if (!prev) return prev
      return {
        ...prev,
        [list]: prev[list].map((item) => (item.id === recordId ? { ...item, text } : item)),
      }
    })
    try {
      const response = await memoryRequest('edit', { recordId, text })
      const data = await response.json().catch(() => null)
      if (data?.status === 'unavailable') setState('unavailable')
      if (data?.status === 'not_found') await loadOverview()
    } catch {
      // Optimistic; reverts on next refresh if the call never landed.
    }
  }

  const beginEdit = (record: FactRecord) => {
    if (!record.id) return
    setEditingId(record.id)
    setEditingText(record.text)
  }

  const cancelEdit = () => {
    setEditingId(null)
    setEditingText('')
  }

  const saveEdit = (list: 'facts' | 'episodes') => {
    const text = editingText.trim()
    if (!editingId || !text) return
    void editRecord(list, editingId, text)
    setEditingId(null)
    setEditingText('')
  }

  const forgetProfileFact = async (key: string) => {
    setOverview((prev) => {
      if (!prev) return prev
      const facts = { ...prev.profile.facts }
      delete facts[key]
      return { ...prev, profile: { ...prev.profile, facts } }
    })
    try {
      await memoryRequest('forget_fact', { text: key })
    } catch {
      // Optimistic; reappears on next refresh if the delete never landed.
    }
  }

  const forgetThread = async (thread: string) => {
    setOverview((prev) =>
      prev
        ? {
            ...prev,
            profile: {
              ...prev.profile,
              openThreads: prev.profile.openThreads.filter((item) => item !== thread),
            },
          }
        : prev,
    )
    try {
      await memoryRequest('forget_thread', { text: thread })
    } catch {
      // Optimistic.
    }
  }

  const rememberFact = async () => {
    const text = rememberText.trim()
    if (!text || rememberBusy) return
    setRememberBusy(true)
    try {
      const response = await memoryRequest('remember', { text })
      const data = await response.json().catch(() => null)
      if (data?.status === 'unavailable') {
        setState('unavailable')
        return
      }
      if (response.ok) {
        setRememberText('')
        const recordId = typeof data?.record_id === 'string' ? data.record_id : null
        setOverview((prev) =>
          prev ? { ...prev, facts: [{ id: recordId, text, pinned: false }, ...prev.facts] } : prev,
        )
      }
    } catch {
      // Leave the input intact so the user can retry.
    } finally {
      setRememberBusy(false)
    }
  }

  // Export/import: her memories belong to the user. Export downloads a zip
  // of the memory files; import replaces them (the server takes a backup of
  // the current files first, and the confirm-arm pattern matches reset).
  const [exportBusy, setExportBusy] = useState(false)
  const [importBusy, setImportBusy] = useState(false)
  const [importArmed, setImportArmed] = useState(false)
  const [backupNotice, setBackupNotice] = useState<string | null>(null)
  const importInputRef = useRef<HTMLInputElement | null>(null)

  const exportMemory = async () => {
    if (exportBusy) return
    setExportBusy(true)
    setBackupNotice(null)
    try {
      const response = await apiFetch(
        `/api/memory/export?user_id=${encodeURIComponent(resolveUserId())}`,
      )
      if (!response.ok) {
        setBackupNotice(
          response.status === 404
            ? t('Nothing to export yet — talk to her first.')
            : t('Export failed.'),
        )
        return
      }
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `amadeus-memory-${new Date().toISOString().slice(0, 10)}.zip`
      anchor.click()
      URL.revokeObjectURL(url)
    } catch {
      setBackupNotice(t('Export failed — is the server running?'))
    } finally {
      setExportBusy(false)
    }
  }

  const importMemory = async (file: File) => {
    if (importBusy) return
    setImportBusy(true)
    setBackupNotice(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const response = await apiFetch(
        `/api/memory/import?user_id=${encodeURIComponent(resolveUserId())}`,
        { method: 'POST', body: form },
      )
      const data = await response.json().catch(() => null)
      if (!response.ok) {
        setBackupNotice(data?.detail || t('Import failed.'))
        return
      }
      setBackupNotice(
        t('Restored: {files}. Her previous memory was backed up first.', {
          files: (data?.restored || []).join(', '),
        }),
      )
      await loadOverview()
    } catch {
      setBackupNotice(t('Import failed — is the server running?'))
    } finally {
      setImportBusy(false)
    }
  }

  const resetRelationship = async () => {
    if (!resetArmed) {
      setResetArmed(true)
      return
    }
    setResetArmed(false)
    if (resetBusy) return
    setResetBusy(true)
    try {
      const response = await memoryRequest('reset_relationship')
      const data = await response.json().catch(() => null)
      if (data?.status === 'unavailable') {
        setState('unavailable')
        return
      }
      if (response.ok) {
        // A relationship reset is meant to feel like starting over — re-arm
        // milestone celebrations and conversational onboarding so they play
        // out again instead of staying latched off from the prior identity.
        const userId = resolveUserId()
        clearSeenMilestones(userId)
        resetIntroDone(userId)
        await loadOverview()
      }
    } catch {
      // Leave the arc as-is; the user can retry.
    } finally {
      setResetBusy(false)
    }
  }

  // Graph mutation handlers (SynapseMap's rec: node actions): the map's
  // node ids for facts/episodes don't distinguish which list a record came
  // from, so pin/edit resolve it from the loaded overview and forget just
  // asks the server directly (no local list to update optimistically until
  // the graph/overview are refreshed).
  const findRecordList = (recordId: string): 'facts' | 'episodes' | null => {
    if (!overview) return null
    if (overview.facts.some((item) => item.id === recordId)) return 'facts'
    if (overview.episodes.some((item) => item.id === recordId)) return 'episodes'
    return null
  }

  /** Toggles `pinned` on the matching graph node (id "rec:<recordId>") so
   * the map's ring redraws immediately, alongside the existing overview
   * pinRecord call — SynapseMap's own effect re-resolves `selected` against
   * this new graph reference, so SynapseDetail's pin button always computes
   * its next state from current data instead of a click-time snapshot. */
  const graphPin = (recordId: string, pinned: boolean) => {
    const list = findRecordList(recordId)
    if (list) void pinRecord(list, recordId, pinned)
    const nodeId = `${RECORD_PREFIX}${recordId}`
    setGraph((prev) =>
      prev
        ? { ...prev, nodes: prev.nodes.map((node) => (node.id === nodeId ? { ...node, pinned } : node)) }
        : prev,
    )
  }

  const graphEdit = (recordId: string, text: string) => {
    const list = findRecordList(recordId)
    if (list) void editRecord(list, recordId, text)
    const nodeId = `${RECORD_PREFIX}${recordId}`
    setGraph((prev) =>
      prev
        ? { ...prev, nodes: prev.nodes.map((node) => (node.id === nodeId ? { ...node, label: text } : node)) }
        : prev,
    )
  }

  /** Removes the record's node (and any edge touching it) from graph state
   * on success, or shows the same protected message the list view uses when
   * the server refuses (pinned record). Previously this called showMap()
   * right after setGraph(null) in the same synchronous handler — since
   * state updates aren't applied until the next render, showMap's own
   * `if (graph || ...) return` guard still saw the pre-null graph and
   * skipped the refetch entirely (a stale-closure no-op). Mutating graph
   * state directly avoids that trap and doesn't need a round-trip. */
  const graphDelete = async (recordId: string) => {
    const nodeId = `${RECORD_PREFIX}${recordId}`
    try {
      const response = await memoryRequest('forget_id', { recordId })
      const data = await response.json().catch(() => null)
      if (data?.status === 'unavailable') {
        setState('unavailable')
        return
      }
      if (data?.status === 'protected') {
        setProtectedIds((prev) => ({ ...prev, [recordId]: true }))
        return
      }
      setGraph((prev) =>
        prev
          ? {
              nodes: prev.nodes.filter((node) => node.id !== nodeId),
              edges: prev.edges.filter((edge) => edge.source !== nodeId && edge.target !== nodeId),
            }
          : prev,
      )
      const list = findRecordList(recordId)
      if (list) removeRecord(list, { id: recordId, text: '', pinned: false })
    } catch {
      // Best-effort; the map just won't reflect the attempted delete.
    }
  }

  const profile = overview?.profile
  const isEmpty =
    state === 'ready' &&
    overview !== null &&
    Object.keys(profile?.facts ?? {}).length === 0 &&
    overview.facts.length === 0 &&
    overview.episodes.length === 0 &&
    (profile?.openThreads.length ?? 0) === 0

  return (
    <div className="memory-overlay" onClick={onClose}>
      <aside
        className="memory-panel"
        role="dialog"
        aria-modal="true"
        aria-label={t('Her memories')}
        onClick={(event) => event.stopPropagation()}
      >
        <header className="memory-panel-header">
          <div>
            <h2>{t('Her memories')}</h2>
            <span className="memory-panel-subtitle">
              {t('What Kurisu keeps between conversations')}
              {profile?.sessionCount
                ? locale === 'ja'
                  ? ` · セッション${profile.sessionCount}回${profile.firstMet ? `(${profile.firstMet}から)` : ''}`
                  : ` · ${profile.sessionCount} session${profile.sessionCount === 1 ? '' : 's'}${profile.firstMet ? ` since ${profile.firstMet}` : ''}`
                : ''}
            </span>
          </div>
          <button
            ref={closeButtonRef}
            type="button"
            className="memory-close"
            onClick={onClose}
            aria-label={t('Close memories')}
          >
            ×
          </button>
        </header>

        <div className="memory-panel-body">
          {state === 'loading' && <p className="memory-note">{t('Recalling…')}</p>}

          {state === 'error' && (
            <p className="memory-note">{t("Couldn't reach her memory — is the server running?")}</p>
          )}

          {state === 'unavailable' && (
            <p className="memory-note">
              {t(
                "Long-term memory is switched off, so there's nothing stored here. Turn it on under Settings → Features to let her remember you between sessions.",
              )}
            </p>
          )}

          {isEmpty && (
            <EmptyState
              icon="memory"
              title={t('Nothing recorded yet.')}
              subLine={t("She's still getting to know you — memories show up here once you talk.")}
            />
          )}

          {state === 'ready' && overview && !isEmpty && (
            <>
              <div className="memory-view-toggle" role="tablist" aria-label={t('Memory view')}>
                <button
                  type="button"
                  role="tab"
                  aria-selected={view === 'list'}
                  className={view === 'list' ? 'active' : ''}
                  onClick={() => setView('list')}
                >
                  {t('List')}
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={view === 'map'}
                  className={view === 'map' ? 'active' : ''}
                  onClick={() => void showMap()}
                >
                  {t('Synapse map')}
                </button>
              </div>

              {view === 'map' && (
                <section className="memory-section">
                  {graphState === 'loading' && <p className="memory-note">{t('Mapping synapses…')}</p>}
                  {graphState === 'error' && (
                    <p className="memory-note">{t("Couldn't build the map — try again.")}</p>
                  )}
                  {graph && graph.nodes.length === 0 && graphState === 'idle' && (
                    <EmptyState
                      icon="synapse"
                      title={t('No synapses formed yet.')}
                      subLine={t('The map fills in as she stores memories to connect.')}
                    />
                  )}
                  {graph && graph.nodes.length > 0 && (
                    <SynapseMap
                      graph={graph}
                      onPin={graphPin}
                      onEdit={graphEdit}
                      onDelete={(id) => void graphDelete(id)}
                      protectedIds={protectedIds}
                    />
                  )}
                </section>
              )}

              {view === 'list' && (
                <>
              {Object.keys(overview.profile.facts).length > 0 && (
                <section className="memory-section">
                  <h3>{t('About you')}</h3>
                  <dl className="memory-profile-facts">
                    {Object.entries(overview.profile.facts).map(([key, value]) => (
                      <div className="memory-profile-row" key={key}>
                        <dt>{key}</dt>
                        <dd>
                          <span className="memory-profile-value">{value}</span>
                          <button
                            type="button"
                            className="memory-forget-button"
                            onClick={() => void forgetProfileFact(key)}
                            aria-label={`${t('Forget:')} ${key}`}
                            title={t('Make her forget this')}
                          >
                            ×
                          </button>
                        </dd>
                      </div>
                    ))}
                  </dl>
                </section>
              )}

              {overview.relationship && (
                <section className="memory-section">
                  <h3>{t('Relationship')}</h3>
                  <div className="memory-relationship">
                    <div className="memory-trust-line">
                      <span>{t('Trust')}</span>
                      <span className="memory-trust-value">
                        {Math.round(overview.relationship.trustLevel * 100)}%
                      </span>
                    </div>
                    <div className="memory-trust-track">
                      <div
                        className="memory-trust-fill"
                        style={{ width: `${Math.round(overview.relationship.trustLevel * 100)}%` }}
                      />
                    </div>

                    {overview.relationship.toneTrend && (
                      <span
                        className={`memory-tone-badge memory-tone-badge-${overview.relationship.toneTrend}`}
                      >
                        {overview.relationship.toneTrend === 'low'
                          ? t("you've seemed low lately")
                          : t('upbeat lately')}
                      </span>
                    )}

                    {overview.relationship.lastFelt && (
                      <p className="memory-last-felt">
                        {t('Last conversation felt')}:{' '}
                        <strong>{overview.relationship.lastFelt.label}</strong>
                        {overview.relationship.lastFelt.subject
                          ? ` — ${overview.relationship.lastFelt.subject}`
                          : ''}
                      </p>
                    )}

                    <BipolarBar
                      value={overview.relationship.arcCuriosity}
                      leftLabel={t('annoyance')}
                      rightLabel={t('curiosity')}
                    />
                    <BipolarBar
                      value={overview.relationship.arcLogic}
                      leftLabel={t('emotion')}
                      rightLabel={t('logic')}
                    />
                    <BipolarBar
                      value={overview.relationship.arcPride}
                      leftLabel={t('vulnerability')}
                      rightLabel={t('pride')}
                    />

                    <button
                      type="button"
                      className={`memory-reset-arc ${resetArmed ? 'armed' : ''}`}
                      onClick={() => void resetRelationship()}
                      onBlur={() => setResetArmed(false)}
                      disabled={resetBusy}
                      aria-label={resetArmed ? t('Confirm resetting the relationship arc') : t('Reset arc')}
                      title={resetArmed ? t('Click again to confirm') : t('Reset the relationship arc')}
                    >
                      {resetArmed ? t('CONFIRM?') : t('Reset arc')}
                    </button>
                  </div>
                </section>
              )}

              <InsightsSection />

              {overview.facts.length > 0 && (
                <section className="memory-section">
                  <h3>{t('Memories')}</h3>
                  <ul className="memory-fact-list">
                    {overview.facts.map((fact, index) => (
                      <li key={fact.id ?? `${fact.text}-${index}`} className={fact.pinned ? 'pinned' : ''}>
                        {editingId && editingId === fact.id ? (
                          <form
                            className="memory-fact-edit-row"
                            onSubmit={(event) => {
                              event.preventDefault()
                              saveEdit('facts')
                            }}
                          >
                            <input
                              type="text"
                              value={editingText}
                              onChange={(event) => setEditingText(event.target.value)}
                              aria-label={`${t('Edit memory:')} ${fact.text}`}
                              maxLength={500}
                              autoFocus
                            />
                            <button type="submit">{t('Save')}</button>
                            <button type="button" className="cancel" onClick={cancelEdit}>
                              {t('Cancel')}
                            </button>
                          </form>
                        ) : (
                          <>
                            <span className="memory-fact-text">{fact.text}</span>
                            {fact.id && (
                              <button
                                type="button"
                                className="memory-pin-button"
                                aria-pressed={fact.pinned}
                                onClick={() => void pinRecord('facts', fact.id as string, !fact.pinned)}
                                aria-label={fact.pinned ? `${t('Unpin:')} ${fact.text}` : `${t('Pin:')} ${fact.text}`}
                                title={fact.pinned ? t('Unpin this memory') : t('Pin this memory')}
                              >
                                📌
                              </button>
                            )}
                            {fact.id && (
                              <button
                                type="button"
                                className="memory-edit-button"
                                onClick={() => beginEdit(fact)}
                                aria-label={`${t('Edit:')} ${fact.text}`}
                                title={t('Edit this memory')}
                              >
                                ✎
                              </button>
                            )}
                            <button
                              type="button"
                              className="memory-forget-button"
                              onClick={() => void forgetFact(fact)}
                              aria-label={`${t('Forget:')} ${fact.text}`}
                              title={t('Make her forget this')}
                            >
                              ×
                            </button>
                          </>
                        )}
                        {fact.id && protectedIds[fact.id] && (
                          <p className="memory-fact-protected">
                            {t('That memory is pinned — unpin it first.')}
                          </p>
                        )}
                      </li>
                    ))}
                  </ul>
                </section>
              )}

              {overview.episodes.length > 0 && (
                <section className="memory-section">
                  <h3>{t('Recent moments')}</h3>
                  <ul className="memory-episode-list">
                    {overview.episodes.map((episode, index) => (
                      <li key={episode.id ?? `${episode.text}-${index}`} className={episode.pinned ? 'pinned' : ''}>
                        <span className="memory-fact-text">{episode.text}</span>
                        {episode.id && (
                          <button
                            type="button"
                            className="memory-pin-button"
                            aria-pressed={episode.pinned}
                            onClick={() => void pinRecord('episodes', episode.id as string, !episode.pinned)}
                            aria-label={episode.pinned ? `${t('Unpin:')} ${episode.text}` : `${t('Pin:')} ${episode.text}`}
                            title={episode.pinned ? t('Unpin this memory') : t('Pin this memory')}
                          >
                            📌
                          </button>
                        )}
                        <button
                          type="button"
                          className="memory-forget-button"
                          onClick={() => void forgetEpisode(episode)}
                          aria-label={`${t('Forget:')} ${episode.text}`}
                          title={t('Make her forget this')}
                        >
                          ×
                        </button>
                        {episode.id && protectedIds[episode.id] && (
                          <p className="memory-fact-protected">
                            {t('That memory is pinned — unpin it first.')}
                          </p>
                        )}
                      </li>
                    ))}
                  </ul>
                </section>
              )}

              {overview.profile.openThreads.length > 0 && (
                <section className="memory-section">
                  <h3>{t('Open threads')}</h3>
                  <ul className="memory-fact-list">
                    {overview.profile.openThreads.map((thread, index) => (
                      <li key={`${thread}-${index}`}>
                        <span className="memory-fact-text">{thread}</span>
                        <button
                          type="button"
                          className="memory-forget-button"
                          onClick={() => void forgetThread(thread)}
                          aria-label={`${t('Drop thread:')} ${thread}`}
                          title={t('Stop her circling back to this')}
                        >
                          ×
                        </button>
                      </li>
                    ))}
                  </ul>
                </section>
              )}
                </>
              )}
            </>
          )}

          {state === 'ready' && overview && diary && (
            <section className="memory-section">
              <h3>{t('Her diary')}</h3>
              <div className="memory-diary-card">
                <div className="memory-diary-header">
                  <span className="memory-diary-date">
                    {/* Local date: the backend stamps diary dates from the
                        LOCAL day, so the UTC stamp mislabeled today's entry
                        all morning east of Greenwich. */}
                    {diary.date === localDate() ? t('Today') : diary.date}
                  </span>
                </div>
                <p className="memory-diary-text">{diary.text}</p>
                <p className="memory-diary-footer">
                  {t("Her own note about her day — you weren't supposed to see this.")}
                </p>
              </div>
            </section>
          )}

          {state !== 'loading' && (
            <section className="memory-section memory-backup-section">
              <h3>{t('Backups')}</h3>
              <p className="memory-note">
                {t(
                  "Her memories of you live on this machine. A rolling backup is taken daily; export keeps a copy that's yours.",
                )}
              </p>
              <div className="memory-backup-actions">
                <button
                  type="button"
                  className="memory-backup-button"
                  onClick={() => void exportMemory()}
                  disabled={exportBusy}
                >
                  {exportBusy ? t('Exporting…') : t('Export memory')}
                </button>
                <button
                  type="button"
                  className={`memory-backup-button${importArmed ? ' armed' : ''}`}
                  onClick={() => {
                    if (!importArmed) {
                      setImportArmed(true)
                      return
                    }
                    setImportArmed(false)
                    importInputRef.current?.click()
                  }}
                  disabled={importBusy}
                >
                  {importBusy
                    ? t('Importing…')
                    : importArmed
                      ? t('Replaces her current memory — choose file')
                      : t('Import memory')}
                </button>
                <input
                  ref={importInputRef}
                  type="file"
                  accept=".zip,application/zip"
                  style={{ display: 'none' }}
                  aria-label={t('Choose a memory export to import')}
                  onChange={(event) => {
                    const file = event.target.files?.[0]
                    event.target.value = ''
                    if (file) void importMemory(file)
                  }}
                />
              </div>
              {backupNotice && <p className="memory-note">{backupNotice}</p>}
            </section>
          )}
        </div>

        {state !== 'unavailable' && state !== 'loading' && (
          <form
            className="memory-remember-row"
            onSubmit={(event) => {
              event.preventDefault()
              void rememberFact()
            }}
          >
            <input
              type="text"
              value={rememberText}
              onChange={(event) => setRememberText(event.target.value)}
              placeholder={t('Remember something…')}
              aria-label={t('Remember something')}
              maxLength={500}
            />
            <button type="submit" disabled={!rememberText.trim() || rememberBusy}>
              {t('Remember')}
            </button>
          </form>
        )}
      </aside>
    </div>
  )
}

export default MemoryPanel
