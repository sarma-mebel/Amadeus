/**
 * "Notice your rhythms" opt-in plus the disclosure panel that renders the
 * ENTIRE stored tally (amadeus/rhythm.py, GET /api/rhythm).
 *
 * Its own section rather than another row in FeaturesSection because the
 * disclosure is the point: this is the only feature that keeps a durable
 * record of how the user spends time at their machine, and the deal is that
 * they can read every byte of it in one screen. That readability is a
 * design constraint on what the store may hold, not a nicety — if a future
 * field can't be shown here plainly, it shouldn't be stored.
 *
 * Consent is doubled (this toggle plus the server's rhythm.enabled), the
 * toggle is deliberately absent from the presence presets, and switching it
 * off ERASES the tally rather than merely halting writes (rhythmPrefs.ts).
 */

import { useCallback, useEffect, useState } from 'react'
import { apiFetch } from '../../api'
import { diagnostics } from '../../diagnostics'
import { resolveUserId } from '../../hooks/useSessions'
import { t } from '../../i18n'
import { clearRhythm, loadRhythmEnabled, setRhythmEnabled } from '../../rhythmPrefs'

interface RhythmDayRow {
  date: string
  late_night_blocks: number
  long_focus_stretches: number
  longest_focus_hours: number
  music_events: number
  app_buckets: Record<string, number>
}

const parseDays = (raw: unknown): RhythmDayRow[] => {
  if (!raw || typeof raw !== 'object') return []
  const record = raw as { days?: unknown }
  if (!Array.isArray(record.days)) return []
  return record.days.filter(
    (row): row is RhythmDayRow =>
      !!row && typeof row === 'object' && typeof (row as RhythmDayRow).date === 'string',
  )
}

const describeDay = (row: RhythmDayRow): string => {
  const parts: string[] = []
  if (row.late_night_blocks > 0) parts.push(t('up past midnight'))
  if (row.long_focus_stretches > 0) {
    parts.push(`${row.long_focus_stretches} ${t('long focus stretches')}`)
  }
  if (row.longest_focus_hours > 0) {
    parts.push(`${t('longest')} ${row.longest_focus_hours}h`)
  }
  const buckets = Object.entries(row.app_buckets || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 2)
    .map(([name]) => t(name))
  if (buckets.length) parts.push(`${t('mostly')} ${buckets.join(', ')}`)
  if (row.music_events > 0) parts.push(`${row.music_events}× ${t('music')}`)
  return parts.length ? parts.join(' · ') : t('nothing notable')
}

export function RhythmSection() {
  const [enabled, setEnabled] = useState(loadRhythmEnabled())
  const [rows, setRows] = useState<RhythmDayRow[]>([])
  const [busy, setBusy] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const response = await apiFetch(
        `/api/rhythm?user_id=${encodeURIComponent(resolveUserId())}`,
      )
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const data = await response.json()
      setRows(parseDays(data?.rhythm))
    } catch (error) {
      diagnostics.increment('rhythm.fetch_failed', String(error))
      setRows([])
    }
  }, [])

  useEffect(() => {
    if (!enabled) {
      setRows([])
      return
    }
    void refresh()
  }, [enabled, refresh])

  const toggle = useCallback(
    async (next: boolean) => {
      setBusy(true)
      setEnabled(next)
      try {
        await setRhythmEnabled(next)
        if (next) await refresh()
        else setRows([])
      } finally {
        setBusy(false)
      }
    },
    [refresh],
  )

  return (
    <section className="settings-section">
      <h3>{t('Rhythms')}</h3>
      <div className="settings-field-group">
        <label className="settings-toggle-row" htmlFor="rhythm-toggle">
          <input
            id="rhythm-toggle"
            type="checkbox"
            checked={enabled}
            disabled={busy}
            onChange={(event) => void toggle(event.target.checked)}
          />
          <span>{t('Notice your rhythms')}</span>
        </label>
        <div className="settings-hint">
          {t(
            'She keeps a two-week tally of how much, roughly when, and which kind of app — how many long focus stretches, how many late nights, how often music was on. No window titles, no document names, no track names: those never leave this machine. She may mention what she noticed at most once a week, and never as a statistic. Turning this off erases the tally. Needs the server\'s rhythm.enabled flag too.',
          )}
        </div>

        {enabled && (
          <div className="rhythm-disclosure">
            <div className="rhythm-disclosure-head">
              <span>{t('Everything she has stored')}</span>
              <button
                type="button"
                className="rhythm-clear"
                disabled={busy}
                onClick={() => {
                  setBusy(true)
                  void clearRhythm()
                    .then(() => setRows([]))
                    .finally(() => setBusy(false))
                }}
              >
                {t('Clear')}
              </button>
            </div>
            {rows.length === 0 ? (
              <p className="rhythm-empty">{t('Nothing recorded yet.')}</p>
            ) : (
              <ul className="rhythm-day-list">
                {rows.map((row) => (
                  <li key={row.date}>
                    <span className="rhythm-day-date">{row.date}</span>
                    <span className="rhythm-day-detail">{describeDay(row)}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </section>
  )
}

export default RhythmSection
