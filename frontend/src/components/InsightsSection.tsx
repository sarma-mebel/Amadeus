/**
 * Standing insights: the transparency surface for her nightly behavioral
 * model of the user (amadeus/insights.py, GET /api/insights).
 *
 * Why this exists at all: the insight set is injected into EVERY turn's
 * system prompt and shapes how she talks to you, and until this component
 * there was no way to see it. An invisible model of you that changes how
 * you are spoken to is the least defensible artifact in the system
 * precisely because it is the most consequential — docs/companion-ethics.md
 * promises "the memory panel shows all of it".
 *
 * The persona rule "never recite them" is a rule about HER SPEECH, not
 * about the user's ACCESS; the two are not in tension, and the backend
 * endpoint's own docstring already called itself a transparency surface.
 *
 * Shows the FULL set, verbatim, with kind tags. Partial transparency would
 * teach the user there is a hidden remainder, which is worse than none.
 * Dismissing one removes it AND records the removal, so the next nightly
 * consolidation is told not to restate it.
 */

import { useCallback, useEffect, useState } from 'react'
import { apiFetch } from '../api'
import { diagnostics } from '../diagnostics'
import { resolveUserId } from '../hooks/useSessions'
import { t } from '../i18n'

interface InsightRow {
  text: string
  kind: string
}

type LoadState = 'loading' | 'ready' | 'unavailable'

const parseInsights = (raw: unknown): InsightRow[] | null => {
  if (!raw || typeof raw !== 'object') return null
  const record = raw as { insights?: unknown }
  if (!Array.isArray(record.insights)) return null
  return record.insights
    .filter((entry): entry is { text: string; kind?: unknown } => {
      return (
        !!entry &&
        typeof entry === 'object' &&
        typeof (entry as { text?: unknown }).text === 'string' &&
        (entry as { text: string }).text.trim().length > 0
      )
    })
    .map((entry) => ({
      text: entry.text,
      kind: typeof entry.kind === 'string' ? entry.kind : 'other',
    }))
}

export function InsightsSection() {
  const [rows, setRows] = useState<InsightRow[]>([])
  const [state, setState] = useState<LoadState>('loading')
  const [busy, setBusy] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const response = await apiFetch(
          `/api/insights?user_id=${encodeURIComponent(resolveUserId())}`,
        )
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const data = await response.json()
        const parsed = parseInsights(data?.insights)
        if (cancelled) return
        if (parsed === null) {
          setState('unavailable')
          return
        }
        setRows(parsed)
        setState('ready')
      } catch (error) {
        if (cancelled) return
        // Degrade loudly: a silently missing transparency surface is the
        // exact failure this feature exists to prevent.
        diagnostics.increment('insights.fetch_failed', String(error))
        setState('unavailable')
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  const dismiss = useCallback(async (text: string) => {
    setBusy(text)
    try {
      const response = await apiFetch('/api/insights/dismiss', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: resolveUserId(), text }),
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const data = await response.json()
      if (data?.dismissed) {
        setRows((current) => current.filter((row) => row.text !== text))
      } else {
        diagnostics.increment('insights.dismiss_rejected', text.slice(0, 60))
      }
    } catch (error) {
      diagnostics.increment('insights.dismiss_failed', String(error))
    } finally {
      setBusy(null)
    }
  }, [])

  if (state === 'loading' || state === 'unavailable') return null

  return (
    <section className="memory-section memory-insights">
      <h3>{t('How she talks to you')}</h3>
      <p className="memory-insights-hint">
        {t(
          'Distilled overnight from your conversations and applied to every reply. She never brings these up on her own — but this is your record, so you can read it and remove anything you disagree with.',
        )}
      </p>
      {rows.length === 0 ? (
        <p className="memory-insights-empty">
          {t('Nothing well-supported enough to note yet.')}
        </p>
      ) : (
        <ul className="memory-insights-list">
          {rows.map((row) => (
            <li key={row.text} className="memory-insights-row">
              <span className={`memory-insight-kind memory-insight-kind-${row.kind}`}>
                {row.kind}
              </span>
              <span className="memory-insight-text">{row.text}</span>
              <button
                type="button"
                className="memory-insight-dismiss"
                onClick={() => void dismiss(row.text)}
                disabled={busy === row.text}
                aria-label={t('Remove this note')}
                title={t('Remove this note')}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

export default InsightsSection
