/**
 * "Diagnostics" settings section: a plain debugging surface, not a
 * dashboard. Two independent counter sources, side by side, plus a latency
 * table:
 *
 *  - CLIENT counters (diagnostics.ts): in-memory, page-lifetime, populated
 *    by the instrumented silent paths (pet poke/courtesy-poll/read-together
 *    failures, presence request failures, missing filler manifest…). Live
 *    via diagnostics.subscribe — no polling needed, it only changes when
 *    something actually fails.
 *  - SERVER counters (amadeus/telemetry.py via GET /api/status): fetched
 *    once on mount, same as every other status consumer in this codebase
 *    (statusApi.fetchBackendStatus, null on any failure/older backend).
 *  - LATENCY (amadeus/telemetry.py record_duration, same /api/status
 *    response's `durations` key): count/avg/max/last per pipeline stage,
 *    plus p50/p95 over the backend's bounded recent-sample window when the
 *    backend is new enough to send them (columns omitted otherwise).
 *    Rides the same single fetch as the server counters above — no extra
 *    request. Absent entirely on older backends (quietly hidden, same
 *    convention as the rest of this panel).
 *
 * Recall-health counters (memory.recall_embed_timeouts & friends) get an
 * amber row + a one-line human hint when nonzero — recall was once silently
 * dead for a week with a climbing counter as the only symptom. All-zeros
 * stays visually quiet.
 *
 * PROVIDER HEALTH (amadeus/provider_health.py, same /api/status fetch,
 * `llm_health`/`tts_health` keys): a small "is it OK right now" block for
 * each provider, side by side. 'ok' renders quietly (a plain state line);
 * llm_health's non-ok states already surface as a separate persistent
 * banner elsewhere (ChatInterface.tsx) so this is a secondary, low-emphasis
 * view of the same field. tts_health's 'degraded'/'failing' states are
 * amber-highlighted with a short hint — voice death took 5 separate root
 * causes to diagnose once (PR #13) specifically because nothing stated
 * plainly that synthesis was failing repeatedly. 'warming' and 'off' are
 * normal, non-alarming states and render plainly, not amber.
 *
 * SELF-EVAL (amadeus/self_eval.py via the same /api/status fetch, `self_eval`
 * key): a small additive block showing the nightly conversation-quality
 * check's last run time and per-scenario scores, amber when the last run
 * flagged a regression. Optional field — absent entirely on older backends
 * or when the job has never run (renders nothing), same "quiet unless there
 * is something to say" convention as everything else in this panel.
 *
 * All-zeros / empty is the healthy state — this exists so a silently dead
 * feature (a 422'ing poke, an ACL-refused screenshot invoke) shows up
 * somewhere instead of just reading as "nothing to say".
 */
import { useEffect, useState } from 'react'
import { apiFetch } from '../../api'
import { diagnostics, type DiagnosticsSnapshot } from '../../diagnostics'
import { t } from '../../i18n'
import { fetchBackendStatus } from '../../statusApi'
import type { BackendStatus, SelfTestReport } from '../../statusApi'

const formatTime = (ms: number): string => {
  try {
    return new Date(ms).toLocaleTimeString()
  } catch {
    return ''
  }
}

/** Stage keys in display order: the headline silence-to-voice number
 * (turn.first_audio_ms) leads and is visually emphasized — everything else
 * is a diagnostic breakdown of where that time actually went. Unknown keys
 * a newer backend might add later still render, just appended after these
 * (see the sort below), so the table never silently drops a stage. */
const DURATION_KEY_ORDER = [
  'turn.first_audio_ms',
  'stt.request_ms',
  'llm.first_token_ms',
  'llm.stream_total_ms',
  'tts.first_audio_ms',
]

const orderedDurationNames = (durations: Record<string, unknown>): string[] => {
  const known = DURATION_KEY_ORDER.filter((key) => key in durations)
  const rest = Object.keys(durations)
    .filter((key) => !DURATION_KEY_ORDER.includes(key))
    .sort()
  return [...known, ...rest]
}

const round = (value: number): number => Math.round(value)

/** The base entry shape plus the p50/p95 fields newer backends compute over
 * a bounded recent-sample window (amadeus/telemetry.py). Local widening of
 * statusApi's durations type — optional so an older backend's payload (no
 * percentile fields) still parses and simply hides the columns. */
type DurationEntry = {
  count: number
  total_ms: number
  max_ms: number
  last_ms: number
  p50_ms?: number
  p95_ms?: number
}

/** Server counters that mean memory recall is quietly degrading. Nonzero →
 * amber row + the hint below the table; zero/absent → nothing (all-zeros is
 * the documented healthy state). The embed-deadline incident (recall
 * silently dead for a week, this counter the only symptom) is why these are
 * called out by name instead of reading as anonymous numbers. */
const RECALL_HEALTH_HINTS: Record<string, string> = {
  'memory.recall_embed_timeouts':
    'per-turn memory recall may be timing out — check the embedder / network',
  'memory.recall_dedupe_embed_timeouts':
    'recall summary-dedupe embeds are timing out — check the embedder / network',
  'memory.profile_load_corrupt': 'a memory file was quarantined as corrupt',
  'memory.vector_store_load_corrupt': 'a memory file was quarantined as corrupt',
}

/** Short human hint shown under tts_health when it's in an amber state.
 * 'ok'/'warming'/'off' need no hint — they render as a plain state line. */
const TTS_HEALTH_HINTS: Record<string, string> = {
  degraded: 'voice degraded — check GPU/VRAM, see logs',
  failing: 'voice failing — check GPU/VRAM, see logs',
}

// Status glyphs rather than colour alone: the report has to be readable
// without relying on hue (and survives a monochrome terminal screenshot in
// a bug report, which is how most of these will actually reach us).
const SELF_TEST_MARK: Record<string, string> = {
  pass: '✓',
  warn: '!',
  fail: '✕',
  skip: '–',
}

const DiagnosticsSection = () => {
  const [clientSnapshot, setClientSnapshot] = useState<DiagnosticsSnapshot>(() =>
    diagnostics.snapshot(),
  )
  useEffect(() => {
    setClientSnapshot(diagnostics.snapshot())
    return diagnostics.subscribe(setClientSnapshot)
  }, [])

  const [serverCounters, setServerCounters] = useState<Record<string, number> | null>(null)
  const [durations, setDurations] = useState<BackendStatus['durations'] | null>(null)
  const [selfEval, setSelfEval] = useState<BackendStatus['self_eval'] | null>(null)
  const [llmHealth, setLlmHealth] = useState<BackendStatus['llm_health'] | null>(null)
  const [ttsHealth, setTtsHealth] = useState<BackendStatus['tts_health'] | null>(null)
  const [serverLoaded, setServerLoaded] = useState(false)
  useEffect(() => {
    let cancelled = false
    void fetchBackendStatus().then((status) => {
      if (cancelled) return
      setServerCounters(status?.counters ?? null)
      // Optional field — undefined on older backends, {} on a fresh process
      // that hasn't completed a turn yet. Both fold to null so the section
      // hides the same way "no server block at all" does.
      setDurations(status?.durations ?? null)
      // Optional field — undefined on older backends, or present but with
      // last_run_at: null when the nightly job is disabled/hasn't run yet.
      // Render nothing in either case (see the block below).
      setSelfEval(status?.self_eval ?? null)
      // Provider health (amadeus/provider_health.py): both optional, absent
      // entirely on older backends. llm_health already drives a separate
      // persistent banner elsewhere (ChatInterface.tsx); it's repeated here
      // so Diagnostics has one place showing both provider healths
      // side by side, same "is it OK right now" question tts_health answers
      // for voice.
      setLlmHealth(status?.llm_health ?? null)
      setTtsHealth(status?.tts_health ?? null)
      setServerLoaded(true)
    })
    return () => {
      cancelled = true
    }
  }, [])

  // Self-test (POST /api/self_test, amadeus/web/self_test.py). Manual only:
  // it round-trips the live LLM and synthesizes audio, so it costs tokens
  // and GPU time and must never fire on mount or on a poll.
  const [selfTest, setSelfTest] = useState<SelfTestReport | null>(null)
  const [selfTestRunning, setSelfTestRunning] = useState(false)
  const [selfTestError, setSelfTestError] = useState<string | null>(null)
  const runSelfTest = () => {
    setSelfTestRunning(true)
    setSelfTestError(null)
    void apiFetch('/api/self_test', { method: 'POST' })
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(String(res.status)))))
      .then((report: SelfTestReport) => setSelfTest(report))
      .catch((err: unknown) => {
        // The backend guarantees no leg raises, so a failure here means the
        // request never landed — worth saying plainly rather than showing
        // an empty panel.
        diagnostics.increment('self_test.request_failed', String(err))
        setSelfTestError(
          t('Could not reach the backend. Is it running?'),
        )
      })
      .finally(() => setSelfTestRunning(false))
  }

  const clientNames = Object.keys(clientSnapshot.counters).sort()
  const serverNames = serverCounters ? Object.keys(serverCounters).sort() : []
  const allZero =
    clientNames.length === 0 && (!serverCounters || serverNames.every((n) => serverCounters[n] === 0))
  const durationNames = durations ? orderedDurationNames(durations) : []
  // Percentile columns only when the backend actually sends the fields —
  // an older backend's entries simply lack them and the table keeps its
  // original four stat columns.
  const hasPercentiles = durationNames.some((name) => {
    const entry = durations?.[name] as DurationEntry | undefined
    return entry?.p50_ms !== undefined
  })
  // Nonzero recall-health counters, in table (sorted-name) order.
  const recallAlerts = serverCounters
    ? serverNames.filter((name) => name in RECALL_HEALTH_HINTS && serverCounters[name] > 0)
    : []

  return (
    <section className="settings-section">
      <h3>{t('Diagnostics')}</h3>
      <div className="settings-field-group">
        <div className="settings-hint">
          {t('All zeros (or empty) means everything is working normally.')}
        </div>

        {/* Self-test sits FIRST: on a broken install this is the only block
            that says what to do, and counters below it are noise until the
            basics work. */}
        <div className="settings-diagnostics-block">
          <div className="settings-diagnostics-label">{t('Self-test')}</div>
          <div className="settings-hint">
            {t('Checks chat, voice, microphone, and memory end to end.')}
          </div>
          <button
            type="button"
            className="settings-primary-button"
            onClick={runSelfTest}
            disabled={selfTestRunning}
          >
            {selfTestRunning ? t('Checking…') : t('Run self-test')}
          </button>
          {selfTestError && <div className="settings-hint">{selfTestError}</div>}
          {selfTest && (
            <div className={`self-test-report self-test-${selfTest.summary.status}`}>
              <div className="self-test-headline">{selfTest.summary.headline}</div>
              {selfTest.summary.next_step && (
                <div className="self-test-next-step">{selfTest.summary.next_step}</div>
              )}
              <table className="settings-diagnostics-table">
                <tbody>
                  {selfTest.legs.map((leg) => (
                    <tr key={leg.key} className={`self-test-leg-${leg.status}`}>
                      <td>{leg.label}</td>
                      <td>{SELF_TEST_MARK[leg.status] ?? leg.status}</td>
                      <td>
                        {leg.detail}
                        {leg.fix && leg.status !== 'pass' && (
                          <div className="self-test-fix">{leg.fix}</div>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {durationNames.length > 0 && (
          <div className="settings-diagnostics-block">
            <div className="settings-diagnostics-label">{t('Latency')}</div>
            <table className="settings-diagnostics-table settings-latency-table">
              <thead>
                <tr>
                  <th>{t('Stage')}</th>
                  <th>{t('Count')}</th>
                  <th>{t('Avg')}</th>
                  {hasPercentiles && <th>{t('p50')}</th>}
                  {hasPercentiles && <th>{t('p95')}</th>}
                  <th>{t('Max')}</th>
                  <th>{t('Last')}</th>
                </tr>
              </thead>
              <tbody>
                {durationNames.map((name) => {
                  const entry = durations?.[name] as DurationEntry | undefined
                  if (!entry) return null
                  const avg = entry.count > 0 ? entry.total_ms / entry.count : 0
                  return (
                    <tr
                      key={name}
                      className={
                        name === 'turn.first_audio_ms' ? 'settings-latency-headline' : undefined
                      }
                    >
                      <td>{name}</td>
                      <td className="settings-diagnostics-count">{entry.count}</td>
                      <td className="settings-diagnostics-count">{round(avg)}ms</td>
                      {hasPercentiles && (
                        <td className="settings-diagnostics-count">
                          {entry.p50_ms !== undefined ? `${round(entry.p50_ms)}ms` : '—'}
                        </td>
                      )}
                      {hasPercentiles && (
                        <td className="settings-diagnostics-count">
                          {entry.p95_ms !== undefined ? `${round(entry.p95_ms)}ms` : '—'}
                        </td>
                      )}
                      <td className="settings-diagnostics-count">{round(entry.max_ms)}ms</td>
                      <td className="settings-diagnostics-count">{round(entry.last_ms)}ms</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
            <div className="settings-hint">
              {t('Target feel: under ~1500ms average silence-to-voice (turn.first_audio_ms).')}
            </div>
          </div>
        )}

        {(llmHealth || ttsHealth) && (
          <div className="settings-diagnostics-block">
            <div className="settings-diagnostics-label">{t('Provider health')}</div>
            <table className="settings-diagnostics-table">
              <thead>
                <tr>
                  <th>{t('Provider')}</th>
                  <th>{t('State')}</th>
                </tr>
              </thead>
              <tbody>
                {llmHealth && (
                  <tr>
                    <td>{t('LLM')}</td>
                    <td>{llmHealth.state}</td>
                  </tr>
                )}
                {ttsHealth && (
                  <tr
                    className={
                      ttsHealth.state in TTS_HEALTH_HINTS
                        ? 'settings-diagnostics-alert'
                        : undefined
                    }
                  >
                    <td>{t('TTS (voice)')}</td>
                    <td>{ttsHealth.state}</td>
                  </tr>
                )}
              </tbody>
            </table>
            {ttsHealth && ttsHealth.state in TTS_HEALTH_HINTS && (
              <div className="settings-hint settings-diagnostics-alert-hint">
                {t(TTS_HEALTH_HINTS[ttsHealth.state])}
              </div>
            )}
          </div>
        )}

        <div className="settings-diagnostics-block">
          <div className="settings-diagnostics-label">{t('Client (this session)')}</div>
          {clientNames.length === 0 ? (
            <div className="settings-hint">{t('No client-side failures recorded yet.')}</div>
          ) : (
            <table className="settings-diagnostics-table">
              <thead>
                <tr>
                  <th>{t('Counter')}</th>
                  <th>{t('Count')}</th>
                  <th>{t('Last seen')}</th>
                </tr>
              </thead>
              <tbody>
                {clientNames.map((name) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td className="settings-diagnostics-count">{clientSnapshot.counters[name]}</td>
                    <td>
                      {clientSnapshot.lastDetail[name]
                        ? `${formatTime(clientSnapshot.lastSeen[name])} — ${clientSnapshot.lastDetail[name]}`
                        : formatTime(clientSnapshot.lastSeen[name])}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        <div className="settings-diagnostics-block">
          <div className="settings-diagnostics-label">{t('Server (this process)')}</div>
          {!serverLoaded && <div className="settings-hint">{t('Loading…')}</div>}
          {serverLoaded && !serverCounters && (
            <div className="settings-hint">
              {t('Not available — the server may be older or unreachable.')}
            </div>
          )}
          {serverLoaded && serverCounters && serverNames.length === 0 && (
            <div className="settings-hint">{t('No server-side failures recorded yet.')}</div>
          )}
          {serverLoaded && serverCounters && serverNames.length > 0 && (
            <table className="settings-diagnostics-table">
              <thead>
                <tr>
                  <th>{t('Counter')}</th>
                  <th>{t('Count')}</th>
                </tr>
              </thead>
              <tbody>
                {serverNames.map((name) => (
                  <tr
                    key={name}
                    className={
                      recallAlerts.includes(name) ? 'settings-diagnostics-alert' : undefined
                    }
                  >
                    <td>{name}</td>
                    <td className="settings-diagnostics-count">{serverCounters[name]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {recallAlerts.map((name) => (
            <div key={name} className="settings-hint settings-diagnostics-alert-hint">
              {name}: {t(RECALL_HEALTH_HINTS[name])}
            </div>
          ))}
        </div>

        {selfEval && selfEval.last_run_at && (
          <div className="settings-diagnostics-block">
            <div className="settings-diagnostics-label">{t('Nightly self-eval')}</div>
            <div className="settings-hint">
              {t('Last run')}: {formatTime(Date.parse(selfEval.last_run_at))}
            </div>
            {selfEval.scores && (
              <table className="settings-diagnostics-table">
                <thead>
                  <tr>
                    <th>{t('Scenario')}</th>
                    <th>{t('Score')}</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(selfEval.scores)
                    .sort()
                    .map((name) => (
                      <tr
                        key={name}
                        className={
                          selfEval.regressions && name in selfEval.regressions
                            ? 'settings-diagnostics-alert'
                            : undefined
                        }
                      >
                        <td>{name}</td>
                        <td className="settings-diagnostics-count">
                          {selfEval.scores?.[name]}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            )}
            {selfEval.regression && (
              <div className="settings-hint settings-diagnostics-alert-hint">
                {t('Conversation quality dropped vs its recent normal — check the transcript.')}
              </div>
            )}
          </div>
        )}

        {allZero && (
          <div className="settings-hint">{t('All zeros — this is the healthy state.')}</div>
        )}
      </div>
    </section>
  )
}

export default DiagnosticsSection
