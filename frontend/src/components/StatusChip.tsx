import { useEffect, useRef, useState } from 'react'
import type { BackendStatus } from '../statusApi'
import { deriveHealth, summarizeTools } from '../statusApi'

/** Toolbar chip showing the backend's RUNTIME provider health (from
 * /api/status): green when everything configured is actually serving,
 * amber when something silently degraded (dummy LLM, TTS fallback,
 * warming). Click for the detail popover. Renders nothing without a
 * status payload, so older backends just don't show it. */
const StatusChip = ({ status }: { status: BackendStatus | null }) => {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!open) return
    const onPointerDown = (event: PointerEvent) => {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) {
        setOpen(false)
      }
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('pointerdown', onPointerDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('pointerdown', onPointerDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [open])

  if (!status) return null
  const health = deriveHealth(status)

  const ttsLine = () => {
    if (status.tts.warming) return `warming (${status.tts.requested ?? '—'})`
    if (!status.tts.requested) return 'disabled'
    if (!status.tts.loaded) return `unavailable (${status.tts.error ?? 'load failed'})`
    const engine = status.tts.engine ? ` · ${status.tts.engine} engine` : ''
    if (status.tts.fallback_from) {
      return `${status.tts.loaded}${engine} — FALLBACK from ${status.tts.fallback_from}`
    }
    if (status.tts.engine_warning) {
      return `${status.tts.loaded}${engine} — ${status.tts.engine_warning}`
    }
    if (status.tts.loaded === 'qwen3_tts' && status.tts.engine && status.tts.engine !== 'faster') {
      return `${status.tts.loaded}${engine} — SLOW ENGINE (faster-qwen3-tts missing, ~10x slower)`
    }
    const streamFailures = status.counters?.['tts.stream_synth_failures'] ?? 0
    if (streamFailures > 0) {
      return `${status.tts.loaded}${engine} — ${streamFailures} streamed sentence${streamFailures === 1 ? '' : 's'} fell back to slow synthesis`
    }
    return `${status.tts.loaded}${engine}`
  }

  const memoryTiers = [
    status.memory.long_term && 'long-term',
    status.memory.fact_extraction && 'facts',
    status.memory.episodes && 'episodes',
    status.memory.greeting && 'greeting',
  ].filter(Boolean)

  // Handles both the legacy boolean flags and the newer registry shape;
  // null (backend predates the tools block) renders nothing extra.
  const toolsSummary = summarizeTools(status.tools)

  return (
    <div className="status-chip-root" ref={rootRef}>
      <button
        type="button"
        className={`status-chip ${health}`}
        onClick={() => setOpen((prev) => !prev)}
        aria-expanded={open}
        aria-label={`System status: ${
          status.llm.is_dummy ? 'degraded, dummy LLM active' : health === 'ok' ? 'all providers live' : 'degraded'
        }`}
        title="Provider status"
      >
        {status.llm.is_dummy ? 'DUMMY LLM' : health === 'ok' ? 'SYS OK' : 'DEGRADED'}
      </button>
      {open && (
        <div className="status-popover" role="dialog" aria-label="Provider status details">
          {status.llm.is_dummy && (
            <div className="status-dummy-warning" role="alert">
              Offline demo mode — replies come from a canned dummy LLM. Add a
              Gemini or OpenAI API key in settings.
            </div>
          )}
          <dl>
            <dt>LLM</dt>
            <dd>
              {status.llm.class}
              {status.llm.model ? ` · ${status.llm.model}` : ''}
              {status.llm.is_dummy && <span className="status-badge">DUMMY</span>}
            </dd>
            <dt>Voice</dt>
            <dd>{ttsLine()}</dd>
            <dt>STT</dt>
            <dd>
              {status.stt.loaded
                ? status.stt.provider
                : status.stt.unavailable_reason ?? `${status.stt.provider} (not loaded)`}
            </dd>
            <dt>Memory</dt>
            <dd>
              {memoryTiers.length ? memoryTiers.join(' · ') : 'short-term only'}
              {status.memory.embedder ? ` · ${status.memory.embedder}` : ''}
              {!status.memory.summary_provider_active && memoryTiers.length > 0 && (
                <span className="status-badge">INERT (no LLM)</span>
              )}
            </dd>
            {toolsSummary && (
              <>
                <dt>Tools</dt>
                <dd>
                  {!toolsSummary.enabled ? 'off' : toolsSummary.registeredLabel ?? 'unknown'}
                  {toolsSummary.unavailable.map((entry) => (
                    <span key={entry.name} className="status-tool-unavailable">
                      {entry.name} — {entry.reason}
                    </span>
                  ))}
                </dd>
              </>
            )}
            <dt>Backend</dt>
            <dd>v{status.version}</dd>
          </dl>
        </div>
      )}
    </div>
  )
}

export default StatusChip
