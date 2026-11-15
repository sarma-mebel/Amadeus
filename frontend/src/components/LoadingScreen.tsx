import { useCallback, useEffect, useRef, useState } from 'react'
import { resolveApiPath, resolveAssetPath } from '../config'
import { t, useLocale } from '../i18n'
import { isTauri } from '../platform'
import NixieMeter from './NixieMeter'
import './LoadingScreen.css'

/**
 * The Amadeus connection sequence — a full-screen boot overlay shown before
 * the chat, styled after the SG0 Amadeus terminal (see Loading_page_sample).
 *
 * It is also the app's real loading gate:
 *  - the diagnostic log rows are timed theatre, except VOICE SYNTHESIS,
 *    which tracks /healthz `voice` (the GPU build warms Qwen3 in the
 *    background after tasks #17);
 *  - the progress bar holds below 100% until the backend answers, and
 *    creeps while the voice is still warming;
 *  - the connect tap is the user gesture that unlocks the AudioContext
 *    (via kurisuLipSync.beginStream), so the voiced greeting can play
 *    the moment the chat appears.
 *
 * The nixie divergence meter (NixieMeter) is purely the `ready` visual: it
 * rolls while `ready` is false and settles onto 1.048596 exactly when the
 * same `ready` flag that gates the tap-to-connect prompt flips true — no
 * separate timing model, so the meter can never finish before the app
 * actually can.
 */

type VoiceState = 'unknown' | 'warming' | 'ready' | 'unavailable'

interface HealthzPayload {
  status?: string
  voice?: string
  version?: string
}

interface DiagnosticRow {
  label: string
  /** Delay before the row types in, ms from mount. */
  delayMs: number
  /** Transitional state shown until the backend link is confirmed;
   * rows settle to OK once /healthz answers. VOICE SYNTHESIS is live. */
  pendingState?: string
}

const DIAGNOSTIC_ROWS: DiagnosticRow[] = [
  { label: 'MEMORY NODES', delayMs: 300 },
  { label: 'SYNAPSE MAP', delayMs: 650 },
  { label: 'PERSONALITY CORE', delayMs: 1000, pendingState: 'INIT' },
  { label: 'EMOTIONAL LAYER', delayMs: 1350, pendingState: 'SYNC' },
  { label: 'LOGIC CIRCUIT', delayMs: 1700 },
  { label: 'VOICE SYNTHESIS', delayMs: 2050 }, // live — see voiceState
]

const AUTO_CONNECT_KEY = 'amadeus_auto_connect'

interface LoadingScreenProps {
  onConnect: () => void
}

const LoadingScreen = ({ onConnect }: LoadingScreenProps) => {
  useLocale()
  const [voiceState, setVoiceState] = useState<VoiceState>('unknown')
  const [backendUp, setBackendUp] = useState(false)
  const [version, setVersion] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [visibleRows, setVisibleRows] = useState(0)
  const [leaving, setLeaving] = useState(false)
  // The tap-to-connect gate exists to satisfy browser autoplay policy — the
  // click is the user gesture that unlocks the AudioContext. On Tauri that
  // policy is neutered at the WebView2 level (additionalBrowserArgs disables
  // autoplay-policy — see desktop-shakedown-fixes memory), so the same gate
  // is pure friction there: nothing needs unlocking, only the first-turn
  // audio unlock call above stays (harmless no-op on desktop). So: an
  // explicit stored choice ('true'/'false', from the checkbox below) always
  // wins; with nothing stored yet, default to on for Tauri and off for the
  // browser. The checkbox still writes an explicit value either way, so a
  // desktop user can opt back into the tap ceremony if they want it.
  const [autoConnect, setAutoConnect] = useState(() => {
    const stored = localStorage.getItem(AUTO_CONNECT_KEY)
    if (stored === 'true') return true
    if (stored === 'false') return false
    return isTauri()
  })
  const connectedRef = useRef(false)
  const waveformRef = useRef<HTMLCanvasElement | null>(null)
  const waveformMirrorRef = useRef<HTMLCanvasElement | null>(null)

  // ---- Backend probe: /healthz drives the real states -------------------
  useEffect(() => {
    let cancelled = false
    let timer: number | undefined
    const probe = async () => {
      try {
        const response = await fetch(resolveApiPath('/healthz'))
        const data: HealthzPayload = await response.json()
        if (cancelled) return
        setBackendUp(true)
        if (data.version) setVersion(data.version)
        const voice = data.voice
        if (voice === 'warming') {
          setVoiceState('warming')
          timer = window.setTimeout(probe, 2000)
          return
        }
        setVoiceState(voice === 'ready' ? 'ready' : 'unavailable')
      } catch {
        if (!cancelled) timer = window.setTimeout(probe, 1500)
      }
    }
    probe()
    return () => {
      cancelled = true
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [])

  // ---- Diagnostic rows type in on a stagger ------------------------------
  useEffect(() => {
    const timers = DIAGNOSTIC_ROWS.map((row, index) =>
      window.setTimeout(() => setVisibleRows(index + 1), row.delayMs)
    )
    return () => timers.forEach((t) => window.clearTimeout(t))
  }, [])

  // ---- Progress model -----------------------------------------------------
  // Theatrical ramp to 70% over ~2s, hold/creep to 92% while the voice
  // warms, snap to 100% once the voice resolves (ready or unavailable —
  // the chat is usable either way; the status line inside covers it).
  useEffect(() => {
    const startedAt = performance.now()
    let raf = 0
    const tick = () => {
      const elapsed = performance.now() - startedAt
      setProgress((prev) => {
        let target: number
        if (!backendUp) {
          target = Math.min(40, (elapsed / 1500) * 40)
        } else if (voiceState === 'warming') {
          // Asymptotic creep 70 -> 92 over ~2 minutes of warmup.
          target = Math.min(92, 70 + 22 * (1 - Math.exp(-elapsed / 45000)))
        } else if (voiceState === 'unknown') {
          target = Math.min(70, (elapsed / 2000) * 70)
        } else {
          target = 100
        }
        // Ease toward target so state flips don't jump-cut the bar.
        return prev + (target - prev) * 0.08
      })
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [backendUp, voiceState])

  const ready = backendUp && voiceState !== 'unknown' && progress > 99

  // ---- Waveform canvases (status line is flanked by a mirrored pair) -----
  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const canvases = [waveformRef.current, waveformMirrorRef.current].filter(
      (c): c is HTMLCanvasElement => c !== null
    )
    if (!canvases.length) return
    let raf = 0
    const draw = (t: number) => {
      for (const canvas of canvases) {
        const ctx = canvas.getContext('2d')
        if (!ctx) continue
        const { width, height } = canvas
        ctx.clearRect(0, 0, width, height)
        ctx.strokeStyle = 'rgba(215, 90, 101, 0.9)' // --auburn-300
        ctx.lineWidth = 1.4
        ctx.beginPath()
        const mid = height / 2
        for (let x = 0; x < width; x += 2) {
          const phase = (x / width) * Math.PI * 6 + t / 260
          // EKG-ish: mostly flat with periodic spikes.
          const spike = Math.pow(Math.max(0, Math.sin(phase)), 9)
          const y = mid - spike * mid * 0.8 + Math.sin(phase * 3.7) * 1.5
          if (x === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        }
        ctx.stroke()
      }
      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [])

  // ---- Connect ------------------------------------------------------------
  const connect = useCallback(() => {
    if (connectedRef.current) return
    connectedRef.current = true
    // The tap is our one guaranteed user gesture: resume the persistent
    // lip-sync AudioContext now so the voiced greeting can autoplay.
    try {
      const lipSync = window.kurisuLipSync
      lipSync?.beginStream?.()
      lipSync?.endStream?.()
    } catch {
      // Audio unlock is best-effort; the in-app toggle still works.
    }
    setLeaving(true)
    window.setTimeout(onConnect, 620)
  }, [onConnect])

  useEffect(() => {
    if (!ready) return
    if (autoConnect) {
      const t = window.setTimeout(connect, 500)
      return () => window.clearTimeout(t)
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Enter' || event.key === ' ') connect()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [ready, autoConnect, connect])

  // Core stability gauge: theatrical baseline (~82%, matching the reference
  // art) that nudges toward the real warmup progress once it overtakes it,
  // so the ring reads as "alive" without inventing a fake metric.
  const coreStability = Math.max(82, Math.min(99, Math.round(progress)))

  const voiceRowState = () => {
    switch (voiceState) {
      case 'ready':
        return { text: 'ONLINE', tone: 'ok' }
      case 'warming':
        return { text: 'WARMING', tone: 'pending' }
      case 'unavailable':
        return { text: 'OFFLINE', tone: 'error' }
      default:
        return { text: '·····', tone: 'pending' }
    }
  }

  const statusLine = !backendUp
    ? t('ESTABLISHING NEURAL LINK···')
    : !ready
      ? voiceState === 'warming'
        ? t('INITIALIZING PERSONALITY CONSTRUCT···')
        : t('RECONSTRUCTING COGNITIVE ARCHITECTURE···')
      : t('CONNECTION READY')

  const voice = voiceRowState()

  return (
    <div
      className={`loading-screen ${leaving ? 'leaving' : ''}`}
      onClick={ready ? connect : undefined}
      role="button"
      aria-label={ready ? t('Establish connection') : t('Loading Amadeus')}
    >
      <div className="loading-scanlines" aria-hidden="true" />

      {/* Top status bar: lab id (left) + link/network status (right). */}
      <div className="loading-topbar" aria-hidden="true">
        <div className="loading-topbar-lab">
          <svg viewBox="0 0 24 24" className="loading-lab-glyph" aria-hidden="true">
            <path
              d="M9 2h6M10 2v5.2L5.5 17a2 2 0 0 0 1.8 2.9h9.4a2 2 0 0 0 1.8-2.9L14 7.2V2"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path d="M8 14h8" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
          </svg>
          <span className="loading-topbar-lab-text">
            <span className="loading-topbar-lab-title">Future Gadget Lab — No. 004</span>
            <span className="loading-topbar-lab-sub">Private Terminal</span>
          </span>
        </div>
        <div className="loading-topbar-net">
          <span className="loading-topbar-net-text">
            <span className="loading-topbar-net-line">
              WIRELESS LINK <span className="loading-net-dot" />
            </span>
            <span className="loading-topbar-net-line muted">LAB NETWORK: OK</span>
          </span>
          <svg viewBox="0 0 24 24" className="loading-wifi-icon" aria-hidden="true">
            <path
              d="M4 9c4.5-4 11.5-4 16 0M7 12.5c2.8-2.4 6.2-2.4 9 0M10.2 16c1.1-1 2.5-1 3.6 0"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.3"
              strokeLinecap="round"
            />
            <circle cx="12" cy="19" r="1.1" fill="currentColor" />
          </svg>
          <svg viewBox="0 0 24 24" className="loading-signal-icon" aria-hidden="true">
            <rect x="3" y="14" width="3" height="6" rx="0.6" fill="currentColor" opacity="0.55" />
            <rect x="8" y="10" width="3" height="10" rx="0.6" fill="currentColor" opacity="0.75" />
            <rect x="13" y="6" width="3" height="14" rx="0.6" fill="currentColor" opacity="0.9" />
            <rect x="18" y="3" width="3" height="17" rx="0.6" fill="currentColor" />
          </svg>
        </div>
      </div>

      {/* Left info panel: subject dossier. */}
      <div className="loading-info-panel" aria-hidden="true">
        <div className="loading-info-header">
          <span className="loading-info-bracket">—</span> PROJECT: AMADEUS{' '}
          <span className="loading-info-caret">▲</span>
        </div>
        <div className="loading-info-row">
          <span className="loading-info-arrow">{'>'}</span> Subject: Kurisu
        </div>
        <div className="loading-info-row">
          <span className="loading-info-arrow">{'>'}</span> Role: Cognitive Core
        </div>
        <div className="loading-info-row">
          <span className="loading-info-arrow">{'>'}</span> Classification: Level Ω
        </div>
        <div className="loading-info-row">
          <span className="loading-info-arrow">{'>'}</span> Status: Preparing Personality
          Construct
        </div>
        {/* Physics readout: part of the dossier, not free-floating. */}
        <div className="loading-info-equations">
          <span className="loading-equation">dS ≥ 0</span>
          <span className="loading-equation">Δt = (1/τ) ln Ω</span>
          <span className="loading-equation">Ψ(t) = 𝒯[Ψ₀]</span>
        </div>
      </div>

      {/* HUD set dressing cropped from the reference art. */}
      <img
        className="loading-deco loading-deco-hexes"
        src={resolveAssetPath('/static/loading/deco_hexes.webp')}
        alt=""
        aria-hidden="true"
      />

      <div className="loading-diagnostics" aria-hidden="true">
        <span className="loading-diagnostics-title">DIAGNOSTIC LOG</span>
        {DIAGNOSTIC_ROWS.slice(0, visibleRows).map((row) => {
          const live = row.label === 'VOICE SYNTHESIS'
          // Theatrical rows show their transitional state (INIT/SYNC) until
          // the backend link is confirmed, then settle to OK.
          const settled = backendUp ? 'OK' : (row.pendingState ?? 'OK')
          const state = live ? voice.text : settled
          const tone = live ? voice.tone : state === 'OK' ? 'ok' : 'pending'
          return (
            <span key={row.label} className="loading-diagnostics-row">
              <span>{row.label}</span>
              <span className={`loading-diag-state ${tone}`}>{state}</span>
            </span>
          )
        })}
        {/* Core stability lives inside the log panel — same instrument. */}
        <div className="loading-stability">
          <span className="loading-stability-label">CORE STABILITY</span>
          <svg viewBox="0 0 100 100" className="loading-stability-ring">
            <circle
              cx="50"
              cy="50"
              r="42"
              fill="none"
              stroke="rgba(255,255,255,0.1)"
              strokeWidth="5"
            />
            <circle
              cx="50"
              cy="50"
              r="42"
              fill="none"
              stroke="var(--auburn-300)"
              strokeWidth="5"
              strokeLinecap="round"
              strokeDasharray={`${(coreStability / 100) * 263.9} 263.9`}
              transform="rotate(-90 50 50)"
              className="loading-stability-arc"
            />
            <text x="50" y="58" textAnchor="middle" className="loading-stability-value">
              {coreStability}%
            </text>
          </svg>
        </div>
      </div>

      <div className="loading-center">
        <div className="loading-portrait-halo" aria-hidden="true">
          <img
            className="loading-portrait"
            src={resolveAssetPath('/static/loading/amadeus_portrait.webp')}
            alt=""
          />
        </div>
        <h1 className="loading-wordmark">AMADEUS</h1>
        <div className="loading-subtitle">COGNITIVE MEMORY INTERFACE</div>
        <div className="loading-tagline" lang="ja">
          すべての記憶は、未来のために。
        </div>
        <div className="loading-tagline-en">“All memories for the future.”</div>

        <div className="loading-statusline-wrap">
          <canvas ref={waveformRef} className="loading-waveform" width={90} height={22} />
          <span className={`loading-statusline ${ready ? 'ready' : ''}`}>
            {ready ? t('TAP TO ESTABLISH CONNECTION') : statusLine}
          </span>
          <canvas
            ref={waveformMirrorRef}
            className="loading-waveform mirrored"
            width={90}
            height={22}
            aria-hidden="true"
          />
        </div>

        {/* Ω seal: pure vector so it has no bitmap-crop edges. */}
        <div className="loading-medallion" aria-hidden="true">
          <svg viewBox="0 0 100 100" className="loading-medallion-ring">
            <circle
              cx="50"
              cy="50"
              r="46"
              fill="none"
              stroke="rgba(var(--auburn-rgb), 0.55)"
              strokeWidth="1"
              strokeDasharray="3 5"
            />
            <circle
              cx="50"
              cy="50"
              r="38"
              fill="none"
              stroke="rgba(var(--auburn-rgb), 0.3)"
              strokeWidth="0.8"
            />
            <text x="50" y="63" textAnchor="middle" className="loading-medallion-omega">
              Ω
            </text>
          </svg>
          <span className="loading-medallion-caption">Veritas in cogitatione.</span>
        </div>

        {/* Divergence meter: the boot signature visual. Settles onto
            1.048596 only once the real load gate (backend + voice state)
            reports ready — it never fakes completion ahead of `ready`. */}
        <NixieMeter
          settled={ready}
          caption={ready ? t('READY') : t('LOADING {percent}%', { percent: Math.floor(progress) })}
        />
        <div className="loading-bar" role="progressbar" aria-valuenow={Math.floor(progress)}>
          <div className="loading-bar-fill" style={{ width: `${progress}%` }} />
        </div>

        <label
          className="loading-autoconnect"
          onClick={(event) => event.stopPropagation()}
        >
          <input
            type="checkbox"
            checked={autoConnect}
            onChange={(event) => {
              setAutoConnect(event.target.checked)
              localStorage.setItem(AUTO_CONNECT_KEY, String(event.target.checked))
            }}
          />
          {t('auto-connect on launch')}
        </label>
      </div>

      <div className="loading-footer">
        <span className="loading-footer-cell loading-footer-left">
          OS: Amadeus OS
          <br />
          Build: {version ?? '···'}
        </span>
        <span className="loading-footer-cell loading-footer-center">
          Future Gadget Laboratory
          <br />
          SERN · 2036
        </span>
        <span className="loading-footer-cell loading-footer-right">
          {version ? `v${version}` : '···'}
          <br />
          Unofficial fan project
        </span>
      </div>
    </div>
  )
}

export default LoadingScreen
