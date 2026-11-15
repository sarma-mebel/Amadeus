/**
 * Small trust-level gauge shown next to the header's "Makise Kurisu" title:
 * a 20px SVG arc (fill proportional to trust) plus a percentage, opening
 * "Her memories" on click. Renders nothing until a real relationship
 * record exists (fresh/dummy-provider sessions have no trust to show).
 */
import './RelationshipGauge.css'

const SIZE = 20
const STROKE = 3
const RADIUS = (SIZE - STROKE) / 2
const CIRCUMFERENCE = 2 * Math.PI * RADIUS

interface RelationshipGaugeProps {
  trust: number | null
  sessionCount: number | null
  onClick: () => void
  celebrating?: boolean
}

const RelationshipGauge = ({ trust, sessionCount, onClick, celebrating = false }: RelationshipGaugeProps) => {
  if (trust === null) return null

  const clamped = Math.max(0, Math.min(1, trust))
  const percent = Math.round(clamped * 100)
  const dashOffset = CIRCUMFERENCE * (1 - clamped)
  const sessionLabel =
    sessionCount != null ? ` · ${sessionCount} session${sessionCount === 1 ? '' : 's'}` : ''

  return (
    <button
      type="button"
      className={`relationship-gauge${celebrating ? ' celebrating' : ''}`}
      onClick={onClick}
      aria-label={`Relationship: trust ${percent}%`}
      title={`Trust ${percent}%${sessionLabel}`}
    >
      <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`} aria-hidden="true">
        <circle
          className="relationship-gauge-track"
          cx={SIZE / 2}
          cy={SIZE / 2}
          r={RADIUS}
          strokeWidth={STROKE}
          fill="none"
        />
        <circle
          className="relationship-gauge-fill"
          cx={SIZE / 2}
          cy={SIZE / 2}
          r={RADIUS}
          strokeWidth={STROKE}
          fill="none"
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={dashOffset}
          transform={`rotate(-90 ${SIZE / 2} ${SIZE / 2})`}
        />
      </svg>
      <span className="relationship-gauge-value">{percent}</span>
    </button>
  )
}

export default RelationshipGauge
