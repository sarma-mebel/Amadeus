/**
 * Reusable "nothing here yet" treatment: centered icon + title + optional
 * sub-line, gentle fade-in on mount. Replaces the plain grey `<p>` text that
 * used to stand in for empty/undesigned states across the chat surfaces.
 *
 * Copy is passed in by the caller, never hardcoded here — callers own
 * character-awareness (e.g. reading `backendStatus.character.name`) since
 * this component has no route to that data itself.
 */
import type { ReactNode } from 'react'
import './EmptyState.css'

export type EmptyStateIcon = 'chat' | 'memory' | 'synapse' | 'sessions'

/** Small inline motifs, ~56px, single-stroke to match the UI's line weight.
 * currentColor so they inherit the muted accent color set via CSS. */
const ICONS: Record<EmptyStateIcon, ReactNode> = {
  // Empty flask: chat invitation — "nothing brewing yet, say something".
  chat: (
    <svg viewBox="0 0 56 56" fill="none" aria-hidden="true">
      <path
        d="M23 8h10M25 8v14.5L15.5 40a4 4 0 0 0 3.5 6h18a4 4 0 0 0 3.5-6L31 22.5V8"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M20.5 33h15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <circle cx="28" cy="39" r="1.6" fill="currentColor" />
      <circle cx="32.5" cy="36.5" r="1" fill="currentColor" />
    </svg>
  ),
  // Small node cluster: a memory graph with nothing recorded yet.
  memory: (
    <svg viewBox="0 0 56 56" fill="none" aria-hidden="true">
      <circle cx="28" cy="16" r="3.5" stroke="currentColor" strokeWidth="2" />
      <circle cx="16" cy="36" r="3.5" stroke="currentColor" strokeWidth="2" />
      <circle cx="40" cy="36" r="3.5" stroke="currentColor" strokeWidth="2" />
      <circle cx="28" cy="44" r="2.4" stroke="currentColor" strokeWidth="2" opacity="0.55" />
      <path
        d="M25.6 18.9 18.2 33M30.4 18.9 37.8 33M18.6 38.6 25.2 42.6M37.4 38.6 30.8 42.6"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        opacity="0.6"
      />
    </svg>
  ),
  // A wider, sparser node web: the synapse map with no records to plot.
  synapse: (
    <svg viewBox="0 0 56 56" fill="none" aria-hidden="true">
      <circle cx="14" cy="20" r="3" stroke="currentColor" strokeWidth="2" />
      <circle cx="42" cy="16" r="3" stroke="currentColor" strokeWidth="2" />
      <circle cx="28" cy="32" r="4" stroke="currentColor" strokeWidth="2" />
      <circle cx="16" cy="42" r="2.6" stroke="currentColor" strokeWidth="2" opacity="0.6" />
      <circle cx="42" cy="40" r="2.6" stroke="currentColor" strokeWidth="2" opacity="0.6" />
      <path
        d="M16.5 22.3 25 29.5M39.6 18 31 29.5M25.6 35 18 40M30.6 35 38.4 38.4"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeDasharray="1 4"
        strokeLinecap="round"
        opacity="0.55"
      />
    </svg>
  ),
  // Stacked cards: an empty conversation log.
  sessions: (
    <svg viewBox="0 0 56 56" fill="none" aria-hidden="true">
      <rect x="12" y="14" width="26" height="9" rx="2.5" stroke="currentColor" strokeWidth="2" opacity="0.5" />
      <rect x="16" y="24" width="26" height="9" rx="2.5" stroke="currentColor" strokeWidth="2" opacity="0.75" />
      <rect x="20" y="34" width="26" height="9" rx="2.5" stroke="currentColor" strokeWidth="2" />
      <path d="M25 38.5h12" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  ),
}

export interface EmptyStateProps {
  icon: EmptyStateIcon
  title: string
  /** Optional second line, quieter than the title. */
  subLine?: string
  /** Extra content rendered below the copy (e.g. the suggestion chips). */
  children?: ReactNode
  className?: string
}

const EmptyState = ({ icon, title, subLine, children, className }: EmptyStateProps) => (
  <div className={`empty-state-block${className ? ` ${className}` : ''}`}>
    <div className="empty-state-icon">{ICONS[icon]}</div>
    <p className="empty-state-title">{title}</p>
    {subLine && <p className="empty-state-subline">{subLine}</p>}
    {children}
  </div>
)

export default EmptyState
