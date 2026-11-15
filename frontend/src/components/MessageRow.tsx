/**
 * Transcript row rendering extracted from ChatInterface: one memoized
 * MessageRow per message (bubble + group header + retry/edit/rewind/
 * regenerate/swipe actions + timestamp), and the non-memoized
 * EditableMessageRow swapped in for a user row while it's being edited
 * (M1b). Moved verbatim out of ChatInterface.tsx.
 */
import { memo, useEffect, useRef } from 'react'
import type { KeyboardEvent as ReactKeyboardEvent } from 'react'
import MessageMarkdown from './MessageMarkdown'
import type { Message } from '../chat/messages'

const timeFormatter = new Intl.DateTimeFormat(undefined, {
  hour: '2-digit',
  minute: '2-digit',
})

const fullTimeFormatter = new Intl.DateTimeFormat(undefined, {
  dateStyle: 'medium',
  timeStyle: 'short',
})

const readableTimestamp = (value: number) => timeFormatter.format(new Date(value))
const fullTimestamp = (value: number) => fullTimeFormatter.format(new Date(value))

/** Memoized so that during token streaming — where setMessages fires many
 * times a second — only the row whose message object actually changed
 * re-renders, instead of the whole transcript. */
const MessageRow = memo(function MessageRow({
  message,
  isGroupStart,
  showRestoredDivider,
  onRetry,
  onRegenerate,
  onSwipe,
  isLastAssistant,
  isStreaming,
  busy,
  onEdit,
  canEdit,
  onRewind,
  assistantName,
}: {
  message: Message
  isGroupStart: boolean
  showRestoredDivider: boolean
  onRetry?: (message: Message) => void
  onRegenerate?: (message: Message) => void
  onSwipe?: (message: Message, delta: 1 | -1) => void
  isLastAssistant?: boolean
  /** True while the live turn is streaming tokens — with isLastAssistant,
   * shows a blinking caret liveness cue so a slow stream doesn't look
   * frozen/done (A3). */
  isStreaming?: boolean
  busy?: boolean
  onEdit?: (message: Message) => void
  canEdit?: boolean
  onRewind?: (message: Message) => void
  /** Display name of the active character (BYO character packs). */
  assistantName?: string
}) {
  const speakerLabel =
    message.role === 'user'
      ? 'You'
      : message.role === 'assistant'
          ? assistantName || 'Kurisu'
          : message.role === 'agent'
              ? message.agentName || 'Council agent'
              : 'Council'

  const roundLabel = message.role === 'agent' && message.round ? `Round ${message.round}` : null
  const fullStamp = fullTimestamp(message.timestamp)
  const groupHeaderLabel = roundLabel ? `${speakerLabel} · ${roundLabel}` : speakerLabel

  const hasVariants = Boolean(message.variants && message.variants.length > 1)
  const showRegenerate =
    message.role === 'assistant' &&
    !message.isError &&
    Boolean(isLastAssistant) &&
    Boolean(message.turnId) &&
    !busy
  const showEdit = message.role === 'user' && Boolean(canEdit) && !busy
  const showRewind = message.role === 'user' && Boolean(message.turnId) && !busy
  const showActions = showRegenerate || showEdit || showRewind

  const variantCount = message.variants?.length ?? 0
  const activeVariant = message.activeVariant ?? (variantCount > 0 ? variantCount - 1 : 0)

  // Liveness cue: a blinking caret trails the last assistant bubble while its
  // turn is still streaming, so a slow stream doesn't read as frozen/done (A3).
  const showStreamingCaret =
    message.role === 'assistant' &&
    !message.isError &&
    Boolean(isLastAssistant) &&
    Boolean(isStreaming)

  return (
    <div>
      {showRestoredDivider && (
        <div className="transcript-restore-divider" role="separator">
          <span>New since you last opened Amadeus</span>
        </div>
      )}
      {isGroupStart && (
        <div className={`message-group-header ${message.role}`}>
          {groupHeaderLabel}
        </div>
      )}
      <div
        className={`message ${message.role} ${message.isError ? 'is-error' : ''}`}
        aria-label={`${speakerLabel}, ${fullStamp}${message.isError ? ', error' : ''}`}
      >
        <div className="message-content">
          {message.role === 'assistant' ? (
            // Assistant replies render as markdown (code blocks, lists,
            // links…); user/agent/system bubbles stay plain text.
            <MessageMarkdown text={message.content} />
          ) : (
            message.content
          )}
          {showStreamingCaret && (
            <span className="streaming-caret" aria-hidden="true">
              ▍
            </span>
          )}
          {message.japanese && (
            <span className="message-japanese" lang="ja">
              {message.japanese}
            </span>
          )}
          {message.thought && (
            <span className="message-thought" aria-label="unspoken thought">
              {message.thought}
            </span>
          )}
          {message.isError && message.retryText && onRetry && (
            <div className="message-error-actions">
              <button
                type="button"
                className="retry-button"
                onClick={() => onRetry(message)}
                aria-label="Retry sending the last message"
              >
                ↻ RETRY
              </button>
            </div>
          )}
          {showActions && (
            <div className={`message-actions ${hasVariants ? 'always-visible' : ''}`}>
              {showEdit && (
                <button
                  type="button"
                  className="message-action-button edit-button"
                  onClick={() => onEdit?.(message)}
                  aria-label="Edit this message"
                  title="Edit"
                >
                  ✎
                </button>
              )}
              {showRewind && (
                <button
                  type="button"
                  className="message-action-button rewind-button"
                  onClick={() => onRewind?.(message)}
                  aria-label="Rewind to before this message"
                  title="Rewind here"
                >
                  ⤺
                </button>
              )}
              {showRegenerate && (
                <>
                  {hasVariants && (
                    <button
                      type="button"
                      className="message-action-button swipe-button"
                      onClick={() => onSwipe?.(message, -1)}
                      aria-label="Show the previous reply"
                      title="Previous reply"
                      disabled={activeVariant <= 0}
                    >
                      ‹
                    </button>
                  )}
                  {hasVariants && (
                    <span className="variant-counter" aria-live="polite">
                      {activeVariant + 1}/{variantCount}
                    </span>
                  )}
                  <button
                    type="button"
                    className="message-action-button regenerate-button"
                    onClick={() => {
                      if (hasVariants && activeVariant >= variantCount - 1) {
                        onRegenerate?.(message)
                      } else if (hasVariants) {
                        onSwipe?.(message, 1)
                      } else {
                        onRegenerate?.(message)
                      }
                    }}
                    aria-label={
                      hasVariants && activeVariant < variantCount - 1
                        ? 'Show the next reply'
                        : 'Regenerate this reply'
                    }
                    title={
                      hasVariants && activeVariant < variantCount - 1
                        ? 'Next reply'
                        : 'Regenerate this reply'
                    }
                  >
                    {hasVariants && activeVariant < variantCount - 1 ? '›' : '↻'}
                  </button>
                </>
              )}
            </div>
          )}
          <span
            className="message-timestamp-hover"
            aria-hidden="true"
            title={fullStamp}
          >
            {readableTimestamp(message.timestamp)}
          </span>
        </div>
      </div>
    </div>
  )
})

export default MessageRow

/** Non-memoized inline editor swapped in for a user MessageRow while it's
 * being edited (M1b). Deliberately separate from MessageRow so the edit
 * state (which changes on every keystroke) doesn't force the memoized row
 * to lose its identity/props-equality fast path for every other message. */
export const EditableMessageRow = ({
  text,
  onTextChange,
  onSave,
  onCancel,
}: {
  text: string
  onTextChange: (text: string) => void
  onSave: () => void
  onCancel: () => void
}) => {
  const areaRef = useRef<HTMLTextAreaElement>(null)
  useEffect(() => {
    areaRef.current?.focus()
    // Cursor at the end rather than a full selection.
    const length = areaRef.current?.value.length ?? 0
    areaRef.current?.setSelectionRange(length, length)
  }, [])

  const isComposing = (event: ReactKeyboardEvent) => {
    const nativeEvent = event.nativeEvent as KeyboardEvent & { isComposing?: boolean }
    return Boolean(nativeEvent.isComposing)
  }

  return (
    <div className="message user editing">
      <div className="message-content message-edit-content">
        <textarea
          ref={areaRef}
          className="message-edit-textarea"
          value={text}
          onChange={(e) => onTextChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey && !isComposing(e)) {
              e.preventDefault()
              onSave()
            } else if (e.key === 'Escape') {
              e.preventDefault()
              onCancel()
            }
          }}
          rows={2}
        />
        <div className="message-edit-actions">
          <button type="button" className="message-edit-save" onClick={onSave}>
            Save
          </button>
          <button type="button" className="message-edit-cancel" onClick={onCancel}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  )
}
