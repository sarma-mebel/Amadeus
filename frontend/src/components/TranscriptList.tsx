/**
 * The transcript's row list, extracted verbatim from ChatInterface and
 * wrapped in React.memo: the messages.map over MessageRow used to
 * re-reconcile inside the full ChatInterface body on every ~16ms token
 * flush AND on every unrelated state change (settings panel, mic state,
 * call HUD…). Memoized here, only props the rows actually consume can
 * trigger a re-walk of the row array — the per-row handlers come in via
 * useStableHandler in ChatInterface, so their identity never changes.
 * Deliberately no virtualization.
 */
import { memo } from 'react'
import MessageRow, { EditableMessageRow } from './MessageRow'
import type { Message } from '../chat/messages'

/** Mirror of ChatInterface's `editing` state: the user message currently
 * being edited inline plus the live textarea value. */
export interface TranscriptEditingState {
  target: Message
  text: string
}

interface TranscriptListProps {
  messages: Message[]
  /** Per-index "starts a new visual group" flags (see groupStarts memo). */
  groupStarts: boolean[]
  lastAssistantIndex: number
  lastTurnId: string | undefined
  busy: boolean
  /** True while the live turn is actively streaming tokens — drives the
   * blinking caret liveness cue on the last assistant row (A3). */
  isStreaming: boolean
  /** Where restored history ends and the live session begins — the row at
   * this index shows the "new since you last opened" divider. */
  restoredCount: number
  assistantName: string
  editing: TranscriptEditingState | null
  onEditTextChange: (text: string) => void
  onEditSave: () => void
  onEditCancel: () => void
  onRetry: (message: Message) => void
  onRegenerate: (message: Message) => void
  onSwipe: (message: Message, delta: 1 | -1) => void
  onEdit: (message: Message) => void
  onRewind: (message: Message) => void
}

const TranscriptList = ({
  messages,
  groupStarts,
  lastAssistantIndex,
  lastTurnId,
  busy,
  isStreaming,
  restoredCount,
  assistantName,
  editing,
  onEditTextChange,
  onEditSave,
  onEditCancel,
  onRetry,
  onRegenerate,
  onSwipe,
  onEdit,
  onRewind,
}: TranscriptListProps) => (
  <>
    {messages.map((message, index) =>
      editing && editing.target === message ? (
        <EditableMessageRow
          key={message.id ?? index}
          text={editing.text}
          onTextChange={onEditTextChange}
          onSave={onEditSave}
          onCancel={onEditCancel}
        />
      ) : (
        <MessageRow
          key={message.id ?? index}
          message={message}
          isGroupStart={groupStarts[index]}
          showRestoredDivider={index === restoredCount && index > 0 && index < messages.length}
          onRetry={onRetry}
          onRegenerate={onRegenerate}
          onSwipe={onSwipe}
          isLastAssistant={index === lastAssistantIndex}
          isStreaming={isStreaming}
          busy={busy}
          onEdit={onEdit}
          canEdit={Boolean(message.turnId) && message.turnId === lastTurnId}
          onRewind={onRewind}
          assistantName={assistantName}
        />
      ),
    )}
  </>
)

export default memo(TranscriptList)
