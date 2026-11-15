import { memo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Components } from 'react-markdown'

/** Markdown renderer for assistant bubbles.
 *
 * react-markdown was chosen over `marked`/`markdown-it` because it renders a
 * React element tree directly — no HTML string, no dangerouslySetInnerHTML,
 * no sanitizer dependency — which matters here since the input is raw LLM
 * output. remark-gfm adds the pieces models emit constantly (fenced code
 * already works; GFM adds tables, strikethrough, autolinks, task lists).
 *
 * Memoized like MessageRow: during token streaming the coalesced flush
 * replaces the message object ~every 16ms, so only the live bubble re-parses;
 * settled bubbles keep their previous tree.
 */

const components: Components = {
  // All links open in a new tab — the SPA shares its window with the VRM
  // scene; navigating away would tear down the whole session.
  a: (props) => {
    const { node, children, ...rest } = props
    void node
    return (
      <a {...rest} target="_blank" rel="noopener noreferrer">
        {children}
      </a>
    )
  },
}

const MessageMarkdown = memo(function MessageMarkdown({ text }: { text: string }) {
  return (
    <div className="message-markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {text}
      </ReactMarkdown>
    </div>
  )
})

export default MessageMarkdown
