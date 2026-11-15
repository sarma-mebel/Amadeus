/** Send-failure classification + user-facing copy.
 *
 * Three failure kinds surface in the transcript:
 *  - 'network': the fetch itself failed — backend unreachable.
 *  - 'server': the backend answered with an HTTP error status.
 *  - 'stream': a WebSocket stream died mid-reply (partial content shown).
 * Keeping the copy here (pure, testable) rather than inline in the
 * component keeps the tone consistent between the POST and stream paths.
 */

export type SendErrorKind = 'network' | 'server' | 'stream'

/** Thrown by the POST path when the backend answered but with an error
 * status, so the catch block can distinguish it from a network failure. */
export class HttpError extends Error {
  readonly status: number

  constructor(status: number) {
    super(`Request failed with HTTP ${status}`)
    this.name = 'HttpError'
    this.status = status
  }
}

/** In-character transcript copy for a failed send. */
export const describeSendError = (kind: SendErrorKind, status?: number): string => {
  switch (kind) {
    case 'network':
      return (
        'Connection failed — I can’t reach the lab’s server at all. ' +
        'Check that the Amadeus backend is running, then retry.'
      )
    case 'server': {
      const statusLabel = typeof status === 'number' && Number.isFinite(status) ? `HTTP ${status}` : 'HTTP error'
      return (
        `Server error (${statusLabel}) — the backend answered, but rejected that request. ` +
        'Not my fault. Check the backend logs, then retry.'
      )
    }
    case 'stream':
      return '(connection lost — reply may be incomplete)'
  }
}

/** Why a stream died, when the server told us (or the message gave it
 * away): the connection was fine, the upstream model failed. Absent means
 * "genuine transport loss" and keeps the connection-flavored caption.
 * `invalid_key` isn't currently sent by the WS error frame (a bad key gets
 * the in-character INVALID_KEY_REPLY as normal reply text before a stream
 * failure would ever fire — see amadeus/engine.py) but the message-sniffing
 * fallback below can still recognize it, and a future backend may start
 * threading the structured kind through, so the type and caption stay
 * ready for it rather than only handling what ships today. */
export type StreamFailureCause = 'rate_limited' | 'invalid_key' | 'provider_error'

const RATE_LIMIT_HINTS = ['429', 'rate limit', 'rate-limit', 'ratelimit', 'quota', 'resource exhausted', 'resource_exhausted']
const INVALID_KEY_HINTS = ['invalid api key', 'invalid_api_key', 'api key', 'unauthorized', '401', '403', 'permission denied', 'authentication']

/** Map the server error frame's structured `kind` (preferred) or, failing
 * that, recognizable rate-limit/invalid-key phrasing in the raw message, to
 * a cause. Returns undefined for a plain transport/socket failure. */
export const classifyStreamFailureCause = (
  kind: string | undefined,
  message: string,
): StreamFailureCause | undefined => {
  if (kind === 'rate_limited' || kind === 'invalid_key' || kind === 'provider_error') return kind
  // Old backends (or kinds this frontend doesn't recognize yet) send no
  // usable `kind`: fall back to sniffing the message. Only rate-limit and
  // invalid-key phrasing are worth string-matching — both point the user at
  // a concrete fix (wait vs. Settings) instead of the generic "connection
  // lost" default, which is the conservative fallback for everything else.
  const text = message.toLowerCase()
  if (RATE_LIMIT_HINTS.some((hint) => text.includes(hint))) return 'rate_limited'
  if (INVALID_KEY_HINTS.some((hint) => text.includes(hint))) return 'invalid_key'
  return undefined
}

/** Caption for a died stream. Short — the pet caption area is small.
 * Rate-limited/invalid-key captions point at the concrete fix (mirroring
 * the backend's RATE_LIMITED_REPLY/INVALID_KEY_REPLY register: matter-of-
 * fact, no guilt, actionable) instead of the generic connection-dropped
 * text, since the connection is fine in both cases — the provider refused. */
export const describeStreamFailure = (cause?: StreamFailureCause): string => {
  switch (cause) {
    case 'rate_limited':
      return '(rate limited — quota’s spent for now; wait for it to reset, or switch key/provider in Settings)'
    case 'invalid_key':
      return '(the provider is rejecting the API key — fix it in Settings)'
    case 'provider_error':
      return '(model error — try again in a moment)'
    default:
      return describeSendError('stream')
  }
}

/** Classify a caught error from the POST send path. */
export const classifySendError = (error: unknown): { kind: SendErrorKind; status?: number } => {
  if (error instanceof HttpError) {
    return { kind: 'server', status: error.status }
  }
  return { kind: 'network' }
}
