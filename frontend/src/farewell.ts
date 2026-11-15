/**
 * Goodbye-intent pre-gate for the farewell flow.
 *
 * Mirrors the server's regex in `amadeus/farewell.py` — keep the two in
 * sync. This is only a cheap client-side gate to avoid a wasted round trip;
 * the server re-validates and replies `not_goodbye` when it disagrees, in
 * which case the message falls through to a normal chat turn.
 */

export const MAX_GOODBYE_LEN = 80
export const MAX_GOODBYE_WORDS = 8

const GOODBYE_RE = new RegExp(
  '\\b(' +
    "good\\s?bye|bye+|bye\\s?now|good\\s?night|g'?night|" +
    'gotta\\s+(go|run|sleep|head\\s+(out|off))|' +
    "(i'?m|i\\s+am)\\s+(off|heading\\s+(out|off|to\\s+bed)|going\\s+to\\s+(bed|sleep))|" +
    'see\\s+(you|ya)(\\s+(later|tomorrow|soon|tonight))?|' +
    'talk\\s+(to\\s+you\\s+)?(later|tomorrow|soon)|' +
    'ttyl|signing\\s+off|logging\\s+off|heading\\s+(out|off)|' +
    'oyasumi' +
    ')\\b',
  'i',
)

export const looksLikeGoodbye = (text: string): boolean => {
  const stripped = (text ?? '').trim()
  if (!stripped || stripped.length > MAX_GOODBYE_LEN) return false
  if (stripped.split(/\s+/).length > MAX_GOODBYE_WORDS) return false
  if (stripped.endsWith('?')) return false
  return GOODBYE_RE.test(stripped)
}
