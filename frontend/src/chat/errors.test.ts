import { describe, expect, it } from 'vitest'
import {
  HttpError,
  classifySendError,
  classifyStreamFailureCause,
  describeSendError,
  describeStreamFailure,
} from './errors'

describe('describeSendError', () => {
  it('describes a network failure as unreachable', () => {
    const copy = describeSendError('network')
    expect(copy).toMatch(/connection failed/i)
    expect(copy).toMatch(/backend is running/i)
  })

  it('includes the HTTP status for server errors', () => {
    expect(describeSendError('server', 503)).toContain('HTTP 503')
    expect(describeSendError('server', 404)).toContain('HTTP 404')
  })

  it('falls back to a generic label when the status is missing', () => {
    expect(describeSendError('server')).toContain('HTTP error')
    expect(describeSendError('server', Number.NaN)).toContain('HTTP error')
  })

  it('keeps the interrupted-stream marker parenthetical', () => {
    expect(describeSendError('stream')).toBe('(connection lost — reply may be incomplete)')
  })
})

describe('classifyStreamFailureCause', () => {
  it('trusts the structured kind from the server error frame first', () => {
    expect(classifyStreamFailureCause('rate_limited', 'anything')).toBe('rate_limited')
    expect(classifyStreamFailureCause('provider_error', 'anything')).toBe('provider_error')
    // Not currently sent by the WS error frame (see errors.ts's doc
    // comment), but a structured kind still wins over sniffing if a future
    // backend starts threading it through.
    expect(classifyStreamFailureCause('invalid_key', 'anything')).toBe('invalid_key')
  })

  it('ignores unknown kinds rather than guessing from them', () => {
    expect(classifyStreamFailureCause('mystery_kind', 'plain boom')).toBeUndefined()
  })

  it('falls back to sniffing rate-limit phrasing on older backends (no kind)', () => {
    // Real-world shapes: Gemini quota errors and generic HTTP 429 text.
    expect(
      classifyStreamFailureCause(undefined, '429 RESOURCE_EXHAUSTED: Quota exceeded for quota metric'),
    ).toBe('rate_limited')
    expect(classifyStreamFailureCause(undefined, 'Rate limit reached — give it a minute')).toBe(
      'rate_limited',
    )
    expect(classifyStreamFailureCause(undefined, 'RateLimitError: too many requests')).toBe(
      'rate_limited',
    )
  })

  it('falls back to sniffing invalid-key/auth phrasing on older backends (no kind)', () => {
    expect(classifyStreamFailureCause(undefined, 'Invalid API key provided')).toBe('invalid_key')
    expect(classifyStreamFailureCause(undefined, '401 Unauthorized')).toBe('invalid_key')
    expect(classifyStreamFailureCause(undefined, 'PermissionDenied: 403')).toBe('invalid_key')
  })

  it('never invents a provider cause from an unrecognizable message', () => {
    expect(classifyStreamFailureCause(undefined, 'boom')).toBeUndefined()
    expect(classifyStreamFailureCause(undefined, '')).toBeUndefined()
  })
})

describe('describeStreamFailure', () => {
  it('keeps rate-limit and model-error captions honest, short, and parenthetical', () => {
    const rateLimited = describeStreamFailure('rate_limited')
    expect(rateLimited).toMatch(/^\(.*\)$/)
    expect(rateLimited).toMatch(/rate limited/i)
    expect(rateLimited).not.toMatch(/connection/i)

    const providerError = describeStreamFailure('provider_error')
    expect(providerError).toMatch(/^\(.*\)$/)
    expect(providerError).toMatch(/model error/i)
    expect(providerError).not.toMatch(/connection/i)
  })

  it('points the rate-limited caption at the concrete fix (quota + Settings), not a vague retry', () => {
    const rateLimited = describeStreamFailure('rate_limited')
    expect(rateLimited).toMatch(/quota/i)
    expect(rateLimited).toMatch(/settings/i)
  })

  it('gives the invalid-key caption an honest, actionable, parenthetical caption', () => {
    const invalidKey = describeStreamFailure('invalid_key')
    expect(invalidKey).toMatch(/^\(.*\)$/)
    expect(invalidKey).toMatch(/api key/i)
    expect(invalidKey).toMatch(/settings/i)
    expect(invalidKey).not.toMatch(/connection/i)
  })

  it('falls back to the connection-lost marker when no cause is known', () => {
    expect(describeStreamFailure(undefined)).toBe(describeSendError('stream'))
    expect(describeStreamFailure()).toBe('(connection lost — reply may be incomplete)')
  })
})

describe('classifySendError', () => {
  it('maps HttpError to a server error with status', () => {
    expect(classifySendError(new HttpError(500))).toEqual({ kind: 'server', status: 500 })
  })

  it('maps anything else (fetch TypeError, aborts) to network', () => {
    expect(classifySendError(new TypeError('Failed to fetch'))).toEqual({ kind: 'network' })
    expect(classifySendError('weird')).toEqual({ kind: 'network' })
  })
})
