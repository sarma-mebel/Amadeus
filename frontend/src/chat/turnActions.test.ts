import { describe, expect, it } from 'vitest'
import { MAX_VARIANTS, buildSystemMessage, foldPendingVariants, stampTurn, variantToMessage } from './turnActions'
import type { Message } from './messages'

const userMsg = (content: string, timestamp: number): Message => ({
  role: 'user',
  content,
  timestamp,
})

const assistantMsg = (content: string, timestamp: number, id?: string): Message => ({
  role: 'assistant',
  content,
  timestamp,
  id,
})

describe('stampTurn', () => {
  it('stamps the matched assistant message and its nearest preceding user message', () => {
    const prev: Message[] = [
      userMsg('hi', 1),
      assistantMsg('hello', 2, 'a1'),
    ]
    const next = stampTurn(prev, 'a1', 't1')
    expect(next[0].turnId).toBe('t1')
    expect(next[1].turnId).toBe('t1')
  })

  it('falls back to the last assistant message when assistantId is not found (POST path)', () => {
    const prev: Message[] = [
      userMsg('hi', 1),
      assistantMsg('hello', 2),
    ]
    const next = stampTurn(prev, null, 't1')
    expect(next[0].turnId).toBe('t1')
    expect(next[1].turnId).toBe('t1')
  })

  it('stops scanning backward for the user message at an earlier assistant message boundary', () => {
    const prev: Message[] = [
      userMsg('first turn prompt', 1),
      assistantMsg('first reply', 2),
      userMsg('second turn prompt', 3),
      assistantMsg('second reply', 4, 'a2'),
    ]
    const next = stampTurn(prev, 'a2', 't2')
    // Only the second turn's pair is stamped.
    expect(next[0].turnId).toBeUndefined()
    expect(next[1].turnId).toBeUndefined()
    expect(next[2].turnId).toBe('t2')
    expect(next[3].turnId).toBe('t2')
  })

  it('restamps an already-stamped user message (post-regenerate) rather than skipping it', () => {
    const prev: Message[] = [
      { ...userMsg('hi', 1), turnId: 'stale' },
      assistantMsg('hello again', 2, 'a1'),
    ]
    const next = stampTurn(prev, 'a1', 'fresh')
    expect(next[0].turnId).toBe('fresh')
  })

  it('returns the input unchanged when there is no assistant message at all', () => {
    const prev: Message[] = [userMsg('hi', 1)]
    const next = stampTurn(prev, 'missing', 't1')
    expect(next).toBe(prev)
  })

  it('does not mutate the input array', () => {
    const prev: Message[] = [userMsg('hi', 1), assistantMsg('hello', 2, 'a1')]
    const snapshot = JSON.parse(JSON.stringify(prev))
    stampTurn(prev, 'a1', 't1')
    expect(prev).toEqual(snapshot)
  })
})

describe('buildSystemMessage', () => {
  it('builds a system-role message with the given content', () => {
    const msg = buildSystemMessage('hello')
    expect(msg.role).toBe('system')
    expect(msg.content).toBe('hello')
    expect(typeof msg.timestamp).toBe('number')
  })
})

describe('foldPendingVariants', () => {
  it('is a no-op when there is nothing pending', () => {
    const messages = [assistantMsg('hi', 1, 'a1')]
    expect(foldPendingVariants(messages, 'a1', null)).toBe(messages)
    expect(foldPendingVariants(messages, null, [{ content: 'x', timestamp: 1 }])).toBe(messages)
  })

  it('folds the pending variant plus the current content onto the matched message', () => {
    const messages = [assistantMsg('new reply', 5, 'a1')]
    const pending = [{ content: 'old reply', timestamp: 1 }]
    const [folded] = foldPendingVariants(messages, 'a1', pending)
    expect(folded.variants).toHaveLength(2)
    expect(folded.variants?.[0].content).toBe('old reply')
    expect(folded.variants?.[1].content).toBe('new reply')
    expect(folded.activeVariant).toBe(1)
  })

  it('caps the merged variant list at MAX_VARIANTS, dropping the oldest first', () => {
    const messages = [assistantMsg('newest', 100, 'a1')]
    const pending = Array.from({ length: MAX_VARIANTS }, (_, i) => ({
      content: `old-${i}`,
      timestamp: i,
    }))
    const [folded] = foldPendingVariants(messages, 'a1', pending)
    expect(folded.variants).toHaveLength(MAX_VARIANTS)
    // Oldest ("old-0") dropped; newest kept content is the live message's.
    expect(folded.variants?.[0].content).toBe('old-1')
    expect(folded.variants?.[folded.variants!.length - 1].content).toBe('newest')
  })
})

describe('variantToMessage', () => {
  it('rebuilds a Message from a variant snapshot, carrying the base turnId/id', () => {
    const variant = { content: 'restored', japanese: 'ja', thought: 'hmm', timestamp: 42 }
    const msg = variantToMessage(variant, { turnId: 't1', id: 'a1' }, [variant])
    expect(msg).toMatchObject({
      role: 'assistant',
      content: 'restored',
      japanese: 'ja',
      thought: 'hmm',
      timestamp: 42,
      turnId: 't1',
      id: 'a1',
    })
    // Single-entry variants array collapses to undefined (matches the
    // "hasVariants" UI gate of more-than-one).
    expect(msg.variants).toBeUndefined()
    expect(msg.activeVariant).toBeUndefined()
  })

  it('keeps the variants array when there is more than one', () => {
    const variants = [
      { content: 'a', timestamp: 1 },
      { content: 'b', timestamp: 2 },
    ]
    const msg = variantToMessage(variants[1], {}, variants)
    expect(msg.variants).toHaveLength(2)
    expect(msg.activeVariant).toBe(1)
  })
})
