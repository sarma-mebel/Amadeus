import { describe, expect, it } from 'vitest'
import { appendLiveReply, petAckTurnId, resolveTurnImages, stripInterruptedMarker } from './petTurn'

describe('appendLiveReply', () => {
  it('starts the running caption with the first sentence verbatim', () => {
    expect(appendLiveReply('', 'Hi.')).toBe('Hi.')
  })

  it('appends later sentences with a single space separator', () => {
    expect(appendLiveReply('Hi.', 'How are you?')).toBe('Hi. How are you?')
  })

  it('chains across many sentences the same way a full turn would', () => {
    let acc = ''
    for (const s of ['Hmph.', "It's not like I", 'wanted to help.']) {
      acc = appendLiveReply(acc, s)
    }
    expect(acc).toBe("Hmph. It's not like I wanted to help.")
  })
})

describe('stripInterruptedMarker', () => {
  it('strips the trailing internal marker', () => {
    expect(stripInterruptedMarker('Wait, I—  [interrupted]')).toBe('Wait, I—')
  })

  it('leaves ordinary replies untouched', () => {
    expect(stripInterruptedMarker('Hmph. Fine.')).toBe('Hmph. Fine.')
  })

  it('returns null for an empty reply', () => {
    expect(stripInterruptedMarker('')).toBeNull()
  })

  it('returns null for null/undefined input', () => {
    expect(stripInterruptedMarker(null)).toBeNull()
    expect(stripInterruptedMarker(undefined)).toBeNull()
  })

  it('returns null when the reply is ONLY the marker', () => {
    expect(stripInterruptedMarker('[interrupted]')).toBeNull()
    expect(stripInterruptedMarker('  [interrupted]')).toBeNull()
  })

  it('only strips a trailing marker, not one embedded mid-string', () => {
    // Defensive: the regex is anchored with $, so a marker-shaped substring
    // in the middle of a reply (unlikely, but not impossible from model
    // output) must not be silently eaten.
    const withMiddle = 'He said [interrupted] and then continued.'
    expect(stripInterruptedMarker(withMiddle)).toBe(withMiddle)
  })
})

describe('petAckTurnId', () => {
  it('formats a pet-prefixed id from the post-increment counter value', () => {
    expect(petAckTurnId(1)).toBe('pet-1')
    expect(petAckTurnId(2)).toBe('pet-2')
  })

  it('produces distinct ids across a sequence, mirroring ++ackTurnRef.current', () => {
    let counter = 0
    const ids = [1, 2, 3].map(() => petAckTurnId(++counter))
    expect(ids).toEqual(['pet-1', 'pet-2', 'pet-3'])
    expect(new Set(ids).size).toBe(3)
  })
})

describe('resolveTurnImages', () => {
  const frame = { mime: 'image/png', data: 'abc123' }

  it('attaches the last page frame when read-together is active and a frame exists', () => {
    expect(resolveTurnImages(true, frame)).toEqual([frame])
  })

  it('returns undefined when read-together is inactive, even with a cached frame', () => {
    expect(resolveTurnImages(false, frame)).toBeUndefined()
  })

  it('returns undefined when read-together is active but no page has been captured yet', () => {
    expect(resolveTurnImages(true, null)).toBeUndefined()
  })

  it('returns undefined when neither condition holds', () => {
    expect(resolveTurnImages(false, null)).toBeUndefined()
  })
})
