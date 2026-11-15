import { describe, expect, it } from 'vitest'
import { classifySpeechBeat } from './speechBeats'

describe('classifySpeechBeat', () => {
  it('classifies questions by the trailing question mark', () => {
    expect(classifySpeechBeat('What do you think that means?')).toBe('question')
    expect(classifySpeechBeat('Really?')).toBe('question')
    expect(classifySpeechBeat('  Are you sure?  ')).toBe('question')
  })

  it('question wins over other cues in the same sentence', () => {
    expect(classifySpeechBeat('No, seriously — are you sure about that?')).toBe('question')
    expect(classifySpeechBeat('Absolutely certain, are you?')).toBe('question')
  })

  it('classifies sentence-leading and phrase negations', () => {
    expect(classifySpeechBeat('No, that is not how entropy works.')).toBe('negation')
    expect(classifySpeechBeat('Never mind the details.')).toBe('negation')
    expect(classifySpeechBeat("That's not at all what the paper claims.")).toBe('negation')
    expect(classifySpeechBeat('There is no way that compiles.')).toBe('negation')
  })

  it('leaves ambiguous mid-sentence negations alone', () => {
    expect(classifySpeechBeat('You never know until you run the experiment.')).toBeNull()
    expect(classifySpeechBeat('It is not bad, honestly.')).toBeNull()
  })

  it('classifies emphasis by exclamation or strong intensifiers', () => {
    expect(classifySpeechBeat('That actually worked!')).toBe('emphasis')
    expect(classifySpeechBeat('That is exactly the point.')).toBe('emphasis')
    expect(classifySpeechBeat('It is obviously a caching bug.')).toBe('emphasis')
  })

  it('negation beats emphasis when both appear', () => {
    expect(classifySpeechBeat('Absolutely not.')).toBe('negation')
  })

  it('returns null for plain sentences and empty input', () => {
    expect(classifySpeechBeat('The lab is quiet this evening.')).toBeNull()
    expect(classifySpeechBeat('')).toBeNull()
    expect(classifySpeechBeat('   ')).toBeNull()
  })
})
