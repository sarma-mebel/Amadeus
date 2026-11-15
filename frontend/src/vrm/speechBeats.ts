/**
 * Speech-synchronized beat gestures (lifelike pass 2026-07-30): NVBG/
 * BEAT-style text rules at hobby scale. Each streamed sentence is
 * classified once — question, negation, or emphasis — and the matching
 * micro-gesture (head tilt + brow raise / small headshake / emphasis nod)
 * fires when that sentence's AUDIO starts playing, not when its text
 * frame arrives: text streams seconds ahead of the ordered TTS audio, so
 * gesturing at classify time would desynchronize the beat from her voice.
 * The seq → beat handoff rides the exact same per-turn path the sentence
 * emotion already takes (frameDemux → lipSync.enqueue → the
 * 'kurisu:sentence-audio-start' karaoke event).
 *
 * Pure text classification only — rate limiting and envelope shaping live
 * in proceduralBody.queueBeatGesture, which owns the head additives.
 * Rules are deliberately conservative: a missed beat is invisible, a
 * wrong one (nodding at "no", headshaking at praise) reads as broken.
 */

export type SpeechBeatKind = 'question' | 'negation' | 'emphasis'

// Sentence-leading negation, or an unambiguous negation phrase anywhere.
// Bare mid-sentence "not"/"never" stays unmatched on purpose — "you never
// know" or "not bad at all" shaking her head would be wrong more often
// than right.
const NEGATION_LEAD = /^(no|nope|never)\b/i
const NEGATION_PHRASE = /\b(no way|not at all|absolutely not|of course not|not a chance|couldn't be more wrong)\b/i

// Strong intensifiers that carry spoken emphasis. Word-bounded so
// "really?" (a question) is classified by the '?' rule first anyway.
const EMPHASIS_WORD = /\b(absolutely|definitely|exactly|precisely|obviously|certainly|completely|totally|of course)\b/i

export const classifySpeechBeat = (en: string): SpeechBeatKind | null => {
  const text = en.trim()
  if (!text) return null
  if (/\?$/.test(text)) return 'question'
  if (NEGATION_LEAD.test(text) || NEGATION_PHRASE.test(text)) return 'negation'
  if (/!$/.test(text) || EMPHASIS_WORD.test(text)) return 'emphasis'
  return null
}
