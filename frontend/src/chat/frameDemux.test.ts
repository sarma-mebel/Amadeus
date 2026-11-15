// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createFrameDemuxer } from './frameDemux'
import { kurisuBridge } from '../vrm/kurisuBridge'
import type { AudioFrame, SentenceFrame } from '../streaming'

const sentenceFrame = (overrides: Partial<SentenceFrame> = {}): SentenceFrame => ({
  seq: 0,
  en: 'Hello.',
  ja: 'こんにちは。',
  emotion: 'happy',
  emotion_intensity: 0.7,
  ...overrides,
})

const audioFrame = (overrides: Partial<AudioFrame> = {}): AudioFrame => ({
  seq: 0,
  mime: 'audio/wav',
  data: 'QUJD',
  ...overrides,
})

describe('createFrameDemuxer', () => {
  let lipSync: {
    beginStream: ReturnType<typeof vi.fn>
    enqueue: ReturnType<typeof vi.fn>
    endStream: ReturnType<typeof vi.fn>
    stop: ReturnType<typeof vi.fn>
    isSpeaking: ReturnType<typeof vi.fn>
    play: ReturnType<typeof vi.fn>
  }
  let expressionsSet: ReturnType<typeof vi.fn>
  let react: ReturnType<typeof vi.fn>

  beforeEach(() => {
    lipSync = {
      beginStream: vi.fn(),
      enqueue: vi.fn().mockResolvedValue(undefined),
      endStream: vi.fn(),
      stop: vi.fn(),
      isSpeaking: vi.fn().mockReturnValue(false),
      play: vi.fn(),
    }
    expressionsSet = vi.fn()
    react = vi.fn()
    kurisuBridge.lipSync = lipSync
    kurisuBridge.expressions = { set: expressionsSet, available: [], list: vi.fn() }
    kurisuBridge.react = react
  })

  afterEach(() => {
    kurisuBridge.lipSync = null
    kurisuBridge.expressions = null
    kurisuBridge.react = null
  })

  describe('handleSentence', () => {
    it('dispatches expression + gesture with the frame emotion and intensity', () => {
      const demux = createFrameDemuxer()
      demux.handleSentence(sentenceFrame({ emotion: 'happy', emotion_intensity: 0.7 }))

      expect(expressionsSet).toHaveBeenCalledWith('happy', 0.7, 2500)
      expect(react).toHaveBeenCalledWith('happy', 0.7)
    })

    it('prefers emotion_raw for react() when present, but expressions.set always uses the basic emotion', () => {
      const demux = createFrameDemuxer()
      demux.handleSentence(sentenceFrame({ emotion: 'happy', emotion_raw: 'giggle', emotion_intensity: 0.5 }))

      expect(expressionsSet).toHaveBeenCalledWith('happy', 0.5, 2500)
      expect(react).toHaveBeenCalledWith('giggle', 0.5)
    })

    it('defaults intensity to 0.8 when emotion_intensity is not a number', () => {
      const demux = createFrameDemuxer()
      demux.handleSentence({ ...sentenceFrame(), emotion_intensity: undefined as unknown as number })

      expect(expressionsSet).toHaveBeenCalledWith('happy', 0.8, 2500)
      expect(react).toHaveBeenCalledWith('happy', 0.8)
    })

    it('does not dispatch expression/gesture when the frame has no emotion', () => {
      const demux = createFrameDemuxer()
      demux.handleSentence(sentenceFrame({ emotion: '' }))

      expect(expressionsSet).not.toHaveBeenCalled()
      expect(react).not.toHaveBeenCalled()
    })

    it('tolerates a null kurisuBridge.expressions/react (avatar not mounted yet)', () => {
      kurisuBridge.expressions = null
      kurisuBridge.react = null
      const demux = createFrameDemuxer()

      expect(() => demux.handleSentence(sentenceFrame())).not.toThrow()
    })

    it('records the emotion per-seq so a later audio frame for the same seq is tinted correctly', () => {
      const demux = createFrameDemuxer()
      demux.handleSentence(sentenceFrame({ seq: 3, emotion: 'sad', emotion_intensity: 0.4 }))
      demux.handleAudio(audioFrame({ seq: 3 }), true)

      expect(lipSync.enqueue).toHaveBeenCalledWith(expect.objectContaining({ seq: 3, emotion: 'sad' }))
    })

    it('classifies a speech beat from the sentence text and threads it to the matching audio frame', () => {
      const demux = createFrameDemuxer()
      demux.handleSentence(sentenceFrame({ seq: 1, en: 'Are you sure about that?' }))
      demux.handleAudio(audioFrame({ seq: 1 }), true)

      expect(lipSync.enqueue).toHaveBeenCalledWith(expect.objectContaining({ seq: 1, beat: 'question' }))
    })

    it('leaves beat undefined for plain sentences', () => {
      const demux = createFrameDemuxer()
      demux.handleSentence(sentenceFrame({ seq: 2, en: 'The lab is quiet today.' }))
      demux.handleAudio(audioFrame({ seq: 2 }), true)

      expect(lipSync.enqueue).toHaveBeenCalledWith(expect.objectContaining({ seq: 2, beat: undefined }))
    })
  })

  describe('handleAudio', () => {
    it('no-ops entirely when audioEnabled is false', () => {
      const demux = createFrameDemuxer()
      demux.handleAudio(audioFrame(), false)

      expect(lipSync.beginStream).not.toHaveBeenCalled()
      expect(lipSync.enqueue).not.toHaveBeenCalled()
      expect(demux.hasAudioStarted()).toBe(false)
    })

    it('calls beginStream + onAudioStart exactly once, on the first audio frame only', () => {
      const onAudioStart = vi.fn()
      const demux = createFrameDemuxer({ onAudioStart })

      demux.handleAudio(audioFrame({ seq: 0 }), true)
      demux.handleAudio(audioFrame({ seq: 1 }), true)
      demux.handleAudio(audioFrame({ seq: 2 }), true)

      expect(lipSync.beginStream).toHaveBeenCalledTimes(1)
      expect(onAudioStart).toHaveBeenCalledTimes(1)
      expect(lipSync.enqueue).toHaveBeenCalledTimes(3)
      expect(demux.hasAudioStarted()).toBe(true)
    })

    it('enqueues audio chunks in the order handleAudio is called, preserving seq', () => {
      const demux = createFrameDemuxer()
      demux.handleAudio(audioFrame({ seq: 0, data: 'AA' }), true)
      demux.handleAudio(audioFrame({ seq: 1, data: 'BB' }), true)
      demux.handleAudio(audioFrame({ seq: 2, data: 'CC' }), true)

      const seqs = lipSync.enqueue.mock.calls.map((call) => call[0].seq)
      const datas = lipSync.enqueue.mock.calls.map((call) => call[0].base64Audio)
      expect(seqs).toEqual([0, 1, 2])
      expect(datas).toEqual(['AA', 'BB', 'CC'])
    })

    it('calls onAudioDropped and never beginStream/enqueue when lipSync.enqueue is unavailable', () => {
      kurisuBridge.lipSync = null
      const onAudioDropped = vi.fn()
      const onAudioStart = vi.fn()
      const demux = createFrameDemuxer({ onAudioDropped, onAudioStart })

      demux.handleAudio(audioFrame(), true)

      expect(onAudioDropped).toHaveBeenCalledTimes(1)
      expect(onAudioStart).not.toHaveBeenCalled()
      expect(demux.hasAudioStarted()).toBe(false)
    })

    it('calls onAudioDropped every time (caller is responsible for once-per-turn de-duplication)', () => {
      kurisuBridge.lipSync = null
      const onAudioDropped = vi.fn()
      const demux = createFrameDemuxer({ onAudioDropped })

      demux.handleAudio(audioFrame({ seq: 0 }), true)
      demux.handleAudio(audioFrame({ seq: 1 }), true)

      expect(onAudioDropped).toHaveBeenCalledTimes(2)
    })

    it('logs and swallows a rejected enqueue promise rather than throwing', async () => {
      const consoleError = vi.spyOn(console, 'error').mockImplementation(() => undefined)
      lipSync.enqueue.mockRejectedValueOnce(new Error('decode failed'))
      const demux = createFrameDemuxer()

      expect(() => demux.handleAudio(audioFrame(), true)).not.toThrow()
      // Let the rejected promise's .catch() handler run.
      await Promise.resolve()
      await Promise.resolve()

      expect(consoleError).toHaveBeenCalledWith('Streamed audio chunk failed:', expect.any(Error))
      consoleError.mockRestore()
    })

    it('an audio frame for a seq with no prior sentence frame enqueues with emotion/beat undefined', () => {
      const demux = createFrameDemuxer()
      demux.handleAudio(audioFrame({ seq: 9 }), true)

      expect(lipSync.enqueue).toHaveBeenCalledWith(
        expect.objectContaining({ seq: 9, emotion: undefined, beat: undefined }),
      )
    })
  })

  describe('hasAudioStarted', () => {
    it('is false before any audio frame and true after the first', () => {
      const demux = createFrameDemuxer()
      expect(demux.hasAudioStarted()).toBe(false)
      demux.handleAudio(audioFrame(), true)
      expect(demux.hasAudioStarted()).toBe(true)
    })
  })

  describe('handleDone', () => {
    it('is a no-op when no audio ever started (voice-off / silent turn)', () => {
      const demux = createFrameDemuxer()
      demux.handleDone({ cancelled: false })

      expect(lipSync.endStream).not.toHaveBeenCalled()
      expect(lipSync.stop).not.toHaveBeenCalled()
    })

    it('drains via endStream() on a normal (non-cancelled) completion', () => {
      const demux = createFrameDemuxer()
      demux.handleAudio(audioFrame(), true)

      demux.handleDone({ cancelled: false })

      expect(lipSync.endStream).toHaveBeenCalledTimes(1)
      expect(lipSync.stop).not.toHaveBeenCalled()
    })

    it('cuts short via stop() on a cancelled completion', () => {
      const demux = createFrameDemuxer()
      demux.handleAudio(audioFrame(), true)

      demux.handleDone({ cancelled: true })

      expect(lipSync.stop).toHaveBeenCalledTimes(1)
      expect(lipSync.endStream).not.toHaveBeenCalled()
    })

    it('resets the started-latch so a reused demuxer re-arms for the next turn', () => {
      const onAudioStart = vi.fn()
      const demux = createFrameDemuxer({ onAudioStart })
      demux.handleAudio(audioFrame(), true)
      demux.handleDone({ cancelled: false })
      expect(demux.hasAudioStarted()).toBe(false)

      // Second turn on the same demuxer instance (duplex's long-lived socket).
      demux.handleAudio(audioFrame({ seq: 0 }), true)
      expect(lipSync.beginStream).toHaveBeenCalledTimes(2)
      expect(onAudioStart).toHaveBeenCalledTimes(2)
      expect(demux.hasAudioStarted()).toBe(true)
    })
  })

  describe('interrupt', () => {
    it('always calls stop(), even when no audio frame ever arrived this turn', () => {
      const demux = createFrameDemuxer()
      demux.interrupt()

      expect(lipSync.stop).toHaveBeenCalledTimes(1)
      expect(lipSync.endStream).not.toHaveBeenCalled()
    })

    it('resets the started-latch', () => {
      const demux = createFrameDemuxer()
      demux.handleAudio(audioFrame(), true)
      expect(demux.hasAudioStarted()).toBe(true)

      demux.interrupt()

      expect(demux.hasAudioStarted()).toBe(false)
    })

    it('tolerates a null lipSync (avatar not mounted)', () => {
      kurisuBridge.lipSync = null
      const demux = createFrameDemuxer()

      expect(() => demux.interrupt()).not.toThrow()
    })
  })

  describe('per-turn isolation', () => {
    it('a fresh demuxer instance does not carry over emotion/beat maps from a previous instance', () => {
      const demux1 = createFrameDemuxer()
      demux1.handleSentence(sentenceFrame({ seq: 0, emotion: 'sad', en: 'Never mind.' }))

      const demux2 = createFrameDemuxer()
      demux2.handleAudio(audioFrame({ seq: 0 }), true)

      expect(lipSync.enqueue).toHaveBeenCalledWith(
        expect.objectContaining({ seq: 0, emotion: undefined, beat: undefined }),
      )
    })
  })
})
