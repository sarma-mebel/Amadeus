// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest'
import {
  LISTENING_REACTIONS_STORAGE_KEY,
  createListeningReactor,
  loadListeningReactionsEnabled,
  saveListeningReactionsEnabled,
} from './listeningReactions'

const makeHooks = () => {
  const calls = { attention: 0, nod: 0, vocal: 0 }
  return {
    calls,
    hooks: {
      onAttention: () => {
        calls.attention += 1
      },
      onNod: () => {
        calls.nod += 1
      },
      onVocal: () => {
        calls.vocal += 1
      },
    },
  }
}

/** Feed `ms` of a constant level in 50ms buffers. */
const feedLevel = (
  reactor: { feed: (rms: number, dtMs: number) => void },
  rms: number,
  ms: number,
) => {
  for (let fed = 0; fed < ms; fed += 50) reactor.feed(rms, 50)
}

describe('createListeningReactor', () => {
  it('stays silent on sub-threshold noise', () => {
    const { calls, hooks } = makeHooks()
    const reactor = createListeningReactor(hooks, { random: () => 0 })
    feedLevel(reactor, 0.005, 10000)
    expect(calls).toEqual({ attention: 0, nod: 0, vocal: 0 })
  })

  it('fires attention periodically and an early nod on sustained speech', () => {
    const { calls, hooks } = makeHooks()
    const reactor = createListeningReactor(hooks, { random: () => 0 })
    feedLevel(reactor, 0.05, 2000)
    expect(calls.attention).toBeGreaterThanOrEqual(2)
    expect(calls.nod).toBe(1) // first nod at ~600ms, next not before 2.4s more
  })

  it('spaces follow-up nods by the configured gap', () => {
    const { calls, hooks } = makeHooks()
    const reactor = createListeningReactor(hooks, {
      random: () => 0, // gap always nodMinGapMs
      nodMinGapMs: 2000,
      nodMaxGapMs: 3000,
    })
    feedLevel(reactor, 0.05, 5000)
    // Nods at 600ms, 2650ms, 4700ms.
    expect(calls.nod).toBe(3)
  })

  it('fires the vocal backchannel once per utterance after sustained speech', () => {
    const { calls, hooks } = makeHooks()
    const reactor = createListeningReactor(hooks, {
      random: () => 0, // always passes the chance roll
      vocalMinSpeechMs: 1000,
      vocalCooldownMs: 60000,
    })
    feedLevel(reactor, 0.05, 4000)
    expect(calls.vocal).toBe(1)
    // Silence ends the utterance; a new one starts but the cooldown holds.
    feedLevel(reactor, 0.0, 1000)
    feedLevel(reactor, 0.05, 4000)
    expect(calls.vocal).toBe(1)
  })

  it('respects the vocal chance roll', () => {
    const { calls, hooks } = makeHooks()
    const reactor = createListeningReactor(hooks, {
      random: () => 0.99, // always fails the chance roll
      vocalMinSpeechMs: 1000,
    })
    feedLevel(reactor, 0.05, 4000)
    expect(calls.vocal).toBe(0)
  })

  it('a short silence gap does not reset the utterance, a long one does', () => {
    const { calls, hooks } = makeHooks()
    const reactor = createListeningReactor(hooks, { random: () => 0 })
    feedLevel(reactor, 0.05, 500)
    feedLevel(reactor, 0.0, 300) // intra-speech pause, tolerated
    feedLevel(reactor, 0.05, 200)
    expect(calls.nod).toBe(1) // 700ms voiced total crosses the 600ms onset

    feedLevel(reactor, 0.0, 1000) // real silence: utterance over
    feedLevel(reactor, 0.05, 700)
    expect(calls.nod).toBe(2) // fresh utterance earns a fresh onset nod
  })

  it('reset() clears utterance state', () => {
    const { calls, hooks } = makeHooks()
    const reactor = createListeningReactor(hooks, { random: () => 0 })
    feedLevel(reactor, 0.05, 500)
    reactor.reset()
    feedLevel(reactor, 0.05, 500)
    expect(calls.nod).toBe(0) // neither run alone reaches the 600ms onset
  })
})

describe('listening reactions preference', () => {
  afterEach(() => {
    window.localStorage.removeItem(LISTENING_REACTIONS_STORAGE_KEY)
  })

  it('defaults on and round-trips through localStorage', () => {
    expect(loadListeningReactionsEnabled()).toBe(true)
    saveListeningReactionsEnabled(false)
    expect(loadListeningReactionsEnabled()).toBe(false)
    saveListeningReactionsEnabled(true)
    expect(loadListeningReactionsEnabled()).toBe(true)
  })
})
