import { describe, expect, it } from 'vitest'
import { describeToolActivity, getTimeOfDayBucket } from './uiHelpers'

describe('getTimeOfDayBucket', () => {
  it('buckets 5-7am as dawn', () => {
    expect(getTimeOfDayBucket(5)).toBe('dawn')
    expect(getTimeOfDayBucket(7)).toBe('dawn')
  })

  it('buckets 8am-4pm as day', () => {
    expect(getTimeOfDayBucket(8)).toBe('day')
    expect(getTimeOfDayBucket(16)).toBe('day')
  })

  it('buckets 5pm-7pm as evening', () => {
    expect(getTimeOfDayBucket(17)).toBe('evening')
    expect(getTimeOfDayBucket(19)).toBe('evening')
  })

  it('buckets everything else (8pm-4am) as night', () => {
    expect(getTimeOfDayBucket(20)).toBe('night')
    expect(getTimeOfDayBucket(23)).toBe('night')
    expect(getTimeOfDayBucket(0)).toBe('night')
    expect(getTimeOfDayBucket(4)).toBe('night')
  })

  it('defaults to the current local hour when called with no argument', () => {
    const expected = getTimeOfDayBucket(new Date().getHours())
    expect(getTimeOfDayBucket()).toBe(expected)
  })
})

describe('describeToolActivity', () => {
  it('returns the friendly label for known tools', () => {
    expect(describeToolActivity('web_search')).toBe('Searching the web…')
    expect(describeToolActivity('code_interpreter')).toBe('Running code…')
    expect(describeToolActivity('calculator')).toBe('Calculating…')
  })

  it('maps every literature-style tool to the same "Searching papers…" label', () => {
    expect(describeToolActivity('literature_search')).toBe('Searching papers…')
    expect(describeToolActivity('pubmed')).toBe('Searching papers…')
    expect(describeToolActivity('arxiv')).toBe('Searching papers…')
  })

  it('falls back to a generic "Using <tool>…" label for unknown tools', () => {
    expect(describeToolActivity('mystery_tool')).toBe('Using mystery_tool…')
  })
})
