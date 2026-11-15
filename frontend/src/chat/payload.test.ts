import { describe, expect, it } from 'vitest'
import { buildChatPayload } from './payload'

describe('buildChatPayload', () => {
  it('produces the exact wire shape', () => {
    expect(
      buildChatPayload({
        message: 'hi',
        sessionId: 'abc',
        userId: 'u1',
        audioEnabled: true,
      }),
    ).toEqual({
      message: 'hi',
      session_id: 'abc',
      user_id: 'u1',
      audio_enabled: true,
    })
  })

  it('passes a null session through', () => {
    const payload = buildChatPayload({
      message: 'hi',
      sessionId: null,
      userId: 'u1',
      audioEnabled: false,
    })
    expect(payload.session_id).toBeNull()
  })
})

describe('buildChatPayload images (vision)', () => {
  it('passes a single image attachment through', () => {
    const payload = buildChatPayload({
      message: 'look at this',
      sessionId: null,
      userId: 'u1',
      audioEnabled: true,
      images: [{ data: 'QUJD', mime: 'image/jpeg' }],
    })
    expect(payload.images).toEqual([{ data: 'QUJD', mime: 'image/jpeg' }])
  })

  it('omits the images key when absent or empty', () => {
    const base = {
      message: 'hi',
      sessionId: null,
      userId: 'u1',
      audioEnabled: true,
    }
    expect(buildChatPayload(base).images).toBeUndefined()
    expect(buildChatPayload({ ...base, images: [] }).images).toBeUndefined()
  })
})
