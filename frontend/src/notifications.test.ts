// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  canNotify,
  ensurePermission,
  loadNotificationsEnabled,
  notify,
  saveNotificationsEnabled,
} from './notifications'

class FakeNotification {
  static permission: NotificationPermission = 'granted'
  static requestPermission = vi.fn(async () => FakeNotification.permission)
  static instances: Array<{ title: string; options?: NotificationOptions }> = []
  constructor(title: string, options?: NotificationOptions) {
    FakeNotification.instances.push({ title, options })
  }
}

const setHidden = (hidden: boolean) => {
  Object.defineProperty(document, 'hidden', { value: hidden, configurable: true })
}

describe('notifications', () => {
  beforeEach(() => {
    window.localStorage.clear()
    FakeNotification.permission = 'granted'
    FakeNotification.instances = []
    vi.stubGlobal('Notification', FakeNotification)
    setHidden(true)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('defaults to OFF', () => {
    expect(loadNotificationsEnabled()).toBe(false)
    expect(canNotify('reminder')).toBe(false)
  })

  it('fires only while the app is hidden', () => {
    saveNotificationsEnabled(true)
    expect(canNotify('reminder')).toBe(true)
    setHidden(false)
    expect(canNotify('reminder')).toBe(false)
  })

  it('requires granted permission', () => {
    saveNotificationsEnabled(true)
    FakeNotification.permission = 'denied'
    expect(canNotify('reminder')).toBe(false)
  })

  it('reminders fire each time; brief and diary cap at once per day', () => {
    saveNotificationsEnabled(true)

    expect(notify('reminder', 'K', 'one')).toBe(true)
    expect(notify('reminder', 'K', 'two')).toBe(true)

    expect(notify('brief', 'K', 'brief')).toBe(true)
    expect(notify('brief', 'K', 'brief again')).toBe(false)

    expect(notify('diary', 'K', 'diary')).toBe(true)
    expect(notify('diary', 'K', 'diary again')).toBe(false)

    expect(notify('reachout', 'K', 'did the draft go out?')).toBe(true)
    expect(notify('reachout', 'K', 'a second ping')).toBe(false)

    expect(FakeNotification.instances).toHaveLength(5)
  })

  it('notify is a no-op when disabled', () => {
    expect(notify('reminder', 'K', 'x')).toBe(false)
    expect(FakeNotification.instances).toHaveLength(0)
  })

  it('ensurePermission maps browser states', async () => {
    FakeNotification.permission = 'granted'
    expect(await ensurePermission()).toBe('granted')
    FakeNotification.permission = 'denied'
    expect(await ensurePermission()).toBe('denied')
    vi.unstubAllGlobals()
    // No Notification global at all.
    expect(await ensurePermission()).toBe('unsupported')
  })
})
