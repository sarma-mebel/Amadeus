// @vitest-environment jsdom
/**
 * PresenceSection: the dial applies its preset to the fine-tune toggles,
 * and hand-tuning a toggle surfaces the "customized" note instead of
 * silently disagreeing with the selected level.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import '@testing-library/jest-dom/vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'

import PresenceSection from './PresenceSection'
import { loadFillersEnabled } from '../../audio/fillerPlayer'
import { loadListeningReactionsEnabled } from '../../audio/listeningReactions'
import { loadPresenceLevel } from '../../presenceLevel'

describe('PresenceSection', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })
  afterEach(() => {
    cleanup()
    window.localStorage.clear()
  })

  it('starts on balanced with the default toggle states', () => {
    render(<PresenceSection />)
    expect(screen.getByLabelText('Balanced')).toBeChecked()
    expect(screen.getByLabelText('Thinking sounds')).not.toBeChecked()
    expect(screen.getByLabelText('Listening reactions')).toBeChecked()
    // Roam is desktop-only; jsdom is a browser, so no toggle at all.
    expect(screen.queryByLabelText('Let her wander the desktop')).toBeNull()
  })

  it('picking Quiet applies the preset to storage and the toggles', () => {
    render(<PresenceSection />)
    fireEvent.click(screen.getByLabelText('Quiet'))
    expect(loadPresenceLevel()).toBe('quiet')
    expect(loadFillersEnabled()).toBe(false)
    expect(loadListeningReactionsEnabled()).toBe(false)
    expect(screen.getByLabelText('Listening reactions')).not.toBeChecked()
  })

  it('hand-tuning a toggle marks the level customized; re-picking clears it', () => {
    render(<PresenceSection />)
    fireEvent.click(screen.getByLabelText('Quiet'))
    expect(screen.queryByText(/Fine-tuned/)).toBeNull()

    fireEvent.click(screen.getByLabelText('Thinking sounds'))
    expect(loadFillersEnabled()).toBe(true)
    expect(screen.getByText(/Fine-tuned/)).toBeInTheDocument()

    fireEvent.click(screen.getByLabelText('Quiet'))
    expect(loadFillersEnabled()).toBe(false)
    expect(screen.queryByText(/Fine-tuned/)).toBeNull()
  })
})
