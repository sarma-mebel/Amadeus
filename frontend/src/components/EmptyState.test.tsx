// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest'
import '@testing-library/jest-dom/vitest'
import { cleanup, render, screen } from '@testing-library/react'
import EmptyState from './EmptyState'

afterEach(() => {
  cleanup()
})

describe('EmptyState', () => {
  it('renders the icon, title and sub-line', () => {
    const { container } = render(
      <EmptyState icon="memory" title="Nothing recorded yet" subLine="She's still getting to know you." />,
    )
    expect(screen.getByText('Nothing recorded yet')).toBeInTheDocument()
    expect(screen.getByText("She's still getting to know you.")).toBeInTheDocument()
    expect(container.querySelector('.empty-state-icon svg')).toBeInTheDocument()
  })

  it('omits the sub-line when none is given', () => {
    render(<EmptyState icon="chat" title="Say something" />)
    expect(screen.getByText('Say something')).toBeInTheDocument()
    expect(document.querySelector('.empty-state-subline')).not.toBeInTheDocument()
  })

  it('renders extra children below the copy', () => {
    render(
      <EmptyState icon="sessions" title="No saved conversations yet">
        <button type="button">Start one</button>
      </EmptyState>,
    )
    expect(screen.getByRole('button', { name: 'Start one' })).toBeInTheDocument()
  })

  it('renders a distinct icon per icon prop', () => {
    const { container: chat } = render(<EmptyState icon="chat" title="a" />)
    const { container: synapse } = render(<EmptyState icon="synapse" title="b" />)
    expect(chat.querySelector('.empty-state-icon svg')?.outerHTML).not.toBe(
      synapse.querySelector('.empty-state-icon svg')?.outerHTML,
    )
  })
})
