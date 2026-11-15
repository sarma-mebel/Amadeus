// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom/vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import RelationshipGauge from './RelationshipGauge'

describe('RelationshipGauge', () => {
  afterEach(() => {
    cleanup()
  })

  it('renders nothing when trust is null', () => {
    const { container } = render(
      <RelationshipGauge trust={null} sessionCount={4} onClick={() => {}} />,
    )
    expect(container).toBeEmptyDOMElement()
  })

  it('renders the trust percentage and fires onClick', async () => {
    const user = userEvent.setup()
    const onClick = vi.fn()
    render(<RelationshipGauge trust={0.72} sessionCount={5} onClick={onClick} />)

    const button = screen.getByRole('button', { name: 'Relationship: trust 72%' })
    expect(button).toBeInTheDocument()
    expect(screen.getByText('72')).toBeInTheDocument()
    expect(button).toHaveAttribute('title', expect.stringContaining('72%'))
    expect(button).toHaveAttribute('title', expect.stringContaining('5 sessions'))

    await user.click(button)
    expect(onClick).toHaveBeenCalledTimes(1)
  })

  it('applies the celebrating class when the prop is set', () => {
    render(<RelationshipGauge trust={0.5} sessionCount={1} onClick={() => {}} celebrating />)
    expect(screen.getByRole('button')).toHaveClass('celebrating')
  })

  it('singularizes the session count label', () => {
    render(<RelationshipGauge trust={0.1} sessionCount={1} onClick={() => {}} />)
    expect(screen.getByRole('button')).toHaveAttribute('title', expect.stringContaining('1 session'))
    expect(screen.getByRole('button')).not.toHaveAttribute(
      'title',
      expect.stringContaining('1 sessions'),
    )
  })
})
