// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom/vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import SynapseMap, { SynapseDetail, RECORD_PREFIX } from './SynapseMap'
import type { GraphNode, MemoryGraph } from './SynapseMap'

describe('SynapseDetail', () => {
  afterEach(() => {
    cleanup()
  })

  it('shows only the hover/select label when no node is selected', () => {
    render(<SynapseDetail detail={null} selected={null} />)
    expect(screen.getByText('Hover a node — click to pin its memory here.')).toBeInTheDocument()
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })

  it('shows no action buttons for a non-record node (e.g. a profile/thread node)', () => {
    const node: GraphNode = { id: 'profile:name', label: 'About you: name', kind: 'profile' }
    render(<SynapseDetail detail={node} selected={node} />)
    expect(screen.getByText('About you: name')).toBeInTheDocument()
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })

  it('renders Pin/Edit/Delete for a rec: node and fires callbacks with the bare id', async () => {
    const user = userEvent.setup()
    const onPin = vi.fn()
    const onEdit = vi.fn()
    const onDelete = vi.fn()
    const node: GraphNode = { id: 'rec:abc123', label: 'Prefers coffee over tea', kind: 'fact' }

    render(
      <SynapseDetail detail={node} selected={node} onPin={onPin} onEdit={onEdit} onDelete={onDelete} />,
    )

    await user.click(screen.getByRole('button', { name: '📌 Pin' }))
    expect(onPin).toHaveBeenCalledWith('abc123', true)

    await user.click(screen.getByRole('button', { name: 'Delete' }))
    expect(onDelete).toHaveBeenCalledWith('abc123')

    await user.click(screen.getByRole('button', { name: '✎ Edit' }))
    const input = screen.getByRole('textbox', { name: 'Edit memory' })
    await user.clear(input)
    await user.type(input, 'Prefers tea now')
    await user.click(screen.getByRole('button', { name: 'Save' }))
    expect(onEdit).toHaveBeenCalledWith('abc123', 'Prefers tea now')
  })

  it('shows Unpin and aria-pressed=true for an already-pinned node', () => {
    const node: GraphNode = { id: 'rec:abc123', label: 'Pinned fact', kind: 'fact', pinned: true }
    render(<SynapseDetail detail={node} selected={node} onPin={() => {}} />)
    const pinButton = screen.getByRole('button', { name: '📌 Unpin' })
    expect(pinButton).toHaveAttribute('aria-pressed', 'true')
  })

  it('cancels an in-progress edit without calling onEdit', async () => {
    const user = userEvent.setup()
    const onEdit = vi.fn()
    const node: GraphNode = { id: 'rec:abc123', label: 'Some fact', kind: 'fact' }
    render(<SynapseDetail detail={node} selected={node} onEdit={onEdit} />)

    await user.click(screen.getByRole('button', { name: '✎ Edit' }))
    await user.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(onEdit).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: '✎ Edit' })).toBeInTheDocument()
  })

  describe('pin state follows updated node props (item 7)', () => {
    // SynapseMap re-resolves `selected` against the freshest `graph` prop
    // whenever MemoryPanel updates graph state (e.g. after a pin toggle),
    // so a rerender with a flipped `pinned` is exactly what SynapseDetail
    // sees in the real flow — this exercises "the button computes its next
    // state from the updated node" without needing the canvas/rAF sim.
    it('flips label/aria-pressed after a rerender with pinned=true, and the next click un-pins', async () => {
      const user = userEvent.setup()
      const onPin = vi.fn()
      const node: GraphNode = { id: 'rec:abc123', label: 'Prefers coffee over tea', kind: 'fact' }

      const { rerender } = render(
        <SynapseDetail detail={node} selected={node} onPin={onPin} />,
      )

      const pinButton = () => screen.getByRole('button', { name: /Pin$|Unpin$/ })
      expect(pinButton()).toHaveTextContent('📌 Pin')
      expect(pinButton()).toHaveAttribute('aria-pressed', 'false')

      await user.click(pinButton())
      expect(onPin).toHaveBeenCalledWith('abc123', true)

      // Simulate the graph-state update flowing back down as a new node.
      const pinnedNode: GraphNode = { ...node, pinned: true }
      rerender(<SynapseDetail detail={pinnedNode} selected={pinnedNode} onPin={onPin} />)

      expect(pinButton()).toHaveTextContent('📌 Unpin')
      expect(pinButton()).toHaveAttribute('aria-pressed', 'true')

      await user.click(pinButton())
      expect(onPin).toHaveBeenLastCalledWith('abc123', false)
    })

    it(`resolves the record id by stripping the "${RECORD_PREFIX}" node-id prefix`, () => {
      const onPin = vi.fn()
      const node: GraphNode = { id: `${RECORD_PREFIX}xyz789`, label: 'A memory', kind: 'episode' }
      render(<SynapseDetail detail={node} selected={node} onPin={onPin} />)
      expect(screen.getByRole('button', { name: '📌 Pin' })).toBeInTheDocument()
    })
  })

  describe('protected status (item 7)', () => {
    it('shows the protected message for a node whose bare id is in protectedIds', () => {
      const node: GraphNode = { id: 'rec:abc123', label: 'Pinned fact', kind: 'fact', pinned: true }
      render(
        <SynapseDetail
          detail={node}
          selected={node}
          onPin={() => {}}
          onDelete={() => {}}
          protectedIds={{ abc123: true }}
        />,
      )
      expect(screen.getByText('That memory is pinned — unpin it first.')).toBeInTheDocument()
    })

    it('shows no protected message when the node is not in protectedIds', () => {
      const node: GraphNode = { id: 'rec:abc123', label: 'Some fact', kind: 'fact' }
      render(
        <SynapseDetail
          detail={node}
          selected={node}
          onDelete={() => {}}
          protectedIds={{ 'other-id': true }}
        />,
      )
      expect(screen.queryByText('That memory is pinned — unpin it first.')).not.toBeInTheDocument()
    })
  })
})

describe('SynapseMap legend', () => {
  // jsdom's canvas has no 2d context, so the force-sim effect early-returns;
  // the surrounding DOM (legend, detail line) still renders, which is all we
  // assert here. This pins the fix that the legend is a readable key rather
  // than aria-hidden colored dots with no text.
  afterEach(() => {
    cleanup()
  })

  const graph: MemoryGraph = {
    nodes: [
      { id: 'rec:1', label: 'A memory', kind: 'fact' },
      { id: 'rec:2', label: 'An episode', kind: 'episode' },
      { id: 'thread:1', label: 'An open thread', kind: 'thread' },
    ],
    edges: [],
  }

  it('renders a visible text label for each node kind present (not aria-hidden)', () => {
    render(<SynapseMap graph={graph} />)
    // The kind→label map: fact→"memories", episode→"episodes", thread→"open threads".
    expect(screen.getByText('memories')).toBeInTheDocument()
    expect(screen.getByText('episodes')).toBeInTheDocument()
    expect(screen.getByText('open threads')).toBeInTheDocument()
  })

  it('exposes the legend to assistive tech (no aria-hidden wrapper)', () => {
    const { container } = render(<SynapseMap graph={graph} />)
    const legend = container.querySelector('.synapse-legend')
    expect(legend).not.toBeNull()
    expect(legend).not.toHaveAttribute('aria-hidden')
    expect(legend).toHaveAttribute('aria-label', 'Node kinds')
  })
})
