/**
 * Force-directed canvas view of long-term memory (the "SYNAPSE MAP" the
 * boot screen promises): nodes are memories/profile facts/open threads,
 * edges are embedding-similarity links from /api/memory action:"graph".
 * Hand-rolled simulation — the graph is bounded (~100 nodes) so a tiny
 * spring/repulsion loop beats pulling in a graph library.
 */
import { useEffect, useRef, useState } from 'react'

export interface GraphNode {
  id: string
  label: string
  kind: string
  // True for a fact/episode node backed by a real record id ("rec:<id>")
  // that's currently pinned — drawn with an extra ring so pinned memories
  // stand out on the map, mirroring the list view's pin marker.
  pinned?: boolean
}

export interface GraphEdge {
  source: string
  target: string
  weight: number
}

export interface MemoryGraph {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

const KIND_COLORS: Record<string, string> = {
  episode: '#d75a65', // auburn — conversation episodes
  fact: '#ffb75e', // amber — remembered facts
  reflection: '#8fd9a8',
  profile: '#ffd08a',
  thread: '#9fd8e8',
  memory: '#b9bec9',
}

const KIND_LABELS: Record<string, string> = {
  episode: 'episodes',
  fact: 'memories',
  reflection: 'reflections',
  profile: 'about you',
  thread: 'open threads',
}

const NODE_RADIUS: Record<string, number> = {
  episode: 7,
  fact: 5.5,
  reflection: 5,
  profile: 5.5,
  thread: 4.5,
  memory: 5,
}

interface SimNode extends GraphNode {
  x: number
  y: number
  vx: number
  vy: number
}

export interface SynapseMapProps {
  graph: MemoryGraph
  onPin?: (recordId: string, pinned: boolean) => void
  onEdit?: (recordId: string, text: string) => void
  onDelete?: (recordId: string) => void
  /** Record ids (bare, no "rec:" prefix) that refused a delete because
   * they're pinned — mirrors the list view's inline "pinned — unpin it
   * first" message so a protected node behaves the same in both views. */
  protectedIds?: Record<string, boolean>
}

const SynapseMap = ({ graph, onPin, onEdit, onDelete, protectedIds }: SynapseMapProps) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [selected, setSelected] = useState<GraphNode | null>(null)
  const [hovered, setHovered] = useState<GraphNode | null>(null)

  // Re-resolve the selected node against a fresh `graph` (pin/edit/delete
  // all land here as a new graph reference from MemoryPanel): a pin toggle
  // needs `selected` to pick up the new `pinned` value so SynapseDetail's
  // button computes its next state from current data, not a click-time
  // snapshot; a delete needs `selected` cleared once its node is gone.
  useEffect(() => {
    setSelected((prev) => {
      if (!prev) return prev
      const next = graph.nodes.find((node) => node.id === prev.id)
      return next ?? null
    })
  }, [graph])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || graph.nodes.length === 0) return
    const context = canvas.getContext('2d')
    if (!context) return

    const width = canvas.clientWidth
    const height = canvas.clientHeight
    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr
    canvas.height = height * dpr
    context.scale(dpr, dpr)

    // Deterministic circular seed (no Math.random: same graph, same shape).
    const nodes: SimNode[] = graph.nodes.map((node, index) => {
      const angle = (index / graph.nodes.length) * Math.PI * 2
      const ring = 0.28 + 0.14 * (index % 3)
      return {
        ...node,
        x: width / 2 + Math.cos(angle) * width * ring,
        y: height / 2 + Math.sin(angle) * height * ring,
        vx: 0,
        vy: 0,
      }
    })
    const byId = new Map(nodes.map((node) => [node.id, node]))
    const edges = graph.edges
      .map((edge) => ({
        a: byId.get(edge.source),
        b: byId.get(edge.target),
        weight: edge.weight,
      }))
      .filter((edge): edge is { a: SimNode; b: SimNode; weight: number } =>
        Boolean(edge.a && edge.b)
      )

    let raf = 0
    // True while a frame is scheduled — lets the interaction handlers
    // re-kick the loop after it settles without ever stacking a second rAF.
    let running = false
    let ticks = 0
    const MAX_TICKS = 600 // settle, then stop burning frames

    const step = () => {
      // Pairwise repulsion.
      for (let i = 0; i < nodes.length; i += 1) {
        for (let j = i + 1; j < nodes.length; j += 1) {
          const a = nodes[i]
          const b = nodes[j]
          let dx = a.x - b.x
          let dy = a.y - b.y
          const distSq = dx * dx + dy * dy || 1
          const dist = Math.sqrt(distSq)
          const force = Math.min(1200 / distSq, 4)
          dx /= dist
          dy /= dist
          a.vx += dx * force
          a.vy += dy * force
          b.vx -= dx * force
          b.vy -= dy * force
        }
      }
      // Springs: stronger similarity pulls closer.
      for (const { a, b, weight } of edges) {
        const dx = b.x - a.x
        const dy = b.y - a.y
        const dist = Math.sqrt(dx * dx + dy * dy) || 1
        const rest = 150 - 90 * weight
        const force = ((dist - rest) / dist) * 0.02 * (0.5 + weight)
        a.vx += dx * force
        a.vy += dy * force
        b.vx -= dx * force
        b.vy -= dy * force
      }
      // Gentle centering + integration with damping and bounds.
      for (const node of nodes) {
        node.vx += (width / 2 - node.x) * 0.002
        node.vy += (height / 2 - node.y) * 0.002
        node.vx *= 0.82
        node.vy *= 0.82
        node.x = Math.max(14, Math.min(width - 14, node.x + node.vx))
        node.y = Math.max(14, Math.min(height - 14, node.y + node.vy))
      }
    }

    const draw = () => {
      context.clearRect(0, 0, width, height)
      context.lineWidth = 1
      for (const { a, b, weight } of edges) {
        context.strokeStyle = `rgba(193, 68, 79, ${0.15 + weight * 0.45})`
        context.beginPath()
        context.moveTo(a.x, a.y)
        context.lineTo(b.x, b.y)
        context.stroke()
      }
      for (const node of nodes) {
        const radius = NODE_RADIUS[node.kind] ?? 5
        context.beginPath()
        context.arc(node.x, node.y, radius, 0, Math.PI * 2)
        context.fillStyle = KIND_COLORS[node.kind] ?? KIND_COLORS.memory
        context.fill()
        context.strokeStyle = 'rgba(9, 13, 22, 0.9)'
        context.stroke()
        if (node.pinned) {
          context.beginPath()
          context.arc(node.x, node.y, radius + 3, 0, Math.PI * 2)
          context.strokeStyle = 'rgba(255, 208, 138, 0.85)'
          context.lineWidth = 1.5
          context.stroke()
          context.lineWidth = 1
        }
      }
    }

    const frame = () => {
      if (ticks < MAX_TICKS) {
        step()
        ticks += 1
        draw()
        raf = requestAnimationFrame(frame)
      } else {
        // Settled: stop the loop entirely rather than idling at 60fps for
        // as long as the panel stays mounted — interaction re-kicks it.
        running = false
      }
    }
    const kickLoop = () => {
      if (running) return
      running = true
      raf = requestAnimationFrame(frame)
    }
    kickLoop()

    // `radius` widens the hit test for imprecise touch input (fat fingers)
    // vs. the tighter 14px the mouse path uses.
    const nodeAt = (event: PointerEvent, radius: number): SimNode | null => {
      const rect = canvas.getBoundingClientRect()
      const x = event.clientX - rect.left
      const y = event.clientY - rect.top
      let best: SimNode | null = null
      let bestDist = radius
      for (const node of nodes) {
        const dist = Math.hypot(node.x - x, node.y - y)
        if (dist < bestDist) {
          bestDist = dist
          best = node
        }
      }
      return best
    }

    // Mouse (and pen) get live hover previews; touch has no hover, so a
    // pointermove there does nothing and selection happens on tap below.
    const handleMove = (event: PointerEvent) => {
      if (event.pointerType === 'touch') return
      const node = nodeAt(event, 14)
      setHovered(node)
      canvas.style.cursor = node ? 'pointer' : 'default'
      ticks = Math.min(ticks, MAX_TICKS - 1) // keep drawing while interacting
      kickLoop() // the settled loop stopped scheduling — restart it
    }
    // pointerdown covers mouse click and touch tap alike: it selects the
    // node under the pointer (larger hit radius for touch), and a tap on
    // empty space deselects. On touch we also clear any stale hover preview.
    const handleDown = (event: PointerEvent) => {
      const radius = event.pointerType === 'touch' ? 24 : 14
      const node = nodeAt(event, radius)
      setSelected(node)
      if (event.pointerType === 'touch') setHovered(null)
    }
    canvas.addEventListener('pointermove', handleMove)
    canvas.addEventListener('pointerdown', handleDown)

    return () => {
      cancelAnimationFrame(raf)
      canvas.removeEventListener('pointermove', handleMove)
      canvas.removeEventListener('pointerdown', handleDown)
    }
  }, [graph])

  const kindsPresent = Array.from(new Set(graph.nodes.map((node) => node.kind)))
  const detail = hovered ?? selected

  return (
    <div className="synapse-map">
      <canvas ref={canvasRef} className="synapse-canvas" aria-label="Memory graph" />
      <div className="synapse-legend" aria-label="Node kinds">
        {kindsPresent.map((kind) => (
          <span key={kind} className="synapse-legend-item">
            <i aria-hidden="true" style={{ background: KIND_COLORS[kind] ?? KIND_COLORS.memory }} />
            {KIND_LABELS[kind] ?? kind}
          </span>
        ))}
      </div>
      <SynapseDetail
        detail={detail}
        selected={selected}
        onPin={onPin}
        onEdit={onEdit}
        onDelete={onDelete}
        protectedIds={protectedIds}
      />
    </div>
  )
}

export default SynapseMap

/** Prefix distinguishing a graph node backed by a real fact/episode record
 * (pin/edit/delete-able) from profile/thread nodes. Exported so MemoryPanel
 * can build/match node ids when optimistically updating graph state after
 * a pin/edit/delete (rec:<id> <-> record id round-trip). */
export const RECORD_PREFIX = 'rec:'

export interface SynapseDetailProps {
  /** Node whose label is shown — hover takes priority over the pinned
   * selection so mousing around still previews labels live. */
  detail: GraphNode | null
  /** The clicked (not merely hovered) node — action buttons only ever key
   * off this, so they don't flicker as the mouse passes over other nodes. */
  selected: GraphNode | null
  onPin?: (recordId: string, pinned: boolean) => void
  onEdit?: (recordId: string, text: string) => void
  onDelete?: (recordId: string) => void
  /** Record ids (bare) that refused a delete because they're pinned. */
  protectedIds?: Record<string, boolean>
}

/** The synapse map's status line + (for a clicked fact/episode node backed
 * by a real record) pin/edit/delete actions. Extracted as its own component
 * so it can be unit-tested without the canvas/rAF simulation. */
export const SynapseDetail = ({ detail, selected, onPin, onEdit, onDelete, protectedIds }: SynapseDetailProps) => {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')

  const recordId =
    selected && selected.id.startsWith(RECORD_PREFIX) ? selected.id.slice(RECORD_PREFIX.length) : null
  const isProtected = Boolean(recordId && protectedIds?.[recordId])

  useEffect(() => {
    setEditing(false)
    setDraft(selected?.label ?? '')
  }, [selected?.id, selected?.label])

  return (
    <div className="synapse-detail-wrap">
      <p className="synapse-detail" role="status">
        {detail ? detail.label : 'Hover a node — click to pin its memory here.'}
      </p>
      {recordId && !editing && (
        <div className="synapse-detail-actions">
          <button
            type="button"
            className="synapse-action-button"
            aria-pressed={Boolean(selected?.pinned)}
            onClick={() => onPin?.(recordId, !selected?.pinned)}
          >
            {selected?.pinned ? '📌 Unpin' : '📌 Pin'}
          </button>
          {onEdit && (
            <button
              type="button"
              className="synapse-action-button"
              onClick={() => {
                setDraft(selected?.label ?? '')
                setEditing(true)
              }}
            >
              ✎ Edit
            </button>
          )}
          {onDelete && (
            <button
              type="button"
              className="synapse-action-button synapse-action-danger"
              onClick={() => onDelete(recordId)}
            >
              Delete
            </button>
          )}
        </div>
      )}
      {recordId && editing && (
        <form
          className="synapse-edit-form"
          onSubmit={(event) => {
            event.preventDefault()
            const text = draft.trim()
            if (text) onEdit?.(recordId, text)
            setEditing(false)
          }}
        >
          <input
            type="text"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            aria-label="Edit memory"
            maxLength={500}
          />
          <button type="submit">Save</button>
          <button type="button" onClick={() => setEditing(false)}>
            Cancel
          </button>
        </form>
      )}
      {isProtected && <p className="memory-fact-protected">That memory is pinned — unpin it first.</p>}
    </div>
  )
}
