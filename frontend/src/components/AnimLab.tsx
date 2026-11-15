/**
 * Animation vetting lab (`?view=animlab`): the full VRM scene next to a
 * clip-inventory panel, for judging candidate clips on Kurisu before wiring
 * them into automatic pools. Read-only against the committed manifest — the
 * intended workflow (docs/animations.md) is: drop personal clips into a
 * gitignored dir (static/animations/mixamo/ or local/), reference them from
 * the gitignored static/animations/manifest.local.json overlay, judge them
 * here, then adjust roles in the overlay. "Copy entry" produces the JSON
 * for exactly that.
 */
import { useEffect, useState } from 'react'
import { kurisuBridge, type VrmClipInfo } from '../vrm/kurisuBridge'
import './AnimLab.css'

const ROLE_ORDER: VrmClipInfo['role'][] = ['idle', 'talk', 'subtle', 'gesture']

const AnimLab = () => {
  const [clips, setClips] = useState<VrmClipInfo[]>([])
  const [current, setCurrent] = useState<string | null>(null)
  const [copied, setCopied] = useState<string | null>(null)

  // The scene registers kurisuBridge.vrmAnimations once the VRM + clips
  // load; poll until then, then keep the current-clip readout live.
  useEffect(() => {
    const timer = window.setInterval(() => {
      const controls = kurisuBridge.vrmAnimations
      if (!controls?.describe) return
      setClips(controls.describe())
      setCurrent(controls.current?.() ?? null)
    }, 500)
    return () => window.clearInterval(timer)
  }, [])

  const play = (name: string, loop: boolean) => {
    kurisuBridge.vrmAnimations?.play(name, { loop })
  }

  // Additive-layer preview (vrm/additiveGestures): judge how a gesture
  // reads layered over the live idle — the re-vetting pass required
  // before enabling kurisu_additive_gestures globally.
  const playAdditive = (name: string) => {
    kurisuBridge.vrmAnimations?.play(name, { additive: true })
  }

  const backToIdle = () => {
    const idle = clips.find((clip) => clip.looping)
    if (idle) kurisuBridge.vrmAnimations?.play(idle.name, { loop: true })
  }

  const copyEntry = (clip: VrmClipInfo) => {
    const entry = {
      file: clip.filename.split('/').pop(),
      ...(clip.directory ? { directory: clip.directory } : {}),
      name: clip.name,
      role: clip.role,
    }
    void navigator.clipboard?.writeText(JSON.stringify(entry)).then(() => {
      setCopied(clip.name)
      window.setTimeout(() => setCopied((value) => (value === clip.name ? null : value)), 1500)
    })
  }

  const grouped = ROLE_ORDER.map((role) => ({
    role,
    items: clips.filter((clip) => clip.role === role),
  })).filter((group) => group.items.length > 0)

  return (
    <div className="animlab" role="region" aria-label="Animation lab">
      <header className="animlab-header">
        <h2>ANIMATION LAB</h2>
        <p className="animlab-sub">
          {clips.length > 0
            ? `${clips.length} clips loaded · playing: ${current ?? '—'}`
            : 'Waiting for the scene to load…'}
        </p>
        <button type="button" className="animlab-idle-btn" onClick={backToIdle}>
          ⟳ Back to idle rotation
        </button>
      </header>

      <div className="animlab-list">
        {grouped.map((group) => (
          <section key={group.role} className="animlab-group">
            <h3 className="animlab-role">{group.role}</h3>
            {group.items.map((clip) => (
              <div
                key={clip.name}
                className={`animlab-row ${current === clip.name ? 'playing' : ''}`}
              >
                <div className="animlab-row-main">
                  <span className="animlab-name">{clip.name}</span>
                  <span className="animlab-meta">
                    {clip.loaded ? `${clip.duration.toFixed(1)}s` : '·'}
                    {clip.looping && <em className="animlab-badge loop">loop pool</em>}
                    {clip.disabled && <em className="animlab-badge off">disabled</em>}
                    {clip.lazy && !clip.loaded && <em className="animlab-badge">lazy</em>}
                    {clip.isFbx && <em className="animlab-badge">fbx</em>}
                    {clip.tags.includes('sourced') && <em className="animlab-badge">sourced</em>}
                    {clip.tags.includes('community') && (
                      <em className="animlab-badge">community</em>
                    )}
                    {clip.tags.includes('manual') && <em className="animlab-badge">manual</em>}
                  </span>
                </div>
                <div className="animlab-actions">
                  <button type="button" title="Play once" onClick={() => play(clip.name, false)}>
                    ▶ once
                  </button>
                  <button type="button" title="Play looping" onClick={() => play(clip.name, true)}>
                    ⟳ loop
                  </button>
                  <button
                    type="button"
                    title="Play as an additive layer over the running idle"
                    onClick={() => playAdditive(clip.name)}
                  >
                    ⧉ additive
                  </button>
                  <button
                    type="button"
                    title="Copy a manifest.local.json entry for this clip"
                    onClick={() => copyEntry(clip)}
                  >
                    {copied === clip.name ? 'copied ✓' : 'copy entry'}
                  </button>
                </div>
              </div>
            ))}
          </section>
        ))}
      </div>

      <footer className="animlab-footer">
        Vetting checklist: watch a full loop for pose snaps at the seam, limb
        distortion, foot slide, and how the crossfade back to idle reads.
        Local clips: <code>static/animations/mixamo/</code> or{' '}
        <code>local/</code> + an entry in{' '}
        <code>static/animations/manifest.local.json</code> (gitignored), then
        reload. See <code>docs/animations.md</code>.
      </footer>
    </div>
  )
}

export default AnimLab
