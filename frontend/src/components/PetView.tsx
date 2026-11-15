/**
 * Desktop-pet overlay: the VRM avatar alone in a transparent, frameless,
 * always-on-top Tauri window (`?view=pet`). Voice-first — mic button /
 * hands-free / poke; the transcript lives in the main window, which shares
 * the same session via same-origin localStorage.
 */
import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import { isTauri } from '../platform'
import { usePetChat } from '../hooks/usePetChat'
import {
  isFullscreenCourtesyEnabled,
  setFullscreenCourtesyEnabled,
  setPetDndUntil,
  usePetDndStatus,
} from '../hooks/usePetPresence'
import { t, useLocale } from '../i18n'
import {
  isPetRoamEnabled,
  PET_ROAM_CHANGE_EVENT,
  setPetRoamEnabled,
  type PetRoamChangeDetail,
} from '../vrm/petRoam'
import {
  isPetTucked,
  peekPetWindow,
  PET_TUCK_CHANGE_EVENT,
  tuckPetWindow,
  untuckPetWindow,
} from '../vrm/petTuck'
import './PetView.css'

const VRMViewer = lazy(() => import('./VRMViewer'))

/** localStorage `kurisu_volume`: string float "0".."1", default 1. Contract
 * only — a sibling agent wires the actual audio gain node off the
 * `kurisu-volume-changed` window event this dispatches. */
const VOLUME_KEY = 'kurisu_volume'
/** localStorage `kurisu_pet_preset`: "full" | "compact". A sibling agent's
 * camera framing reads this (directly and via the change event below); this
 * hook only owns the toggle button + persistence + the Tauri window resize. */
const PRESET_KEY = 'kurisu_pet_preset'
type PetPreset = 'full' | 'compact'

const loadVolume = (): number => {
  try {
    const raw = window.localStorage.getItem(VOLUME_KEY)
    const value = raw === null ? 1 : Number(raw)
    return Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 1
  } catch {
    return 1
  }
}

const loadPreset = (): PetPreset =>
  window.localStorage.getItem(PRESET_KEY) === 'compact' ? 'compact' : 'full'

/** Rolling caption history: current line plus up to this many previous
 * replies, revealed stacked-and-dimmed on hover (feature 3b). */
const CAPTION_HISTORY_LIMIT = 3
const CAPTION_FADE_MS = 11400
const CAPTION_CLEAR_MS = 12000

const PetView = () => {
  useLocale()
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)
  const {
    isBusy,
    lastReply,
    lastSent,
    lastEmotion,
    sendText,
    mic,
    handsFree,
    screenCompanion,
    toggleScreenCompanion,
    lastPeekAt,
    readTogether,
    toggleReadTogether,
    lastPageAt,
    watchTogether,
    toggleWatchTogether,
    lastWatchFrameAt,
    firePoke,
    wakeOnMicPress,
    voiceWarmingHint,
    focus,
  } = usePetChat(textareaRef)
  const [caption, setCaption] = useState<{ text: string; leaving: boolean } | null>(null)
  // Previous replies, most recent first, capped at CAPTION_HISTORY_LIMIT —
  // shown stacked above the live caption on hover (3b). Pushed only when a
  // NEW final reply lands (see the caption effect below), never for the
  // live streaming updates to the current line.
  const [captionHistory, setCaptionHistory] = useState<string[]>([])
  // Pin (3c): while true the caption's fade/clear timers never run. Click
  // toggles it; the icon shows in the caption itself.
  const [captionPinned, setCaptionPinned] = useState(false)
  // Hover pause (3a): while the pointer is over the caption, fade/clear
  // timers are suspended; leaving restarts the full timeout (simpler than
  // resuming a partial one, and kinder — you get the full read time back).
  const [captionHovered, setCaptionHovered] = useState(false)
  const [draft, setDraft] = useState('')
  // Brief "» what you said/typed" echo — the only sent-confirmation the pet
  // window has, since the transcript lives in the main window.
  const [sentEcho, setSentEcho] = useState<{ text: string; leaving: boolean } | null>(null)

  // Roam glide (opt-in desktop roaming, vrm/petRoam): while the Rust tween
  // strolls the window along the screen edge, the stage leans/bobs slightly
  // in the travel direction via CSS (see .pet-stage.roaming-* in
  // PetView.css). This is the vetted-clip-free fallback — petRoam itself
  // plays a manifest role-"walk" clip instead if one is ever promoted to
  // the eager set. 0 = not roaming.
  const [roamLean, setRoamLean] = useState<-1 | 0 | 1>(0)
  useEffect(() => {
    const handleRoamChange = (event: Event) => {
      const detail = (event as CustomEvent<PetRoamChangeDetail>).detail
      setRoamLean(detail?.roaming ? detail.direction : 0)
    }
    window.addEventListener(PET_ROAM_CHANGE_EVENT, handleRoamChange)
    return () => window.removeEventListener(PET_ROAM_CHANGE_EVENT, handleRoamChange)
  }, [])

  // Transparent page background for the overlay window.
  useEffect(() => {
    document.documentElement.classList.add('pet-mode')
    document.body.classList.add('pet-mode')
    return () => {
      document.documentElement.classList.remove('pet-mode')
      document.body.classList.remove('pet-mode')
    }
  }, [])

  // Screen-peek pulse (feature 4): the button glows briefly whenever a
  // capture actually succeeds. Keyed on lastPeekAt (a timestamp, not a
  // boolean) so two peeks in a row each re-trigger the CSS animation even
  // though the "pulsing" boolean itself is the same true->true transition —
  // toggling pulsing off then back on via a timer is what makes that work.
  const [peekPulsing, setPeekPulsing] = useState(false)
  useEffect(() => {
    if (lastPeekAt === null) return
    setPeekPulsing(true)
    const timer = window.setTimeout(() => setPeekPulsing(false), 1500)
    return () => window.clearTimeout(timer)
  }, [lastPeekAt])

  // Read-together page pulse: same pattern as the screen-peek pulse above,
  // but keyed on lastPageAt so it fires only when a page was actually
  // POSTed (not on every detected-but-skipped page turn).
  const [pagePulsing, setPagePulsing] = useState(false)
  useEffect(() => {
    if (lastPageAt === null) return
    setPagePulsing(true)
    const timer = window.setTimeout(() => setPagePulsing(false), 1500)
    return () => window.clearTimeout(timer)
  }, [lastPageAt])

  // Watch-together frame pulse: same pattern as the read-together page
  // pulse above, keyed on lastWatchFrameAt so it fires only when a frame
  // was actually POSTed (not on every sampled-but-skipped-as-static tick).
  const [watchPulsing, setWatchPulsing] = useState(false)
  useEffect(() => {
    if (lastWatchFrameAt === null) return
    setWatchPulsing(true)
    const timer = window.setTimeout(() => setWatchPulsing(false), 1500)
    return () => window.clearTimeout(timer)
  }, [lastWatchFrameAt])

  // Fading caption of her last line, since there's no transcript here.
  // It eases out (.leaving) before unmounting instead of vanishing —
  // EXCEPT while hovered or pinned, when the timers never start at all
  // (3a/3c). Unpinning happens on the NEXT line — a still-pinned caption
  // from three replies ago would otherwise silently block every future one
  // from ever fading, which reads as a stuck/broken window rather than a
  // deliberate pin.
  const prevReplyRef = useRef<string | null>(null)
  useEffect(() => {
    if (!lastReply) return
    // A new reply supersedes any prior pin — pin is "keep THIS one", not
    // "freeze the caption area forever".
    setCaptionPinned(false)
    // History gets the reply this caption is REPLACING, not the new one —
    // so it always reads as "what she said before this".
    setCaptionHistory((prev) => {
      const previous = prevReplyRef.current
      if (!previous || previous === lastReply) return prev
      return [previous, ...prev].slice(0, CAPTION_HISTORY_LIMIT)
    })
    prevReplyRef.current = lastReply
    setCaption({ text: lastReply, leaving: false })
  }, [lastReply])

  // Timer effect is separate from the content effect above so hover/pin can
  // restart or suspend the SAME caption's timers without touching history
  // or re-triggering the "supersede the pin" logic on every hover toggle.
  useEffect(() => {
    if (!caption || captionHovered || captionPinned) return
    const fade = window.setTimeout(
      () => setCaption((current) => (current ? { ...current, leaving: true } : current)),
      CAPTION_FADE_MS,
    )
    const clear = window.setTimeout(() => setCaption(null), CAPTION_CLEAR_MS)
    return () => {
      window.clearTimeout(fade)
      window.clearTimeout(clear)
    }
    // Re-running on caption?.text (not the whole object, which also carries
    // `leaving`) means the timers don't restart the instant `.leaving` flips
    // true — that would cancel the very clear() it just scheduled.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [caption?.text, captionHovered, captionPinned])

  useEffect(() => {
    if (!lastSent) return
    setSentEcho({ text: lastSent.text, leaving: false })
    const fade = window.setTimeout(
      () => setSentEcho((current) => (current ? { ...current, leaving: true } : current)),
      3600,
    )
    const clear = window.setTimeout(() => setSentEcho(null), 4200)
    return () => {
      window.clearTimeout(fade)
      window.clearTimeout(clear)
    }
  }, [lastSent])

  const submitDraft = () => {
    const text = draft.trim()
    if (!text || isBusy) return
    sendText(text)
    setDraft('')
  }

  // Do-not-disturb (feature 1): status comes from the shared hook (polls
  // localStorage so a timed period visibly clears itself); the chooser is
  // just local UI state for which inline popover is open.
  const dndStatus = usePetDndStatus()
  const [dndChooserOpen, setDndChooserOpen] = useState(false)
  // Tucked-away state (vrm/petTuck): survives restarts via localStorage —
  // the persisted window rect is the tucked position, so the overlay must
  // rearm on mount or a restarted tucked pet would be stranded half
  // off-screen with no way back.
  const [tucked, setTucked] = useState(isPetTucked)
  useEffect(() => {
    const sync = () => setTucked(isPetTucked())
    window.addEventListener(PET_TUCK_CHANGE_EVENT, sync)
    return () => window.removeEventListener(PET_TUCK_CHANGE_EVENT, sync)
  }, [])
  // Dismiss the popover on outside click/Escape, same as any inline menu —
  // otherwise it stays pinned open until the next unrelated re-render
  // happens to close it.
  useEffect(() => {
    if (!dndChooserOpen) return
    const dismiss = (event: MouseEvent | KeyboardEvent) => {
      if (event instanceof KeyboardEvent) {
        if (event.key === 'Escape') setDndChooserOpen(false)
        return
      }
      const target = event.target as Node | null
      if (target && !(target as HTMLElement).closest?.('.pet-dnd-anchor')) {
        setDndChooserOpen(false)
      }
    }
    window.addEventListener('pointerdown', dismiss)
    window.addEventListener('keydown', dismiss)
    return () => {
      window.removeEventListener('pointerdown', dismiss)
      window.removeEventListener('keydown', dismiss)
    }
  }, [dndChooserOpen])
  const chooseDnd = (minutes: number | 'indefinite') => {
    setPetDndUntil(minutes === 'indefinite' ? 'indefinite' : Date.now() + minutes * 60_000)
    setDndChooserOpen(false)
  }
  const clearDnd = () => {
    setPetDndUntil(null)
    setDndChooserOpen(false)
  }

  // Shared-activity chooser (Screen/Read/Watch merged behind one button —
  // three permanent toggles for variations of "she sees my screen" crowded
  // the controls row; the popover matches the DND pattern above). Read and
  // watch are mutually exclusive here: both are explicit "she follows this
  // with me" activities and running two capture loops at once is never what
  // the user meant. Screen glances stay an independent ambient preference —
  // they already self-pause while either activity runs (usePetAmbient's
  // peek gate), so choosing an activity never silently flips the persisted
  // screen pref off.
  const [togetherChooserOpen, setTogetherChooserOpen] = useState(false)
  useEffect(() => {
    if (!togetherChooserOpen) return
    const dismiss = (event: MouseEvent | KeyboardEvent) => {
      if (event instanceof KeyboardEvent) {
        if (event.key === 'Escape') setTogetherChooserOpen(false)
        return
      }
      const target = event.target as Node | null
      if (target && !(target as HTMLElement).closest?.('.pet-together-anchor')) {
        setTogetherChooserOpen(false)
      }
    }
    window.addEventListener('pointerdown', dismiss)
    window.addEventListener('keydown', dismiss)
    return () => {
      window.removeEventListener('pointerdown', dismiss)
      window.removeEventListener('keydown', dismiss)
    }
  }, [togetherChooserOpen])
  const chooseReadTogether = () => {
    if (!readTogether && watchTogether) toggleWatchTogether()
    toggleReadTogether()
    setTogetherChooserOpen(false)
  }
  const chooseWatchTogether = () => {
    if (!watchTogether && readTogether) toggleReadTogether()
    toggleWatchTogether()
    setTogetherChooserOpen(false)
  }
  const chooseScreenCompanion = () => {
    toggleScreenCompanion()
    setTogetherChooserOpen(false)
  }
  const togetherActive = screenCompanion || readTogether || watchTogether
  // Session activities outrank the ambient pref for the button's face: they
  // are what she is actually doing right now.
  const togetherIcon = watchTogether ? '👀' : readTogether ? '📖' : screenCompanion ? '👁' : '👀'

  // Fullscreen courtesy opt-out (feature: step aside for fullscreen apps).
  // Lives as a checkbox row inside the existing DND popover rather than its
  // own always-visible button — it's an infrequently-touched preference,
  // not a status the user needs to glance at, so it doesn't earn a
  // permanent spot in the drag-bar row. Tauri-only (the underlying feature
  // is), so the row itself doesn't render outside Tauri.
  const [courtesyEnabled, setCourtesyEnabled] = useState(isFullscreenCourtesyEnabled)
  const toggleCourtesyEnabled = () => {
    setCourtesyEnabled((prev) => {
      const next = !prev
      setFullscreenCourtesyEnabled(next)
      return next
    })
  }
  // Focus / co-working mode (feature: quiet company while heads-down).
  // Server-scoped state (useFocusMode polls /api/focus) rather than a
  // localStorage contract like DND — the chooser is local UI only.
  const [focusChooserOpen, setFocusChooserOpen] = useState(false)
  useEffect(() => {
    if (!focusChooserOpen) return
    const dismiss = (event: MouseEvent | KeyboardEvent) => {
      if (event instanceof KeyboardEvent) {
        if (event.key === 'Escape') setFocusChooserOpen(false)
        return
      }
      const target = event.target as Node | null
      if (target && !(target as HTMLElement).closest?.('.pet-focus-anchor')) {
        setFocusChooserOpen(false)
      }
    }
    window.addEventListener('pointerdown', dismiss)
    window.addEventListener('keydown', dismiss)
    return () => {
      window.removeEventListener('pointerdown', dismiss)
      window.removeEventListener('keydown', dismiss)
    }
  }, [focusChooserOpen])
  const chooseFocus = (minutes: number) => {
    void focus.start(minutes)
    setFocusChooserOpen(false)
  }
  const focusTooltip = focus.active
    ? t('Focusing — {time} left. Click to end early.', {
        time:
          focus.remainingSeconds != null && focus.remainingSeconds >= 3600
            ? `${Math.ceil(focus.remainingSeconds / 3600)}h`
            : `${Math.max(1, Math.ceil((focus.remainingSeconds ?? 0) / 60))}m`,
      })
    : t('Going heads-down for a while? She stays quiet and checks in when you get back.')

  const dndTooltip = (() => {
    if (dndStatus.until === 'indefinite') return t('Quiet until turned off')
    if (typeof dndStatus.until === 'number') {
      const remainingMin = Math.max(1, Math.ceil((dndStatus.until - Date.now()) / 60_000))
      const time = remainingMin >= 60 ? `${Math.ceil(remainingMin / 60)}h` : `${remainingMin}m`
      return t('Quiet for {time} more', { time })
    }
    return t('Do not disturb: mute greetings, reminders, reach-outs, and idle chatter')
  })()

  // Volume slider (feature 2): persistence + broadcast only — a sibling
  // agent reads kurisu_volume / listens for kurisu-volume-changed to apply
  // actual gain. Local state just keeps the slider itself responsive.
  const [volume, setVolume] = useState(loadVolume)
  // Roaming toggle: promoted out of Settings on 2026-07-28 for
  // discoverability, then folded into the DND popover's preference rows on
  // 2026-08-03 (a persisted set-and-forget flag never earned a permanent
  // controls-row button). localStorage-backed like the other instant client
  // prefs; the roam scheduler re-reads the flag at every cadence tick, so
  // this applies without any relaunch.
  const [roamEnabled, setRoamEnabled] = useState(() => isPetRoamEnabled())
  const handleVolumeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const next = Number(event.target.value)
    setVolume(next)
    try {
      window.localStorage.setItem(VOLUME_KEY, String(next))
    } catch {
      // Preference just won't persist across reloads.
    }
    window.dispatchEvent(new CustomEvent('kurisu-volume-changed', { detail: { volume: next } }))
  }

  // Size preset (feature 6a): full <-> compact. Persistence + broadcast are
  // this hook's job; a sibling agent reframes the camera off the event, and
  // the Tauri command (once it exists) actually resizes the window — wrapped
  // in try/catch so an as-yet-missing command degrades to camera-only.
  const [preset, setPreset] = useState<PetPreset>(loadPreset)
  const togglePreset = async () => {
    const next: PetPreset = preset === 'full' ? 'compact' : 'full'
    setPreset(next)
    try {
      window.localStorage.setItem(PRESET_KEY, next)
    } catch {
      // Preference just won't persist across reloads.
    }
    window.dispatchEvent(new CustomEvent('kurisu-pet-preset-changed', { detail: { preset: next } }))
    if (!isTauri()) return
    try {
      const { invoke } = await import('@tauri-apps/api/core')
      await invoke('set_pet_window_preset', { preset: next })
    } catch {
      // Command may not exist yet (owned by a sibling agent) — the camera
      // reframe from the event above still happens either way.
    }
  }

  const handleClickThrough = async () => {
    if (!isTauri()) return
    // Parting line before she goes quiet/unclickable (6b) — best-effort,
    // fire-and-forget so a slow/failed poke never delays the actual
    // click-through toggle the user asked for.
    void firePoke('click_through')
    try {
      const { invoke } = await import('@tauri-apps/api/core')
      await invoke('set_pet_click_through', { enabled: true })
    } catch {
      // Not fatal; the window simply stays clickable.
    }
  }

  const handleOpenChat = async () => {
    if (!isTauri()) return
    try {
      const { invoke } = await import('@tauri-apps/api/core')
      await invoke('show_main_window')
    } catch {
      // Not fatal; the tray's "Open chat window" does the same thing.
    }
  }

  const handleClose = async () => {
    if (!isTauri()) {
      window.close()
      return
    }
    try {
      const { getCurrentWindow } = await import('@tauri-apps/api/window')
      await getCurrentWindow().close()
    } catch {
      // Ignore; tray can still close it.
    }
  }

  return (
    <div className="pet-view">
      {/* Tucked overlay (vrm/petTuck): while tucked, the whole window is a
          single "come back" affordance — hover the sliver and she leans
          out, click and she returns. Intercepts pokes by construction. */}
      {tucked && (
        <div
          className="pet-tucked-overlay"
          role="button"
          aria-label={t('Bring her back on screen')}
          title={t('Click to bring her back')}
          onMouseEnter={() => void peekPetWindow(true)}
          onMouseLeave={() => void peekPetWindow(false)}
          onClick={() => {
            setTucked(false)
            void untuckPetWindow()
          }}
        />
      )}
      {/* data-tauri-drag-region: the bar owns window dragging (the canvas
          keeps clicks for poke). */}
      <div className="pet-drag-bar" data-tauri-drag-region>
        <span className="pet-drag-title" data-tauri-drag-region>
          KURISU
        </span>
        <div className="pet-drag-actions">
          {isTauri() && (
            <button
              type="button"
              title={t('Open the chat window (transcript, memory, settings)')}
              onClick={() => void handleOpenChat()}
            >
              💬
            </button>
          )}
          <button
            type="button"
            className={preset === 'compact' ? 'active' : ''}
            title={preset === 'compact' ? t('Switch to full size') : t('Switch to compact size')}
            aria-pressed={preset === 'compact'}
            onClick={() => void togglePreset()}
          >
            {preset === 'compact' ? '⤢' : '⤡'}
          </button>
          <div className="pet-dnd-anchor pet-focus-anchor">
            <button
              type="button"
              className={focus.active ? 'active' : ''}
              aria-pressed={focus.active}
              title={focusTooltip}
              onClick={() => {
                if (focus.active) {
                  void focus.end()
                } else {
                  setFocusChooserOpen((open) => !open)
                }
              }}
            >
              🎯
            </button>
            {focusChooserOpen && !focus.active && (
              <div className="pet-dnd-chooser" role="menu">
                <button type="button" role="menuitem" onClick={() => chooseFocus(25)}>
                  {t('Focus for 25 minutes')}
                </button>
                <button type="button" role="menuitem" onClick={() => chooseFocus(50)}>
                  {t('Focus for 50 minutes')}
                </button>
                <button type="button" role="menuitem" onClick={() => chooseFocus(90)}>
                  {t('Focus for 90 minutes')}
                </button>
              </div>
            )}
          </div>
          <div className="pet-dnd-anchor">
            <button
              type="button"
              className={dndStatus.active ? 'active' : ''}
              aria-pressed={dndStatus.active}
              title={dndTooltip}
              onClick={() => {
                if (dndStatus.active) {
                  clearDnd()
                } else {
                  setDndChooserOpen((open) => !open)
                }
              }}
            >
              🔕
            </button>
            {dndChooserOpen && !dndStatus.active && (
              <div className="pet-dnd-chooser" role="menu">
                <button type="button" role="menuitem" onClick={() => chooseDnd(30)}>
                  {t('Quiet for 30 minutes')}
                </button>
                <button type="button" role="menuitem" onClick={() => chooseDnd(60)}>
                  {t('Quiet for 1 hour')}
                </button>
                <button type="button" role="menuitem" onClick={() => chooseDnd('indefinite')}>
                  {t('Quiet until I turn it off')}
                </button>
                {/* Click-through moved here from the drag bar (2026-08-03
                    calm pass): it's a one-way action whose only recovery is
                    the tray menu, and it sat directly beside the × close
                    button — a popover row makes it deliberate instead of a
                    misclick away. */}
                {isTauri() && (
                  <button
                    type="button"
                    role="menuitem"
                    onClick={() => {
                      setDndChooserOpen(false)
                      void handleClickThrough()
                    }}
                    title={t(
                      'Click-through mode: clicks pass to the desktop (restore from the tray menu)',
                    )}
                  >
                    ○ {t('Click-through mode')}
                  </button>
                )}
                {isTauri() && (
                  <label className="pet-dnd-chooser-option">
                    <input
                      type="checkbox"
                      checked={courtesyEnabled}
                      onChange={toggleCourtesyEnabled}
                    />
                    {t('Step aside during fullscreen apps')}
                  </label>
                )}
                {/* Roam moved here from its own controls-row button
                    (2026-08-03 calm pass): it's a set-and-forget persisted
                    preference like the courtesy row above, not a per-session
                    activity — and roam already yields to DND, so the two
                    belong to the same "how she behaves ambiently" menu. */}
                {/* Tuck away (vrm/petTuck): slides her mostly off the
                    nearest side edge and quiets presence until she's
                    brought back — hover the visible sliver and she peeks
                    out; click it to bring her back. */}
                {isTauri() && (
                  <button
                    type="button"
                    role="menuitem"
                    onClick={() => {
                      setDndChooserOpen(false)
                      void tuckPetWindow().then((ok) => {
                        if (ok) setTucked(true)
                      })
                    }}
                    title={t(
                      'She tucks herself mostly off-screen and stays quiet. Hover the visible edge and she peeks out; click it to bring her back.',
                    )}
                  >
                    ⇤ {t('Tuck away at the edge')}
                  </button>
                )}
                {isTauri() && (
                  <label
                    className="pet-dnd-chooser-option"
                    title={t(
                      'Roam: she occasionally strolls along the edges of your screen — first stroll a minute or two after you leave her alone, then every few minutes. She yields while you interact and stops if you grab her.',
                    )}
                  >
                    <input
                      type="checkbox"
                      checked={roamEnabled}
                      onChange={() => {
                        const next = !roamEnabled
                        setRoamEnabled(next)
                        setPetRoamEnabled(next)
                      }}
                    />
                    🚶 {t('Roam the screen edges')}
                  </label>
                )}
              </div>
            )}
          </div>
          <button type="button" title={t('Close pet window')} onClick={() => void handleClose()}>
            ×
          </button>
        </div>
      </div>

      <div
        className={`pet-stage ${
          roamLean === 0 ? '' : roamLean === 1 ? 'roaming-right' : 'roaming-left'
        }`}
      >
        {/* Emotion-tinted ambient glow behind the avatar while she's
            thinking/speaking; sits under the canvas via z-index. */}
        <div
          className={`pet-mood-glow ${isBusy || caption ? 'active' : ''}`}
          data-emotion={lastEmotion ?? 'neutral'}
          aria-hidden="true"
        />
        <Suspense fallback={null}>
          <VRMViewer variant="pet" />
        </Suspense>
      </div>

      {caption && (
        <div
          className={`pet-caption ${caption.leaving ? 'leaving' : ''} ${
            captionPinned ? 'pinned' : ''
          }`}
          onMouseEnter={() => setCaptionHovered(true)}
          onMouseLeave={() => setCaptionHovered(false)}
          onClick={() => setCaptionPinned((pinned) => !pinned)}
          title={captionPinned ? t('Pinned — click to let this fade again') : t('Click to pin — keeps this on screen')}
          role="button"
          tabIndex={0}
          onKeyDown={(event) => {
            if (event.key === 'Enter' || event.key === ' ') {
              event.preventDefault()
              setCaptionPinned((pinned) => !pinned)
            }
          }}
        >
          {/* History (3b): only appears on hover, dimmer/smaller, oldest at
              the top so reading order matches chat order. */}
          {captionHovered && captionHistory.length > 0 && (
            <div className="pet-caption-history">
              {[...captionHistory].reverse().map((line, index) => (
                <div key={index} className="pet-caption-history-line">
                  {line}
                </div>
              ))}
            </div>
          )}
          <div className="pet-caption-current">
            {captionPinned && <span className="pet-caption-pin" aria-hidden="true">📌</span>}
            {caption.text}
          </div>
        </div>
      )}
      {sentEcho && (
        <div
          className={`pet-caption pet-sent-echo ${sentEcho.leaving ? 'leaving' : ''}`}
          role="status"
          aria-label={t('Message sent')}
        >
          » {sentEcho.text}
        </div>
      )}
      {mic.micHint && <div className="pet-caption pet-hint">{mic.micHint}</div>}
      {/* Voice-warming honesty chip (one-shot, see usePetChat): the greeting
          arrived as silent text on a GPU install still warming the clone —
          without this a silent first line just reads as broken audio. */}
      {voiceWarmingHint && <div className="pet-caption pet-hint">{voiceWarmingHint}</div>}

      <div
        className={`pet-controls ${
          isBusy || handsFree.handsFree || mic.micStatus === 'recording' || dndStatus.active
            ? 'engaged'
            : ''
        }`}
      >
        <button
          type="button"
          className={`pet-mic ${mic.micStatus === 'recording' ? 'recording' : ''}`}
          onPointerDown={(event) => {
            event.preventDefault()
            // Doze wake (c): pressing the mic is an interaction that isn't
            // itself a chat message yet, so it gets the groggy poke when it
            // actually interrupts a doze (unlike sendVoiceTurn's wake,
            // which never pokes — see usePetChat's wakeOnMicPress comment).
            wakeOnMicPress()
            void mic.startRecording()
          }}
          onPointerUp={() => void mic.stopRecordingAndTranscribe()}
          onPointerCancel={mic.cancelRecording}
          disabled={mic.micAvailable !== true || mic.micPermissionDenied}
          aria-pressed={mic.micStatus === 'recording'}
          title={
            mic.micAvailable === false
              ? mic.micUnavailableReason ?? t('Speech-to-text not installed')
              : t('Hold to talk')
          }
        >
          🎙
        </button>
        <button
          type="button"
          className={`pet-toggle ${handsFree.handsFree ? 'active' : ''}`}
          onClick={handsFree.toggleHandsFree}
          disabled={mic.micAvailable !== true || mic.micPermissionDenied}
          aria-pressed={handsFree.handsFree}
          title={t('Hands-free: open mic — speak and she answers when you pause')}
        >
          {t('Hands-free')}
        </button>
        <div className="pet-together-anchor">
          <button
            type="button"
            className={`pet-toggle ${watchTogether ? 'pet-watch-toggle' : ''} ${
              readTogether ? 'pet-read-toggle' : ''
            } ${togetherActive ? 'active' : ''} ${peekPulsing ? 'peek-pulse' : ''} ${
              pagePulsing ? 'page-pulse' : ''
            } ${watchPulsing ? 'watch-pulse' : ''}`}
            onClick={() => setTogetherChooserOpen((open) => !open)}
            aria-pressed={togetherActive}
            aria-haspopup="menu"
            aria-expanded={togetherChooserOpen}
            title={t(
              'Together: shared activities — screen glances, reading along, or watching with you',
            )}
          >
            {togetherIcon} {t('Together')}
          </button>
          {togetherChooserOpen && (
            <div className="pet-dnd-chooser pet-chooser-up" role="menu">
              {isTauri() && (
                <button
                  type="button"
                  role="menuitemcheckbox"
                  aria-checked={screenCompanion}
                  className={screenCompanion ? 'active' : ''}
                  onClick={chooseScreenCompanion}
                  title={t(
                    "Screen glances: every few minutes she reacts to what you're doing. Nothing is stored.",
                  )}
                >
                  👁 {t('Screen glances')}
                </button>
              )}
              <button
                type="button"
                role="menuitemcheckbox"
                aria-checked={readTogether}
                className={readTogether ? 'active' : ''}
                onClick={chooseReadTogether}
                title={
                  isTauri()
                    ? t(
                        'Read together: she reads along — a snapshot per page turn, kept only for this session.',
                      )
                    : t(
                        'Read together: pick the book/manga window to share — she reads along, one snapshot per page turn, kept only for this session.',
                      )
                }
              >
                📖 {t('Read together')}
              </button>
              <button
                type="button"
                role="menuitemcheckbox"
                aria-checked={watchTogether}
                className={watchTogether ? 'active' : ''}
                onClick={chooseWatchTogether}
                title={
                  isTauri()
                    ? t(
                        'Watch together: she glances at the screen every minute or so — frames stay only in this session.',
                      )
                    : t(
                        'Watch together: pick a window to share — she glances at it every minute or so, frames stay only in this session.',
                      )
                }
              >
                👀 {t('Watch together')}
              </button>
            </div>
          )}
        </div>
        <label className="pet-volume" title={t('Volume')}>
          <span aria-hidden="true">🔊</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={volume}
            onChange={handleVolumeChange}
            aria-label={t('Volume')}
          />
        </label>
        {isBusy && <span className="pet-busy" aria-label={t('Thinking')} />}
        {screenCompanion && (
          <span className="pet-listening pet-watching" role="status">
            <span className="pet-listening-dot" aria-hidden="true" />
            {t('WATCHING')}
          </span>
        )}
        {readTogether && (
          <span className="pet-listening pet-reading" role="status">
            <span className="pet-listening-dot" aria-hidden="true" />
            {t('READING')}
          </span>
        )}
        {watchTogether && (
          <span className="pet-listening pet-watching-together" role="status">
            <span className="pet-listening-dot" aria-hidden="true" />
            {t('WATCHING TOGETHER')}
          </span>
        )}
        {handsFree.handsFreeStatus === 'listening' && (
          <span className="pet-listening" role="status">
            <span className="pet-listening-dot" aria-hidden="true" />
            {t('LISTENING')}
          </span>
        )}
        {dndStatus.active && (
          <span className="pet-listening pet-quiet" role="status">
            <span className="pet-listening-dot" aria-hidden="true" />
            {t('QUIET')}
          </span>
        )}
      </div>

      {/* Typed input — same turn pipeline as voice. Bound to textareaRef so
          hold-Space push-to-talk stays suppressed while typing. */}
      <div className="pet-composer">
        <textarea
          ref={textareaRef}
          className="pet-composer-input"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault()
              submitDraft()
            }
          }}
          placeholder={t('Type to her…')}
          rows={1}
          aria-label={t('Message Kurisu')}
        />
        <button
          type="button"
          className="pet-composer-send"
          onClick={submitDraft}
          disabled={!draft.trim() || isBusy}
          title={t('Send')}
          aria-label={t('Send message')}
        >
          ➤
        </button>
      </div>
    </div>
  )
}

export default PetView
