import { useEffect, useRef, useState } from 'react'
import './NixieMeter.css'

/**
 * Nixie-tube divergence meter — the boot-sequence signature visual shared by
 * LoadingScreen and DesktopBackendWait. Renders SETTLE_VALUE's digits as
 * individually lit IN-14-style tubes; while `settled` is false the digits
 * flicker through random values on a per-tube desynced interval so the roll
 * reads as mechanical rather than a single uniform blink.
 */

export const SETTLE_VALUE = '1.048596'

const randomDigit = () => Math.floor(Math.random() * 10).toString()

// Per-tube tick period desync: real nixie driver boards don't share a clock,
// and identical intervals across tubes read as one flat strobe instead of a
// mechanical roll.
const tickMs = (index: number) => 60 + ((index * 37) % 31)

interface NixieMeterProps {
  /** False keeps every digit rolling at random; true settles left-to-right
   * onto SETTLE_VALUE and holds. */
  settled: boolean
  /** Optional status text rendered under the meter (reuses the caller's
   * existing copy — this component owns only the digit tubes). */
  caption?: string
  className?: string
}

const NixieMeter = ({ settled, caption, className }: NixieMeterProps) => {
  const chars = SETTLE_VALUE.split('')
  const digitCount = chars.filter((c) => c !== '.').length

  const [reducedMotion, setReducedMotion] = useState(() =>
    typeof window !== 'undefined' && window.matchMedia
      ? window.matchMedia('(prefers-reduced-motion: reduce)').matches
      : false
  )
  // One state slot per digit tube (decimal point excluded — it never rolls).
  const [digits, setDigits] = useState<string[]>(() => Array(digitCount).fill('0'))
  const settledRef = useRef(settled)
  settledRef.current = settled

  useEffect(() => {
    const mql = window.matchMedia?.('(prefers-reduced-motion: reduce)')
    if (!mql) return
    const onChange = () => setReducedMotion(mql.matches)
    mql.addEventListener?.('change', onChange)
    return () => mql.removeEventListener?.('change', onChange)
  }, [])

  // One interval per tube, each on its own desynced period. Cleared on
  // unmount and whenever reducedMotion flips on so a static render never
  // leaves a hidden roll running underneath it.
  useEffect(() => {
    if (reducedMotion) return
    const timers = Array.from({ length: digitCount }, (_, index) =>
      window.setInterval(() => {
        if (settledRef.current) return
        setDigits((prev) => {
          const next = [...prev]
          next[index] = randomDigit()
          return next
        })
      }, tickMs(index))
    )
    return () => timers.forEach((id) => window.clearInterval(id))
  }, [reducedMotion, digitCount])

  // Settle left-to-right: each tube stops on its target digit with a short
  // stagger so the meter reads as a mechanical lock-in rather than a snap.
  useEffect(() => {
    if (!settled) return
    const targets = chars.filter((c) => c !== '.')
    const timers = targets.map((digit, index) =>
      window.setTimeout(
        () => {
          setDigits((prev) => {
            const next = [...prev]
            next[index] = digit
            return next
          })
        },
        reducedMotion ? 0 : index * 70
      )
    )
    return () => timers.forEach((id) => window.clearTimeout(id))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settled, reducedMotion])

  const staticDigits = reducedMotion ? chars.filter((c) => c !== '.') : digits
  let digitCursor = 0

  return (
    <div
      className={`nixie-meter ${className ?? ''}`.trim()}
      role="img"
      aria-label={settled ? `Divergence meter: ${SETTLE_VALUE}` : 'Divergence meter calibrating'}
    >
      <div className="nixie-tubes">
        {chars.map((char, i) => {
          if (char === '.') {
            return (
              <span key={`dot-${i}`} className="nixie-tube nixie-tube-dot" aria-hidden="true">
                <span className="nixie-ghost">8</span>
                <span className="nixie-dot" />
              </span>
            )
          }
          const value = staticDigits[digitCursor] ?? '0'
          digitCursor += 1
          return (
            <span
              key={i}
              className={`nixie-tube ${settled ? 'nixie-tube-settled' : ''}`}
              aria-hidden="true"
            >
              <span className="nixie-ghost">8</span>
              <span className="nixie-digit">{value}</span>
            </span>
          )
        })}
      </div>
      {caption ? <div className="nixie-caption">{caption}</div> : null}
    </div>
  )
}

export default NixieMeter
