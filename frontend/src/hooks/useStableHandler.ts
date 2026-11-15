/**
 * Stable-identity wrapper for a handler that closes over frequently-changing
 * state. Used by ChatInterface for the row-level callbacks (retry/
 * regenerate/swipe/edit/rewind) passed into the memoized MessageRow: the ref
 * always points at the latest closure, but the returned callback's identity
 * never changes, so MessageRow's props-equality fast path still holds
 * across renders that only touch unrelated state.
 *
 * Replaces five copy-pasted `useRef` + `useEffect` + `useCallback` triplets
 * that all did exactly this.
 */
import { useCallback, useEffect, useRef } from 'react'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function useStableHandler<T extends (...args: any[]) => any>(handler: T): T {
  const handlerRef = useRef(handler)
  useEffect(() => {
    handlerRef.current = handler
  })
  return useCallback((...args: Parameters<T>) => handlerRef.current(...args), []) as T
}
