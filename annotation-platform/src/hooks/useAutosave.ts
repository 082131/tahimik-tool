import { useEffect, useRef, useState, useCallback } from 'react';

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface UseAutosaveReturn<T> {
  /** The value loaded from localStorage on mount (or undefined). */
  savedValue: T | undefined;
  /** Remove the saved value from localStorage. */
  clearSaved: () => void;
}

/* ------------------------------------------------------------------ */
/*  Hook                                                               */
/* ------------------------------------------------------------------ */

/**
 * Debounced autosave to localStorage.
 *
 * @param key   - localStorage key
 * @param value - current value to persist
 * @param delay - debounce delay in ms (default 3 000)
 */
export function useAutosave<T>(
  key: string,
  value: T,
  delay = 3000,
): UseAutosaveReturn<T> {
  const [savedValue, setSavedValue] = useState<T | undefined>(undefined);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const latestValue = useRef(value);

  // Keep ref in sync so the debounce always writes the freshest value.
  latestValue.current = value;

  /* ---- Load on mount ---- */
  useEffect(() => {
    try {
      const raw = localStorage.getItem(key);
      if (raw !== null) {
        setSavedValue(JSON.parse(raw) as T);
      }
    } catch {
      // Corrupted data → ignore.
      localStorage.removeItem(key);
    }
    // Only run once per key.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  /* ---- Debounced save ---- */
  useEffect(() => {
    if (timerRef.current) clearTimeout(timerRef.current);

    timerRef.current = setTimeout(() => {
      try {
        localStorage.setItem(key, JSON.stringify(latestValue.current));
      } catch {
        // Quota exceeded or private mode → fail silently.
      }
    }, delay);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [key, value, delay]);

  /* ---- Clear ---- */
  const clearSaved = useCallback(() => {
    localStorage.removeItem(key);
    setSavedValue(undefined);
  }, [key]);

  return { savedValue, clearSaved };
}
