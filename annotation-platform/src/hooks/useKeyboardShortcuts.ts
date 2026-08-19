import { useEffect, useRef } from 'react';

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

type ShortcutMap = Record<string, () => void>;

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

/**
 * Normalize a combo string to a canonical form.
 * "Ctrl+Enter" → "ctrl+enter"
 * "cmd+s"      → "meta+s"
 */
function normalizeCombo(combo: string): string {
  return combo
    .toLowerCase()
    .replace(/\s/g, '')
    .replace('cmd', 'meta')
    .replace('command', 'meta')
    .replace('ctrl', 'control');
}

/** Build the canonical combo from a KeyboardEvent. */
function comboFromEvent(e: KeyboardEvent): string {
  const parts: string[] = [];
  if (e.ctrlKey) parts.push('control');
  if (e.metaKey) parts.push('meta');
  if (e.altKey) parts.push('alt');
  if (e.shiftKey) parts.push('shift');

  const key = e.key.toLowerCase();
  // Avoid duplication for modifier-only presses.
  if (!['control', 'meta', 'alt', 'shift'].includes(key)) {
    parts.push(key);
  }

  return parts.join('+');
}

/* ------------------------------------------------------------------ */
/*  Hook                                                               */
/* ------------------------------------------------------------------ */

/**
 * Register global keyboard shortcuts.
 *
 * @example
 * useKeyboardShortcuts({
 *   'ctrl+enter': handleSubmit,
 *   'cmd+enter': handleSubmit,
 *   'escape': handleCancel,
 * });
 */
export function useKeyboardShortcuts(shortcuts: ShortcutMap) {
  // Keep a ref so we always close over the latest map without re-attaching listeners.
  const mapRef = useRef(shortcuts);
  mapRef.current = shortcuts;

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      const pressed = comboFromEvent(e);

      for (const [combo, fn] of Object.entries(mapRef.current)) {
        if (normalizeCombo(combo) === pressed) {
          e.preventDefault();
          fn();
          return;
        }
      }
    }

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);
}
