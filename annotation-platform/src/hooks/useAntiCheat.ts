import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { supabase } from '../lib/supabase';
import type { TaskType, EventType } from '../lib/types';

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface UseAntiCheatOptions {
  userId: string;
  sentenceId: string;
  task: TaskType;
  enabled?: boolean;
}

interface InputProtectionHandlers {
  onPaste: (e: React.ClipboardEvent) => void;
  onCut: (e: React.ClipboardEvent) => void;
  onDrop: (e: React.DragEvent) => void;
  onDragOver: (e: React.DragEvent) => void;
  onContextMenu: (e: React.MouseEvent) => void;
}

interface SourceProtectionHandlers {
  onCopy: (e: React.ClipboardEvent) => void;
  onContextMenu: (e: React.MouseEvent) => void;
}

interface UseAntiCheatReturn {
  tabSwitchCount: number;
  bindInputProtection: () => InputProtectionHandlers;
  bindSourceProtection: () => SourceProtectionHandlers;
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

/** Fire-and-forget event log to Supabase. */
function logEvent(
  userId: string,
  sentenceId: string,
  task: TaskType,
  type: EventType,
  value?: Record<string, unknown>,
) {
  supabase
    .from('events')
    .insert({
      user_id: userId,
      sentence_id: sentenceId,
      task,
      type,
      value: value ?? null,
    })
    .then(({ error }) => {
      if (error) console.warn('[anti-cheat] event log failed:', error.message);
    });
}

/* ------------------------------------------------------------------ */
/*  Hook                                                               */
/* ------------------------------------------------------------------ */

export function useAntiCheat({
  userId,
  sentenceId,
  task,
  enabled = true,
}: UseAntiCheatOptions): UseAntiCheatReturn {
  const [tabSwitchCount, setTabSwitchCount] = useState(0);
  const blurTsRef = useRef<number | null>(null);

  // Reset count when the sentence changes.
  useEffect(() => {
    setTabSwitchCount(0);
  }, [sentenceId]);

  /* ---- Tab / window visibility tracking ---- */
  useEffect(() => {
    if (!enabled) return;

    function handleVisibilityChange() {
      if (document.hidden) {
        blurTsRef.current = Date.now();
        setTabSwitchCount((c) => c + 1);
        logEvent(userId, sentenceId, task, 'tab_blur');
      } else {
        const away = blurTsRef.current ? Date.now() - blurTsRef.current : 0;
        logEvent(userId, sentenceId, task, 'focus_return', {
          away_ms: away,
        });
        blurTsRef.current = null;
      }
    }

    function handleBlur() {
      if (!document.hidden) {
        // Window lost focus but tab is still visible (e.g. alt-tab).
        blurTsRef.current = Date.now();
        setTabSwitchCount((c) => c + 1);
        logEvent(userId, sentenceId, task, 'tab_blur');
      }
    }

    function handleFocus() {
      if (blurTsRef.current) {
        const away = Date.now() - blurTsRef.current;
        logEvent(userId, sentenceId, task, 'focus_return', {
          away_ms: away,
        });
        blurTsRef.current = null;
      }
    }

    document.addEventListener('visibilitychange', handleVisibilityChange);
    window.addEventListener('blur', handleBlur);
    window.addEventListener('focus', handleFocus);

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('blur', handleBlur);
      window.removeEventListener('focus', handleFocus);
    };
  }, [enabled, userId, sentenceId, task]);

  /* ---- Input protection ---- */
  const bindInputProtection = useCallback(
    (): InputProtectionHandlers => ({
      onPaste: (e: React.ClipboardEvent) => {
        if (!enabled) return;
        e.preventDefault();
        logEvent(userId, sentenceId, task, 'paste_attempt');
      },
      onCut: (e: React.ClipboardEvent) => {
        if (!enabled) return;
        e.preventDefault();
        logEvent(userId, sentenceId, task, 'cut_attempt');
      },
      onDrop: (e: React.DragEvent) => {
        if (!enabled) return;
        e.preventDefault();
        logEvent(userId, sentenceId, task, 'drag_attempt');
      },
      onDragOver: (e: React.DragEvent) => {
        if (!enabled) return;
        e.preventDefault();
      },
      onContextMenu: (e: React.MouseEvent) => {
        if (!enabled) return;
        e.preventDefault();
      },
    }),
    [enabled, userId, sentenceId, task],
  );

  /* ---- Source protection ---- */
  const bindSourceProtection = useCallback(
    (): SourceProtectionHandlers => ({
      onCopy: (e: React.ClipboardEvent) => {
        if (!enabled) return;
        e.preventDefault();
        logEvent(userId, sentenceId, task, 'copy_attempt');
      },
      onContextMenu: (e: React.MouseEvent) => {
        if (!enabled) return;
        e.preventDefault();
      },
    }),
    [enabled, userId, sentenceId, task],
  );

  return useMemo(
    () => ({ tabSwitchCount, bindInputProtection, bindSourceProtection }),
    [tabSwitchCount, bindInputProtection, bindSourceProtection],
  );
}
