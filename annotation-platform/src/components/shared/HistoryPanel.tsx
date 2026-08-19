import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { supabase } from '../../lib/supabase';
import type {
  NormalizationWithSentence,
  LabelingWithSentence,
  TaskType,
} from '../../lib/types';
import { LABELS } from '../../lib/constants';

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface HistoryPanelProps {
  task: TaskType;
  userId: string;
  onEdit?: (item: NormalizationWithSentence | LabelingWithSentence) => void;
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function HistoryPanel({ task, userId, onEdit }: HistoryPanelProps) {
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<
    (NormalizationWithSentence | LabelingWithSentence)[]
  >([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(false);

  // Session start = component mount time. Items submitted after this are editable.
  const sessionStart = useRef(new Date().toISOString());

  /* ---- Fetch ---- */
  const fetchHistory = useCallback(async () => {
    setLoading(true);
    const table = task === 'normalization' ? 'normalizations' : 'labelings';

    const { data, error } = await supabase
      .from(table)
      .select('*, sentences(*)')
      .eq('user_id', userId)
      .order('submitted_at', { ascending: false })
      .limit(100);

    if (!error && data) {
      setItems(data as (NormalizationWithSentence | LabelingWithSentence)[]);
    }
    setLoading(false);
  }, [task, userId]);

  // Refetch when panel opens or task changes.
  useEffect(() => {
    if (open) fetchHistory();
  }, [open, fetchHistory]);

  /* ---- Computed: editable items (last 5 after session start) ---- */
  const editableIds = useMemo(() => {
    const recent = items
      .filter((it) => it.submitted_at > sessionStart.current)
      .slice(0, 5);
    return new Set(recent.map((it) => it.id));
  }, [items]);

  /* ---- Filtered by search ---- */
  const filtered = useMemo(() => {
    if (!search.trim()) return items;
    const q = search.toLowerCase();
    return items.filter((it) =>
      it.sentences.noisy_text.toLowerCase().includes(q),
    );
  }, [items, search]);

  /* ---- Label name helper ---- */
  const labelName = useCallback(
    (key: string) => LABELS.find((l) => l.key === key)?.category ?? key,
    [],
  );

  /* ---- Toggle button (always visible) ---- */
  return (
    <>
      {/* Toggle tab on the right edge */}
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className={
          'fixed right-0 top-1/2 z-30 -translate-y-1/2 rounded-l-[var(--radius-md)] ' +
          'border border-r-0 border-[var(--color-border)] bg-[var(--color-surface-2)] ' +
          'px-2 py-4 text-xs font-medium text-[var(--color-text-secondary)] ' +
          'transition-colors duration-200 hover:bg-[var(--color-surface-3)] ' +
          'hover:text-[var(--color-text-primary)] [writing-mode:vertical-rl]'
        }
        aria-label={open ? 'Close history' : 'Open history'}
      >
        📋 History
      </button>

      {/* Panel */}
      <div
        className={
          'fixed right-0 top-14 z-30 flex h-[calc(100vh-3.5rem)] w-80 flex-col ' +
          'border-l border-[var(--color-border)] bg-[var(--color-surface-1)] ' +
          'shadow-xl transition-transform duration-250 ease-out ' +
          (open ? 'translate-x-0 animate-slide-in' : 'translate-x-full')
        }
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-[var(--color-border)] px-4 py-3">
          <h2 className="text-sm font-semibold text-[var(--color-text-primary)]">
            Submission History
          </h2>
          <button
            type="button"
            onClick={() => setOpen(false)}
            className="text-[var(--color-text-muted)] transition-colors hover:text-[var(--color-text-primary)]"
            aria-label="Close history panel"
          >
            ✕
          </button>
        </div>

        {/* Search */}
        <div className="px-4 py-2">
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search sentences…"
            className={
              'w-full rounded-[var(--radius-md)] border border-[var(--color-border)] ' +
              'bg-[var(--color-surface-2)] px-3 py-1.5 text-sm text-[var(--color-text-primary)] ' +
              'placeholder:text-[var(--color-text-muted)] focus:border-[var(--color-border-focus)] ' +
              'focus:outline-none'
            }
          />
        </div>

        {/* List */}
        <div className="flex-1 overflow-y-auto px-4 py-2">
          {loading ? (
            <p className="py-8 text-center text-sm text-[var(--color-text-muted)]">
              Loading…
            </p>
          ) : filtered.length === 0 ? (
            <p className="py-8 text-center text-sm text-[var(--color-text-muted)]">
              No submissions yet.
            </p>
          ) : (
            <ul className="flex flex-col gap-2">
              {filtered.map((item) => {
                const isEditable = editableIds.has(item.id);
                const isNorm = task === 'normalization';

                return (
                  <li
                    key={item.id}
                    className={
                      'glass-panel flex flex-col gap-1.5 p-3 text-xs ' +
                      'transition-colors duration-200 ' +
                      (isEditable
                        ? 'border-[var(--color-accent)]/20'
                        : '')
                    }
                  >
                    {/* Noisy text */}
                    <p className="line-clamp-2 text-[var(--color-text-muted)]">
                      {item.sentences.noisy_text}
                    </p>

                    {/* Result preview */}
                    {isNorm ? (
                      <p className="line-clamp-2 font-medium text-[var(--color-text-primary)]">
                        → {(item as NormalizationWithSentence).normalized_text ?? '—'}
                      </p>
                    ) : (
                      <div className="flex flex-wrap gap-1">
                        {Object.entries(
                          (item as LabelingWithSentence).labels ?? {},
                        )
                          .filter(([, v]) => v === 1)
                          .map(([k]) => (
                            <span
                              key={k}
                              className={
                                'inline-block rounded-full bg-[var(--color-accent-muted)] ' +
                                'px-2 py-0.5 text-[10px] font-medium text-[var(--color-accent)]'
                              }
                            >
                              {labelName(k)}
                            </span>
                          ))}
                      </div>
                    )}

                    {/* Actions */}
                    <div className="mt-1 flex items-center justify-between">
                      <span className="text-[10px] text-[var(--color-text-muted)]">
                        {new Date(item.submitted_at).toLocaleTimeString()}
                      </span>
                      {isEditable && onEdit ? (
                        <button
                          type="button"
                          onClick={() => onEdit(item)}
                          className={
                            'rounded-[var(--radius-sm)] bg-[var(--color-accent-muted)] ' +
                            'px-2 py-0.5 text-[10px] font-medium text-[var(--color-accent)] ' +
                            'transition-colors hover:bg-[var(--color-accent)] hover:text-[var(--color-text-inverse)]'
                          }
                        >
                          Edit
                        </button>
                      ) : (
                        <span
                          className="text-[var(--color-text-muted)]"
                          title="Locked"
                          aria-label="Locked — not editable"
                        >
                          🔒
                        </span>
                      )}
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </div>
    </>
  );
}
