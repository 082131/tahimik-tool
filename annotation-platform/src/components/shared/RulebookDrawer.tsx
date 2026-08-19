import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  RULES,
  LABELS,
  NEEDS_REVIEW,
  RULEBOOK_TITLE,
  RULEBOOK_NOTE,
  type LabelGroup,
} from '../../lib/constants';

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function RulebookDrawer() {
  const [open, setOpen] = useState(false);
  const [closing, setClosing] = useState(false);
  const [search, setSearch] = useState('');

  /* ---- Open / close via custom event ---- */
  useEffect(() => {
    function handleToggle() {
      setOpen((prev) => {
        if (prev) {
          // Start close animation.
          setClosing(true);
          setTimeout(() => {
            setClosing(false);
            setOpen(false);
          }, 250);
          return prev; // keep open during animation
        }
        return true;
      });
    }

    window.addEventListener('toggle-rulebook', handleToggle);
    return () => window.removeEventListener('toggle-rulebook', handleToggle);
  }, []);

  const handleClose = useCallback(() => {
    setClosing(true);
    setTimeout(() => {
      setClosing(false);
      setOpen(false);
    }, 250);
  }, []);

  /* ---- Filtered data ---- */
  const q = search.toLowerCase();

  const filteredRules = useMemo(
    () =>
      RULES.filter(
        (r) =>
          !q ||
          r.rule.toLowerCase().includes(q) ||
          r.decision.toLowerCase().includes(q) ||
          r.doThis.toLowerCase().includes(q) ||
          r.example.toLowerCase().includes(q) ||
          r.id.toLowerCase().includes(q),
      ),
    [q],
  );

  const filteredLabels = useMemo(
    () =>
      LABELS.filter(
        (l) =>
          !q ||
          l.key.toLowerCase().includes(q) ||
          l.category.toLowerCase().includes(q) ||
          l.definition.toLowerCase().includes(q) ||
          l.examples.toLowerCase().includes(q) ||
          l.decision.toLowerCase().includes(q),
      ),
    [q],
  );

  const showNeedsReview =
    !q ||
    NEEDS_REVIEW.category.toLowerCase().includes(q) ||
    NEEDS_REVIEW.definition.toLowerCase().includes(q) ||
    NEEDS_REVIEW.key.toLowerCase().includes(q);

  /* ---- Group labels ---- */
  const labelGroups = useMemo(() => {
    const groups: Record<LabelGroup, typeof filteredLabels> = {
      Noise: [],
      Function: [],
      Metadata: [],
    };
    for (const l of filteredLabels) {
      groups[l.group].push(l);
    }
    return groups;
  }, [filteredLabels]);

  /* ---- Keyboard: Escape to close ---- */
  useEffect(() => {
    if (!open) return;
    function handleKey(e: KeyboardEvent) {
      if (e.key === 'Escape') handleClose();
    }
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [open, handleClose]);

  if (!open) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className={
          'fixed inset-0 z-50 bg-black/50 transition-opacity duration-200 ' +
          (closing ? 'opacity-0' : 'opacity-100')
        }
        onClick={handleClose}
        aria-hidden="true"
      />

      {/* Drawer */}
      <aside
        className={
          'fixed right-0 top-0 z-50 flex h-full w-full flex-col ' +
          'border-l border-[var(--color-border)] bg-[var(--color-surface-1)] ' +
          'shadow-2xl sm:w-[28rem] md:w-[32rem] ' +
          (closing ? 'animate-slide-out' : 'animate-slide-in')
        }
        role="dialog"
        aria-label="Rulebook"
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-[var(--color-border)] px-5 py-4">
          <div>
            <h2 className="text-lg font-bold text-[var(--color-text-primary)]">
              {RULEBOOK_TITLE}
            </h2>
            <p className="mt-0.5 text-xs text-[var(--color-text-muted)]">
              {RULEBOOK_NOTE}
            </p>
          </div>
          <button
            type="button"
            onClick={handleClose}
            className={
              'rounded-[var(--radius-md)] p-1.5 text-[var(--color-text-muted)] ' +
              'transition-colors hover:bg-[var(--color-surface-2)] hover:text-[var(--color-text-primary)]'
            }
            aria-label="Close rulebook"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Search */}
        <div className="border-b border-[var(--color-border)] px-5 py-3">
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search rules & labels…"
            className={
              'w-full rounded-[var(--radius-md)] border border-[var(--color-border)] ' +
              'bg-[var(--color-surface-2)] px-3 py-2 text-sm text-[var(--color-text-primary)] ' +
              'placeholder:text-[var(--color-text-muted)] focus:border-[var(--color-border-focus)] ' +
              'focus:outline-none'
            }
            autoFocus
          />
        </div>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto px-5 py-4">
          {/* ---- Rules Section ---- */}
          {filteredRules.length > 0 && (
            <section className="mb-8">
              <h3 className="mb-3 text-xs font-semibold uppercase tracking-widest text-[var(--color-accent)]">
                Rules
              </h3>
              <div className="flex flex-col gap-3">
                {filteredRules.map((r) => (
                  <div
                    key={r.id}
                    className="glass-panel p-4"
                  >
                    <div className="mb-2 flex items-baseline gap-2">
                      <span className="rounded bg-[var(--color-accent-muted)] px-1.5 py-0.5 text-[10px] font-bold text-[var(--color-accent)]">
                        {r.id}
                      </span>
                      <span className="text-sm font-semibold text-[var(--color-text-primary)]">
                        {r.rule}
                      </span>
                    </div>
                    <p className="mb-2 text-xs leading-relaxed text-[var(--color-text-secondary)]">
                      {r.decision}
                    </p>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="rounded-[var(--radius-sm)] bg-[var(--color-emerald-muted)] p-2">
                        <span className="mb-0.5 block font-medium text-[var(--color-emerald)]">
                          ✓ Do
                        </span>
                        <span className="text-[var(--color-text-secondary)]">
                          {r.doThis}
                        </span>
                      </div>
                      <div className="rounded-[var(--radius-sm)] bg-[var(--color-rose-muted)] p-2">
                        <span className="mb-0.5 block font-medium text-[var(--color-rose)]">
                          ✗ Don't
                        </span>
                        <span className="text-[var(--color-text-secondary)]">
                          {r.doNot}
                        </span>
                      </div>
                    </div>
                    <div className="mt-2 rounded-[var(--radius-sm)] bg-[var(--color-surface-2)] p-2 text-xs">
                      <span className="font-medium text-[var(--color-text-muted)]">
                        Example:{' '}
                      </span>
                      <span className="text-[var(--color-text-primary)]">{r.example}</span>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* ---- Labels Section ---- */}
          {filteredLabels.length > 0 && (
            <section className="mb-8">
              <h3 className="mb-3 text-xs font-semibold uppercase tracking-widest text-[var(--color-accent)]">
                Labels
              </h3>

              {(Object.entries(labelGroups) as [LabelGroup, typeof filteredLabels][]).map(
                ([group, labels]) =>
                  labels.length > 0 && (
                    <div key={group} className="mb-5">
                      <h4 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
                        {group} Labels
                      </h4>
                      <div className="flex flex-col gap-2">
                        {labels.map((l) => (
                          <div
                            key={l.key}
                            className="glass-panel p-3"
                          >
                            <div className="mb-1.5 flex items-center gap-2">
                              <span className="rounded bg-[var(--color-violet-muted)] px-1.5 py-0.5 text-[10px] font-bold text-[var(--color-violet)]">
                                {l.key}
                              </span>
                              <span className="text-sm font-medium text-[var(--color-text-primary)]">
                                {l.category}
                              </span>
                            </div>
                            {l.definition && (
                              <p className="mb-1 text-xs text-[var(--color-text-secondary)]">
                                {l.definition}
                              </p>
                            )}
                            {l.examples && (
                              <p className="mb-1 text-xs text-[var(--color-text-muted)]">
                                <span className="font-medium">Examples: </span>
                                {l.examples}
                              </p>
                            )}
                            <div className="mt-1.5 whitespace-pre-line rounded-[var(--radius-sm)] bg-[var(--color-surface-2)] p-2 text-xs leading-relaxed text-[var(--color-text-secondary)]">
                              {l.decision}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ),
              )}
            </section>
          )}

          {/* ---- NEEDS_REVIEW Section ---- */}
          {showNeedsReview && (
            <section className="mb-8">
              <h3 className="mb-3 text-xs font-semibold uppercase tracking-widest text-[var(--color-amber)]">
                Workflow Flag
              </h3>
              <div className="glass-panel border-[var(--color-amber)]/20 p-4">
                <div className="mb-1.5 flex items-center gap-2">
                  <span className="rounded bg-[var(--color-amber-muted)] px-1.5 py-0.5 text-[10px] font-bold text-[var(--color-amber)]">
                    {NEEDS_REVIEW.key}
                  </span>
                  <span className="text-sm font-medium text-[var(--color-text-primary)]">
                    {NEEDS_REVIEW.category}
                  </span>
                </div>
                <p className="mb-1 text-xs text-[var(--color-text-secondary)]">
                  {NEEDS_REVIEW.definition}
                </p>
                <p className="text-xs text-[var(--color-text-muted)]">
                  <span className="font-medium">Decision: </span>
                  {NEEDS_REVIEW.decision}
                </p>
                <p className="mt-1 text-xs text-[var(--color-text-muted)]">
                  <span className="font-medium">Examples: </span>
                  {NEEDS_REVIEW.examples}
                </p>
              </div>
            </section>
          )}

          {/* Empty state */}
          {filteredRules.length === 0 &&
            filteredLabels.length === 0 &&
            !showNeedsReview && (
              <p className="py-12 text-center text-sm text-[var(--color-text-muted)]">
                No matches for "{search}"
              </p>
            )}
        </div>
      </aside>
    </>
  );
}
