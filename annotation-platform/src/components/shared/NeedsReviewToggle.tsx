/* ------------------------------------------------------------------ */
/*  NeedsReviewToggle                                                  */
/*  Accessible pill toggle with amber ON state.                        */
/* ------------------------------------------------------------------ */

import { useId } from 'react';

interface NeedsReviewToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
}

export default function NeedsReviewToggle({
  checked,
  onChange,
}: NeedsReviewToggleProps) {
  const id = useId();

  return (
    <label
      htmlFor={id}
      className={
        'group relative inline-flex cursor-pointer select-none items-center gap-2.5 ' +
        'rounded-full px-4 py-2 text-sm font-semibold transition-all duration-200 ' +
        (checked
          ? 'bg-[var(--color-amber-muted)] text-[var(--color-amber)] ring-1 ring-[var(--color-amber)]/30'
          : 'bg-[var(--color-surface-2)] text-[var(--color-text-muted)] ring-1 ring-[var(--color-border)]')
      }
    >
      {/* Hidden native checkbox for accessibility */}
      <input
        id={id}
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="sr-only"
        aria-label="Needs review"
      />

      {/* Toggle dot */}
      <span
        className={
          'inline-block h-4 w-4 rounded-full transition-all duration-200 ' +
          (checked
            ? 'bg-[var(--color-amber)] shadow-[0_0_6px_var(--color-amber)]'
            : 'bg-[var(--color-surface-3)]')
        }
        aria-hidden="true"
      />

      {/* Label text */}
      <span className="tracking-wide">
        {checked ? 'NEEDS REVIEW' : 'Mark Review'}
      </span>
    </label>
  );
}
