/* ------------------------------------------------------------------ */
/*  ProgressBar                                                        */
/* ------------------------------------------------------------------ */

interface ProgressBarProps {
  current: number;
  total: number;
  label?: string;
}

export default function ProgressBar({ current, total, label }: ProgressBarProps) {
  const pct = total > 0 ? Math.min(Math.round((current / total) * 100), 100) : 0;

  return (
    <div className="flex flex-col gap-1.5">
      {/* Top row: label + count + percentage */}
      <div className="flex items-baseline justify-between text-xs">
        {label && (
          <span className="font-medium text-[var(--color-text-secondary)]">
            {label}
          </span>
        )}
        <div className="ml-auto flex items-baseline gap-2">
          <span className="tabular-nums text-[var(--color-text-primary)]">
            {current}{' '}
            <span className="text-[var(--color-text-muted)]">/ {total}</span>
          </span>
          <span className="min-w-[3ch] text-right tabular-nums text-[var(--color-text-muted)]">
            {pct}%
          </span>
        </div>
      </div>

      {/* Bar track */}
      <div
        className={
          'h-2 w-full overflow-hidden rounded-full ' +
          'bg-[var(--color-surface-2)]'
        }
        role="progressbar"
        aria-valuenow={current}
        aria-valuemin={0}
        aria-valuemax={total}
        aria-label={label ?? 'Progress'}
      >
        {/* Fill */}
        <div
          className="h-full rounded-full bg-[var(--color-accent)] transition-[width] duration-200 ease-out"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
