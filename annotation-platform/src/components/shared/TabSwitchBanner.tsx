/* ------------------------------------------------------------------ */
/*  TabSwitchBanner                                                    */
/*  Fixed banner in the annotation area — calm, non-accusatory.        */
/* ------------------------------------------------------------------ */

interface TabSwitchBannerProps {
  count: number;
  visible: boolean;
}

export default function TabSwitchBanner({ count, visible }: TabSwitchBannerProps) {
  return (
    <div
      className={
        'overflow-hidden transition-all duration-300 ease-out ' +
        (visible
          ? 'max-h-20 opacity-100'
          : 'pointer-events-none max-h-0 opacity-0')
      }
      role="alert"
      aria-live="polite"
    >
      <div
        className={
          'flex items-center gap-3 rounded-[var(--radius-md)] ' +
          'bg-[var(--color-amber-muted)] px-4 py-2.5 text-sm ' +
          'text-[var(--color-amber)] animate-fade-in'
        }
      >
        {/* Icon */}
        <span className="shrink-0 text-base" aria-hidden="true">
          ⚠️
        </span>

        {/* Message */}
        <p className="leading-snug">
          Tab change detected — please keep annotation in this window.{' '}
          <span className="font-semibold tabular-nums">
            Switches on this item: {count}
          </span>
        </p>
      </div>
    </div>
  );
}
