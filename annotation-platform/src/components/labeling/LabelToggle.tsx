import { useState, useRef } from 'react';
import type { LabelDef } from '../../lib/constants';

interface LabelToggleProps {
  label: LabelDef;
  value: 0 | 1;
  onChange: (value: 0 | 1) => void;
}

const groupColors = {
  Noise: {
    active: 'bg-[var(--color-accent)] text-white shadow-[0_0_12px_var(--color-accent-muted)]',
    inactive: 'bg-[var(--color-accent-muted)] text-[var(--color-accent)] hover:bg-[var(--color-accent)]/20',
  },
  Function: {
    active: 'bg-[var(--color-violet)] text-white shadow-[0_0_12px_var(--color-violet-muted)]',
    inactive: 'bg-[var(--color-violet-muted)] text-[var(--color-violet)] hover:bg-[var(--color-violet)]/20',
  },
  Metadata: {
    active: 'bg-[var(--color-amber)] text-[var(--color-text-inverse)] shadow-[0_0_12px_var(--color-amber-muted)]',
    inactive: 'bg-[var(--color-amber-muted)] text-[var(--color-amber)] hover:bg-[var(--color-amber)]/20',
  },
};

export default function LabelToggle({ label, value, onChange }: LabelToggleProps) {
  const [showTooltip, setShowTooltip] = useState(false);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const colors = groupColors[label.group];

  const handleMouseEnter = () => {
    timeoutRef.current = setTimeout(() => setShowTooltip(true), 300);
  };

  const handleMouseLeave = () => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setShowTooltip(false);
  };

  const handleFocus = () => setShowTooltip(true);
  const handleBlur = () => setShowTooltip(false);

  return (
    <div className="relative inline-block">
      <button
        ref={toggleRef}
        type="button"
        role="switch"
        aria-pressed={value === 1}
        aria-label={`${label.key}: ${label.category}`}
        onClick={() => onChange(value === 1 ? 0 : 1)}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onFocus={handleFocus}
        onBlur={handleBlur}
        className={`
          inline-flex items-center rounded-full px-3.5 py-1.5
          text-xs font-semibold tracking-wide
          transition-all duration-150 cursor-pointer
          focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--color-accent)]
          ${value === 1 ? colors.active : colors.inactive}
        `}
      >
        {label.key}
      </button>

      {/* Tooltip popover */}
      {showTooltip && (
        <div
          className="absolute bottom-full left-1/2 z-50 mb-2 w-80 -translate-x-1/2
            rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-1)]
            p-4 shadow-xl animate-fade-in"
          onMouseEnter={() => setShowTooltip(true)}
          onMouseLeave={handleMouseLeave}
        >
          {/* Arrow */}
          <div className="absolute -bottom-1.5 left-1/2 h-3 w-3 -translate-x-1/2 rotate-45 border-b border-r border-[var(--color-border)] bg-[var(--color-surface-1)]" />

          <p className="mb-1 text-sm font-bold text-[var(--color-text-primary)]">
            {label.category}
          </p>

          {label.definition && (
            <p className="mb-2 text-xs text-[var(--color-text-secondary)]">
              {label.definition}
            </p>
          )}

          {label.examples && (
            <p className="mb-2 text-xs text-[var(--color-text-muted)]">
              <span className="font-medium text-[var(--color-text-secondary)]">Examples:</span>{' '}
              {label.examples}
            </p>
          )}

          <div className="max-h-48 overflow-y-auto text-xs text-[var(--color-text-secondary)]">
            <p className="mb-1 font-medium text-[var(--color-text-primary)]">Decision:</p>
            <pre className="whitespace-pre-wrap font-sans leading-relaxed">{label.decision}</pre>
          </div>
        </div>
      )}
    </div>
  );
}
