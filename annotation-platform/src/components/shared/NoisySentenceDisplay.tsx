/* ------------------------------------------------------------------ */
/*  NoisySentenceDisplay                                               */
/*  HIGH CONTRAST · no glassmorphism · no backdrop-blur · user-select  */
/*  none (via .noisy-text-display CSS class)                           */
/* ------------------------------------------------------------------ */

interface NoisySentenceDisplayProps {
  text: string;
  className?: string;
}

export default function NoisySentenceDisplay({
  text,
  className = '',
}: NoisySentenceDisplayProps) {
  return (
    <div className={className}>
      {/* Label */}
      <p className="mb-2 text-xs font-medium uppercase tracking-widest text-[var(--color-text-muted)]">
        Noisy Sentence
      </p>

      {/* Sentence display */}
      <div
        className="noisy-text-display"
        onCopy={(e) => e.preventDefault()}
        onContextMenu={(e) => e.preventDefault()}
        aria-label="Noisy sentence text"
      >
        {text}
      </div>
    </div>
  );
}
