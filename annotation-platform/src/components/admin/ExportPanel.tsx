import { useState, useEffect, useCallback } from 'react';
import { supabase } from '../../lib/supabase';

interface ExportCounts {
  normTotal: number;
  normCompleted: number;
  normReview: number;
  labelTotal: number;
  labelCompleted: number;
  labelReview: number;
}

export default function ExportPanel() {
  const [counts, setCounts] = useState<ExportCounts | null>(null);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState<'normalization' | 'labeling' | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    async function fetchCounts() {
      try {
        const [normAll, normReview, labelAll, labelReview] = await Promise.all([
          supabase.from('normalizations').select('id', { count: 'exact', head: true }),
          supabase.from('normalizations').select('id', { count: 'exact', head: true }).eq('needs_review', true),
          supabase.from('labelings').select('id', { count: 'exact', head: true }),
          supabase.from('labelings').select('id', { count: 'exact', head: true }).eq('needs_review', true),
        ]);

        setCounts({
          normTotal: normAll.count ?? 0,
          normCompleted: (normAll.count ?? 0) - (normReview.count ?? 0),
          normReview: normReview.count ?? 0,
          labelTotal: labelAll.count ?? 0,
          labelCompleted: (labelAll.count ?? 0) - (labelReview.count ?? 0),
          labelReview: labelReview.count ?? 0,
        });
      } catch {
        setError('Failed to load counts');
      } finally {
        setLoading(false);
      }
    }
    fetchCounts();
  }, []);

  const handleExport = useCallback(async (task: 'normalization' | 'labeling') => {
    setExporting(task);
    setError('');

    try {
      const { data, error: fnError } = await supabase.functions.invoke('export-data', {
        method: 'POST',
        body: { task },
      });

      if (fnError) throw fnError;

      // data should be TSV text
      const tsvContent = typeof data === 'string' ? data : JSON.stringify(data);
      const blob = new Blob([tsvContent], { type: 'text/tab-separated-values' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${task}_export.tsv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : `Export failed for ${task}`);
    } finally {
      setExporting(null);
    }
  }, []);

  if (loading) {
    return <div className="py-12 text-center text-[var(--color-text-muted)]">Loading export stats...</div>;
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-[var(--color-text-primary)]">Export Data</h2>
        <p className="text-sm text-[var(--color-text-muted)]">
          Download annotation results as TSV files
        </p>
      </div>

      {/* Normalization Export */}
      <div className="glass-panel p-6 space-y-4">
        <h3 className="text-sm font-medium text-[var(--color-text-primary)]">
          Task 1 — Normalizations
        </h3>
        <p className="text-xs text-[var(--color-text-muted)]">
          Standalone 5-column dataset: Sentence_ID, Raw_Noisy_Sentence, Normalized_Sentence, NEEDS_REVIEW, Notes
        </p>
        {counts && (
          <div className="flex gap-4 text-xs text-[var(--color-text-secondary)]">
            <span>Total: {counts.normTotal}</span>
            <span className="text-[var(--color-emerald)]">Completed: {counts.normCompleted}</span>
            <span className="text-[var(--color-amber)]">Needs Review: {counts.normReview}</span>
          </div>
        )}
        <button
          id="export-normalization-btn"
          onClick={() => handleExport('normalization')}
          disabled={exporting !== null}
          className="rounded-lg bg-[var(--color-accent)] px-5 py-2 text-sm font-medium text-[var(--color-text-inverse)] transition-all hover:bg-[var(--color-accent-hover)] disabled:opacity-50"
        >
          {exporting === 'normalization' ? 'Exporting...' : 'Export Normalizations (.tsv)'}
        </button>
      </div>

      {/* Labeling Export */}
      <div className="glass-panel p-6 space-y-4">
        <h3 className="text-sm font-medium text-[var(--color-text-primary)]">
          Task 2 — Noise Labelings
        </h3>
        <p className="text-xs text-[var(--color-text-muted)]">
          Wide-format TSV matching 05_Reliability-Subset with 3 annotator blocks
        </p>
        {counts && (
          <div className="flex gap-4 text-xs text-[var(--color-text-secondary)]">
            <span>Total: {counts.labelTotal}</span>
            <span className="text-[var(--color-emerald)]">Completed: {counts.labelCompleted}</span>
            <span className="text-[var(--color-amber)]">Needs Review: {counts.labelReview}</span>
          </div>
        )}
        <button
          id="export-labeling-btn"
          onClick={() => handleExport('labeling')}
          disabled={exporting !== null}
          className="rounded-lg bg-[var(--color-accent)] px-5 py-2 text-sm font-medium text-[var(--color-text-inverse)] transition-all hover:bg-[var(--color-accent-hover)] disabled:opacity-50"
        >
          {exporting === 'labeling' ? 'Exporting...' : 'Export Labelings (.tsv)'}
        </button>
      </div>

      {/* NEEDS_REVIEW warning */}
      <div className="rounded-lg border border-[var(--color-amber)]/30 bg-[var(--color-amber-muted)] px-4 py-3 text-xs text-[var(--color-amber)]">
        <strong>⚠ NEEDS_REVIEW encoding:</strong> When NEEDS_REVIEW=1, label columns are exported
        as empty strings (not 0) to distinguish abstention from real judgments. Exclude NEEDS_REVIEW
        rows from α computation in your sheet.
      </div>

      {error && (
        <div className="rounded-lg bg-[var(--color-rose-muted)] px-4 py-3 text-sm text-[var(--color-rose)] animate-fade-in">
          ❌ {error}
        </div>
      )}
    </div>
  );
}
