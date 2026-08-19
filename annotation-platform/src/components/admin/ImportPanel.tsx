import { useState, useCallback } from 'react';
import { supabase } from '../../lib/supabase';

interface ParsedRow {
  Sentence_ID: string;
  Raw_Noisy_Sentence: string;
  Source_Platform?: string;
}

export default function ImportPanel() {
  const [file, setFile] = useState<File | null>(null);
  const [delimiter, setDelimiter] = useState<'auto' | 'tab' | 'comma'>('auto');
  const [markReliability, setMarkReliability] = useState(false);
  const [preview, setPreview] = useState<ParsedRow[]>([]);
  const [skipCount, setSkipCount] = useState(0);
  const [totalRows, setTotalRows] = useState(0);
  const [importing, setImporting] = useState(false);
  const [result, setResult] = useState<{ imported: number; skipped: number; errors: string[] } | null>(null);
  const [error, setError] = useState('');

  const BROKEN_CELLS = ['#NAME?', '#REF!', '#N/A', '#VALUE!', '#NULL!', '#DIV/0!'];

  const parseFile = useCallback(async (f: File) => {
    const text = await f.text();
    const det = delimiter === 'auto'
      ? (text.includes('\t') ? '\t' : ',')
      : delimiter === 'tab' ? '\t' : ',';

    const lines = text.split(/\r?\n/).filter((l) => l.trim());
    if (lines.length < 2) { setError('File has no data rows'); return; }

    const headers = lines[0].split(det).map((h) => h.trim().replace(/^["']|["']$/g, ''));
    const idIdx = headers.findIndex((h) => /sentence.?id/i.test(h));
    const textIdx = headers.findIndex((h) => /raw.?noisy|noisy.?text|sentence/i.test(h));
    const platformIdx = headers.findIndex((h) => /source.?platform|platform/i.test(h));

    if (idIdx === -1 || textIdx === -1) {
      setError('Could not find Sentence_ID and Raw_Noisy_Sentence columns');
      return;
    }

    const rows: ParsedRow[] = [];
    let skipped = 0;

    for (let i = 1; i < lines.length; i++) {
      const cols = lines[i].split(det).map((c) => c.trim().replace(/^["']|["']$/g, ''));
      const id = cols[idIdx] || '';
      const noisy = cols[textIdx] || '';

      if (!id || !noisy || BROKEN_CELLS.some((b) => id.includes(b) || noisy.includes(b))) {
        skipped++;
        continue;
      }

      rows.push({
        Sentence_ID: id,
        Raw_Noisy_Sentence: noisy,
        Source_Platform: platformIdx >= 0 ? cols[platformIdx] : undefined,
      });
    }

    setPreview(rows.slice(0, 10));
    setSkipCount(skipped);
    setTotalRows(rows.length);
    setError('');
  }, [delimiter]);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) {
      setFile(f);
      setResult(null);
      parseFile(f);
    }
  }, [parseFile]);

  const handleImport = useCallback(async () => {
    if (!file) return;
    setImporting(true);
    setError('');
    setResult(null);

    try {
      const text = await file.text();

      const { data, error: fnError } = await supabase.functions.invoke('import-sentences', {
        body: text,
        headers: { 'Content-Type': 'text/plain' },
        method: 'POST',
      });
      if (fnError) throw fnError;
      setResult(data);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Import failed';
      setError(msg);
    } finally {
      setImporting(false);
    }
  }, [file, delimiter, markReliability]);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-[var(--color-text-primary)]">Import Sentences</h2>
        <p className="text-sm text-[var(--color-text-muted)]">
          Upload CSV/TSV with columns: Sentence_ID, Raw_Noisy_Sentence, Source_Platform (optional)
        </p>
      </div>

      {/* File input */}
      <div className="glass-panel p-6 space-y-4">
        <div>
          <label htmlFor="import-file" className="block text-sm font-medium text-[var(--color-text-secondary)] mb-2">
            Select File
          </label>
          <input
            id="import-file"
            type="file"
            accept=".csv,.tsv,.txt"
            onChange={handleFileChange}
            className="block w-full text-sm text-[var(--color-text-secondary)]
              file:mr-4 file:rounded-lg file:border-0 file:bg-[var(--color-accent-muted)]
              file:px-4 file:py-2 file:text-sm file:font-medium file:text-[var(--color-accent)]
              file:cursor-pointer hover:file:bg-[var(--color-accent)]/20"
          />
        </div>

        {/* Delimiter */}
        <div className="flex items-center gap-4">
          <label className="text-sm text-[var(--color-text-secondary)]">Delimiter:</label>
          {(['auto', 'tab', 'comma'] as const).map((d) => (
            <label key={d} className="flex items-center gap-1.5 text-sm text-[var(--color-text-secondary)] cursor-pointer">
              <input
                type="radio"
                name="delimiter"
                value={d}
                checked={delimiter === d}
                onChange={() => { setDelimiter(d); if (file) parseFile(file); }}
                className="accent-[var(--color-accent)]"
              />
              {d === 'auto' ? 'Auto-detect' : d === 'tab' ? 'Tab' : 'Comma'}
            </label>
          ))}
        </div>

        {/* Mark reliability */}
        <label className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)] cursor-pointer">
          <input
            type="checkbox"
            checked={markReliability}
            onChange={(e) => setMarkReliability(e.target.checked)}
            className="accent-[var(--color-accent)]"
          />
          Mark imported sentences as reliability subset
        </label>
      </div>

      {/* Preview */}
      {preview.length > 0 && (
        <div className="glass-panel p-6 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-[var(--color-text-primary)]">
              Preview ({totalRows} valid rows)
            </h3>
            {skipCount > 0 && (
              <span className="text-xs text-[var(--color-amber)]">
                ⚠ {skipCount} rows skipped (broken cells or empty)
              </span>
            )}
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-[var(--color-text-secondary)]">
              <thead>
                <tr className="border-b border-[var(--color-border)]">
                  <th className="px-3 py-2 text-left font-medium">Sentence_ID</th>
                  <th className="px-3 py-2 text-left font-medium">Raw_Noisy_Sentence</th>
                  <th className="px-3 py-2 text-left font-medium">Source_Platform</th>
                </tr>
              </thead>
              <tbody>
                {preview.map((row, i) => (
                  <tr key={i} className="border-b border-[var(--color-border)]/50">
                    <td className="px-3 py-2 font-mono text-[var(--color-text-muted)]">{row.Sentence_ID}</td>
                    <td className="px-3 py-2 max-w-md truncate">{row.Raw_Noisy_Sentence}</td>
                    <td className="px-3 py-2 text-[var(--color-text-muted)]">{row.Source_Platform || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Import button */}
      <button
        id="import-sentences-btn"
        onClick={handleImport}
        disabled={importing || !file || totalRows === 0}
        className="rounded-lg bg-[var(--color-accent)] px-6 py-2.5 font-medium text-[var(--color-text-inverse)] transition-all duration-200 hover:bg-[var(--color-accent-hover)] disabled:cursor-not-allowed disabled:opacity-50"
      >
        {importing ? 'Importing...' : `Import ${totalRows} Sentences`}
      </button>

      {/* Result */}
      {result && (
        <div className="rounded-lg bg-[var(--color-emerald-muted)] px-4 py-3 text-sm text-[var(--color-emerald)] animate-fade-in">
          ✅ Imported {result.imported} sentences. Skipped {result.skipped}.
          {result.errors?.length > 0 && (
            <ul className="mt-2 list-disc pl-4">
              {result.errors.map((e, i) => <li key={i}>{e}</li>)}
            </ul>
          )}
        </div>
      )}

      {error && (
        <div className="rounded-lg bg-[var(--color-rose-muted)] px-4 py-3 text-sm text-[var(--color-rose)] animate-fade-in">
          ❌ {error}
        </div>
      )}
    </div>
  );
}
