import { useState, useEffect, useCallback } from 'react';
import { supabase } from '../../lib/supabase';
import type { User } from '../../lib/types';

interface Stats {
  totalSentences: number;
  reliabilityCount: number;
  totalAssignments: number;
  normalizationAssignments: number;
  labelingAssignments: number;
  annotators: { id: string; name: string; normCount: number; labelCount: number }[];
}

export default function AssignmentPanel() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [markingReliability, setMarkingReliability] = useState(false);
  const [reliabilityIds, setReliabilityIds] = useState('');
  const [confirmGenerate, setConfirmGenerate] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const fetchStats = useCallback(async () => {
    setLoading(true);
    try {
      const [sentencesRes, reliabilityRes, assignmentsRes, annotatorsRes] = await Promise.all([
        supabase.from('sentences').select('id', { count: 'exact', head: true }),
        supabase.from('sentences').select('id', { count: 'exact', head: true }).eq('in_reliability', true),
        supabase.from('assignments').select('id, task', { count: 'exact' }),
        supabase.from('users').select('id, name').eq('role', 'annotator').eq('status', 'active'),
      ]);

      const annotators = (annotatorsRes.data || []) as User[];
      const annotatorStats = await Promise.all(
        annotators.map(async (a) => {
          const [normRes, labelRes] = await Promise.all([
            supabase.from('assignments').select('id', { count: 'exact', head: true })
              .eq('user_id', a.id).eq('task', 'normalization'),
            supabase.from('assignments').select('id', { count: 'exact', head: true })
              .eq('user_id', a.id).eq('task', 'labeling'),
          ]);
          return { id: a.id, name: a.name, normCount: normRes.count ?? 0, labelCount: labelRes.count ?? 0 };
        }),
      );

      const normTotal = assignmentsRes.data?.filter((a) => a.task === 'normalization').length ?? 0;
      const labelTotal = assignmentsRes.data?.filter((a) => a.task === 'labeling').length ?? 0;

      setStats({
        totalSentences: sentencesRes.count ?? 0,
        reliabilityCount: reliabilityRes.count ?? 0,
        totalAssignments: assignmentsRes.count ?? 0,
        normalizationAssignments: normTotal,
        labelingAssignments: labelTotal,
        annotators: annotatorStats,
      });
    } catch {
      setError('Failed to load stats');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchStats(); }, [fetchStats]);

  const handleMarkReliability = useCallback(async () => {
    if (!reliabilityIds.trim()) return;
    setMarkingReliability(true);
    setError('');
    setMessage('');

    try {
      const ids = reliabilityIds.split(/[\n,\t]+/).map((s) => s.trim()).filter(Boolean);
      const { data, error: fnError } = await supabase.functions.invoke('assign-sentences', {
        body: { action: 'mark_reliability', sentence_ids: ids },
      });
      if (fnError) throw fnError;
      setMessage(`Marked ${data?.updated ?? ids.length} sentences as reliability.`);
      setReliabilityIds('');
      fetchStats();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to mark reliability');
    } finally {
      setMarkingReliability(false);
    }
  }, [reliabilityIds, fetchStats]);

  const handleGenerate = useCallback(async () => {
    if (!confirmGenerate) { setError('Please confirm before generating'); return; }
    setGenerating(true);
    setError('');
    setMessage('');

    try {
      const { data, error: fnError } = await supabase.functions.invoke('assign-sentences', {
        body: { action: 'generate' },
      });
      if (fnError) throw fnError;
      setMessage(
        `Assignments generated! ${data?.normalization_assignments ?? 0} normalization, ` +
        `${data?.labeling_assignments ?? 0} labeling. ` +
        `Slices: ${data?.slices?.join(', ') ?? 'N/A'}. Reliability: ${data?.reliability_count ?? 0}.`
      );
      setConfirmGenerate(false);
      fetchStats();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to generate assignments');
    } finally {
      setGenerating(false);
    }
  }, [confirmGenerate, fetchStats]);

  if (loading) {
    return <div className="py-12 text-center text-[var(--color-text-muted)]">Loading stats...</div>;
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-[var(--color-text-primary)]">Assignment Management</h2>
        <p className="text-sm text-[var(--color-text-muted)]">
          Manage sentence assignments for 3 annotators
        </p>
      </div>

      {/* Stats */}
      {stats && (
        <div className="glass-panel p-6">
          <h3 className="mb-4 text-sm font-medium text-[var(--color-text-primary)]">Current Stats</h3>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <StatCard label="Total Sentences" value={stats.totalSentences} />
            <StatCard label="Reliability" value={stats.reliabilityCount} color="var(--color-violet)" />
            <StatCard label="Norm. Assignments" value={stats.normalizationAssignments} color="var(--color-accent)" />
            <StatCard label="Label Assignments" value={stats.labelingAssignments} color="var(--color-amber)" />
          </div>
          {stats.annotators.length > 0 && (
            <div className="mt-4">
              <h4 className="mb-2 text-xs font-medium uppercase tracking-widest text-[var(--color-text-muted)]">Per Annotator</h4>
              <div className="space-y-1">
                {stats.annotators.map((a) => (
                  <div key={a.id} className="flex items-center justify-between text-xs text-[var(--color-text-secondary)]">
                    <span>{a.name}</span>
                    <span>Norm: {a.normCount} · Label: {a.labelCount}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Mark Reliability */}
      <div className="glass-panel p-6 space-y-4">
        <h3 className="text-sm font-medium text-[var(--color-text-primary)]">Mark Reliability Sentences</h3>
        <p className="text-xs text-[var(--color-text-muted)]">
          Paste or enter Sentence_IDs (one per line, or comma/tab separated) to flag as in_reliability
        </p>
        <textarea
          id="reliability-ids"
          value={reliabilityIds}
          onChange={(e) => setReliabilityIds(e.target.value)}
          rows={4}
          placeholder="S001&#10;S002&#10;S003..."
          className="w-full resize-y rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-2)] px-4 py-3 font-mono text-xs text-[var(--color-text-primary)] placeholder-[var(--color-text-muted)] focus:border-[var(--color-border-focus)] focus:outline-none"
        />
        <button
          id="mark-reliability-btn"
          onClick={handleMarkReliability}
          disabled={markingReliability || !reliabilityIds.trim()}
          className="rounded-lg bg-[var(--color-violet)] px-4 py-2 text-sm font-medium text-white transition-all hover:opacity-90 disabled:opacity-50"
        >
          {markingReliability ? 'Marking...' : 'Mark as Reliability'}
        </button>
      </div>

      {/* Generate Assignments */}
      <div className="glass-panel p-6 space-y-4">
        <h3 className="text-sm font-medium text-[var(--color-text-primary)]">Generate Assignments</h3>
        <div className="rounded-lg bg-[var(--color-amber-muted)] px-4 py-3 text-xs text-[var(--color-amber)]">
          ⚠ This will delete all existing assignments and regenerate from scratch. Normalization: 3 disjoint slices of ~5,000. Labeling: all reliability sentences × 3 annotators.
        </div>
        <label className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)] cursor-pointer">
          <input
            type="checkbox"
            checked={confirmGenerate}
            onChange={(e) => setConfirmGenerate(e.target.checked)}
            className="accent-[var(--color-accent)]"
          />
          I understand this will delete existing assignments
        </label>
        <button
          id="generate-assignments-btn"
          onClick={handleGenerate}
          disabled={generating || !confirmGenerate}
          className="rounded-lg bg-[var(--color-accent)] px-6 py-2.5 font-medium text-[var(--color-text-inverse)] transition-all hover:bg-[var(--color-accent-hover)] disabled:opacity-50"
        >
          {generating ? 'Generating...' : 'Generate Assignments'}
        </button>
      </div>

      {message && (
        <div className="rounded-lg bg-[var(--color-emerald-muted)] px-4 py-3 text-sm text-[var(--color-emerald)] animate-fade-in">
          ✅ {message}
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

function StatCard({ label, value, color }: { label: string; value: number; color?: string }) {
  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-2)] p-4 text-center">
      <p className="text-2xl font-bold" style={{ color: color || 'var(--color-text-primary)' }}>
        {value.toLocaleString()}
      </p>
      <p className="mt-1 text-xs text-[var(--color-text-muted)]">{label}</p>
    </div>
  );
}
