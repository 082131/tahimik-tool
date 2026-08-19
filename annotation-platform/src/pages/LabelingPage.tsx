import { useState, useEffect, useCallback } from 'react';
import { supabase } from '../lib/supabase';
import { useAuth } from '../contexts/AuthContext';
import type { AssignmentWithSentence } from '../lib/types';
import LabelingForm from '../components/labeling/LabelingForm';
import ProgressBar from '../components/shared/ProgressBar';
import HistoryPanel from '../components/shared/HistoryPanel';

export default function LabelingPage() {
  const { user } = useAuth();
  const [currentAssignment, setCurrentAssignment] = useState<AssignmentWithSentence | null>(null);
  const [loading, setLoading] = useState(true);
  const [completed, setCompleted] = useState(0);
  const [total, setTotal] = useState(0);
  const [showHistory, setShowHistory] = useState(false);

  const fetchNextAssignment = useCallback(async () => {
    if (!user) return;
    setLoading(true);

    try {
      const { count: totalCount } = await supabase
        .from('assignments')
        .select('id', { count: 'exact', head: true })
        .eq('user_id', user.id)
        .eq('task', 'labeling');

      const { count: completedCount } = await supabase
        .from('assignments')
        .select('id', { count: 'exact', head: true })
        .eq('user_id', user.id)
        .eq('task', 'labeling')
        .eq('status', 'completed');

      setTotal(totalCount ?? 0);
      setCompleted(completedCount ?? 0);

      const { data, error } = await supabase
        .from('assignments')
        .select('*, sentences(*)')
        .eq('user_id', user.id)
        .eq('task', 'labeling')
        .in('status', ['pending', 'in_progress'])
        .order('sentence_id', { ascending: true })
        .limit(1)
        .single();

      if (error && error.code !== 'PGRST116') {
        console.error('Fetch error:', error);
      }

      setCurrentAssignment(data as AssignmentWithSentence | null);

      if (data && data.status === 'pending') {
        await supabase
          .from('assignments')
          .update({ status: 'in_progress' })
          .eq('id', data.id);
      }
    } catch (err) {
      console.error('Failed to fetch assignment:', err);
    } finally {
      setLoading(false);
    }
  }, [user]);

  useEffect(() => { fetchNextAssignment(); }, [fetchNextAssignment]);

  const handleSubmit = useCallback(() => {
    setTimeout(() => fetchNextAssignment(), 800);
  }, [fetchNextAssignment]);

  const handleEdit = useCallback((item: { sentence_id: string }) => {
    async function loadAssignment() {
      if (!user) return;
      const { data } = await supabase
        .from('assignments')
        .select('*, sentences(*)')
        .eq('user_id', user.id)
        .eq('task', 'labeling')
        .eq('sentence_id', item.sentence_id)
        .single();
      if (data) setCurrentAssignment(data as AssignmentWithSentence);
    }
    loadAssignment();
    setShowHistory(false);
  }, [user]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="text-center">
          <div className="mb-3 h-8 w-8 mx-auto animate-spin rounded-full border-2 border-[var(--color-surface-3)] border-t-[var(--color-accent)]" />
          <p className="text-sm text-[var(--color-text-muted)]">Loading assignment...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl animate-fade-in">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-xl font-semibold text-[var(--color-text-primary)]">Noise Labeling</h1>
        <button
          id="toggle-history-labeling"
          onClick={() => setShowHistory(!showHistory)}
          className="rounded-lg border border-[var(--color-border)] px-3 py-1.5 text-xs text-[var(--color-text-secondary)] transition-colors hover:bg-[var(--color-surface-2)]"
        >
          {showHistory ? 'Hide' : 'Show'} History
        </button>
      </div>

      <div className="mb-6">
        <ProgressBar current={completed} total={total} />
      </div>

      {currentAssignment ? (
        <LabelingForm
          key={currentAssignment.id}
          assignment={currentAssignment}
          onSubmit={handleSubmit}
        />
      ) : (
        <div className="glass-panel p-12 text-center">
          <div className="mb-4 text-5xl">🎉</div>
          <h2 className="mb-2 text-xl font-semibold text-[var(--color-text-primary)]">
            All labeling tasks complete!
          </h2>
          <p className="text-sm text-[var(--color-text-muted)]">
            You've labeled all {total} assigned sentences.
          </p>
        </div>
      )}

      {showHistory && user && (
        <div className="mt-6">
          <HistoryPanel task="labeling" userId={user.id} onEdit={handleEdit} />
        </div>
      )}
    </div>
  );
}
