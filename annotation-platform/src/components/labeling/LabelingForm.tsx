import { useState, useEffect, useCallback } from 'react';
import type { AssignmentWithSentence } from '../../lib/types';
import type { LabelKey } from '../../lib/constants';
import { LABELS } from '../../lib/constants';
import { supabase } from '../../lib/supabase';
import { useAuth } from '../../contexts/AuthContext';
import { useAntiCheat } from '../../hooks/useAntiCheat';
import { useAutosave } from '../../hooks/useAutosave';
import { useKeyboardShortcuts } from '../../hooks/useKeyboardShortcuts';
import NoisySentenceDisplay from '../shared/NoisySentenceDisplay';
import NeedsReviewToggle from '../shared/NeedsReviewToggle';
import TabSwitchBanner from '../shared/TabSwitchBanner';
import LabelToggle from './LabelToggle';

interface LabelingFormProps {
  assignment: AssignmentWithSentence;
  onSubmit: () => void;
}

type LabelValues = Record<LabelKey, 0 | 1>;

const defaultLabels = (): LabelValues =>
  Object.fromEntries(LABELS.map((l) => [l.key, 0])) as LabelValues;

interface DraftState {
  labels: LabelValues;
  needsReview: boolean;
}

const groups = [
  { name: 'Noise', color: 'var(--color-accent)', keys: LABELS.filter((l) => l.group === 'Noise') },
  { name: 'Function', color: 'var(--color-violet)', keys: LABELS.filter((l) => l.group === 'Function') },
  { name: 'Metadata', color: 'var(--color-amber)', keys: LABELS.filter((l) => l.group === 'Metadata') },
];

export default function LabelingForm({ assignment, onSubmit }: LabelingFormProps) {
  const { user } = useAuth();
  const sentenceId = assignment.sentence_id;
  const noisyText = assignment.sentences.noisy_text;

  const [labels, setLabels] = useState<LabelValues>(defaultLabels);
  const [needsReview, setNeedsReview] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [warning, setWarning] = useState('');

  const { tabSwitchCount, bindSourceProtection } = useAntiCheat({
    userId: user?.id ?? '',
    sentenceId,
    task: 'labeling',
    enabled: !!user,
  });

  const autosaveKey = `${user?.id}-labeling-${sentenceId}`;
  const { savedValue, clearSaved } = useAutosave<DraftState>(
    autosaveKey,
    { labels, needsReview },
    3000,
  );

  // Restore draft
  useEffect(() => {
    if (savedValue) {
      setLabels(savedValue.labels || defaultLabels());
      setNeedsReview(savedValue.needsReview || false);
    }
  }, [savedValue]);

  // Reset on assignment change
  useEffect(() => {
    setLabels(defaultLabels());
    setNeedsReview(false);
    setWarning('');
    setSubmitted(false);
  }, [sentenceId]);

  const handleLabelChange = useCallback((key: LabelKey, value: 0 | 1) => {
    setLabels((prev) => ({ ...prev, [key]: value }));
  }, []);

  const handleSubmit = useCallback(async () => {
    if (submitting || !user) return;
    setWarning('');
    setSubmitting(true);

    try {
      const { error: labelError } = await supabase.from('labelings').upsert({
        sentence_id: sentenceId,
        user_id: user.id,
        labels,
        needs_review: needsReview,
        submitted_at: new Date().toISOString(),
      }, { onConflict: 'sentence_id,user_id' });

      if (labelError) throw labelError;

      await supabase.from('assignments')
        .update({ status: 'completed' })
        .eq('id', assignment.id);

      await supabase.from('events').insert({
        user_id: user.id,
        sentence_id: sentenceId,
        task: 'labeling',
        type: 'submit',
        value: {
          labels,
          needs_review: needsReview,
          active_label_count: Object.values(labels).filter((v) => v === 1).length,
        },
      });

      clearSaved();
      setSubmitted(true);
      onSubmit();
    } catch (err) {
      console.error('Submit failed:', err);
      setWarning('Submission failed. Please try again.');
    } finally {
      setSubmitting(false);
    }
  }, [submitting, user, sentenceId, labels, needsReview, assignment.id, clearSaved, onSubmit]);

  useKeyboardShortcuts({
    'ctrl+enter': handleSubmit,
    'cmd+enter': handleSubmit,
  });

  const sourceProtection = bindSourceProtection();

  if (submitted) {
    return (
      <div className="flex items-center justify-center py-12 animate-fade-in">
        <div className="text-center">
          <div className="mb-3 text-4xl">✅</div>
          <p className="text-lg text-[var(--color-text-primary)]">Submitted successfully!</p>
          <p className="text-sm text-[var(--color-text-muted)]">Loading next sentence...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <TabSwitchBanner count={tabSwitchCount} visible={tabSwitchCount > 0} />

      <div {...sourceProtection}>
        <NoisySentenceDisplay text={noisyText} />
      </div>

      <p className="text-xs text-[var(--color-text-muted)] italic">
        Labels are judged from the noisy text alone — no normalization is shown.
      </p>

      {/* Label groups */}
      {groups.map((group) => (
        <div key={group.name}>
          <h3
            className="mb-3 text-xs font-semibold uppercase tracking-widest"
            style={{ color: group.color }}
          >
            {group.name}
          </h3>
          <div className="flex flex-wrap gap-2">
            {group.keys.map((labelDef) => (
              <LabelToggle
                key={labelDef.key}
                label={labelDef}
                value={labels[labelDef.key]}
                onChange={(val) => handleLabelChange(labelDef.key, val)}
              />
            ))}
          </div>
        </div>
      ))}

      {/* NEEDS_REVIEW */}
      <div className="flex items-center gap-4">
        <NeedsReviewToggle checked={needsReview} onChange={setNeedsReview} />
        <span className="text-xs text-[var(--color-text-muted)]">
          Mark as needing adjudication
        </span>
      </div>

      {/* Warning */}
      {warning && (
        <div className="rounded-lg bg-[var(--color-amber-muted)] px-4 py-3 text-sm text-[var(--color-amber)] animate-fade-in">
          ⚠ {warning}
        </div>
      )}

      {/* Submit */}
      <div className="flex items-center gap-4">
        <button
          id="submit-labeling"
          onClick={handleSubmit}
          disabled={submitting}
          className="rounded-lg bg-[var(--color-accent)] px-6 py-2.5 font-medium text-[var(--color-text-inverse)] transition-all duration-200 hover:bg-[var(--color-accent-hover)] hover:shadow-lg disabled:cursor-not-allowed disabled:opacity-50"
        >
          {submitting ? 'Submitting...' : 'Submit'}
        </button>
        <span className="text-xs text-[var(--color-text-muted)]">
          or press Ctrl+Enter
        </span>
      </div>
    </div>
  );
}
