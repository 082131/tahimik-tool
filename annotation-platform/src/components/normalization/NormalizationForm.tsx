import { useState, useRef, useEffect, useCallback } from 'react';
import type { AssignmentWithSentence } from '../../lib/types';
import { supabase } from '../../lib/supabase';
import { useAuth } from '../../contexts/AuthContext';
import { useAntiCheat } from '../../hooks/useAntiCheat';
import { useAutosave } from '../../hooks/useAutosave';
import { useKeyboardShortcuts } from '../../hooks/useKeyboardShortcuts';
import NoisySentenceDisplay from '../shared/NoisySentenceDisplay';
import NeedsReviewToggle from '../shared/NeedsReviewToggle';
import TabSwitchBanner from '../shared/TabSwitchBanner';

interface NormalizationFormProps {
  assignment: AssignmentWithSentence;
  onSubmit: () => void;
}

interface DraftState {
  normalizedText: string;
  needsReview: boolean;
  notes: string;
}

export default function NormalizationForm({ assignment, onSubmit }: NormalizationFormProps) {
  const { user } = useAuth();
  const sentenceId = assignment.sentence_id;
  const noisyText = assignment.sentences.noisy_text;

  const [normalizedText, setNormalizedText] = useState('');
  const [needsReview, setNeedsReview] = useState(false);
  const [notes, setNotes] = useState('');
  const [warning, setWarning] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const firstKeystrokeRef = useRef<number | null>(null);
  const editCountRef = useRef(0);
  const mountTimeRef = useRef(Date.now());

  // Anti-cheat
  const { tabSwitchCount, bindInputProtection, bindSourceProtection } = useAntiCheat({
    userId: user?.id ?? '',
    sentenceId,
    task: 'normalization',
    enabled: !!user,
  });

  // Autosave
  const autosaveKey = `${user?.id}-normalization-${sentenceId}`;
  const { savedValue, clearSaved } = useAutosave<DraftState>(
    autosaveKey,
    { normalizedText, needsReview, notes },
    3000,
  );

  // Restore draft on mount
  useEffect(() => {
    if (savedValue) {
      setNormalizedText(savedValue.normalizedText || '');
      setNeedsReview(savedValue.needsReview || false);
      setNotes(savedValue.notes || '');
    }
  }, [savedValue]);

  // Reset on assignment change
  useEffect(() => {
    setNormalizedText('');
    setNeedsReview(false);
    setNotes('');
    setWarning('');
    setSubmitted(false);
    firstKeystrokeRef.current = null;
    editCountRef.current = 0;
    mountTimeRef.current = Date.now();
  }, [sentenceId]);

  const handleNormalizedChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const val = e.target.value;
    setNormalizedText(val);
    editCountRef.current += 1;
    if (!firstKeystrokeRef.current) {
      firstKeystrokeRef.current = Date.now();
      // Log time-to-first-keystroke
      if (user) {
        supabase.from('events').insert({
          user_id: user.id,
          sentence_id: sentenceId,
          task: 'normalization',
          type: 'time_to_first_keystroke',
          value: { ms: Date.now() - mountTimeRef.current },
        }).then(() => {});
      }
    }
  }, [user, sentenceId]);

  const handleSubmit = useCallback(async () => {
    if (submitting || !user) return;
    setWarning('');

    // Non-blocking warnings
    if (!needsReview && normalizedText.trim() === noisyText.trim()) {
      setWarning('Normalized text is identical to noisy text. Submit anyway?');
    }
    if (!needsReview && !normalizedText.trim()) {
      setWarning('No normalized text provided. Submit anyway?');
    }

    setSubmitting(true);
    try {
      // Upsert normalization
      const { error: normError } = await supabase.from('normalizations').upsert({
        sentence_id: sentenceId,
        user_id: user.id,
        normalized_text: normalizedText.trim() || null,
        needs_review: needsReview,
        notes: notes.trim() || null,
        submitted_at: new Date().toISOString(),
      }, { onConflict: 'sentence_id,user_id' });

      if (normError) throw normError;

      // Update assignment status
      await supabase.from('assignments')
        .update({ status: 'completed' })
        .eq('id', assignment.id);

      // Log submit event
      await supabase.from('events').insert({
        user_id: user.id,
        sentence_id: sentenceId,
        task: 'normalization',
        type: 'submit',
        value: {
          edit_count: editCountRef.current,
          needs_review: needsReview,
          time_ms: Date.now() - mountTimeRef.current,
        },
      });

      // Log keystroke summary
      await supabase.from('events').insert({
        user_id: user.id,
        sentence_id: sentenceId,
        task: 'normalization',
        type: 'keystroke_summary',
        value: {
          edit_count: editCountRef.current,
          final_length: normalizedText.trim().length,
          time_to_first_keystroke_ms: firstKeystrokeRef.current
            ? firstKeystrokeRef.current - mountTimeRef.current
            : null,
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
  }, [submitting, user, needsReview, normalizedText, noisyText, sentenceId, notes, assignment.id, clearSaved, onSubmit]);

  // Keyboard shortcut
  useKeyboardShortcuts({
    'ctrl+enter': handleSubmit,
    'cmd+enter': handleSubmit,
  });

  const inputProtection = bindInputProtection();
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
      {/* Tab switch banner */}
      <TabSwitchBanner count={tabSwitchCount} visible={tabSwitchCount > 0} />

      {/* Noisy sentence */}
      <div {...sourceProtection}>
        <NoisySentenceDisplay text={noisyText} />
      </div>

      {/* Normalized text input */}
      <div>
        <label
          htmlFor="normalized-text"
          className="mb-2 block text-sm font-medium text-[var(--color-text-secondary)]"
        >
          Normalized Text {needsReview && <span className="text-[var(--color-amber)]">(optional — NEEDS REVIEW active)</span>}
        </label>
        <textarea
          id="normalized-text"
          value={normalizedText}
          onChange={handleNormalizedChange}
          spellCheck={false}
          autoCorrect="off"
          autoComplete="off"
          rows={4}
          placeholder="Type the normalized version of the sentence..."
          className="w-full resize-y rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-2)] px-4 py-3 text-base text-[var(--color-text-primary)] placeholder-[var(--color-text-muted)] transition-colors duration-200 focus:border-[var(--color-border-focus)] focus:outline-none focus:ring-1 focus:ring-[var(--color-accent)]"
          {...inputProtection}
        />
      </div>

      {/* NEEDS_REVIEW toggle */}
      <div className="flex items-center gap-4">
        <NeedsReviewToggle checked={needsReview} onChange={setNeedsReview} />
        <span className="text-xs text-[var(--color-text-muted)]">
          Mark as needing adjudication — normalized text becomes optional
        </span>
      </div>

      {/* Notes */}
      <div>
        <label
          htmlFor="notes"
          className="mb-2 block text-sm font-medium text-[var(--color-text-secondary)]"
        >
          Notes <span className="text-[var(--color-text-muted)]">(optional)</span>
        </label>
        <textarea
          id="notes"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          rows={2}
          placeholder="Any observations, difficulties, or context..."
          className="w-full resize-y rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-2)] px-4 py-3 text-sm text-[var(--color-text-primary)] placeholder-[var(--color-text-muted)] transition-colors duration-200 focus:border-[var(--color-border-focus)] focus:outline-none focus:ring-1 focus:ring-[var(--color-accent)]"
          {...inputProtection}
        />
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
          id="submit-normalization"
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
