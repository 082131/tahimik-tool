// TAHIMIK database types — mirrors the Supabase Postgres schema.
// Keep in sync with supabase/migrations/001_initial_schema.sql.

export type UserRole = 'annotator' | 'admin';
export type UserStatus = 'active' | 'suspended';
export type TaskType = 'normalization' | 'labeling';
export type AssignmentStatus = 'pending' | 'in_progress' | 'completed';

export type EventType =
  | 'keystroke_summary'
  | 'paste_attempt'
  | 'copy_attempt'
  | 'cut_attempt'
  | 'drag_attempt'
  | 'tab_blur'
  | 'focus_return'
  | 'time_to_first_keystroke'
  | 'edit_count'
  | 'submit';

export interface User {
  id: string;
  name: string;
  role: UserRole;
  status: UserStatus;
}

export interface Sentence {
  id: string;
  noisy_text: string;
  source_platform: string | null;
  batch: string | null;
  in_reliability: boolean;
}

export interface Assignment {
  id: string;
  sentence_id: string;
  user_id: string;
  task: TaskType;
  status: AssignmentStatus;
}

export interface Normalization {
  id: string;
  sentence_id: string;
  user_id: string;
  normalized_text: string | null;
  needs_review: boolean;
  notes: string | null;
  submitted_at: string;
}

export interface Labeling {
  id: string;
  sentence_id: string;
  user_id: string;
  labels: Record<string, number>;
  needs_review: boolean;
  submitted_at: string;
}

export interface Event {
  id: string;
  user_id: string;
  sentence_id: string | null;
  task: TaskType | null;
  type: EventType;
  value: Record<string, unknown> | null;
  ts: string;
}

// Joined types for UI convenience
export interface AssignmentWithSentence extends Assignment {
  sentences: Sentence;
}

export interface NormalizationWithSentence extends Normalization {
  sentences: Sentence;
}

export interface LabelingWithSentence extends Labeling {
  sentences: Sentence;
}
