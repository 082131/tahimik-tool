-- ============================================================================
-- TAHIMIK Annotation Platform — Initial Schema
-- Migration: 001_initial_schema.sql
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- TABLES
-- ============================================================================

-- Users (synced with Supabase Auth)
CREATE TABLE users (
  id UUID PRIMARY KEY REFERENCES auth.users(id),
  name TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('annotator', 'admin')),
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'suspended'))
);

-- Sentences (no 'pool' column, no gold columns)
CREATE TABLE sentences (
  id TEXT PRIMARY KEY,
  noisy_text TEXT NOT NULL,
  source_platform TEXT,
  batch TEXT,
  in_reliability BOOLEAN NOT NULL DEFAULT FALSE
);

-- Assignments
CREATE TABLE assignments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  sentence_id TEXT NOT NULL REFERENCES sentences(id),
  user_id UUID NOT NULL REFERENCES users(id),
  task TEXT NOT NULL CHECK (task IN ('normalization', 'labeling')),
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed')),
  UNIQUE(sentence_id, user_id, task)
);

-- Task 1 output: Normalizations
CREATE TABLE normalizations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  sentence_id TEXT NOT NULL REFERENCES sentences(id),
  user_id UUID NOT NULL REFERENCES users(id),
  normalized_text TEXT,
  needs_review BOOLEAN NOT NULL DEFAULT FALSE,
  notes TEXT,
  submitted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(sentence_id, user_id)
);

-- Task 2 output: Labelings
CREATE TABLE labelings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  sentence_id TEXT NOT NULL REFERENCES sentences(id),
  user_id UUID NOT NULL REFERENCES users(id),
  labels JSONB NOT NULL DEFAULT '{"ABBR":0,"ORTHO":0,"ELONG":0,"CS":0,"EMOJI":0,"SLANG":0,"MORPH":0,"PUNC":0,"CAPS":0,"LAUGH_MARKER":0,"REACTION_MARKER":0,"HASHTAG":0,"MENTION":0,"URL":0}',
  needs_review BOOLEAN NOT NULL DEFAULT FALSE,
  submitted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(sentence_id, user_id)
);

-- Event log
CREATE TABLE events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id),
  sentence_id TEXT REFERENCES sentences(id),
  task TEXT CHECK (task IN ('normalization', 'labeling')),
  type TEXT NOT NULL CHECK (type IN (
    'keystroke_summary',
    'paste_attempt',
    'copy_attempt',
    'cut_attempt',
    'drag_attempt',
    'tab_blur',
    'focus_return',
    'time_to_first_keystroke',
    'edit_count',
    'submit'
  )),
  value JSONB,
  ts TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Assignments indexes
CREATE INDEX idx_assignments_user_id ON assignments(user_id);
CREATE INDEX idx_assignments_sentence_id ON assignments(sentence_id);
CREATE INDEX idx_assignments_task ON assignments(task);
CREATE INDEX idx_assignments_status ON assignments(status);
CREATE INDEX idx_assignments_user_task ON assignments(user_id, task);
CREATE INDEX idx_assignments_user_status ON assignments(user_id, status);

-- Normalizations indexes
CREATE INDEX idx_normalizations_user_id ON normalizations(user_id);
CREATE INDEX idx_normalizations_sentence_id ON normalizations(sentence_id);
CREATE INDEX idx_normalizations_needs_review ON normalizations(needs_review);

-- Labelings indexes
CREATE INDEX idx_labelings_user_id ON labelings(user_id);
CREATE INDEX idx_labelings_sentence_id ON labelings(sentence_id);
CREATE INDEX idx_labelings_needs_review ON labelings(needs_review);

-- Events indexes
CREATE INDEX idx_events_user_id ON events(user_id);
CREATE INDEX idx_events_sentence_id ON events(sentence_id);
CREATE INDEX idx_events_type ON events(type);
CREATE INDEX idx_events_ts ON events(ts);
CREATE INDEX idx_events_user_task ON events(user_id, task);

-- Sentences indexes
CREATE INDEX idx_sentences_in_reliability ON sentences(in_reliability);
CREATE INDEX idx_sentences_batch ON sentences(batch);

-- Users indexes
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_status ON users(status);

-- ============================================================================
-- ROW LEVEL SECURITY
-- ============================================================================

-- Enable RLS on all tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE sentences ENABLE ROW LEVEL SECURITY;
ALTER TABLE assignments ENABLE ROW LEVEL SECURITY;
ALTER TABLE normalizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE labelings ENABLE ROW LEVEL SECURITY;
ALTER TABLE events ENABLE ROW LEVEL SECURITY;

-- ---------------------------------------------------------------------------
-- Helper: check if current user is admin
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION is_admin()
RETURNS BOOLEAN AS $$
  SELECT EXISTS (
    SELECT 1 FROM users WHERE id = auth.uid() AND role = 'admin'
  );
$$ LANGUAGE sql SECURITY DEFINER STABLE;

-- ---------------------------------------------------------------------------
-- USERS policies
-- ---------------------------------------------------------------------------

-- Users can see their own row; admins can see all
CREATE POLICY "users_select_own_or_admin"
  ON users FOR SELECT
  USING (id = auth.uid() OR is_admin());

-- Only admins can insert users (or service role)
CREATE POLICY "users_insert_admin"
  ON users FOR INSERT
  WITH CHECK (is_admin());

-- Only admins can update users
CREATE POLICY "users_update_admin"
  ON users FOR UPDATE
  USING (is_admin())
  WITH CHECK (is_admin());

-- ---------------------------------------------------------------------------
-- SENTENCES policies
-- ---------------------------------------------------------------------------

-- Users can see sentences assigned to them; admins can see all
CREATE POLICY "sentences_select_assigned_or_admin"
  ON sentences FOR SELECT
  USING (
    is_admin()
    OR EXISTS (
      SELECT 1 FROM assignments
      WHERE assignments.sentence_id = sentences.id
        AND assignments.user_id = auth.uid()
    )
  );

-- Only admins can insert sentences
CREATE POLICY "sentences_insert_admin"
  ON sentences FOR INSERT
  WITH CHECK (is_admin());

-- Only admins can update sentences
CREATE POLICY "sentences_update_admin"
  ON sentences FOR UPDATE
  USING (is_admin())
  WITH CHECK (is_admin());

-- ---------------------------------------------------------------------------
-- ASSIGNMENTS policies
-- ---------------------------------------------------------------------------

-- Users can see their own assignments; admins can see all
CREATE POLICY "assignments_select_own_or_admin"
  ON assignments FOR SELECT
  USING (user_id = auth.uid() OR is_admin());

-- Only admins can insert assignments
CREATE POLICY "assignments_insert_admin"
  ON assignments FOR INSERT
  WITH CHECK (is_admin());

-- Users can update their own assignments (e.g., status); admins can update all
CREATE POLICY "assignments_update_own_or_admin"
  ON assignments FOR UPDATE
  USING (user_id = auth.uid() OR is_admin())
  WITH CHECK (user_id = auth.uid() OR is_admin());

-- Only admins can delete assignments
CREATE POLICY "assignments_delete_admin"
  ON assignments FOR DELETE
  USING (is_admin());

-- ---------------------------------------------------------------------------
-- NORMALIZATIONS policies
-- ---------------------------------------------------------------------------

-- Users can see their own normalizations; admins can see all
CREATE POLICY "normalizations_select_own_or_admin"
  ON normalizations FOR SELECT
  USING (user_id = auth.uid() OR is_admin());

-- Users can insert their own normalizations
CREATE POLICY "normalizations_insert_own"
  ON normalizations FOR INSERT
  WITH CHECK (auth.uid() = user_id);

-- Users can update their own normalizations; admins can update all
CREATE POLICY "normalizations_update_own_or_admin"
  ON normalizations FOR UPDATE
  USING (user_id = auth.uid() OR is_admin())
  WITH CHECK (user_id = auth.uid() OR is_admin());

-- Users can delete their own normalizations; admins can delete all
CREATE POLICY "normalizations_delete_own_or_admin"
  ON normalizations FOR DELETE
  USING (user_id = auth.uid() OR is_admin());

-- ---------------------------------------------------------------------------
-- LABELINGS policies
-- ---------------------------------------------------------------------------

-- Users can see their own labelings; admins can see all
CREATE POLICY "labelings_select_own_or_admin"
  ON labelings FOR SELECT
  USING (user_id = auth.uid() OR is_admin());

-- Users can insert their own labelings
CREATE POLICY "labelings_insert_own"
  ON labelings FOR INSERT
  WITH CHECK (auth.uid() = user_id);

-- Users can update their own labelings; admins can update all
CREATE POLICY "labelings_update_own_or_admin"
  ON labelings FOR UPDATE
  USING (user_id = auth.uid() OR is_admin())
  WITH CHECK (user_id = auth.uid() OR is_admin());

-- Users can delete their own labelings; admins can delete all
CREATE POLICY "labelings_delete_own_or_admin"
  ON labelings FOR DELETE
  USING (user_id = auth.uid() OR is_admin());

-- ---------------------------------------------------------------------------
-- EVENTS policies
-- ---------------------------------------------------------------------------

-- Users can insert events for themselves
CREATE POLICY "events_insert_own"
  ON events FOR INSERT
  WITH CHECK (auth.uid() = user_id);

-- Only admins can view events
CREATE POLICY "events_select_admin"
  ON events FOR SELECT
  USING (is_admin());
