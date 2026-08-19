-- ============================================================================
-- TAHIMIK Annotation Platform — Seed Data
-- ============================================================================
--
-- NOTE: Real user creation should happen via the Supabase Dashboard or
-- supabase auth admin API. The auth.users inserts below are placeholders
-- that work with `supabase db reset` in local development.
--
-- In production, create users through the Dashboard first, then insert
-- the corresponding rows into the `users` table.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- Placeholder UUIDs
-- ---------------------------------------------------------------------------
-- Admin:      a0000000-0000-0000-0000-000000000001
-- Annotator1: b0000000-0000-0000-0000-000000000001
-- Annotator2: b0000000-0000-0000-0000-000000000002
-- Annotator3: b0000000-0000-0000-0000-000000000003

-- ---------------------------------------------------------------------------
-- 1. Create auth.users (local dev only — will fail in hosted Supabase)
-- ---------------------------------------------------------------------------
-- These use Supabase's internal auth schema. Comment out if deploying to
-- hosted Supabase and create users via Dashboard instead.

INSERT INTO auth.users (id, instance_id, email, encrypted_password, email_confirmed_at, created_at, updated_at, aud, role)
VALUES
  (
    'a0000000-0000-0000-0000-000000000001',
    '00000000-0000-0000-0000-000000000000',
    'admin@tahimik.dev',
    crypt('admin-password-123', gen_salt('bf')),
    now(), now(), now(), 'authenticated', 'authenticated'
  ),
  (
    'b0000000-0000-0000-0000-000000000001',
    '00000000-0000-0000-0000-000000000000',
    'annotator1@tahimik.dev',
    crypt('annotator-password-1', gen_salt('bf')),
    now(), now(), now(), 'authenticated', 'authenticated'
  ),
  (
    'b0000000-0000-0000-0000-000000000002',
    '00000000-0000-0000-0000-000000000000',
    'annotator2@tahimik.dev',
    crypt('annotator-password-2', gen_salt('bf')),
    now(), now(), now(), 'authenticated', 'authenticated'
  ),
  (
    'b0000000-0000-0000-0000-000000000003',
    '00000000-0000-0000-0000-000000000000',
    'annotator3@tahimik.dev',
    crypt('annotator-password-3', gen_salt('bf')),
    now(), now(), now(), 'authenticated', 'authenticated'
  )
ON CONFLICT (id) DO NOTHING;

-- Also insert into auth.identities (required by Supabase Auth)
INSERT INTO auth.identities (id, provider_id, user_id, identity_data, provider, last_sign_in_at, created_at, updated_at)
VALUES
  (
    'a0000000-0000-0000-0000-000000000001',
    'a0000000-0000-0000-0000-000000000001',
    'a0000000-0000-0000-0000-000000000001',
    jsonb_build_object('sub', 'a0000000-0000-0000-0000-000000000001', 'email', 'admin@tahimik.dev'),
    'email', now(), now(), now()
  ),
  (
    'b0000000-0000-0000-0000-000000000001',
    'b0000000-0000-0000-0000-000000000001',
    'b0000000-0000-0000-0000-000000000001',
    jsonb_build_object('sub', 'b0000000-0000-0000-0000-000000000001', 'email', 'annotator1@tahimik.dev'),
    'email', now(), now(), now()
  ),
  (
    'b0000000-0000-0000-0000-000000000002',
    'b0000000-0000-0000-0000-000000000002',
    'b0000000-0000-0000-0000-000000000002',
    jsonb_build_object('sub', 'b0000000-0000-0000-0000-000000000002', 'email', 'annotator2@tahimik.dev'),
    'email', now(), now(), now()
  ),
  (
    'b0000000-0000-0000-0000-000000000003',
    'b0000000-0000-0000-0000-000000000003',
    'b0000000-0000-0000-0000-000000000003',
    jsonb_build_object('sub', 'b0000000-0000-0000-0000-000000000003', 'email', 'annotator3@tahimik.dev'),
    'email', now(), now(), now()
  )
ON CONFLICT (id) DO NOTHING;

-- ---------------------------------------------------------------------------
-- 2. Insert into public.users table
-- ---------------------------------------------------------------------------
INSERT INTO users (id, name, role, status)
VALUES
  ('a0000000-0000-0000-0000-000000000001', 'Admin User',     'admin',     'active'),
  ('b0000000-0000-0000-0000-000000000001', 'Annotator One',  'annotator', 'active'),
  ('b0000000-0000-0000-0000-000000000002', 'Annotator Two',  'annotator', 'active'),
  ('b0000000-0000-0000-0000-000000000003', 'Annotator Three','annotator', 'active')
ON CONFLICT (id) DO NOTHING;

-- ---------------------------------------------------------------------------
-- 3. Insert 10 sample sentences
-- ---------------------------------------------------------------------------
INSERT INTO sentences (id, noisy_text, source_platform, batch, in_reliability)
VALUES
  ('S001', 'anu ba yan hahaha di ko gets',                 'Facebook',  'batch_01', FALSE),
  ('S002', 'grbe talga yung init noh 🔥🔥🔥',              'Twitter',   'batch_01', FALSE),
  ('S003', 'HALAAA ang ganda naman neto!!!',                'Instagram', 'batch_01', FALSE),
  ('S004', 'pls lang po sana makapass ako 🙏',              'Facebook',  'batch_01', TRUE),
  ('S005', 'ang sarap ng ulam namin todayyy hehe',          'Twitter',   'batch_01', FALSE),
  ('S006', 'OMG nakita ko si @crush sa mall kanina lol',    'Twitter',   'batch_01', TRUE),
  ('S007', 'sna ol may jowa charot 😂😂',                   'Facebook',  'batch_01', FALSE),
  ('S008', 'gsto ko ng matulog ang pagod ko na talaga',     'Facebook',  'batch_02', TRUE),
  ('S009', 'LF: affordable laptop for online class pls dm', 'Facebook',  'batch_02', FALSE),
  ('S010', 'hpy bday sayo!! #blessed #thankful 🎂',         'Instagram', 'batch_02', FALSE)
ON CONFLICT (id) DO NOTHING;

-- ---------------------------------------------------------------------------
-- 4. Create sample assignments
-- ---------------------------------------------------------------------------

-- Normalization assignments: distribute sentences across annotators
-- Annotator 1: S001, S002, S003, S004
INSERT INTO assignments (sentence_id, user_id, task, status)
VALUES
  ('S001', 'b0000000-0000-0000-0000-000000000001', 'normalization', 'pending'),
  ('S002', 'b0000000-0000-0000-0000-000000000001', 'normalization', 'pending'),
  ('S003', 'b0000000-0000-0000-0000-000000000001', 'normalization', 'pending'),
  ('S004', 'b0000000-0000-0000-0000-000000000001', 'normalization', 'pending')
ON CONFLICT (sentence_id, user_id, task) DO NOTHING;

-- Annotator 2: S004, S005, S006, S007
INSERT INTO assignments (sentence_id, user_id, task, status)
VALUES
  ('S004', 'b0000000-0000-0000-0000-000000000002', 'normalization', 'pending'),
  ('S005', 'b0000000-0000-0000-0000-000000000002', 'normalization', 'pending'),
  ('S006', 'b0000000-0000-0000-0000-000000000002', 'normalization', 'pending'),
  ('S007', 'b0000000-0000-0000-0000-000000000002', 'normalization', 'pending')
ON CONFLICT (sentence_id, user_id, task) DO NOTHING;

-- Annotator 3: S007, S008, S009, S010
INSERT INTO assignments (sentence_id, user_id, task, status)
VALUES
  ('S007', 'b0000000-0000-0000-0000-000000000003', 'normalization', 'pending'),
  ('S008', 'b0000000-0000-0000-0000-000000000003', 'normalization', 'pending'),
  ('S009', 'b0000000-0000-0000-0000-000000000003', 'normalization', 'pending'),
  ('S010', 'b0000000-0000-0000-0000-000000000003', 'normalization', 'pending')
ON CONFLICT (sentence_id, user_id, task) DO NOTHING;

-- Labeling assignments: reliability sentences (S004, S006, S008) -> all 3 annotators
INSERT INTO assignments (sentence_id, user_id, task, status)
VALUES
  ('S004', 'b0000000-0000-0000-0000-000000000001', 'labeling', 'pending'),
  ('S004', 'b0000000-0000-0000-0000-000000000002', 'labeling', 'pending'),
  ('S004', 'b0000000-0000-0000-0000-000000000003', 'labeling', 'pending'),
  ('S006', 'b0000000-0000-0000-0000-000000000001', 'labeling', 'pending'),
  ('S006', 'b0000000-0000-0000-0000-000000000002', 'labeling', 'pending'),
  ('S006', 'b0000000-0000-0000-0000-000000000003', 'labeling', 'pending'),
  ('S008', 'b0000000-0000-0000-0000-000000000001', 'labeling', 'pending'),
  ('S008', 'b0000000-0000-0000-0000-000000000002', 'labeling', 'pending'),
  ('S008', 'b0000000-0000-0000-0000-000000000003', 'labeling', 'pending')
ON CONFLICT (sentence_id, user_id, task) DO NOTHING;
