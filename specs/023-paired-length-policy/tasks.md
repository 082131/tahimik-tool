# Tasks: Paired Length Policy

**Input**: `spec.md`, `plan.md`, `research.md`, and `data-model.md`

## Phase 1: Policy contract

- [X] T001 [US1] Add failing whole-aligned-group, split/merge, exact-label,
  exact-limit, audit, and impossible-first-group tests in
  `tests/test_paired_length_policy.py`.
- [X] T002 [US1] Add failing dataset no-silent-truncation regression coverage
  in `tests/test_paired_length_policy.py`.

## Phase 2: Shared preparation foundation

- [X] T003 [US1] Implement `PreparedPairs`, `PairPreparationAudit`, tokenizer
  measurement, aligned word groups, and `prepare_paired_examples` in
  `src/data/preprocessing.py`.
- [X] T004 [US1] Run focused policy tests and verify T001/T002 pass.

## Phase 3: Dataset and entry-point integration

- [X] T005 [US2] Replace silent tokenizer truncation with explicit length
  validation in `src/data/dataset.py`.
- [X] T006 [US2] Prepare gold and synthetic pairs before label computation and
  splitting in `scripts/train.py`, `scripts/evaluate.py`, and
  `scripts/run_experiment.py`.
- [X] T007 [US2] Add entry-point and split-disjointness coverage proving only
  prepared pairs reach partitions.

## Phase 4: Convergence and verification

- [ ] T008 Compare specification requirements to the final code and classify
  each as implemented, intentionally excluded, or divergent.
- [ ] T009 Run `python -m pytest tests/test_paired_length_policy.py -v` and
  `python -m pytest tests/ -v`.
- [ ] T010 Inspect `git diff --check` and staged paths; commit only source,
  tests, and SpecKit artifacts.
