# Tasks: Imported Corpus Contract

## Phase 1 — Setup

- [ ] T001 Create invalid/valid CSV fixtures in `tests/fixtures/import_contracts/` without real participant text.
- [ ] T002 [P] Define contract enums and dataclasses in `src/data/import_contracts.py`.

## Phase 2 — Foundational validation

- [ ] T003 [US1] Add failing tests for headers, statuses, four-word minimum, UTF-8 byte limit, duplicates, conflicts, and privacy-safe reports in `tests/test_import_contracts.py`.
- [ ] T004 [US1] Implement gold CSV parsing and validation in `src/data/import_contracts.py` until T003 passes.
- [ ] T005 [US1] Add failing tests for long-form binary labels, category coverage, annotator coverage, and foreign IDs in `tests/test_import_contracts.py`.
- [ ] T006 [US1] Implement reliability parsing and coverage validation in `src/data/import_contracts.py` until T005 passes.

## Phase 3 — Full experiment eligibility

- [ ] T007 [US2] Add failing tests for row-order-independent seeded 12,000/1,500/1,500 splitting and exact-count boundaries in `tests/test_import_contracts.py`.
- [ ] T008 [US2] Implement canonical fingerprints, deterministic splitting, and full eligibility reporting in `src/data/import_contracts.py`.
- [ ] T009 [US2] Add a failing orchestration test proving invalid exports stop before model construction in `tests/test_run_experiment_preflight.py`.
- [ ] T010 [US2] Integrate strict preflight and the exact 1,000,000 synthetic-count check in `scripts/run_experiment.py` and `src/data/preprocessing.py`.

## Phase 4 — Verification

- [ ] T011 Run `python -m pytest tests/test_import_contracts.py tests/test_run_experiment_preflight.py -q`.
- [ ] T012 Verify reports contain no `noisy` or `clean` values and update `README.md` with the two external-export contracts.

## Dependencies and parallel work

T001 and T002 may run in parallel. T003 precedes T004; T005 precedes T006; T007 precedes T008; T009 precedes T010. Phase 3 depends on Phase 2.
