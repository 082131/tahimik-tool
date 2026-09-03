# Tasks: Stage Checkpoint Handoff

## Phase 1 — Foundations

- [ ] T001 Define checkpoint validation, digest, and atomic-load helpers in `src/utils/checkpointing.py`.
- [ ] T002 [P] Create deterministic trainer/checkpoint fixtures in `tests/test_checkpoint_handoff.py`.

## Phase 2 — Best Stage 1 state

- [ ] T003 [US1] Add failing tests where the best Stage 1 epoch is not the last epoch in `tests/test_checkpoint_handoff.py`.
- [ ] T004 [US1] Refactor best-state tracking and checkpoint returns in `src/training/trainer.py` until T003 passes.
- [ ] T005 [US1] Add failing tests for missing, malformed, wrong-stage, and incompatible checkpoints in `tests/test_checkpoint_handoff.py`.
- [ ] T006 [US1] Implement validated restoration and clear failures in `src/training/trainer.py` until T005 passes.

## Phase 3 — Stage 2 handoff

- [ ] T007 [US1] Add a failing order test proving restoration occurs before Stage 2 optimizer/scheduler construction in `tests/test_checkpoint_handoff.py`.
- [ ] T008 [US1] Implement the restore-before-construction transition and independent Stage 2 best tracker in `src/training/trainer.py`.
- [ ] T009 [US1] Add parent checkpoint and handoff metadata to Stage 2 payloads in `src/training/trainer.py`.

## Phase 4 — Verification

- [ ] T010 Run `python -m pytest tests/test_checkpoint_handoff.py -q` and inspect a round-trip checkpoint.

## Dependencies and parallel work

T001 and T002 may run in parallel. Each failing-test task precedes its implementation. T007 depends on T004 and T006.
