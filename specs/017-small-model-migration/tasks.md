# Tasks: Small Model Migration

**Input**: `spec.md`, `research.md`, `data-model.md`, and `plan.md`

- [X] T001 [US1] Set the shared compressed-model source in `configs/base.py` to `stanfordnlp/mrt5-small`.
- [X] T002 [US1] Set `configs/byt5_config.py` to `google/byt5-small`.
- [X] T003 [US1] Verify MrT5 and TAHIMIK inherit the Stanford Small source.
- [X] T004 [US2] Persist and validate model-source and architecture fingerprints in `src/training/trainer.py`.
- [X] T005 [US2] Add Base/Small and same-shape cross-source rejection coverage in `tests/test_checkpoint_compatibility.py`.
- [X] T006 [US3] Configure physical/accumulated batches as 4x4 and 4x2 while preserving effective sizes 16 and 8.
- [X] T007 [US1] Add exact source-resolution coverage in configuration tests.
- [X] T008 Verify the focused configuration, compatibility, and accumulation tests.
- [X] T009 Verify the complete Python suite.

## Open system follow-ups

- [ ] T010 Route standalone evaluation and benchmark checkpoint loads through
  the shared architecture/source validator.
- [ ] T011 Apply `configure_determinism` consistently to every
  results-producing entry point.
- [ ] T012 Persist and reuse a versioned split manifest with checkpoints and
  results.
