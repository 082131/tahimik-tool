# Tasks: Experiment Provenance

## Phase 1 — Schema and canonicalization

- [ ] T001 Define stable eligibility reason codes and provenance field requirements in `src/utils/reproducibility.py`.
- [ ] T002 [US1] Add failing recursive serialization, key-order, and fingerprint tests in `tests/test_provenance.py`.
- [ ] T003 [US1] Implement canonical value normalization and fingerprinting in `src/utils/reproducibility.py` until T002 passes.

## Phase 2 — Runtime collection

- [ ] T004 [US1] Add failing mocked CPU/CUDA/cuDNN/device/precision tests in `tests/test_provenance.py`.
- [ ] T005 [US1] Implement software, hardware, Git, and deterministic-setting collectors in `src/utils/reproducibility.py` until T004 passes.
- [ ] T006 [US2] Add failing eligibility tests for dirty, CPU efficiency, undersized data, missing manifest/resources, and eligible conditions in `tests/test_provenance.py`.
- [ ] T007 [US2] Implement computed eligibility and reason codes in `src/utils/reproducibility.py` until T006 passes.

## Phase 3 — Integration

- [ ] T008 [US1] Replace ad hoc metadata in `scripts/train.py` and `scripts/run_experiment.py` with the shared builder.
- [ ] T009 [US1] Add the shared provenance record to `scripts/evaluate.py` and `scripts/benchmark.py`.
- [ ] T010 [US1] Add the shared record and checkpoint parentage to `src/training/trainer.py`.

## Phase 4 — Verification

- [ ] T011 Add an artifact contract test across all commands/checkpoints in `tests/test_provenance_integration.py`.
- [ ] T012 Run `python -m pytest tests/test_provenance.py tests/test_provenance_integration.py -q` and confirm no raw text/environment dump is serialized.

## Dependencies and parallel work

T002 precedes T003, T004 precedes T005, and T006 precedes T007. T008–T010 can run in parallel after T003, T005, and T007; T011 depends on all integrations.
