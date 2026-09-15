# Tasks: ByT5 Base Migration

> **Status:** Completed historical work, superseded by Spec 017. These tasks are
> not current implementation instructions.

**Input**: Design documents from `/specs/015-byt5-base-migration/`
**Prerequisites**: [spec.md](spec.md), [plan.md](plan.md), [data-model.md](data-model.md), [research.md](research.md), [quickstart.md](quickstart.md)

---

## Task Breakdown

### Task 1: Establish Authoritative Base Model Identifier
- [x] T001 [US1] Write failing authority tests in `tests/test_model_configuration.py` asserting all 3 configs resolve `google/byt5-base` and `scripts/run_experiment.py` has no hardcoded Small identifier.
- [x] T002 [US1] Set `BaseConfig.model_name = "google/byt5-base"` in `configs/base.py`.
- [x] T003 [US1] Update `scripts/run_experiment.py` to instantiate `AutoTokenizer.from_pretrained(base_config.model_name)`.
- [x] T004 [US1] Run tests: `pytest tests/test_model_configuration.py -v` (PASSED).

### Task 2: Prove Model Modules Derive Dimensions Dynamically
- [x] T005 [US2] Add parameterized architecture tests in `tests/test_model_forward.py` across `(32, 4)` and `(48, 18)` depths/widths.
- [x] T006 [US2] Verify `FixedCompressionByT5`, `NoiseAdaptiveByT5`, `DeleteGate`, and `NoiseEstimator` initialize linear layers dynamically from `d_model`.
- [x] T007 [US2] Run tests: `pytest tests/test_model_forward.py -k backbone_shape -v` (PASSED).

### Task 3: Gradient Accumulation & Effective Batch Sizes
- [x] T008 [US3] Write failing accumulation tests in `tests/test_gradient_accumulation.py` verifying step counts and loss division.
- [x] T009 [US3] Add `stage1_gradient_accumulation_steps: int = 8` and `stage2_gradient_accumulation_steps: int = 4` to `configs/base.py` with physical batches of 2.
- [x] T010 [US3] Update `_train_epoch` in `src/training/trainer.py` to scale loss and update at accumulation boundaries.
- [x] T011 [US3] Compute scheduler steps via `math.ceil(len(train_loader) / accum_steps) * epochs`.
- [x] T012 [US3] Run tests: `pytest tests/test_gradient_accumulation.py -v` (PASSED).

### Task 4: Shared Base Memory Controls
- [x] T013 [US3] Add `gradient_checkpointing: bool = True` and `precision: str = "fp16"` to `configs/base.py`.
- [x] T014 [US3] Integrate `torch.amp.autocast` and `GradScaler` into `src/training/trainer.py`.
- [x] T015 [US3] Write `tests/test_memory_configuration.py` validating shared memory settings and device/precision combinations.
- [x] T016 [US3] Run tests: `pytest tests/test_memory_configuration.py -v` (PASSED).

### Task 5: Reject Small or Incompatible Checkpoints
- [x] T017 [US4] Write failing compatibility tests in `tests/test_checkpoint_compatibility.py`.
- [x] T018 [US4] Implement `validate_checkpoint_architecture(checkpoint, model, allow_legacy=False)` in `src/training/trainer.py`.
- [x] T019 [US4] Record architecture fingerprint in all saved checkpoints.
- [x] T020 [US4] Enforce architecture validation before `load_state_dict` in Stage 1 handoff and checkpoint loading.
- [x] T021 [US4] Run tests: `pytest tests/test_checkpoint_compatibility.py -v` (PASSED).

### Task 6: Hardware & Configuration Preflight CLI
- [x] T022 [US5] Create `scripts/preflight_base.py` supporting `--device`, `--max-input-length`, `--metadata-only`, and `--output`.
- [x] T023 [US5] Add unit tests in `tests/test_base_preflight.py` verifying offline metadata checks and OOM recommendations.
- [x] T024 [US5] Run offline preflight: `python scripts/preflight_base.py --metadata-only` (PASSED).

### Task 7: Manuscript Naming & Documentation Alignment
- [x] T025 Produce unambiguous model names (`ByT5-base`, `ByT5-base + fixed-rate deletion`, `TAHIMIK (ByT5-base + noise-adaptive deletion)`).
- [x] T026 Update `README.md`, `docs/DEFENSE-PREP.md`, `docs/CHAPTER3-COMPLIANCE.md`, `docs/PROCESS.md`, `specs/DECISIONS.md`, `specs/FINDINGS.md`, and `specs/PROVENANCE.md`.

### Task 8: Final Controlled-Comparison Verification
- [x] T027 Run full repository test suite: `python -m pytest tests -v` (81 passed).
- [x] T028 Confirm 0 occurrences of `google/byt5-small` in `configs`, `src`, `scripts`, and `backend`.
- [x] T029 Verify resolved control variables across all variants: effective batches 16 and 8.
