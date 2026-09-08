# Feature Specification: ByT5 Base Migration

**Feature Branch**: `feat/015-byt5-base-migration`
**Created**: 2026-09-08
**Status**: Implemented
**Input**: Migrate the controlled ByT5 baseline, MrT5-style fixed compression, and TAHIMIK noise-adaptive experiment from `google/byt5-small` to the manuscript-required `google/byt5-base` backbone without introducing tokenizer, checkpoint, memory, or comparison confounds.

---

## User Scenarios & Testing

### User Story 1 - Authoritative ByT5-Base Backbone (Priority: P1) 🎯
A researcher runs any of the three variants (ByT5 baseline, MrT5 fixed compression, or TAHIMIK noise-adaptive) and verifies that every variant and tokenizer loads from `google/byt5-base` as the single authority, matching Chapter 3 of the manuscript.

**Independent Test**: `pytest tests/test_model_configuration.py -v` confirms all three variants resolve `google/byt5-base` and no script hardcodes `google/byt5-small`.

### User Story 2 - Dynamic Architecture Dimensions (Priority: P1)
A model module (DeleteGate, NoiseEstimator, FixedCompressionByT5, NoiseAdaptiveByT5) dynamically derives its linear projections from `model.config.d_model` (1,536 for Base, 1,472 for Small) and encoder depths without hardcoded dimension constants.

**Independent Test**: `pytest tests/test_model_forward.py -k backbone_shape -v` verifies that both small and 18-layer Base-scale configurations construct and execute forward/generate.

### User Story 3 - Effective Batch Size Preservation via Gradient Accumulation (Priority: P1)
To prevent out-of-memory errors on available GPUs when scaling from Small to Base (582M parameters), training physical microbatches are set to 2, and gradient accumulation steps preserve the manuscript's effective batch size of 16 (Stage 1) and 8 (Stage 2).

**Independent Test**: `pytest tests/test_gradient_accumulation.py -v` verifies exact accumulation boundaries, loss scaling, and ceiling division for scheduler step counts.

### User Story 4 - Checkpoint Architecture Validation (Priority: P1)
When restoring checkpoints (e.g. Stage 1 to Stage 2 handoff or inference evaluation), the trainer inspects architecture metadata (`d_model`, layer counts, `model_name`). It explicitly rejects incompatible Small checkpoints (`d_model=1472`) when attempting to load into Base models (`d_model=1536`).

**Independent Test**: `pytest tests/test_checkpoint_compatibility.py -v` validates rejection of Small checkpoints and fail-closed behavior for legacy checkpoints.

### User Story 5 - Hardware Preflight Verification (Priority: P2)
Before initiating multi-epoch training, an operator runs an explicit preflight command to verify model resolution, compute parameters, check tokenization, and test generation under memory limits.

**Independent Test**: `python scripts/preflight_base.py --metadata-only` and `pytest tests/test_base_preflight.py -v`.

---

## Edge Cases

- **CUDA OOM**: Preflight and trainer detect out-of-memory errors, clear device caches, and output recommendations to decrease physical microbatch while increasing accumulation proportionally.
- **Legacy Checkpoints**: Checkpoints lacking architecture identity fail closed unless `allow_legacy_checkpoint=True` is provided; using legacy checkpoints disqualifies results from thesis reporting.
- **CPU Fallback**: Running on a CPU device automatically disables fp16/bf16 with a clean configuration error or fallback rather than silent underflow.

---

## Requirements

### Functional Requirements

- **FR-001**: `BaseConfig.model_name` MUST be `"google/byt5-base"` and serve as the sole authority across configs, model loaders, and tokenizers.
- **FR-002**: All three backbones MUST initialize independently from Google's pretrained `google/byt5-base` checkpoint; study-specific gate and estimator modules MUST be randomly initialized.
- **FR-003**: The released `stanfordnlp/mrt5-small` checkpoint MUST NOT be loaded or transplanted into ByT5-base.
- **FR-004**: Delete gate location MUST remain at layer 3 for both compressed variants for manuscript fidelity.
- **FR-005**: Physical batch sizes (2) MUST be paired with gradient accumulation steps (8 for Stage 1, 4 for Stage 2) to maintain effective batch sizes 16 and 8.
- **FR-006**: Loss MUST be divided by the accumulation factor before backpropagation, and optimizer/scheduler updates MUST occur only at accumulation boundaries or epoch end.
- **FR-007**: Saved checkpoints MUST store architecture metadata (`model_name`, `model_type`, `d_model`, `num_encoder_layers`, `num_decoder_layers`, `vocab_size`).
- **FR-008**: Loading a checkpoint with mismatched `d_model` or layer counts MUST raise a descriptive `ValueError` listing all mismatches.

---

## Success Criteria

- **SC-001**: `python -m pytest tests -v` passes 100% (81 tests) offline with zero downloads.
- **SC-002**: Grep searches across `configs`, `src`, `scripts`, and `backend` return zero occurrences of `google/byt5-small` or `stanfordnlp/mrt5-small`.
- **SC-003**: Preflight reports `google/byt5-base` for all variants with effective batches 16 and 8.
- **SC-004**: Small checkpoints fail closed when evaluated against Base architectures.
