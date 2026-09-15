# Feature Specification: Small Model Migration

**Feature Branch**: `feat/017-small-model-migration`

**Created**: 2026-09-13

**Status**: Implemented

**Input**: Run the three study variants with their designated Small pretrained sources while preserving comparable training and evaluation controls.

## User Scenarios & Testing

### User Story 1 - Resolve the intended Small source (Priority: P1)

A researcher can construct any variant and obtain the correct pretrained model identifier without changing scripts.

**Independent Test**: Instantiate all three configuration classes and compare their resolved `model_name` values.

**Acceptance Scenarios**:

1. **Given** `ByT5Config`, **when** it is resolved, **then** its model is `google/byt5-small`.
2. **Given** `MrT5Config` or `TAHIMIKConfig`, **when** it is resolved, **then** its model is `stanfordnlp/mrt5-small`.

### User Story 2 - Reject incompatible checkpoints (Priority: P1)

A researcher cannot accidentally load a Base checkpoint or exchange Google and Stanford checkpoints merely because two Small architectures have compatible shapes.

**Independent Test**: Validate matching Small metadata, Base metadata, and same-shape metadata from a different source.

### User Story 3 - Preserve experiment controls (Priority: P2)

The migration changes model sources and physical microbatches without changing effective batch sizes, data splits, seed, sequence ceilings, optimizer, or evaluation policy.

**Independent Test**: Assert Stage 1 uses `4 x 4 = 16` and Stage 2 uses `4 x 2 = 8`.

### Edge Cases

- A legacy checkpoint without architecture metadata fails closed unless legacy loading is explicitly allowed.
- `google/byt5-small` and `stanfordnlp/mrt5-small` remain non-interchangeable even when both report `d_model=1472`, 12 encoder layers, 4 decoder layers, and vocabulary size 384.
- No result produced with a Base checkpoint is eligible for a Small-model run.

## Requirements

### Functional Requirements

- **FR-001**: `ByT5Config.model_name` MUST be `google/byt5-small`.
- **FR-002**: The shared compressed-model default MUST be `stanfordnlp/mrt5-small`.
- **FR-003**: MrT5 and TAHIMIK MUST resolve to `stanfordnlp/mrt5-small`.
- **FR-004**: Saved checkpoints MUST record model source and architecture identity.
- **FR-005**: Loading MUST reject Base/Small architecture mismatches and same-shape source mismatches.
- **FR-006**: Stage 1 effective batch size MUST remain 16 and Stage 2 MUST remain 8.
- **FR-007**: Tokenizers MUST resolve from the same model identifier as their owning variant.

### Key Entities

- **Model Source Identity**: Checkpoint identifier plus architecture fingerprint.
- **Architecture Fingerprint**: `model_name`, hidden width, encoder-layer count, decoder-layer count, and vocabulary size.

## Success Criteria

- **SC-001**: Configuration tests resolve exactly one Google Small baseline and two Stanford MrT5 Small compressed variants.
- **SC-002**: Compatibility tests reject every Base checkpoint and every mismatched pretrained source.
- **SC-003**: Effective batch-size tests remain 16 and 8.
- **SC-004**: The complete Python test suite passes.

## Assumptions

- Small is the authoritative model-size family for current training and reporting.
- Existing Base-produced checkpoints are separate experimental artifacts and are not loaded into Small variants.

## Current Convergence Status

The model-source migration itself is implemented and covered by configuration
and checkpoint-compatibility tests. The following adjacent runtime issues remain
open and must not be mistaken for completed migration guarantees:

- `scripts/evaluate.py` and `scripts/benchmark.py` load state dictionaries
  directly and do not yet invoke the shared architecture/source validator.
- `scripts/train.py`, `scripts/evaluate.py`, and `scripts/benchmark.py` seed
  Torch but do not yet call `configure_determinism`; `run_experiment.py` does.
- Dataset membership is reconstructed from the current input order and seed;
  a durable split manifest is not yet stored with checkpoints/results.
- No trained study checkpoints currently exist. Configuration and loading tests
  do not establish trained-model availability.

The earlier length/label mismatch is resolved by
`specs/023-paired-length-policy/`: pair preparation now occurs before noise
labels and splitting, and datasets reject silent token truncation.
