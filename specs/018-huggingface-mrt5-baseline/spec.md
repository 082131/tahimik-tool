# Feature Specification: Hugging Face MrT5 Baseline

**Feature Branch**: `feat/018-huggingface-mrt5-baseline`

**Created**: 2026-09-13

**Status**: Implemented

**Input**: Use the released Stanford MrT5 Small model and pretrained delete gate as the fixed-rate efficiency baseline.

## User Scenarios & Testing

### User Story 1 - Load the official MrT5 source (Priority: P1)

A researcher constructing the fixed-rate variant receives Stanford's custom MrT5 model and pretrained weights from `stanfordnlp/mrt5-small`.

**Independent Test**: Replace the loader with a spy and assert `AutoModelForSeq2SeqLM.from_pretrained("stanfordnlp/mrt5-small", trust_remote_code=True)`.

### User Story 2 - Apply one pretrained gate (Priority: P1)

The wrapper copies the embedded Stanford gate into the local gate interface, disables the embedded gate, and performs deletion exactly once.

**Independent Test**: Supply a model with an embedded gate, verify it is disabled, and compare copied gate parameters.

### User Story 3 - Fail closed (Priority: P1)

If pretrained gate weights cannot be recovered, model construction stops rather than silently using a random fixed-rate baseline.

**Independent Test**: Make gate loading return false and assert a `RuntimeError`.

### Edge Cases

- An offline environment may use an already cached/local Hugging Face model, but a missing model or gate cannot become a reporting run.
- The embedded and wrapper gates must never both execute.
- The local compatibility loader accepts Stanford gate names and maps them to the wrapper's names.

## Requirements

### Functional Requirements

- **FR-001**: MrT5 MUST load `stanfordnlp/mrt5-small` with `trust_remote_code=True`.
- **FR-002**: The matching tokenizer MUST load from `stanfordnlp/mrt5-small`.
- **FR-003**: The wrapper MUST capture the embedded pretrained Stanford delete gate before disabling it.
- **FR-004**: The embedded gate MUST be disabled before the wrapper executes encoder layers.
- **FR-005**: The local gate MUST use Stanford's T5 RMS normalization, `k * sigmoid(-logit)` direction, training-only Gumbel noise, and vectorized hard deletion.
- **FR-006**: Gate placement MUST be configuration index 2, meaning after the third encoder block.
- **FR-007**: The fixed deletion target MUST be 0.5 and noise conditioning MUST be disabled.
- **FR-008**: Construction MUST fail when pretrained gate loading fails.

### Key Entities

- **Remote MrT5 Model**: Stanford custom architecture and released Small weights.
- **Embedded Gate**: Gate contained in the loaded remote encoder.
- **Wrapper Gate**: Local compatible gate used by the training/evaluation interface.

## Success Criteria

- **SC-001**: Loader tests prove the exact model identifier and trusted custom-code setting.
- **SC-002**: Tests prove one-gate execution and exact gate-weight transfer.
- **SC-003**: No random-gate fallback is possible.
- **SC-004**: Forward, generation, and delete-gate tests pass.

## Assumptions

- Hugging Face hosts the selected released model; `jkallini/mrt5` is the design/code attribution source.
- The wrapper remains necessary for the project's shared telemetry, losses, and hard-deletion interface.

## Current Convergence Status

The source selection, remote-code loading, pretrained-gate transfer, one-gate
execution, and fail-closed constructor are implemented. These issues remain:

- **Decoder-logit parity:** the manual teacher-forced decoder calls `lm_head`
  without applying the backbone's conditional `model_dim ** -0.5` scaling when
  `scale_decoder_outputs` is enabled. Native generation uses the backbone path,
  so training/validation logits can differ from generation logits.
- **Soft/hard rate gap:** `L_rate` trains the differentiable soft rate
  `1 - mean(keep_prob)`, while inference reports the hard threshold rate from
  `kept_mask`. No acceptance criterion currently proves that hard rates meet
  the 0.5 target.
- **Standalone checkpoint validation:** `scripts/evaluate.py` and
  `scripts/benchmark.py` bypass the shared architecture/source validator.
- **Error visibility:** the gate compatibility loader catches download/format
  failures broadly before the wrapper raises its fail-closed `RuntimeError`.
- No fine-tuned MrT5 study checkpoint currently exists.
