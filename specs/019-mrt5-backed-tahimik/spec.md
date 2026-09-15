# Feature Specification: MrT5-Backed TAHIMIK

**Feature Branch**: `feat/019-mrt5-backed-tahimik`

**Created**: 2026-09-13

**Status**: Implemented

**Input**: Build TAHIMIK from the same released MrT5 Small source as the fixed-rate baseline, adding only the study's noise-adaptive mechanism.

## User Scenarios & Testing

### User Story 1 - Share the MrT5 starting point (Priority: P1)

A researcher comparing MrT5 and TAHIMIK knows both compressed variants start from `stanfordnlp/mrt5-small` and the same pretrained gate path.

**Independent Test**: Spy on both loaders and compare model identifiers and gate transfer.

### User Story 2 - Add noise adaptivity (Priority: P1)

TAHIMIK predicts sentence noise from pre-gate states and changes deletion pressure per example while retaining MrT5's base gate semantics.

**Independent Test**: For otherwise equal inputs, verify noisier examples receive lower deletion targets and compression than cleaner examples.

### User Story 3 - Preserve gradient isolation (Priority: P1)

The noise estimator learns only from its supervised noise-estimation loss; gate/rate objectives do not train it indirectly.

**Independent Test**: Backpropagate loss terms separately and inspect estimator gradients.

### Edge Cases

- Failure to load the pretrained Stanford gate aborts construction.
- The embedded remote gate remains disabled so adaptive deletion is applied once by the wrapper.
- Padding does not contribute to noise pooling or deletion-rate measurement.
- In evaluation, efficiency telemetry uses hard physical deletion rather than soft training probabilities.

## Requirements

### Functional Requirements

- **FR-001**: TAHIMIK MUST load `stanfordnlp/mrt5-small` using `trust_remote_code=True`.
- **FR-002**: TAHIMIK MUST import the same pretrained Stanford gate path used by the fixed MrT5 variant.
- **FR-003**: The remote embedded gate MUST be disabled before the wrapper gate executes.
- **FR-004**: A 256-unit MLP MUST predict `n` in `[0,1]` from masked mean-pooled pre-gate hidden states.
- **FR-005**: The adaptive deletion target MUST be `d_max * (1 - n)` with `d_max=0.5`.
- **FR-006**: Gate scores MUST receive the learned shift `cn * (n - navg)` with EMA momentum 0.99.
- **FR-007**: Noise scores used by gate/rate behavior MUST be detached so only `L_NE` trains the estimator.
- **FR-008**: TAHIMIK MUST preserve the MrT5 gate location, scale, soft-training, and hard-inference contracts.
- **FR-009**: Construction MUST fail if pretrained gate weights cannot be loaded.

### Key Entities

- **Noise Score (`n`)**: Predicted per-sentence noise level.
- **Noise Target (`n*`)**: Byte edit-distance supervision label.
- **Noise Average (`navg`)**: Detached exponential moving average used to center the gate shift.
- **Adaptive Coefficient (`cn`)**: Learned non-negative shift strength.

## Success Criteria

- **SC-001**: MrT5 and TAHIMIK resolve to the identical released model source.
- **SC-002**: Both copy pretrained gate weights and execute only the wrapper gate.
- **SC-003**: Tests prove clean inputs target more deletion than noisy inputs.
- **SC-004**: Tests prove estimator gradient isolation and finite end-to-end losses.

## Assumptions

- TAHIMIK's experimental contribution is the noise estimator, adaptive target, and centered gate shift—not a different pretrained backbone.
- MrT5 and TAHIMIK retain separate fine-tuned study checkpoints despite sharing their initial released source.

## Current Convergence Status

The shared Stanford initialization, estimator lifecycle, non-negative adaptive
coefficient, EMA update, adaptive target, and gradient isolation are implemented.
TAHIMIK currently inherits these unresolved compressed-path issues:

- Its manual teacher-forced decoder omits conditional decoder-output scaling
  before `lm_head` when the backbone enables `scale_decoder_outputs`.
- Its rate loss constrains a soft probability surrogate, while reported
  inference efficiency uses hard threshold deletion. Hard-rate agreement with
  per-example adaptive targets has not been calibrated or accepted by a test.
- Standalone evaluation and benchmarking bypass shared checkpoint
  architecture/source validation.
- The compatibility loader hides the underlying gate download/format exception
  before construction fails closed.
- No fine-tuned TAHIMIK study checkpoint currently exists.

The paired input/target length and `n*` supervision issue is resolved by Spec
023 and is not an open TAHIMIK defect.
