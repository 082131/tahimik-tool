# Feature Specification: Fixed-Rate Compression (MrT5 Baseline)

**Feature Branch**: `feat/004-fixed-rate-compression`

**Created**: 2026-08-27

**Status**: Draft — **Class B provenance**, see [../PROVENANCE.md](../PROVENANCE.md)

**Input**: Retrofit spec for `src/models/fixed_compression_byt5.py` and the
fixed-rate mode of `src/models/delete_gate.py`.

## Plain-language summary

This is the second baseline. It speeds the model up by throwing away a fixed
share of the input — always the same share, about half, no matter how messy the
sentence is. That is the point and the weakness. It proves compression makes
things faster, and it sets up TAHIMIK's argument: *a fixed rate is the wrong
rate for a messy sentence.*

## What the manuscript specifies

> "MrT5 with fixed-rate compression … will serve as the baseline of the study."

> The proposed model is "inspired by MrT5's compression with the addition of
> the proposed solution of making it adaptive to noise."

Reference: Kallini et al. (2025), *MrT5: Dynamic Token Merging for Efficient
Byte-Level Language Models* (ICLR 2025). `configs/mrt5_config.py` records the
gate placement (layer 3), the fixed target (0.5), and `gate_k = -30.0`.

## User Scenarios & Testing

### User Story 1 - Compress at a fixed rate during training (Priority: P1)

**Actor**: the training loop.

The delete gate scores every byte, and the rate loss pushes the measured
deletion rate toward one fixed target for every sentence, regardless of input.

**Independent Test**: Run a short optimization loop on a tiny T5 and confirm
the measured deletion rate moves toward the configured target.

**Acceptance Scenarios**:

1. **Given** a training batch, **When** the gate runs, **Then** it produces
   per-byte scores bounded in `[k, 0]` and a deletion rate in `[0, 1]`.
2. **Given** training mode, **When** deletion is applied, **Then** it is *soft*
   — the sequence length is unchanged and the operation stays differentiable,
   so the rate loss can actually train the gate.
3. **Given** the rate loss, **When** it computes its target, **Then** the
   target is the configured fixed value, identical for every sentence in the
   batch — this is what distinguishes this variant from TAHIMIK.

### User Story 2 - Physically compress at inference (Priority: P1)

**Actor**: the inference path and the efficiency benchmark.

**Independent Test**: Run an eval forward pass and confirm the encoder output
sequence is no longer than the input.

**Acceptance Scenarios**:

1. **Given** eval mode, **When** deletion is applied, **Then** bytes are
   *physically removed*, producing a shorter sequence — this is where the
   speedup actually comes from. Soft masking at inference would produce no
   saving at all.

### Edge Cases

- **Every byte deleted in a sentence**: `apply_hard_deletion` forces a minimum
  sequence length of 1 (`delete_gate.py:215`), so the decoder always receives
  something. Behavior is defined.
- **Padding counted as deleted**: both `keep_prob` and `kept_mask` are
  multiplied by the attention mask, and the rate divides by real-token count
  only, so padding never inflates the reported compression.

## Requirements

- **FR-001**: The gate MUST sit after encoder layer `delete_gate_layer` (3),
  splitting the encoder into pre-gate and post-gate halves.
- **FR-002**: The gate MUST produce scores via `k · sigmoid(Linear(LayerNorm(H)))`,
  bounded in `[k, 0]` with `k = -30.0`.
- **FR-003**: In training, deletion MUST be soft — the gate score added to the
  attention logits as a log-space bias, never multiplied into the binary mask.
- **FR-004**: In inference, deletion MUST be hard — bytes physically removed
  from the sequence.
- **FR-005**: The rate loss target MUST be a single fixed value applied
  identically to every sentence, with **no** noise conditioning.
- **FR-006**: The variant MUST emit its own configured
  `fixed_deletion_target` so the loss does not fall back to a hardcoded default.
- **FR-007**: Reported deletion rate MUST exclude padding from both numerator
  and denominator.

## Success Criteria

### Checkable now

- **SC-001**: Deletion rate is within `[0, 1]` for any batch.
- **SC-002**: Training-mode deletion rate is differentiable (`grad_fn` present).
- **SC-003**: Eval-mode encoder output is no longer than the input sequence.
- **SC-004**: A short optimization run moves the measured rate toward the target.

### BLOCKED on the gold-standard dataset

- **SC-005** *(blocked)*: Accuracy cost of fixed compression versus the
  uncompressed baseline on real data.
- **SC-006** *(blocked)*: Whether fixed compression degrades more on noisy
  sentences than clean ones — the premise TAHIMIK is built on.

## Assumptions

- This variant is a replication, not a contribution. Deviations from the MrT5
  paper are defects unless the manuscript states otherwise.
- The delete gate is shared with `005`; this spec covers only its
  **fixed-rate** mode (`noise_adaptive=False`).

## Convergence — assessed against the implementation

| ID | Gap | Severity | Evidence |
|----|-----|----------|----------|
| F1 | **contradicts** | **HIGH** | `byt5-small` vs manuscript's `byt5-base`. Cross-cutting, see [../PROVENANCE.md](../PROVENANCE.md). |
| F2 | **contradicts** | MEDIUM | `src/training/losses.py:106-129` implements `L_attn_reg` as a gate-commitment term `mean(4p(1-p))`, **not** the attention-weight regularizer MrT5 Appendix D specifies. The code documents this deviation honestly in a comment, but the manuscript does not — a reader comparing the two would find a mismatch the manuscript does not acknowledge. |
| F3 | partial | LOW | `fixed_compression_byt5.py:176` builds `encoder_outputs = (hidden_states,)` and never uses it — dead assignment. |

**FR-001 through FR-007: satisfied.** The soft/hard deletion split is correctly
implemented, and the log-space-bias subtlety (FR-003) is not only correct but
documented at the call site with the reasoning for why multiplication would
break it.

### Convergence tasks

- [ ] T001 Decide whether the `L_attn_reg` deviation from MrT5 Appendix D should be (a) reconciled in the code, or (b) documented in the manuscript as a deliberate, justified substitution. Currently the code is honest and the manuscript is silent (contradicts) — **this one needs the author, it is not a code question**
- [ ] T002 [P] Remove the dead `encoder_outputs` assignment at `src/models/fixed_compression_byt5.py:176` (partial)
- [ ] T003 [P] Confirm `tests/test_model_forward.py` and `tests/test_delete_gate.py` between them already cover SC-001 through SC-004; if so, record that rather than writing duplicates (partial)
