# Feature Specification: ByT5 Baseline (No Compression)

> **Current model-source authority:** `specs/017-small-model-migration/spec.md`.
> The active uncompressed baseline is `google/byt5-small`; clauses below that
> require a shared model identifier across compressed variants are superseded.

**Feature Branch**: `feat/003-byt5-baseline`

**Created**: 2026-08-27

**Status**: Draft — **Class B provenance**, see [../PROVENANCE.md](../PROVENANCE.md)

**Input**: Retrofit spec for `src/models/byt5_baseline.py`. Decisions taken
from the ratified manuscript, not from author interrogation.

## Plain-language summary

This is the "do nothing special" version of the model. It reads messy text and
cleans it up, using the full ByT5 model with nothing thrown away. Because it
never deletes anything, it should be the most accurate of the three and the
slowest. It exists to answer one question: **how much accuracy do the other two
variants give up in exchange for their speed?**

Without this baseline, "TAHIMIK is fast" is a claim with nothing to compare it
against.

## What the manuscript specifies

> "The independent variable highlights the various compression methods. First
> is ByT5 with no compression, MrT5 with fixed-rate compression, both will
> serve as the baseline of the study."

> "The base variant of ByT5 will be used, and both MrT5 and the proposed
> noise-adaptive model will adopt the base variant."

> Control variables: "the training data, the train/validation/test
> partitioning, the model hyperparameters, and the hardware used for training
> and evaluation."

## User Scenarios & Testing

### User Story 1 - Establish the accuracy ceiling (Priority: P1)

**Actor**: the evaluation pipeline.

The baseline is trained and evaluated on the same data, splits, and hardware as
the other two variants, so its accuracy scores form the ceiling the compressed
variants are measured against.

**Independent Test**: Run a forward and backward pass on a tiny randomly
initialized T5 and confirm the model trains, produces logits, and reports a
deletion rate of exactly zero.

**Acceptance Scenarios**:

1. **Given** a batch of noisy input and clean labels, **When** the baseline
   runs a training forward pass, **Then** it returns a finite loss and logits
   shaped for the byte vocabulary.
2. **Given** any input, **When** the baseline reports its deletion rate,
   **Then** the value is exactly `0.0` for every sentence — this variant
   deletes nothing by definition.
3. **Given** the same config fields the other two variants read, **When** the
   baseline is constructed, **Then** it uses the identical tokenizer and
   `model_name`, so the encoder is a genuine control variable.

### User Story 2 - Serve as the efficiency floor (Priority: P2)

**Actor**: the efficiency benchmark.

Because it processes every byte through every layer, this variant establishes
the *worst* inference time and *highest* GPU memory, against which compression
savings are measured.

**Independent Test**: Benchmark on any input and confirm it produces timing and
memory numbers; the comparison itself belongs to spec 007.

### Edge Cases

- **A fully-padded batch row**: handled by HuggingFace's own T5
  implementation; this variant adds no pooling or masking logic of its own, so
  it inherits upstream behavior. No project-specific handling required.

## Requirements

- **FR-001**: The baseline MUST wrap a standard `T5ForConditionalGeneration`
  with no delete gate, no noise estimator, and no compression of any kind.
- **FR-002**: The baseline MUST load its model and tokenizer from
  `config.model_name`, the same field the other two variants read, so the
  shared encoder stays a control variable.
- **FR-003**: The baseline MUST report a deletion rate of exactly `0.0` per
  sentence, so downstream evaluation can treat all three variants uniformly.
- **FR-004**: The baseline MUST expose the same forward-pass output keys the
  other variants expose where they are meaningful (`loss`, `logits`,
  `encoder_last_hidden_state`, `deletion_rate`), so the trainer and evaluator
  need no variant-specific branching.
- **FR-005**: The baseline MUST support beam-search generation with the same
  interface as the other two variants.

## Success Criteria

### Checkable now

- **SC-001**: A training forward pass on a tiny T5 returns a finite loss.
- **SC-002**: `deletion_rate` is exactly `0.0` for every sentence in any batch.
- **SC-003**: `generate()` returns byte IDs with batch dimension preserved.

### BLOCKED on the gold-standard dataset

- **SC-004** *(blocked)*: Actual GLEU+, chrF, ERR, and alpha-word accuracy
  scores on real Tagalog/Taglish test data.
- **SC-005** *(blocked)*: Confirmation that this variant is in fact the most
  accurate of the three — expected, but unproven without data.

## Assumptions

- The manuscript treats this as a replication of standard ByT5, not a novel
  contribution, so no design decisions are open here.
- Training schedule, optimizer, and data splits are shared with all variants
  and are specified in `006-two-stage-training`, not here.

## Convergence — assessed against `src/models/byt5_baseline.py`

| ID | Gap | Severity | Evidence |
|----|-----|----------|----------|
| F1 | **contradicts** | **HIGH** | `configs/base.py:25` pins `google/byt5-small`; the manuscript specifies the **base** variant. Inherited by all three model variants. See [../PROVENANCE.md](../PROVENANCE.md). |
| F2 | partial | LOW | `byt5_baseline.py` returns no `keep_prob` or `kept_mask` key. Harmless today because the evaluator does not read them for this variant, but it makes the three variants' output contracts non-uniform (FR-004). |

**FR-001, FR-002, FR-003, FR-005: satisfied.** The implementation is a clean,
minimal wrapper that does exactly what the manuscript describes. `deletion_rate`
is explicitly zeroed at `byt5_baseline.py:68`.

### Convergence tasks

- [ ] T001 Flag the `byt5-small` / `byt5-base` divergence in a comment at `configs/base.py:25` naming the manuscript as authoritative, per Constitution Principle III (contradicts)
- [ ] T002 [P] Decide whether the three variants should return a uniform output-key contract, or whether the evaluator should branch per variant; record the decision here (partial)
- [ ] T003 [P] Write `tests/test_byt5_baseline.py` covering SC-001 through SC-003 using the tiny-T5 pattern from `tests/test_model_forward.py` (missing)
