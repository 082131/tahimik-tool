# Architectural decisions

Decisions that cut across more than one spec, recorded once here rather than
repeated in each. Each states what was decided, why, what was rejected, and
what it costs.

---

## AD-001: Adopt Stanford's MrT5 implementation; do not use their checkpoint

**Date**: 2026-08-27 (Historical Note: Initial decision specified `byt5-small`; superseded on 2026-09-08 by AD-002 specifying `google/byt5-base`)
**Status**: Superseded on 2026-09-15 by AD-003
**Affects**: [004](004-fixed-rate-compression/spec.md), [005](005-noise-adaptive-byt5/spec.md)

### Decision

Replace the hand-written delete gate in `src/models/delete_gate.py` with
Stanford's implementation from
[github.com/jkallini/mrt5](https://github.com/jkallini/mrt5)
(`models/modeling_mrt5.py`), and **train it from `google/byt5-base` on this study's
data** (originally drafted as `byt5-small` on 2026-08-27 before the base migration).

Explicitly **do not** use the released
[`stanfordnlp/mrt5-small`](https://huggingface.co/stanfordnlp/mrt5-small)
checkpoint or transplant any 1,472-dim weights into ByT5-base's 1,536-dim hidden states.

TAHIMIK's noise-adaptive conditioning is then built as an extension of that
same implementation, so both compressed variants share one gate and differ only
in whether the noise shift is applied.

### Why

**Using their code removes a liability.** The current gate is a reimplementation
from the paper's equations. A reimplementation must be shown faithful, and
comparison against Stanford's actual code found real differences — no Gumbel
noise during training, and a Python-loop hard deletion where theirs is
vectorised. Running their implementation ends that line of questioning entirely.

**Not using their checkpoint preserves the comparison.** A control variable does
not have to be zero, it has to be *constant*. Google's ByT5 pretraining is
shared by all three variants, so it cancels out. Stanford's additional
multilingual pretraining would be exclusive to MrT5, giving one variant a
training history the other two lack. Results would then have two possible
causes — the compression method, or the extra pretraining — and no way to
separate them.

The manuscript fixes *"the training data, the train/validation/test
partitioning, the model hyperparameters, and the hardware"* across all three
variants. The checkpoint breaks the first of those.

**TAHIMIK must extend the same gate.** If MrT5 ran Stanford's implementation and
TAHIMIK ran a hand-written one, the two would differ in more than the noise
conditioning, and the contribution would again be unmeasurable.

### Alternatives considered

| Option | Fair comparison | Faithful to MrT5 | Verdict |
|--------|-----------------|------------------|---------|
| **A.** Keep the reimplementation | Yes | Unproven | Rejected |
| **B.** Stanford's code, trained from `google/byt5-base` | Yes | Yes | **Chosen** (Originally `byt5-small` on 2026-08-27; upgraded to `google/byt5-base` in AD-002) |
| **C.** Stanford's checkpoint, fine-tuned on study data | **No** | Yes | Rejected — buys fidelity at the cost of the study design |

C was rejected specifically because its failure mode is asymmetric and
unrecoverable. If TAHIMIK wins, the result is merely conservative. If MrT5 wins
or ties, the outcome cannot be attributed and the thesis has no clean answer.
Betting the comparison on getting the preferred result is a poor position to
defend from.

---

## AD-002: Authoritative ByT5-Base Backbone, Dynamic Collation, Position Bias Preservation, and Gradient Accumulation

**Date**: 2026-09-08
**Status**: Superseded on 2026-09-15 by AD-003
**Affects**: `configs/base.py`, `src/models/`, `src/data/`, `src/training/`, `scripts/`

### Decision
1. **Unambiguous Model Variants**: Establish the manuscript-required three experimental conditions:
   - `ByT5-base` (uncompressed accuracy ceiling)
   - `ByT5-base + fixed-rate deletion` (MrT5-style fixed delete gate)
   - `TAHIMIK (ByT5-base + noise-adaptive deletion)` (noise-adaptive delete gate)
2. **Authoritative Backbone**: Configure `BaseConfig.model_name = "google/byt5-base"` as the sole authority across all models and tokenizers. All backbones are initialized independently from Google's pretrained checkpoint; gate and estimator modules are initialized randomly. No released Stanford MrT5 checkpoint is loaded.
3. **Delete Gate Layer**: Layer 3 is retained as the absolute delete gate location for both compressed variants for manuscript fidelity.
4. **Position Bias Preservation**: Maintain relative position bias across the compression interface by capturing the relative position bias tensor at layer index 1 and vector-gathering it across retained token indices via `compress_position_bias`.
5. **Dynamic Padding**: Eliminate static 1,024-byte padding at dataset initialization; use a 1,024-byte truncation ceiling and dynamically collate batches to batch-maximum length using `NormalizationCollator`.
6. **Effective Batch Preservation via Gradient Accumulation**: Maintain effective batch sizes of 16 (Stage 1) and 8 (Stage 2) using physical microbatches of 2 and gradient accumulation steps of 8 and 4 respectively.
7. **Two-Stage Checkpoint Handoff and Architecture Validation**: Stage 2 training deterministically restores the best Stage 1 checkpoint (`best_stage1.pt`) evaluated on validation loss before building the Stage 2 optimizer. Checkpoint architecture validation explicitly rejects Small or mismatched model checkpoints.

---

## AD-003: Designated Small sources and released MrT5 initialization

**Date**: 2026-09-15
**Status**: Decided and **implemented**
**Affects**: [017](017-small-model-migration/spec.md), [018](018-huggingface-mrt5-baseline/spec.md), [019](019-mrt5-backed-tahimik/spec.md)

### Decision

The active controlled comparison uses these designated Small sources:

1. `google/byt5-small` for the uncompressed ByT5 accuracy baseline.
2. `stanfordnlp/mrt5-small` for the fixed-rate MrT5 efficiency baseline.
3. `stanfordnlp/mrt5-small` as TAHIMIK's starting model and gate, extended by
   the noise estimator, adaptive target, and centered gate shift.

The compressed variants load Stanford's custom model with
`AutoModelForSeq2SeqLM.from_pretrained(..., trust_remote_code=True)`. Each
wrapper captures the embedded pretrained gate, disables its embedded execution,
and imports its weights into the local compatible gate so deletion occurs once.
Construction fails if the pretrained gate cannot be loaded.

### Control and compatibility consequences

- MrT5 and TAHIMIK share the same pretrained compressed-model starting point;
  their experimental difference is fixed versus noise-adaptive deletion.
- The uncompressed baseline remains the native Google ByT5 Small model.
- Checkpoints record both model source and architecture fingerprint. Google
  Small and Stanford Small checkpoints are rejected across variants even when
  their dimensions match.
- Base checkpoints are not eligible for Small-model runs.
- Effective batches remain 16 in Stage 1 and 8 in Stage 2.

### Authority

Specs 017–019 govern current model selection and initialization. AD-001 and
AD-002 remain architecture-history records and are not current build or
reporting instructions.

