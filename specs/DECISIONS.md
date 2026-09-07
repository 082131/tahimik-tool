# Architectural decisions

Decisions that cut across more than one spec, recorded once here rather than
repeated in each. Each states what was decided, why, what was rejected, and
what it costs.

---

## AD-001: Adopt Stanford's MrT5 implementation; do not use their checkpoint

**Date**: 2026-08-27
**Status**: Decided, **not yet implemented**
**Affects**: [004](004-fixed-rate-compression/spec.md), [005](005-noise-adaptive-byt5/spec.md)

### Decision

Replace the hand-written delete gate in `src/models/delete_gate.py` with
Stanford's implementation from
[github.com/jkallini/mrt5](https://github.com/jkallini/mrt5)
(`models/modeling_mrt5.py`), and **train it from `byt5-small` on this study's
data**.

Explicitly **do not** use the released
[`stanfordnlp/mrt5-small`](https://huggingface.co/stanfordnlp/mrt5-small)
checkpoint.

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
| **A.** Keep the reimplementation | Yes | Unproven | Current state. Works, but fidelity is undefended |
| **B.** Stanford's code, trained from `byt5-small` | Yes | Yes | **Chosen** |
| **C.** Stanford's checkpoint, fine-tuned on study data | **No** | Yes | Rejected — buys fidelity at the cost of the study design |

C was rejected specifically because its failure mode is asymmetric and
unrecoverable. If TAHIMIK wins, the result is merely conservative. If MrT5 wins
or ties, the outcome cannot be attributed and the thesis has no clean answer.
Betting the comparison on getting the preferred result is a poor position to
defend from.

### What this costs

- **Licensing**: Apache 2.0, so vendoring the model file is permitted **with
  attribution**. The repository currently has no top-level `LICENSE`,
  `NOTICE`, or `ATTRIBUTIONS.md`; one is now required.
- **Integration**: the repo is a research release, cloned rather than pip
  installed, and its `utils.py` expects a `BASE_PATH` macro to be edited.
  Vendoring the single model file is likely cleaner than depending on the repo
  wholesale.
- **Rework**: `delete_gate.py` and both compressed variants change.
  `tests/test_delete_gate.py` and `tests/test_model_forward.py` assume the
  current gate's 4-tuple return signature and will need revision.
- **Risk**: this replaces working, tested code. It should not be attempted
  immediately before a deadline.

### Consequences for existing findings

- **Finding 8** (`L_attn_reg` deviates from MrT5 Appendix D) is likely resolved
  by adopting their loss alongside their gate — to be confirmed, not assumed.
- **Finding 9** (`navg` initialisation, gate-shift clamp) stays open. Both are
  TAHIMIK's own additions and appear in no MrT5 implementation.
- **Finding 7** (`cn` sign unconstrained) stays open, for the same reason.
- The Python-loop hard deletion becomes moot, since Stanford's vectorised
  version replaces it.

### Not yet done

This is a decision, not an implementation. Nothing in `src/` has changed.
Tasks live in [004](004-fixed-rate-compression/tasks.md) and
[005](005-noise-adaptive-byt5/tasks.md).

---

## AD-002: Authoritative ByT5-Base Backbone, Dynamic Collation, and Position Bias Preservation

**Date**: 2026-09-08
**Status**: Decided and **Implemented**
**Affects**: `configs/base.py`, `src/models/`, `src/data/`, `src/training/`

### Decision
1. Configure `google/byt5-base` as the base model backbone across all three experimental conditions (ByT5 baseline, MrT5 fixed compression, TAHIMIK noise-adaptive).
2. Maintain position bias preservation across the compression interface by capturing the relative position bias tensor at layer index 1 and vector-gathering it across retained token indices via `compress_position_bias`.
3. Eliminate static 1,024-byte padding at dataset initialization; use a 1,024-byte truncation ceiling and dynamically collate batches to batch-maximum length using `NormalizationCollator`.
4. Restore `best_stage1.pt` (evaluated on validation loss) before constructing the optimizer and scheduler for Stage 2 fine-tuning.

