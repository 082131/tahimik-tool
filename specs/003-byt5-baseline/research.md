# Phase 0 Research: ByT5 Baseline

Every question raised during grilling, with its answer and the source that
settled it. Two answers are implementation judgment rather than manuscript
fact; both are labelled as such.

## Decision: this variant is a control, not a contribution

**Decision**: Adopt stock ByT5 unchanged. Any deviation is a defect.

**Rationale**: Manuscript, Conceptual Framework — *"First is ByT5 with no
compression, MrT5 with fixed-rate compression, both will serve as the baseline
of the study."* A baseline that has been tuned or modified stops being a
reference point. Its value comes entirely from being ordinary.

**Alternatives considered**: Treating it as a tunable variant, so the
comparison is "best ByT5 vs best TAHIMIK". Rejected — the manuscript's control
variables fix hyperparameters across variants, so tuning one variant
independently would break the study design.

## Decision: identical config source, independent model instances

**Decision**: All three variants construct their own `T5ForConditionalGeneration`
from the same `config.model_name`. No object sharing.

**Rationale**: Manuscript lists control variables as *"the training data, the
train/validation/test partitioning, the model hyperparameters, and the hardware
used for training and evaluation"* — hyperparameters and data, not weights. Each
variant trains its own copy; the requirement is a common starting point and
common settings.

**Alternatives considered**: A literally shared encoder instance across
variants. Rejected as incoherent — the variants train separately and would
overwrite each other's weights.

## Decision: report `deletion_rate = 0.0`

**Decision**: Always emit `deletion_rate` as exactly `0.0` per sentence.

**Rationale**: **Implementation judgment, not manuscript-derived.** RQ2 compares
efficiency across all three variants, so the evaluation harness reads the same
keys from each. Omitting the key would force variant-specific branching in the
evaluator. `0.0` is additionally *truthful* — the model genuinely deletes zero
bytes — so it is a real measurement, not a placeholder.

**Alternatives considered**: Omit the key (rejected: forces branching); return
`None` (rejected: `None` means "not measured", but this *is* measured, and it
is zero).

## Decision: no delete gate at all, not a gate set to zero

**Decision**: The baseline contains no `DeleteGate` instance.

**Rationale**: This variant is the accuracy ceiling. A gate configured to delete
nothing would still add `LayerNorm` and `Linear` parameters to the encoder and
still participate in training dynamics, so the "ceiling" would be measured on a
model that is not actually plain ByT5. The manuscript's framing — *"ByT5 with no
compression"* — describes an absence, not a neutralised presence.

**Alternatives considered**: A gate with `fixed_deletion_target = 0.0`, for
code-path uniformity. Rejected for the reason above; uniformity is not worth
contaminating the control.

## Open: output-key uniformity across variants

**Status**: Unresolved judgment call, recorded not decided.

The `deletion_rate = 0.0` decision argues for uniform output keys. The
no-gate-at-all decision argues the baseline should stay genuinely bare.
`keep_prob` and `kept_mask` sit between the two: emitting them as `None` would
be scaffolding for a concept that does not apply to this variant, but omitting
them makes the three variants' contracts non-uniform.

Recorded as finding F2 and task T002. Not blocking — the evaluator does not
currently read those keys for this variant.

## Confirmation: beam width is a control variable

**Decision**: The baseline uses the same beam width as the other variants.

**Rationale**: Beam width is a model hyperparameter, and the manuscript fixes
hyperparameters as control variables. It is also uniquely dangerous to vary
here because it affects accuracy (RQ1) and inference time (RQ2) simultaneously —
a difference in beam width would confound both research questions at once.

**Related known gap**: `--num_beams` is currently a CLI flag rather than a
config value. Constitution Principle III requires anything altering model output
to live in `configs/`. Tracked in `specs/FINDINGS.md` item 4, not re-raised here.
