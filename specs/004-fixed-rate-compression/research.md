# Phase 0 Research: Fixed-Rate Compression (MrT5)

Grilling answers with their sources. One item is left deliberately unresolved
because it is a manuscript question, not a code question.

## Decision: this is a replication, not an adaptation

**Decision**: MrT5 is reproduced as published. Deviations are defects unless the
manuscript explicitly declares them.

**Rationale**: Manuscript — TAHIMIK is *"inspired by MrT5's compression with the
addition of the proposed solution of making it adaptive to noise."* The addition
belongs to TAHIMIK. MrT5 itself must stay untouched, because it is the thing
TAHIMIK's improvement is measured against. A tuned MrT5 would make the
comparison meaningless in either direction.

**Alternatives considered**: Treating MrT5 as a starting point to improve on.
Rejected — an improved MrT5 is no longer the published baseline the study claims
to compare against.

**Immediate consequence**: this decision converts the `L_attn_reg` substitution
(below) from "a choice" into "a defect or an undeclared deviation". It cannot be
both a replication and a modification.

## An official MrT5 release exists, and this repo does not use it

**Fact, recorded because it changes what a reviewer can ask.**

MrT5 has both an official code repository
([github.com/jkallini/mrt5](https://github.com/jkallini/mrt5)) and a released
pretrained checkpoint on HuggingFace
([`stanfordnlp/mrt5-small`](https://huggingface.co/stanfordnlp/mrt5-small),
~1.2GB safetensors with a custom model implementation).

The released checkpoint's configuration is close to this repo's: trained at
deletion rate **δ = 0.5**, deleting after the **third** encoder layer. Those
match `fixed_deletion_target = 0.5` and `delete_gate_layer = 3` in
`configs/mrt5_config.py`.

This repository uses neither. `src/models/delete_gate.py` is written from the
paper using only `torch` and `torch.nn`; no MrT5 package is imported or
installed.

**Why reimplementing is defensible** — the decisive reason first:

1. **Control variables.** `stanfordnlp/mrt5-small` was produced by multilingual
   continued pretraining, not by this study's two-stage schedule on
   Tagalog/Taglish data. The manuscript fixes *"the training data, the
   train/validation/test partitioning, the model hyperparameters, and the
   hardware"* across all three variants. Dropping in a checkpoint carrying
   someone else's pretraining would break exactly the condition that makes the
   comparison fair — it would compare "MrT5 trained on their corpus" against
   "TAHIMIK trained on ours".
2. **TAHIMIK extends the same gate.** The noise-adaptive shift is added to this
   gate implementation. Extending code the project owns is simpler and safer
   than modifying a third-party model class.

**Why it is still a liability**: a reimplementation must be shown faithful,
whereas a library would inherit that. The `L_attn_reg` deviation below is
precisely the kind of gap that carries. The released checkpoint offers a way to
close it — see the validation task.

**Recorded so it is not discovered by a reviewer first**: the manuscript should
state that an official release exists and why it was not used. Silence on this
reads as not having known.

## OPEN: the attention regulariser substitutes for MrT5 Appendix D

**Status**: Unresolved. **Author decision required — this is a manuscript
question.**

MrT5 Appendix D defines `L_attn_reg` over attention weights, penalising
attention paid to positions the gate marked for deletion. That requires
materialising per-head attention matrices.

`src/training/losses.py:106-129` instead implements a gate-commitment term,
`mean(4·p·(1−p))`, maximal at `p = 0.5` and zero at both extremes. It serves a
related purpose — stopping the gate hedging — at negligible cost.

The code is honest about this: the comment names it a deviation, explains the
substitution, and records that the previous implementation was worse (it was
minimised by deleting the entire sequence, and was the only gradient reaching
the gate).

The manuscript does not mention it. Under the replication decision above, a
reader comparing manuscript to code finds an unacknowledged difference in the
loss function.

**Two acceptable resolutions**, both requiring the author:

1. Implement Appendix D's attention-weight regulariser, making the replication
   faithful.
2. Document the substitution in the manuscript as a deliberate, justified
   deviation with the reasoning the code comment already contains.

**Not acceptable**: leaving the code and manuscript silently disagreeing.

## Decision: gate at layer 3, and it is a control variable

**Decision**: `delete_gate_layer = 3`, identical for MrT5 and TAHIMIK.

**Rationale**: MrT5 places the gate after an early encoder block;
`configs/mrt5_config.py` records layer 3 with the paper as its source. Equality
with TAHIMIK is required by the study design — if the two variants gated at
different layers, any difference in results would confound gate *placement* with
noise *adaptation*, and the contribution would be unmeasurable.

**Why early rather than late**: the computational saving is proportional to the
number of layers that run *after* the gate. Gating at layer 3 of N means layers
3…N process a shorter sequence. Gating near the end would save almost nothing.
The lower bound is context: gate too early and the scores are computed from
representations too shallow to judge redundancy. Layer 3 is the paper's balance.

**Alternatives considered**: Tuning gate placement per variant. Rejected —
directly violates the control-variable requirement.

## Decision: soft deletion in training, hard deletion at inference

**Decision**: Two mechanisms, selected by `self.training`.

**Rationale**: They have incompatible requirements and neither works alone.

- Hard deletion physically removes bytes. It is a discrete decision, so it has
  no gradient — the gate could never learn from it.
- Soft deletion keeps every byte but makes deleted ones progressively harder to
  attend to. Differentiable, so the gate learns. But the sequence length is
  unchanged, so it produces **zero** computational saving.

Training needs the gradient; inference needs the saving. Hence both.

**Consequence for evaluation**: efficiency numbers are only meaningful from the
inference path. A benchmark accidentally run in training mode would measure a
compression that is not happening.

## Decision: soft deletion adds to attention logits, never multiplies the mask

**Decision**: The gate score is added to the already-extended attention mask as
a log-space bias.

**Rationale**: This is the subtlest correctness requirement in the feature, and
the failure mode is silent. HuggingFace builds its additive mask as
`(1 - mask) * finfo.min`. A soft mask value of `0.99997` therefore becomes a
bias of roughly `-1.1e34`, which softmax treats as `-inf`. The intended soft
mask becomes hard, the gradient dies, and the gate stops learning while the code
continues to run and produce plausible output.

`src/models/fixed_compression_byt5.py:82-94` carries this reasoning inline.

**Alternatives considered**: Multiplying the gate score into the binary mask —
the intuitive approach, and wrong for the reason above.

## Decision: the fixed rate is enforced by the loss, not the architecture

**Decision**: `fixed_deletion_target = 0.5`, driven by the rate loss.

**Rationale**: Nothing structurally prevents the gate deleting a different
fraction. The rate loss penalises squared deviation from the target, pushing the
measured rate toward it during training.

**Consequence for interpreting results**: a trained MrT5 measuring 40% deletion
is not broken. It means the rate loss traded off against cross-entropy — the
tension the design intends. Only a rate far from the target across all inputs
would indicate a defect.

## Decision: padding is excluded from the deletion rate

**Decision**: Both numerator and denominator count real tokens only.

**Rationale**: If padding counted as deleted, a heavily-padded batch would
report high compression for free, and reported efficiency would be an artifact
of batch composition rather than of the gate. `delete_gate.py:159` masks
`keep_prob`, `:165` masks `kept_mask`, and `:170` divides by real-token count.

**Alternatives considered**: Counting all positions. Rejected — makes the
headline efficiency metric depend on how batches happen to be assembled.
