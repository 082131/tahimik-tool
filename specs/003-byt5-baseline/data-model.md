# Phase 1 Data Model: ByT5 Baseline

This variant introduces no new data entity. It consumes the same training pair
every variant consumes and emits the same result shape, minus the compression
fields that do not apply to it.

## Forward-pass result

**What it represents**: What the baseline returns to the trainer and evaluator.

**Fields**:

| Key | Type | Notes |
|-----|------|-------|
| `loss` | scalar tensor | Present only when `labels` is supplied. This is `L_CE` alone — the baseline has no rate, attention, or noise term. |
| `logits` | `(batch, target_len, vocab)` | Standard seq2seq output over the byte vocabulary. |
| `encoder_last_hidden_state` | `(batch, seq_len, d_model)` | Sequence length is unchanged from the input, since nothing is deleted. |
| `deletion_rate` | `(batch,)` | Always exactly `0.0`. A true measurement, not a placeholder — see `research.md`. |

**Deliberately absent**: `gate_outputs`, `keep_prob`, `kept_mask`,
`target_deletion_rate`, `noise_scores`, `ne_loss`. None apply to a variant with
no gate. Whether their absence should instead be an explicit `None` is the open
question recorded as F2 / T002.

## (Noisy, clean) training pair

**What it represents**: The unit of training data. Not owned by this feature —
shared with all variants and specified in `006-two-stage-training`.

**Relevant detail here**: the pair carries a precomputed `n*` noise label, which
this variant receives and **ignores**. That is correct and intentional: `n*`
supervises the noise estimator, which this variant does not have. The trainer
passes `noise_level` to every variant uniformly; only TAHIMIK consumes it.

## Config fields read

| Field | Source | Why it matters here |
|-------|--------|---------------------|
| `model_name` | `configs/base.py` | The shared starting point that makes the encoder a control variable. Currently `google/byt5-small` against a manuscript specifying `byt5-base` — see `specs/FINDINGS.md` item 4. |
| `use_compression` | `configs/byt5_config.py` | `False` for this variant. Documents intent; the class contains no gate regardless. |

No field is written by this feature.
