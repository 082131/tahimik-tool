# Phase 1 Data Model: Two-Stage Training Pipeline

The entities that flow through training, and the split structure that makes the
three-variant comparison fair.

## (Noisy, clean, `n*`) training triple

**What it represents**: One training example. The unit every variant consumes.

**Fields**:

| Field | Type | Notes |
|-------|------|-------|
| noisy text | `str` | Input. Synthetically corrupted in Stage 1, human-observed in Stage 2 |
| clean text | `str` | The normalisation target |
| `n*` | `float` in `[0, 1]` | Precomputed noise label. Consumed only by TAHIMIK; passed to all variants uniformly |

**Produced by**: `src/data/noise_generator.py` (Stage 1) or the annotation
pipeline (Stage 2, blocked).

**Batched by**: `collate_fn` in `src/data/dataset.py` into `input_ids`,
`attention_mask`, `labels`, `noise_level`.

**Design note**: `noise_level` is passed to every variant, and ByT5 and MrT5
ignore it. That uniformity is deliberate — it keeps the trainer free of
per-variant branching, which is what makes "same pipeline" enforceable rather
than aspirational.

## Split structure

The shape of the data, and the reason for its asymmetry:

| Source | Train | Val | Test | Why |
|--------|-------|-----|------|-----|
| Synthetic (~1M) | 90% | 10% | **none** | A test score on synthetic data measures how well the model reverses the noise script. High numbers there are misleading, not encouraging |
| Gold (~15k) | 80% | 10% | 10% | The only test set. Every reported accuracy number comes from here |

**Control-variable requirement**: identical splits across all three variants,
from the same seed. Different splits would mean the variants were evaluated on
different sentences, and the comparison would be meaningless.

## Stage configuration

Two stages, each with its own schedule, from `configs/base.py`:

| Field | Stage 1 | Stage 2 | Notes |
|-------|---------|---------|-------|
| epochs | `stage1_epochs` (3) | `stage2_epochs` (10) | Fewer passes over abundant synthetic data; more over scarce gold |
| batch size | `stage1_batch_size` (16) | `stage2_batch_size` (8) | Smaller gold batches suit the smaller corpus |

Shared across both stages and all variants: `learning_rate`, AdamW betas and
epsilon, `weight_decay`, `max_grad_norm`, `warmup_ratio`, `lr_scheduler_type`,
`seed`.

## Loss composition

One class, three configurations, selected by flags rather than by branching in
the trainer:

| Variant | Active terms |
|---------|--------------|
| ByT5 | `L_CE` |
| MrT5 | `L_CE + w_rate·L_rate + w_attn_reg·L_attn_reg` |
| TAHIMIK | `L_CE + w_rate·L_rate + w_attn_reg·L_attn_reg + L_NE` |

`L_NE` carries implicit weight 1.0. The rate target differs by variant — a
shared constant for MrT5, per-sentence for TAHIMIK — which is the only place the
loss knows which variant it is serving.

## Checkpoint

**Currently stored**: `epoch`, `stage`, `model_state_dict`, `val_loss`.

**Required by Constitution Principle III but absent**: git SHA, dirty-tree flag,
resolved config. Without these, a checkpoint that produced a reported number
cannot be traced to the code that produced it.

**Selection rule**: written only when validation loss improves, with the best
tracker reset at each stage boundary because the two stages' losses are not
comparable quantities.

## Training history

Returned by `train()` as `{"stage1": {...}, "stage2": {...}}`, each holding
per-epoch train and validation loss dicts including component losses (`l_ce`,
`l_rate`, `l_attn_reg`, `l_ne`) where they apply.

Useful for diagnosis — a rate loss that never falls, or an `l_ne` that plateaus
immediately, are both visible here before any evaluation is run.
