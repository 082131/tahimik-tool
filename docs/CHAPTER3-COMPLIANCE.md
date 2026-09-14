# Chapter 3 Code Compliance Notes

## What the code calculates

- **GLEU+**: compares predicted and clean word n-grams (orders 1–4) and applies a source-copy penalty when a prediction simply repeats the noisy input.
- **chrF**: character n-gram F-score from SacreBLEU.
- **ERR**: `(distance(noisy, clean) - distance(predicted, clean)) / distance(noisy, clean)` per sentence.
- **Alpha-word accuracy**: sequence-aligned (Levenshtein distance on extracted alphabetic words normalized by reference word count), case-insensitive accuracy for Unicode alphabetic words.
- **Bootstrap**: resamples paired sentences 1,000 times; the two tails use add-one smoothing, a percentile 95% CI is reported, and significance requires both `p < .05` and a CI that excludes zero.
- **Wilcoxon**: tests paired per-run GPU memory and reports median, IQR, and rank-biserial effect size. CPU runs report unavailable rather than fake zeros.
- **Holm-Bonferroni**: applies the step-down family-wise correction and stores both raw and adjusted p-values.

## Architecture and Model Backbone

- **Backbone and Model Variants**: Three unambiguous conditions are compared under identical control variables:
  1. `ByT5-base` (uncompressed ceiling)
  2. `ByT5-base + fixed-rate deletion` (MrT5-style fixed delete gate)
  3. `TAHIMIK (ByT5-base + noise-adaptive deletion)` (noise-adaptive delete gate)
  All three backbones are initialized independently from Google's pretrained `google/byt5-base` checkpoint. Gate and estimator modules are randomly initialized. No released Stanford MrT5 checkpoint (`stanfordnlp/mrt5-small` or non-existent Base) is used.
- **Delete Gate Layer**: Layer 3 is retained as the absolute delete gate location for both compressed variants for manuscript fidelity.
- **Position Bias**: Relative position bias is extracted from layer index 1 and threaded across compression, preserving learned position embeddings.
- **Gating Parameterization**: Adaptive noise coefficient $c_n$ uses smooth softplus parameterization ($c_n = \text{softplus}(\text{raw\_cn}) \ge 0.0$) ensuring stable monotonic noise-adaptive compression.
- **Dynamic Padding**: Batches are padded dynamically to batch-max length via `NormalizationCollator` with a 1,024-byte truncation ceiling rather than static 1,024-byte zero-padding.
- **Batching & Gradient Accumulation**: Preserves manuscript effective batch sizes (Stage 1: 16, Stage 2: 8) using physical batch size 2 with 8 and 4 gradient accumulation steps respectively.
- **Two-Stage Checkpoint Handoff**: Stage 2 training deterministically restores the best Stage 1 checkpoint (`best_stage1.pt`) evaluated on validation loss before building the Stage 2 optimizer. Architecture validation enforces that Small or mismatched checkpoints cannot be loaded.

## Reproducibility and data controls

Experiment JSON and checkpoints include git SHA, dirty-tree status, seed, resolved configuration, Python/package versions, and platform details. Gold pairs are normalized before splitting; identical duplicates collapse and conflicting targets stop the run for review. Stage 1 requires an approved noise manifest; categories without implemented, reviewed local resources fail before generation. A Chapter 3 run remains ineligible until reviewed code-switching and Taglish-morphology resources are supplied and implemented.

The annotator website and human annotation workflow are not represented by this training/evaluation code and must be supplied as a separate artifact.
