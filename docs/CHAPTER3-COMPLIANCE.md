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

- **Backbone**: `google/byt5-base` across all three experimental conditions (ByT5 baseline, MrT5 fixed compression, TAHIMIK noise-adaptive).
- **Position Bias**: Relative position bias is extracted from layer index 1 and threaded across compression, preserving learned position embeddings.
- **Gating Parameterization**: Adaptive noise coefficient $c_n$ uses smooth softplus parameterization ($c_n = \text{softplus}(\text{raw\_cn}) \ge 0.0$) ensuring stable monotonic noise-adaptive compression.
- **Dynamic Padding**: Batches are padded dynamically to batch-max length via `NormalizationCollator` with a 1,024-byte truncation ceiling rather than static 1,024-byte zero-padding.
- **Two-Stage Checkpoint Handoff**: Stage 2 training deterministically restores the best Stage 1 checkpoint (`best_stage1.pt`) evaluated on validation loss before building the Stage 2 optimizer.

## Reproducibility and data controls

Experiment JSON and checkpoints include git SHA, dirty-tree status, seed, resolved configuration, Python/package versions, and platform details. Gold pairs are normalized before splitting; identical duplicates collapse and conflicting targets stop the run for review. Synthetic data never rewrites slang, emoji, code-switching, or Taglish morphology.

The annotator website and human annotation workflow are not represented by this training/evaluation code and must be supplied as a separate artifact.
