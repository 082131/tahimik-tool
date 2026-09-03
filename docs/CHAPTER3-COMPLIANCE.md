# Chapter 3 Code Compliance Notes

## What the code calculates

- **GLEU+**: compares predicted and clean word n-grams (orders 1–4) and applies a source-copy penalty when a prediction simply repeats the noisy input.
- **chrF**: character n-gram F-score from SacreBLEU.
- **ERR**: `(distance(noisy, clean) - distance(predicted, clean)) / distance(noisy, clean)` per sentence.
- **Alpha-word accuracy**: positional, case-insensitive accuracy for Unicode alphabetic reference words.
- **Bootstrap**: resamples paired sentences 1,000 times; the two tails use add-one smoothing, a percentile 95% CI is reported, and significance requires both `p < .05` and a CI that excludes zero.
- **Wilcoxon**: tests paired per-run GPU memory and reports median, IQR, and rank-biserial effect size. CPU runs report unavailable rather than fake zeros.
- **Holm-Bonferroni**: applies the step-down family-wise correction and stores both raw and adjusted p-values.

## Reproducibility and data controls

Experiment JSON and checkpoints include git SHA, dirty-tree status, seed, resolved configuration, Python/package versions, and platform details. Gold pairs are normalized before splitting; identical duplicates collapse and conflicting targets stop the run for review. Synthetic data never rewrites slang, emoji, code-switching, or Taglish morphology.

The annotator website and human annotation workflow are not represented by this training/evaluation code and must be supplied as a separate artifact.
