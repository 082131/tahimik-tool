# Research and Decisions

## GLEU+ edge behavior

**Decision:** Source-copy penalties apply only to source n-grams absent from the clean reference. If source, reference, and prediction are identical clean text, the score is perfect.

## ERR aggregation

**Decision:** Report corpus ERR as `(sum source errors - sum prediction errors) / sum source errors`; retain sentence-level ERR for paired bootstrap. Define the zero-source-error case explicitly as perfect when prediction is also correct and non-improvement otherwise.

## Paired bootstrap

**Decision:** Sample paired indices with replacement 1,000 times. Orient deltas so positive favors the named target model, use the two-tailed add-one p-value capped at 1, percentile 2.5/97.5 bounds, and require both adjusted p significance and a CI excluding zero.

## Wilcoxon effect size

**Decision:** Apply the two-sided signed-rank test to nonzero paired differences. Compute matched-pairs rank-biserial as `(R_positive - R_negative)/(R_positive + R_negative)`, using average ranks for ties.

## Holm families

**Decision:** Accuracy family membership is saved explicitly. Each model-pair efficiency family contains only latency and memory. Holm adjusted p-values are cumulative maxima of ordered multiplicative adjustments, capped at one; rejection stops after the first failure.

## Reliability

**Decision:** Pivot long-form binary judgments to annotator-by-sentence matrices and compute nominal Krippendorff alpha separately per category. Any category below 0.80 blocks the full run.

## Memory protocol

**Decision:** Discard five warm-ups. For each of 20 timed runs synchronize, reset peak stats, run inference, synchronize, immediately read peak allocation, and retain the independent observation. CPU reports memory unavailable.
