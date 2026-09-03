# Feature Specification: Chapter 3 Methodology Compliance

## Objective

Align the executable evaluation pipeline with Chapter 3's documented protocol and make every reported result auditable.

## Requirements

- **FR-001 Metrics:** Compute source-aware GLEU+, chrF, ERR, and alpha-word accuracy per sentence and as corpus summaries.
- **FR-002 Bootstrap:** Compare exactly ByT5 vs TAHIMIK and MrT5 vs TAHIMIK using 1,000 paired, two-tailed bootstrap resamples with add-one smoothing, percentile 95% CIs, and CI-aware significance.
- **FR-003 Efficiency:** Record per-sentence inference latency and one peak-GPU-memory observation per timed run after five warm-ups and a memory reset.
- **FR-004 Wilcoxon:** Use two-sided Wilcoxon signed-rank for paired GPU-memory runs and report median, IQR, and rank-biserial effect size. CPU-only runs explicitly skip this test.
- **FR-005 Holm:** Apply step-down Holm-Bonferroni within each declared comparison family and report raw and adjusted p-values.
- **FR-006 Data validation:** Normalize, filter, deduplicate exact pairs, and reject conflicting clean targets for one noisy input before splitting data.
- **FR-007 Synthetic data:** Keep code-switching, Taglish morphology, slang, and emoji intact; synthetic corruption may use only configured orthographic/noise operations.
- **FR-008 Reproducibility:** Results and checkpoints include git SHA, dirty-tree flag, resolved configuration, seed, Python/package versions, and device information. Deterministic mode is available.

## Acceptance criteria

1. Unit tests cover each statistical formula and each validation/protection rule.
2. A complete experiment JSON can be traced to its source configuration and repository state.
3. CPU CI reports accuracy and latency while clearly marking GPU-memory inference unavailable.
4. Existing model behavior and user changes remain intact.
