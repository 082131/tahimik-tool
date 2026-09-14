# Methodology Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Make Chapter 3 metrics, annotation reliability, efficiency measurements, and paired statistics formula-correct and regression-tested.

**Architecture:** Keep metric computation pure in `metrics.py`, statistical procedures in `statistical_tests.py`, reliability reshaping/alpha in `annotation.py`, and device measurement in `efficiency.py`. `run_experiment.py` assembles explicitly named comparison families and saves vectors plus summaries.

**Tech Stack:** Python, NumPy, SciPy, sacrebleu, krippendorff, PyTorch, pytest.

**Spec:** `specs/014-methodology-verification/spec.md`

## Global Constraints

- Use behavioral, hand-computed fixtures rather than output-key-only tests.
- Preserve held-out sentence order for paired analyses.
- Bootstrap uses exactly 1,000 resamples for full results.
- Efficiency uses five discarded warm-ups and 20 independent paired runs.
- Do not edit the manuscript.

## Technical Context

| Area | Required behavior |
|---|---|
| GLEU+ | source-aware n-grams, declared order weights/brevity penalty, no penalty for correct unchanged-clean text |
| ERR | case-insensitive whitespace-token accuracy over Leave-As-Is; corpus totals for reported score and sentence vector retained for resampling |
| Bootstrap | paired, two-tailed add-one p, percentile 95% CI, direction-aware, p-and-CI significance |
| Wilcoxon | two-sided, paired 20 runs; signed-rank-sum rank-biserial |
| Holm | sorted step-down stopping, monotonic adjusted p-values, explicit families |
| Reliability | nominal binary Krippendorff alpha per category, threshold 0.80 |
| Efficiency | per-sentence latency and per-run peak allocation vectors |

## Constitution Check

Scientific behavior is explicit, tests precede implementation, random resampling is seeded, full vectors and family membership are retained for audit, and results carry provenance from spec 013.

## Project Structure

```text
src/evaluation/metrics.py
src/evaluation/statistical_tests.py
src/evaluation/annotation.py
src/evaluation/efficiency.py
scripts/run_experiment.py
tests/test_metrics_exact.py
tests/test_statistics_exact.py
tests/test_annotation_reliability.py
tests/test_efficiency_protocol.py
tests/test_methodology_integration.py
```

## Implementation Phases

1. Lock equations and output contracts with hand-computed fixtures.
2. Correct metric aggregation and shared token definitions.
3. Correct bootstrap, Wilcoxon/rank-biserial, Holm, and family assembly.
4. Correct nominal per-category alpha and threshold gating.
5. Correct warm-up/timing/memory protocol and summaries.
6. Run focused and full regression suites and reconcile every FR with a behavioral test.

## Post-Design Constitution Check

No exception is required. Existing tests that merely check keys do not satisfy this plan and must be strengthened rather than counted as coverage.
