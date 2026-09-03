# Feature Specification: Evaluation and Statistical Testing

**Feature Branch**: `feat/007-evaluation-and-statistics`

**Created**: 2026-08-27

**Status**: Draft — **Class B provenance**, see [../PROVENANCE.md](../PROVENANCE.md)

**Input**: Retrofit spec for `src/evaluation/metrics.py`,
`src/evaluation/efficiency.py`, `src/evaluation/statistical_tests.py`.

> **This spec contains the highest-severity finding in the repository.**
> See F1 under Convergence.

## Plain-language summary

This is how the study answers its own research questions. It scores each of the
three models on how *accurately* they clean text and how *fast and cheap* they
are, then runs statistics to say whether the differences are real or could be
chance.

The statistics matter as much as the scores. "TAHIMIK got a higher number" is
not a finding. "TAHIMIK got a higher number, and here is the probability that
happened by luck" is.

## What the manuscript specifies

**Four accuracy metrics** (Research Question 1): GLEU+, character-level F-score
(chrF), Error Reduction Rate (ERR), alpha-word accuracy.

**Two efficiency metrics** (Research Question 2): average inference time per
sentence, peak GPU memory during inference.

**Two pairwise comparisons**: ByT5 vs noise-adaptive, and MrT5 vs
noise-adaptive.

**Significance testing** — and the manuscript is precise here:

> "Paired bootstrap resampling will be used… Each bootstrap sample will be
> created by resampling the test sentences with replacement… repeated for
> 1,000 bootstrap samples."

The two-tailed p-value is defined explicitly:

```
p = 2 × min( (1 + Σ I(Δb ≤ 0))/(B+1),  (1 + Σ I(Δb ≥ 0))/(B+1) )
```

95% CI from the 2.5th and 97.5th percentiles. Significance requires **both**
`p < α` **and** a confidence interval excluding zero, at `α = 0.05`.

**Peak GPU memory is different**:

> "Peak GPU memory cannot be resampled at the sentence level because it is
> measured once per inference run. Therefore, the 20 paired peak-memory
> measurements for each model comparison will be analyzed using the Wilcoxon
> signed-rank test instead of paired bootstrap resampling."

Reported with p-value, median difference in GB, and matched-pairs rank-biserial
effect size.

**Multiple-comparison correction**: Holm-Bonferroni across the family of tests.
Both unadjusted and adjusted p-values reported.

**Benchmark protocol**: 5 CUDA warmup passes discarded, then 20 timed runs,
peak-counter reset after warmup.

## User Scenarios & Testing

### User Story 1 - Score accuracy on the test set (Priority: P1)

**Independent Test**: Compute all four metrics on hand-written prediction /
reference / noisy triples where the expected values can be reasoned about.

**Acceptance Scenarios**:

1. **Given** predictions identical to references, **When** metrics are
   computed, **Then** ERR is `1.0` and chrF is at its maximum.
2. **Given** a prediction that made things worse, **When** ERR is computed,
   **Then** the value is negative — the metric must be able to express harm.
3. **Given** a reference already identical to the noisy input (nothing to fix),
   **When** ERR is computed, **Then** the divide-by-zero case is handled by a
   defined convention rather than crashing.

### User Story 2 - Measure efficiency (Priority: P1)

**Acceptance Scenarios**:

1. **Given** a benchmark run, **When** it executes, **Then** 5 warmup passes
   run and are discarded before any timing is recorded.
2. **Given** warmup completion, **When** timing begins, **Then** the CUDA peak
   memory counter is reset first, so warmup allocation is not counted.
3. **Given** 20 timed runs, **When** results are reported, **Then** per-sentence
   average inference time and peak GPU memory are both returned.

### User Story 3 - Test significance correctly (Priority: P1)

Equal priority to Stories 1 and 2, because a correct score with an incorrect
significance test produces a wrong published conclusion.

**Acceptance Scenarios**:

1. **Given** per-sentence scores for two variants, **When** the paired bootstrap
   runs, **Then** it resamples 1,000 times with replacement and returns a
   **two-tailed** p-value computed by the manuscript's formula.
2. **Given** bootstrap differences, **When** the CI is computed, **Then** it
   spans the 2.5th to 97.5th percentiles.
3. **Given** a comparison, **When** significance is decided, **Then** it
   requires **both** `p < α` **and** a CI excluding zero.
4. **Given** 20 paired peak-memory measurements, **When** they are tested,
   **Then** Wilcoxon signed-rank is used, **not** bootstrap, and rank-biserial
   effect size is reported.
5. **Given** a family of tests, **When** correction is applied, **Then**
   Holm-Bonferroni is used and both raw and adjusted p-values are reported.

### Edge Cases

- **ERR when the noisy input is already clean**: no errors to reduce.
  `metrics.py:109-112` treats a perfect prediction as `1.0` and any change as
  `0.0`. Defined; worth confirming against the manuscript's intent.
- **Wilcoxon with n < 10**: the manuscript specifies the normal approximation
  for `n ≥ 10`. With 20 runs this holds, but the code should not silently
  mis-handle a shorter run.

## Requirements

- **FR-001**: All four accuracy metrics MUST be computed: GLEU+, chrF, ERR,
  alpha-word accuracy.
- **FR-002**: Both efficiency metrics MUST be measured: average per-sentence
  inference time and peak GPU memory.
- **FR-003**: The benchmark MUST discard 5 warmup passes and reset the peak
  memory counter before its 20 timed runs.
- **FR-004**: The paired bootstrap MUST use 1,000 resamples with replacement
  at the sentence level.
- **FR-005**: The bootstrap p-value MUST be **two-tailed**, computed as
  `2 × min((1 + Σ I(Δb ≤ 0))/(B+1), (1 + Σ I(Δb ≥ 0))/(B+1))`.
- **FR-006**: The 95% CI MUST be the 2.5th–97.5th percentile interval.
- **FR-007**: Significance MUST require both `p < α` **and** a CI excluding
  zero.
- **FR-008**: Peak GPU memory MUST use Wilcoxon signed-rank, never bootstrap,
  and MUST report median difference in GB and rank-biserial effect size.
- **FR-009**: Holm-Bonferroni MUST be applied, with both raw and adjusted
  p-values reported.
- **FR-010**: Comparisons MUST be ByT5-vs-TAHIMIK and MrT5-vs-TAHIMIK.

## Success Criteria

### Checkable now

- **SC-001**: All four metrics compute on synthetic triples without crashing.
- **SC-002**: ERR is `1.0` for a perfect prediction and negative when the
  prediction is worse than the input.
- **SC-003**: The bootstrap p-value matches the manuscript's formula on a
  constructed case with a known answer.
- **SC-004**: A CI containing zero is reported as not significant even when
  `p < α`.
- **SC-005**: Holm-Bonferroni reproduces hand-computed thresholds on a small
  set of p-values.

### BLOCKED on the gold-standard dataset

- **SC-006** *(blocked)*: Actual metric values for all three variants.
- **SC-007** *(blocked)*: Whether H₀₁ and H₀₂ are rejected.
- **SC-008** *(blocked)*: Peak GPU memory comparisons — needs a GPU **and**
  real data; CI is CPU-only.

## Assumptions

- Metric implementations delegate to established libraries (`sacrebleu` for
  chrF, `nltk` for GLEU, `editdistance` for ERR), so metric *correctness* is
  inherited; what this spec checks is that they are wired up as specified.
- Efficiency numbers cannot come from CI. Per Constitution Principle III they
  require a recorded benchmark run traceable to a commit.

## Convergence — assessed against the implementation

| ID | Gap | Severity | Evidence |
|----|-----|----------|----------|
| F1 | **contradicts** | **CRITICAL** | `src/evaluation/statistical_tests.py:94` computes `p_value = wins_a / self.n_bootstrap` — a **one-tailed** proportion with no add-one smoothing. The manuscript specifies a **two-tailed** p-value with `+1` correction: `2 × min((1+Σ I(Δb≤0))/(B+1), (1+Σ I(Δb≥0))/(B+1))`. The docstring confirms the intent is one-tailed: *"The null hypothesis is that A >= B."* The manuscript's hypotheses H₀₁/H₀₂ are stated as "no significant difference" — **two-tailed**. This directly changes which results are reported as significant. |
| F2 | **contradicts** | HIGH | FR-007: `statistical_tests.py:106` sets `significant = p_value < self.alpha` only. The manuscript requires **both** `p < α` **and** a CI excluding zero. The CI is computed (`:98-99`) but never used in the significance decision. |
| F3 | **partial** | **HIGH** | FR-008: `wilcoxon_test` is correctly reserved for GPU memory and correctly two-sided (`statistical_tests.py:154`), but it returns only `statistic, p_value, significant, mean_a, mean_b`. The manuscript requires **median difference in GB**, its **interquartile range**, and **matched-pairs rank-biserial effect size** — none are computed, and it reports means where the manuscript asks for medians. |
| F5 | *(observation)* | — | Corroborating F1: `wilcoxon_test` passes `alternative="two-sided"` while `paired_bootstrap` is one-tailed. The two significance tests in the same class disagree on tail count, which is evidence F1 is an oversight rather than a deliberate methodological choice. |
| F4 | partial | LOW | `requirements.txt` annotates `scipy` as "Wilcoxon signed-rank test" only, understating its role now that bootstrap and Holm-Bonferroni are also present. Cosmetic. |

**FR-001, FR-002, FR-003, FR-004, FR-006, FR-009, FR-010: satisfied.** All four
metrics are implemented (`metrics.py`), the 5-warmup / 20-run protocol matches
(`efficiency.py:55-56`), `n_bootstrap` defaults to 1000, the CI uses the correct
percentiles, Holm-Bonferroni is implemented (`:175`), and `run_full_comparison`
expects exactly the four manuscript metric keys.

### Convergence tasks

- [ ] T001 **CRITICAL** Replace the one-tailed p-value at `src/evaluation/statistical_tests.py:94` with the manuscript's two-tailed formula including add-one smoothing, per FR-005. Any significance result computed before this fix must be recomputed (contradicts)
- [ ] T002 **HIGH** Make the significance decision require both `p < α` and a CI excluding zero at `src/evaluation/statistical_tests.py:106`, per FR-007 (contradicts)
- [ ] T003 **HIGH** Add median difference in GB, interquartile range, and matched-pairs rank-biserial effect size to `wilcoxon_test`, per FR-008. Currently reports means where the manuscript specifies medians (partial)
- [ ] T004 [P] Write `tests/test_statistical_tests.py` covering SC-003 through SC-005 — especially a constructed case with a hand-computed two-tailed p-value, which would have caught F1 (missing)
- [ ] T005 [P] Write `tests/test_metrics.py` covering SC-001 and SC-002 (missing)

## 009 compliance integration

The executable implementation now follows `specs/009-methodology-compliance`: source-aware GLEU+ is calculated per sentence using noisy input, bootstrap inference is two-tailed with add-one smoothing and CI gating, and the prescribed comparisons are ByT5-vs-TAHIMIK and MrT5-vs-TAHIMIK. Latency is paired per sentence; GPU memory is collected independently per timed run and is omitted on CPU. Holm-Bonferroni is step-down with adjusted p-values. Gold conflicts are rejected, exact duplicates collapse, and synthetic slang/emoji/code-switching/Taglish morphology are protected. The annotator website and human annotation remain external deliverables.
