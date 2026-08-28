# Tasks: Evaluation and Statistical Testing

**Input**: Design documents from `/specs/007-evaluation-and-statistics/`

**Prerequisites**: spec.md, plan.md, research.md, data-model.md, quickstart.md

**Tests**: Applied as mandatory even though `src/evaluation/` sits outside
Constitution Principle IV's strict `src/models/` scope. These components decide
what the thesis reports; untested statistics are a worse risk than untested
models.

> **This feature holds the repository's highest-severity finding.** T001 changes
> which results count as significant.

## Phase 1: Setup

None. `sacrebleu`, `nltk`, `editdistance`, `scipy`, and `numpy` are all already
installed.

## Phase 2: Foundational

None.

**Checkpoint**: Story 1 can start immediately.

---

## Phase 3: User Story 3 - Test significance correctly (Priority: P1) 🎯 MVP

**Deliberately ordered first**, ahead of Stories 1 and 2. A correct metric fed
into an incorrect significance test produces a wrong published conclusion, and
that is the current state.

**Goal**: The bootstrap and Wilcoxon tests implement the manuscript's specified
methods exactly.

**Independent Test**: `check_pvalue.py` from `quickstart.md`, plus the
hand-computed two-tailed formula check.

### Tests for User Story 3 ⚠️

- [ ] T001 [US3] Write a two-tailed p-value test in `tests/test_statistical_tests.py`: construct a bootstrap distribution with a known 70/30 split and assert the p-value matches the manuscript's formula by hand computation (SC-003). **This test would have caught F1**
- [ ] T002 [US3] Write a no-difference test: two identical distributions must yield a CI spanning zero and a p-value far from zero (SC-004)
- [ ] T003 [US3] Write a CI-gating test: a case where `p < α` but the CI contains zero must be reported as **not** significant (SC-004, FR-007)
- [ ] T004 [P] [US3] Write a Holm-Bonferroni test against hand-computed thresholds for a two-test family (SC-005)
- [ ] T005 [US3] Run the suite and confirm T001–T003 **fail** against the current implementation. They should — that is the point

### Implementation for User Story 3

- [ ] T006 **CRITICAL** [US3] Replace the one-tailed p-value at `src/evaluation/statistical_tests.py:94` with the manuscript's two-tailed formula including add-one smoothing: `p = 2·min((1+Σ I(Δb≤0))/(B+1), (1+Σ I(Δb≥0))/(B+1))`. Update the docstring, which currently states a one-sided null (FR-005, contradicts)
- [ ] T007 **HIGH** [US3] Make significance require both `p < α` **and** a CI excluding zero at `src/evaluation/statistical_tests.py:106`. The CI is already computed two lines above and simply not consulted (FR-007, contradicts)
- [ ] T008 **HIGH** [US3] Add median difference in GB, interquartile range, and matched-pairs rank-biserial effect size to `wilcoxon_test`. It currently reports means where the manuscript specifies medians, and no effect size at all (FR-008, partial)
- [ ] T009 [P] [US3] Verify the family composition and that both raw and adjusted p-values are reported (FR-009) — already present, confirm rather than rebuild
- [ ] T010 [US3] **Recompute any significance result produced before T006.** Numbers generated under the one-tailed test are not comparable to the manuscript's stated method

**Checkpoint**: Significance testing matches the manuscript. Until this
checkpoint passes, **no result should be reported as significant.**

---

## Phase 4: User Story 1 - Score accuracy on the test set (Priority: P1)

### Tests for User Story 1

- [ ] T011 [P] [US1] Write metric smoke tests in `tests/test_metrics.py`: all four metrics compute on hand-written triples without crashing (SC-001)
- [ ] T012 [US1] Write ERR edge-case tests: `1.0` for a perfect correction, **negative** when the prediction is worse than the input, and the defined convention when `errors_before = 0` (SC-002)

### Implementation for User Story 1

- [ ] T013 [P] [US1] Verify all four metrics are computed and returned under the keys `run_full_comparison` expects: `gleu_plus`, `chrf`, `err`, `alpha_word_accuracy` (FR-001)
- [ ] T014 [P] [US1] Confirm the ERR degenerate-case convention at `metrics.py:109-112` matches author intent — defensible as written, but unconfirmed (FR-001)

---

## Phase 5: User Story 2 - Measure efficiency (Priority: P1)

**Partially blocked**: peak GPU memory cannot be measured without CUDA.

### Tests for User Story 2

- [ ] T015 [P] [US2] Write a protocol test: 5 warmup passes are discarded and the peak-memory counter is reset **after** warmup, before timing (SC-001, FR-003)

### Implementation for User Story 2

- [ ] T016 [P] [US2] Verify `avg_inference_time` is per-sentence, not per-batch — the manuscript specifies "per sentence" (FR-002)
- [ ] T017 [US2] Handle the CPU sentinel explicitly: `peak_gpu_memory_mb` returns `0.0` without CUDA. Ensure a `0.0` can never be mistaken for a real measurement in a results file (FR-002)

---

## Final Phase: Polish

- [ ] T018 [P] Run every check in `quickstart.md`, including both p-value verifications
- [ ] T019 [P] Update the `scipy` comment in `requirements.txt` — it says "Wilcoxon signed-rank test" but scipy now also backs bootstrap and Holm-Bonferroni (partial)
- [ ] T020 Record in the PR which tasks converge found already satisfied

---

## Phase 6: Convergence

**Generated by `/speckit-converge`** after reading `src/evaluation/metrics.py`,
`efficiency.py`, and `statistical_tests.py`.

**Headline result**: metrics, the benchmark protocol, and Holm-Bonferroni are
all correctly implemented. Significance testing is not — three separate gaps,
all in `statistical_tests.py`.

Findings F1, F2, and F3 are already scheduled above as T006, T007, and T008
rather than duplicated here. Remaining:

- [ ] T021 [P] Create `tests/test_statistical_tests.py` and `tests/test_metrics.py`. **Neither exists.** These are the least-tested and most result-determining components in the repository (missing)
- [ ] T022 **HIGH** Stamp the git SHA, dirty flag, and resolved config into results files written by `scripts/evaluate.py` and `scripts/benchmark.py`, per Constitution Principle III. Shared with `006` T013 — fix once, in one place (contradicts)

**Checkpoint**: After T006, T007, T008, T021, run `/speckit-analyze`.

---

## Dependencies & Execution Order

- **Story 3 first**, despite being listed third in `spec.md`. Correct metrics
  fed into a wrong test still produce a wrong conclusion.
- **Stories 1 and 2** are independent of each other and of Story 3.
- **T010 depends on T006** — recomputation is meaningless before the fix.
- **T022** is shared with `006`.

### Within Story 3

T001–T005 (tests, confirmed failing) precede T006–T009 (fixes). T005 exists
specifically to confirm the tests fail first — if T001 passes against the
current code, the test is wrong, not the implementation.

## Parallel Example

```bash
# Different files, no dependency:
Task: "Write ERR edge-case tests in tests/test_metrics.py"
Task: "Write two-tailed p-value test in tests/test_statistical_tests.py"
```

## Implementation Strategy

**MVP**: Phase 3 alone. Nothing else in this feature matters until significance
testing matches the manuscript.

**Order within Phase 3**: T001 first. Write the failing test *before* the fix,
so the fix is demonstrably what changed the outcome. This is the case where
test-first pays for itself most obviously — it converts "I changed a formula and
believe it is better" into "here is a test that failed and now passes."

**Do not skip T010.** Fixing the test without recomputing prior results leaves
numbers in circulation that were produced by the old method.

## Notes

- T006 changes what the study reports. It belongs on its own branch with an
  isolated, reviewable diff — never bundled into a documentation change.
- The `wilcoxon_test` is already correctly two-sided. Only the bootstrap is
  wrong, which is what makes it clearly an oversight rather than a
  methodological choice.
