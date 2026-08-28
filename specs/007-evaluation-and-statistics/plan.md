# Implementation Plan: Evaluation and Statistical Testing

**Branch**: `feat/007-evaluation-and-statistics` | **Date**: 2026-08-27 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/007-evaluation-and-statistics/spec.md`

> **This feature contains the highest-severity finding in the repository.** The
> bootstrap significance test does not implement the method the manuscript
> specifies, and the difference changes which results count as significant.

## Summary

Score all three variants on four accuracy metrics and two efficiency metrics,
then test whether the differences against each baseline are statistically
significant using the exact methods the manuscript specifies: paired bootstrap
for sentence-level quantities, Wilcoxon signed-rank for run-level peak memory,
Holm-Bonferroni across the comparison family.

The metrics, the benchmark protocol, and Holm-Bonferroni are all correctly
implemented. The significance testing is not: the bootstrap is one-tailed where
the manuscript is two-tailed, significance ignores the confidence interval the
manuscript requires, and the Wilcoxon reporting omits three required statistics.

## Technical Context

**Language/Version**: Python, CI on 3.10 and 3.11.

**Primary Dependencies**: `sacrebleu` (chrF), `nltk` (GLEU+), `editdistance`
(ERR), `scipy` (Wilcoxon), `numpy` (bootstrap, percentiles), `torch` (CUDA
memory instrumentation). All already installed. No new dependency.

**Storage**: Results written as JSON via `--output_file` on `scripts/evaluate.py`
and `scripts/benchmark.py`. Per Constitution Principle III these files MUST
carry a git SHA and resolved config; they currently do not (shared gap with
`006`).

**Testing**: `pytest`. **No test exists for either metrics or statistics** —
this is the largest correctness risk in the repository, because these are the
components that decide what the thesis reports.

**Target Platform**: Accuracy metrics run on CPU. **Peak GPU memory cannot be
measured in CI at all** — it requires CUDA, so those numbers must come from a
recorded local benchmark run traceable to a commit.

**Project Type**: Single project.

**Performance Goals**: N/A — this component measures performance, it does not
have performance requirements of its own. 1,000 bootstrap resamples over a
15,000-sentence test set is trivially fast.

**Constraints**:
- Bootstrap MUST be two-tailed with add-one smoothing.
- Significance MUST require both `p < α` **and** a CI excluding zero.
- Peak memory MUST use Wilcoxon, never bootstrap.
- Exactly two comparisons; adding a third raises the correction burden on the
  two that matter.

**Scale/Scope**: Three modules — metrics, efficiency, statistics — plus two CLI
entry points.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Check | Result |
|---|---|---|
| I. Branch Discipline | Carried on the shared spec-retrofit branch | **DEVIATION** — see Complexity Tracking |
| II. Spec-Driven Development | Written from the manuscript's formulas without treating `statistical_tests.py` as the source of intent. **This is exactly why F1 was found** — reading the code first would have produced a spec that ratified the one-tailed test | PASS |
| III. Reproducibility | Results files carry no git SHA or resolved config (shared with `006` T013). Efficiency numbers require a recorded benchmark run, which the constitution mandates and nothing currently enforces | **VIOLATION, tracked** |
| IV. Test-First & CI Gate | `src/evaluation/` is not under `src/models/`, so test-first is not strictly mandated. Applying it anyway — these components decide what the thesis reports | PASS |
| V. Transparent AI Assistance | `Co-Authored-By:`; provenance recorded | PASS |

**Principle II earned its keep here.** The rule that a spec must be written
without reading the implementation is the sole reason the one-tailed/two-tailed
divergence surfaced. A spec derived from the code would have described
`wins_a / n_bootstrap` as the intended behaviour and the error would have been
ratified rather than caught.

## Project Structure

### Documentation (this feature)

```text
specs/007-evaluation-and-statistics/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 — eight decisions, one verified gap
├── data-model.md        # Phase 1
├── quickstart.md        # Phase 1
└── tasks.md             # Phase 2
```

### Source Code (repository root)

```text
src/evaluation/
├── metrics.py             # GLEU+, chrF, ERR, alpha-word accuracy
├── efficiency.py          # 5 warmup + 20 timed runs, peak memory
├── statistical_tests.py   # Bootstrap, Wilcoxon, Holm-Bonferroni — F1/F2/F3 live here
└── annotation.py          # Inter-annotator agreement (gold standard, blocked)

scripts/
├── evaluate.py            # Accuracy CLI
└── benchmark.py           # Efficiency CLI

tests/
├── test_metrics.py            # NEW — does not exist
└── test_statistical_tests.py  # NEW — does not exist
```

**Structure Decision**: Single project, existing paths. Two new test files.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Shares the spec-retrofit branch instead of `feat/007-` | Five specs in one documentation pass before a deadline. | One branch per feature remains correct for implementation. The F1 fix in particular MUST land on its own branch — it changes reported significance and needs a reviewable, isolated diff. |
| **F1 recorded rather than fixed in this pass** | Convergence records; it does not edit code. Changing a significance test inside a documentation commit would alter what the thesis reports via a PR described as docs-only. | Fixing it immediately was tempting given severity. Rejected because a silent statistical-method change buried in a docs PR is worse than a recorded finding, even when the fix is correct. Scheduled as T001 on its own branch. |

## Phase 0: Outline & Research

Complete. See [research.md](./research.md) — eight decisions settled directly
from the manuscript's formulas, one verified reporting gap. No open questions:
the manuscript specifies this section more precisely than any other.

## Phase 1: Design & Contracts

See [data-model.md](./data-model.md) and [quickstart.md](./quickstart.md).

**A `contracts/` artifact is arguably justified here** and is deliberately not
produced. `scripts/evaluate.py` and `scripts/benchmark.py` emit JSON that a
human reads and transcribes into the manuscript, so the output *shape* matters
more than elsewhere in this codebase. But it is still a developer entry point
rather than a published interface with external consumers. The JSON structure is
documented in `data-model.md` instead. Revisit if anything ever consumes those
files programmatically.

## Constitution Check — post-design re-check

Unchanged. Principle III remains violated (results files lack SHA and config)
and remains scheduled rather than resolved. No new dependency or storage format
introduced.
