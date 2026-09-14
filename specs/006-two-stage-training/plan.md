# Implementation Plan: Two-Stage Training Pipeline

**Branch**: `feat/006-two-stage-training` | **Date**: 2026-08-27 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/006-two-stage-training/spec.md`

## Summary

Train every variant through the same two-stage schedule — synthetic pretraining
then gold fine-tuning — with identical data, splits, optimizer, and hardware, so
that any difference in results is attributable to the compression mechanism and
nothing else. One trainer serves all three variants; the loss activates only the
terms each variant needs.

The structure is implemented and correct. The outstanding work is two
constitution-compliance gaps (determinism, checkpoint traceability) and one
unverified claim about noise-category coverage.

## Technical Context

**Language/Version**: Python, CI on 3.10 and 3.11.

**Primary Dependencies**: `torch>=2.1.0`, `transformers>=4.36.0` (for
`get_scheduler`), `editdistance` (for `n*`). No new dependency.

**Storage**: Checkpoints written to `config.checkpoint_dir`, gitignored per
`docs/agents/contributing.md`. No database.

**Testing**: `pytest`. No trainer test currently exists — this is the largest
untested component in the repository.

**Target Platform**: CUDA for real runs (fp16 enabled only when CUDA is
present); CPU in CI, which means CI can exercise the schedule's control flow but
not a realistic training run.

**Project Type**: Single project.

**Performance Goals**: None as a target. Training throughput is not a study
variable — only *inference* efficiency is measured, in spec `007`.

**Constraints**:
- All three variants MUST share data, splits, hyperparameters, and hardware.
- Constitution Principle III requires deterministic mode and SHA-stamped
  outputs. Neither is currently met.
- Stage 1 MUST remain skippable, because the gold standard does not exist and
  every run today is Stage-1-only.

**Scale/Scope**: The trainer, the loss, and the data pipeline feeding them.
Synthetic generation targets ~1M pairs; gold targets ~15,000 sentences.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Check | Result |
|---|---|---|
| I. Branch Discipline | Carried on the shared spec-retrofit branch | **DEVIATION** — see Complexity Tracking |
| II. Spec-Driven Development | Written from the manuscript, not from reading `trainer.py`. Class B provenance. No open design questions — the manuscript settles the design | PASS |
| III. Reproducibility | **FAIL** — determinism incomplete, checkpoints carry no git SHA | **VIOLATION, tracked below** |
| IV. Test-First & CI Gate | `src/training/` → test-first mandatory. **No trainer test exists at all** | PASS, with a large obligation |
| V. Transparent AI Assistance | `Co-Authored-By:`; provenance recorded | PASS |

**Principle III does not pass, and this plan does not pretend otherwise.** The
constitution was ratified today with full determinism as a MUST; the code
predates that decision. Recording the violation and scheduling the fix is the
correct response — silently downgrading the principle to match the code would
invert the relationship between the two.

Both items are the constitution's own recorded follow-up tasks, so this feature
inherits them rather than introducing them.

## Project Structure

### Documentation (this feature)

```text
specs/006-two-stage-training/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 — seven decisions, two known gaps, one unverified
├── data-model.md        # Phase 1
├── quickstart.md        # Phase 1
└── tasks.md             # Phase 2
```

### Source Code (repository root)

```text
src/training/
├── trainer.py            # Two-stage schedule, optimizer, checkpointing
└── losses.py             # Variant-aware loss assembly

src/data/
├── noise_generator.py    # Synthetic corruption — nine categories, unverified
├── noise_label.py        # n* computation (spec 002)
├── dataset.py            # NormalizationDataset, collate_fn
└── preprocessing.py      # Splits and pair loading

scripts/
├── train.py              # Entry point — where seeding happens today
└── run_experiment.py     # Full-pipeline driver

configs/
└── base.py               # Shared schedule, splits, optimizer, seed

tests/
└── test_trainer.py       # NEW — does not exist
```

**Structure Decision**: Single project, existing paths. One new test file.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Shares the spec-retrofit branch instead of `feat/006-` | Five specs in one documentation pass before a deadline. | One branch per feature remains correct for implementation work and MUST be used when these tasks are built. |
| **Principle III violation carried, not fixed** | This is a documentation pass. Convergence records; it does not edit code. Fixing determinism and SHA-stamping inside a spec commit would mix a behaviour change into a docs change, against `CONTRIBUTING.md`'s one-logical-change rule. | Fixing it here was the obvious alternative. Rejected because it would make a docs PR silently change training behaviour — reviewable only by reading the diff rather than the description. The fix is scheduled as T001 and T002 on its own branch. |

## Phase 0: Outline & Research

Complete. See [research.md](./research.md) — seven decisions settled from the
manuscript, two known constitution gaps with mechanical fixes, one unverified
claim recorded as a task rather than assumed.

## Phase 1: Design & Contracts

See [data-model.md](./data-model.md) and [quickstart.md](./quickstart.md).

No `contracts/`: the trainer is invoked by `scripts/train.py` inside the same
codebase. `scripts/train.py`'s CLI is arguably a surface, but it is a developer
entry point rather than a published interface, and spec `007` covers the
evaluation CLI where output format actually matters.

## Constitution Check — post-design re-check

Unchanged. Principle III remains violated and remains scheduled rather than
resolved. Phase 1 introduces no new dependency, storage format, or
hyperparameter.
