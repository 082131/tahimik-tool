# Implementation Plan: ByT5 Baseline (No Compression)

**Branch**: `feat/003-byt5-baseline` | **Date**: 2026-08-27 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/003-byt5-baseline/spec.md`

## Summary

Wrap a stock `T5ForConditionalGeneration` with no compression of any kind, so it
serves as both the accuracy ceiling and the efficiency floor for the
three-variant comparison. Loads from the same `config.model_name` the other two
variants read, reports a deletion rate of exactly zero, and exposes a
forward-pass interface the shared trainer and evaluator can consume without
variant-specific branching.

The component already exists and is correct. This plan's real work is the
missing test file and the two recorded divergences.

## Technical Context

**Language/Version**: Python, CI on 3.10 and 3.11.

**Primary Dependencies**: `torch>=2.1.0`, `transformers>=4.36.0`. No dependency
is added by this feature.

**Storage**: N/A.

**Testing**: `pytest`, using the tiny randomly-initialized `T5Config` pattern
from `tests/test_model_forward.py` rather than a real ~1.2GB checkpoint.

**Target Platform**: CPU in CI; GPU for the efficiency measurements that belong
to spec `007`.

**Project Type**: Single project — existing `src/`, `tests/`, `configs/` layout.

**Performance Goals**: None as a target. This variant's inference time and
memory *are* a measurement in RQ2, not a goal to optimize — making it faster
would destroy its value as the efficiency floor.

**Constraints**: MUST NOT contain a delete gate, noise estimator, or any
compression. MUST read `config.model_name`, not a hardcoded model id.

**Scale/Scope**: One internal model class. No external interface, so no
`contracts/` artifact.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Check | Result |
|---|---|---|
| I. Branch Discipline | Spec directory is `003-byt5-baseline`; work is being carried on the in-flight spec branch rather than a separate `feat/003-` branch | **DEVIATION** — see Complexity Tracking |
| II. Spec-Driven Development | Spec written from the manuscript without treating `byt5_baseline.py` as the source of intent. Provenance is Class B (manuscript-derived, not author-interrogated) and is recorded in `specs/PROVENANCE.md` | PASS |
| III. Reproducibility | This feature adds no hyperparameter. It inherits the repo-wide `--num_beams` and determinism gaps, which are tracked in `specs/FINDINGS.md` rather than duplicated here | PASS |
| IV. Test-First & CI Gate | `src/models/byt5_baseline.py` is under `src/models/` → test-first is mandatory. `tests/test_byt5_baseline.py` does not exist | PASS, with obligation: T003 writes the test first |
| V. Transparent AI Assistance | Commits carry `Co-Authored-By:`; provenance recorded per-spec | PASS |

## Project Structure

### Documentation (this feature)

```text
specs/003-byt5-baseline/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 — grilling answers and their sources
├── data-model.md        # Phase 1
├── quickstart.md        # Phase 1
└── tasks.md             # Phase 2
# No contracts/ — internal component, no external interface.
```

### Source Code (repository root)

```text
src/models/
└── byt5_baseline.py        # The variant. Already implemented and correct.

configs/
├── base.py                 # model_name — carries the byt5-small/base divergence
└── byt5_config.py          # use_compression = False

tests/
└── test_byt5_baseline.py   # NEW — does not exist yet
```

**Structure Decision**: Single project, existing paths. One new test file. No
new directories.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Specs 003–007 share the `feat/002-noise-estimator` branch instead of one branch each | All five are documentation for a single retrofit pass, produced in one sitting under a presentation deadline. Five branches would mean five PRs reviewing the same class of change, splitting the findings across reviews when they are most useful read together. | One-branch-per-feature is the constitutional default and is correct for implementation work. It was rejected here only for the spec-retrofit pass, and only because the findings interlock — the `byt5-small` divergence alone spans 003, 004, and 005. Implementation of these specs MUST return to one branch per feature. |

## Phase 0: Outline & Research

Complete. See [research.md](./research.md) — four decisions settled from the
manuscript, one open judgment call recorded, one confirmation.

## Phase 1: Design & Contracts

See [data-model.md](./data-model.md) and [quickstart.md](./quickstart.md).

No `contracts/`: this variant is constructed and called directly by the trainer
and evaluator inside the same codebase. There is no API, CLI surface, or
service boundary to document.

## Constitution Check — post-design re-check

Phase 1 introduces no new dependency, no new storage, and no new
hyperparameter. The branch deviation above is unchanged and remains the only
one.
