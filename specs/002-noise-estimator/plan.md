# Implementation Plan: Noise Estimator

**Branch**: `feat/002-noise-estimator` | **Date**: 2026-08-27 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/002-noise-estimator/spec.md`

## Summary

Build a small feed-forward network that reads layer-3 encoder hidden states
and outputs one noise score per sentence in `[0, 1]`, trained from the first
step of Stage 1 by `L_NE` alone, with gradients from `L_NE` reaching only its
own parameters. The precomputed label `n*` (byte edit-distance ratio) is
computed once at dataset-build time, not recomputed during training.

This plan targets the existing repository structure directly — no new
top-level directories, no new external dependencies. The three files it
touches already exist and already carry comments describing this exact
design, so the primary risk this plan manages is confirming the code matches
those comments, not inventing new structure.

## Technical Context

**Language/Version**: Python, exercised in CI on 3.10 and 3.11
(`.github/workflows/tests.yml`).

**Primary Dependencies**: `torch>=2.1.0`, `transformers>=4.36.0` (HF `T5`
family, per `configs/base.py`), `editdistance>=0.6.0` — already a project
dependency, its `requirements.txt` comment already names it for "ERR and
noise level (n*)".

**Storage**: N/A (no database). Precomputed `n*` labels extend whatever
schema `src/data/dataset.py` already uses for (noisy, clean) pairs; this
plan does not introduce a new storage format.

**Testing**: `pytest`, following the pattern already established in
`tests/test_delete_gate.py` and `tests/test_model_forward.py` — a tiny,
randomly-initialized `T5Config` model, not a real ~1.2GB checkpoint, since
that is enough to prove shapes, wiring, and gradient flow.

**Target Platform**: CPU in CI. GPU-specific behavior (see Constitution
Principle III, determinism) is out of scope for this component's own tests,
since none of its acceptance criteria depend on GPU-only nondeterminism —
gradient-presence checks on a tiny CPU model are exact regardless.

**Project Type**: Single project (existing `src/`, `tests/`, `configs/`
layout per `CONTRIBUTING.md`).

**Performance Goals**: N/A. The estimator is a small MLP; its own runtime
cost is not the subject of this feature. Efficiency benchmarking belongs to
the (not-yet-specced) Evaluation feature, not here.

**Constraints**: MUST NOT introduce a second encoder forward pass (spec
FR-001 — reads the same layer-3 hidden states the delete-gate already
consumes). MUST NOT change the existing detach guarantees already described
in `src/training/losses.py`'s header comment (spec FR-006, FR-007).

**Scale/Scope**: One internal component. No external interface — nothing
here is called by another service or exposed as an API, so no `contracts/`
artifact is produced (per the Phase 1 rule: skip contracts for purely
internal components).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Check | Result |
|---|---|---|
| I. Branch Discipline | Branch is `feat/002-noise-estimator`; `002` is the correct next number (`001` is already `docs/001-ratify-constitution`) | PASS |
| II. Spec-Driven Development | `spec.md` was written before this plan and before reading `noise_estimator.py`; produced via `/grill-me` with the author. Contains one `NEEDS CLARIFICATION` (all-padding batch edge case) | PASS — the open item is carried forward, unresolved, per the constitution's explicit allowance (a spec may merge with open questions; only the dependent task is blocked, not the whole plan) |
| III. Reproducibility | Noise-estimator hyperparameters (`noise_estimator_hidden_dim`, `noise_avg_momentum`) already live in `configs/tahimik_config.py`, not inline | PASS |
| IV. Test-First & CI Gate | `src/models/noise_estimator.py` is under `src/models/` → test-first is mandatory. `tests/test_noise_estimator.py` does not exist yet | PASS, with an obligation: `tasks.md` must create this test file first and confirm it fails before any implementation change |
| V. Transparent AI Assistance | Commits will carry `Co-Authored-By:` per existing practice | PASS — unrelated pre-existing gap: `docs/AI-USE.md` still doesn't exist repo-wide; not introduced or worsened by this feature |

No violations. **Complexity Tracking is not applicable** — nothing here
requires a justified deviation from the constitution.

## Project Structure

### Documentation (this feature)

```text
specs/002-noise-estimator/
├── spec.md              # Feature specification (already written)
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md         # Phase 1 output
└── quickstart.md        # Phase 1 output
# No contracts/ — purely internal component, no external interface.
# tasks.md is Phase 2, produced by /speckit-tasks, not this command.
```

### Source Code (repository root)

```text
src/
├── models/
│   └── noise_estimator.py     # Retrofit target — the estimator itself
├── data/
│   └── noise_label.py         # Precomputes n* (already exists; header
│                               # comment already states the formula this
│                               # spec requires — /speckit-converge will
│                               # confirm the code matches the comment)
└── training/
    └── losses.py               # Consumes the estimator's output for L_NE;
                                 # NOT modified by this feature — its header
                                 # already documents the detach guarantee
                                 # this spec's FR-006/FR-007 require

tests/
└── test_noise_estimator.py    # NEW — does not exist yet; created test-first
                                 # per Constitution Principle IV
```

**Structure Decision**: Single project, reusing the four real paths above.
No new top-level directories. This is a retrofit of an existing file inside
the existing `src/models/` package, plus one new test file following the
existing test-file convention.

## Phase 0: Outline & Research

Only one open item exists in the spec's Technical Context / Edge Cases: the
all-padding-batch behavior. Everything else was resolved during grilling with
real, sourced answers (README, manuscript, and the project's own
`requirements.txt` / existing modules), so there is nothing else to research.

See [research.md](./research.md).

## Phase 1: Design & Contracts

See [data-model.md](./data-model.md) for the three entities this feature
concerns, and [quickstart.md](./quickstart.md) for how a developer runs the
resulting tests and manually spot-checks precomputed labels (User Story 3).

No `contracts/` directory: this component has no external interface to
document — it's an internal function call inside the training loop, not an
API, CLI surface, or service boundary.

## Constitution Check — post-design re-check

Design artifacts introduce no new dependency, no new storage format, and no
change to the detach boundaries already established. The gate table above
still holds unchanged after Phase 1.

## Complexity Tracking

*Not applicable — no constitution violations to justify.*
