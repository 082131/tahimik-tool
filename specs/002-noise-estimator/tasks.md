# Tasks: Noise Estimator

**Input**: Design documents from `/specs/002-noise-estimator/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md (no contracts/ — no external interface)

**Tests**: Not optional here. Constitution Principle IV mandates test-first for
any change under `src/models/`, and `src/models/noise_estimator.py` is
exactly that. Every implementation task below has its test written and
confirmed failing first.

**Note on retrofitting**: This feature retrofits an existing file. The tasks
below are written as if building from scratch, per the normal spec-kit flow —
`/speckit-converge` is the next command after this one, and its job is to
check each task against the actual code and mark it already done, missing,
or done differently. Do not skip converge and assume these tasks are new
work; some or all of the implementation tasks may already be satisfied.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependency on an
  incomplete task). Tasks that edit the same file are never marked `[P]`,
  even if the template elsewhere uses it more loosely — accuracy here
  matters more than following that convention decoratively.
- **[Story]**: Maps to `spec.md`'s User Story 1/2/3.

## Phase 1: Setup

**Purpose**: Project initialization.

None required. This retrofit introduces no new dependency (`editdistance` is
already installed, see `research.md`), no new directory, and no new
configuration surface beyond what `configs/tahimik_config.py` already
defines (`noise_estimator_hidden_dim`, `noise_avg_momentum`). Inventing a
setup task here would be busywork, not real work.

## Phase 2: Foundational

**Purpose**: Blocking prerequisites shared by all user stories.

None required, for the same reason as Phase 1 — everything this feature
needs already exists in the repository (`src/models/noise_estimator.py`,
`src/data/noise_label.py`, `src/training/losses.py`, and the test-construction
pattern in `tests/test_delete_gate.py`).

**Checkpoint**: Nothing blocks starting User Story 1 directly.

---

## Phase 3: User Story 1 - Training loop obtains a noise score during Stage 1 (Priority: P1) 🎯 MVP

**Goal**: The noise estimator produces a valid score for a batch of sentences,
trained correctly and in isolation from the very first step of Stage 1.

**Independent Test**: Run `tests/test_noise_estimator.py` against a tiny,
randomly-initialized T5 (no real checkpoint, no real dataset needed).

### Tests for User Story 1 ⚠️

> Write these first. Run them. Confirm they FAIL before touching
> implementation — this is Constitution Principle IV, not a suggestion.

- [ ] T001 [US1] Write shape-and-range test in `tests/test_noise_estimator.py`: feed a tiny T5's layer-3 hidden states for a batch of sentences through the noise estimator, assert output shape `(batch_size,)` and every value in `[0, 1]` (spec SC-001)
- [ ] T002 [US1] Write label-correctness test in `tests/test_noise_estimator.py`: for at least 3 hand-written (noisy, clean) pairs with a hand-computed expected `n*`, assert the computed value matches exactly (spec SC-002)
- [ ] T003 [US1] Write gradient-isolation test in `tests/test_noise_estimator.py`: run one training step computing `L_NE`, call `.backward()`, assert the noise estimator's parameters have non-`None` gradients and the shared encoder's parameters do not (spec SC-003, FR-006)
- [ ] T004 [US1] Run `python -m pytest tests/test_noise_estimator.py -v` and confirm all three tests fail against the current state of the code, before making any implementation change

### Implementation for User Story 1

- [ ] T005 [P] [US1] In `src/models/noise_estimator.py`, ensure the module accepts encoder hidden states from layer 3 — the same layer the delete-gate reads — with no additional encoder forward pass (FR-001)
- [ ] T006 [P] [US1] In `src/models/noise_estimator.py`, ensure mean-pooling is computed over non-padding positions only, using the batch's padding/attention mask (FR-002)
- [ ] T007 [US1] In `src/models/noise_estimator.py`, ensure the forward pass is exactly `Linear(d_model→256) → GELU → Dropout(0.1) → Linear(256→1) → Sigmoid` (FR-003) (depends on T005, T006 — same file)
- [ ] T008 [P] [US1] In `src/data/noise_label.py`, ensure `n*` is computed as `editdistance(noisy, clean) / len(longer)` once per pair at dataset-build time, not recomputed inside the training loop (FR-004)
- [ ] T009 [US1] In `src/training/trainer.py` (or wherever the Stage 1/Stage 2 schedule is controlled), ensure `L_NE` is applied starting from the first training step of Stage 1, not deferred to Stage 2 (FR-005)
- [ ] T010 [US1] At every point the noise score `n` is consumed outside the estimator itself — the delete-gate's conditioning term and the deletion-target calculation in `src/training/losses.py` / `src/models/delete_gate.py` — confirm the value used is `n.detach()`, so `L_NE`'s gradient path terminates at the estimator's own parameters (FR-006, FR-007). This task touches only the detach boundary; it does not modify the delete-gate's own logic, which is out of scope here (spec's Round 3 boundary decision)
- [ ] T011 [US1] Run `python -m pytest tests/test_noise_estimator.py -v` again and confirm T001–T003 now pass

**Checkpoint**: User Story 1 is complete and independently testable — the
noise estimator produces correct, gradient-isolated scores from fake data,
with no dependency on the missing gold-standard dataset.

---

## Phase 4: User Story 2 - Training loop obtains a noise score during Stage 2 (Priority: P2)

**Goal**: The identical mechanism from User Story 1 works unchanged once
pointed at Stage 2 (gold-standard) data.

**Independent Test**: Substitute hand-written stand-in "gold-style" pairs for
the missing real dataset, and confirm no code path change is required.

**Status**: Mostly BLOCKED. Per `spec.md` and `research.md`, this story has
no open design question — only a missing data dependency. Only one task is
buildable today; the rest is intentionally not converted into tasks, per the
constitution's rule against claiming readiness for something no test can yet
verify.

### Tests for User Story 2

- [ ] T012 [US2] Write an integration test in `tests/test_noise_estimator.py` using hand-written stand-in pairs shaped like gold-standard data, proving the same estimator code path handles them with zero changes (spec Acceptance Scenario 1)

### Implementation for User Story 2

None. The mechanism is already built by User Story 1; there is nothing
Stage-2-specific to implement. **BLOCKED, not skipped**: SC-004
(correlation with real messiness) and SC-005 (clean-vs-noisy separation on
real text) remain marked blocked in `spec.md` until the gold-standard
dataset exists. Do not write a task that would need real data to verify.

**Checkpoint**: The interface is proven stage-agnostic. Nothing further to
do here until the dataset exists.

---

## Phase 5: User Story 3 - Developer inspects precomputed noise labels before training (Priority: P3)

**Goal**: A developer can open the precomputed `n*` labels and sanity-check
them by eye before committing to a full training run.

**Independent Test**: Build labels for a few hand-picked pairs with an
intuitively obvious answer, and confirm they're readable without writing a
new tool.

### Tests for User Story 3

- [ ] T013 [US3] Write a test in `tests/test_noise_estimator.py` asserting that for a (noisy, clean) pair where noisy equals clean exactly, `n*` is exactly `0.0` (spec Acceptance Scenario 2)

### Implementation for User Story 3

- [ ] T014 [P] [US3] Confirm `src/data/noise_label.py` writes `n*` values in a plain, directly-readable format alongside each pair (e.g., a labeled field a developer can open and read), not opaquely embedded inside a serialized tensor — per quickstart.md's manual spot-check step. This is a format check, not a new inspection tool; do not build a separate script for this

**Checkpoint**: All three user stories are independently complete.

---

## Final Phase: Polish

- [ ] T015 [P] Run `quickstart.md` end to end and confirm every step in it works as written, correcting the doc if reality diverges
- [ ] T016 Note in the PR description (per `CONTRIBUTING.md`'s PR-body requirement) which of T005–T014 `/speckit-converge` found already implemented versus newly written

**Explicitly not a task**: the all-padding-batch edge case
(`NEEDS CLARIFICATION` in `spec.md`) is deliberately excluded from this list,
per `research.md`. Do not add a task for it without first resolving the
open question.

---

## Dependencies & Execution Order

- **Setup / Foundational**: Empty — User Story 1 can start immediately.
- **User Story 1 (P1)**: No dependency on other stories. This is the MVP.
- **User Story 2 (P2)**: Depends on User Story 1 being complete (reuses its
  code path); mostly blocked on external data regardless.
- **User Story 3 (P3)**: Depends on T008 (User Story 1's `noise_label.py`
  work) existing first, since it inspects that same output.

### Within User Story 1

Tests (T001–T004) before implementation (T005–T011). T005 and T006 touch the
same file and are sequenced, not parallel, despite both being taggable as
"model work" — same-file edits are never marked `[P]` in this task list.
T008 is genuinely parallel to T005–T007 (different file, `noise_label.py`
vs. `noise_estimator.py`).

## Parallel Example: User Story 1

```bash
# T005 and T008 touch different files and have no dependency on each other:
Task: "Ensure layer-3 hidden-state input in src/models/noise_estimator.py"
Task: "Ensure n* formula in src/data/noise_label.py"
```

## Implementation Strategy

### MVP First

Complete Phase 3 (User Story 1) only. That alone proves the noise estimator
works correctly in isolation, using only synthetic data — no dependency on
the missing gold-standard dataset. Stop there and validate before moving to
User Story 2 or 3.

### Incremental Delivery

1. User Story 1 → run `quickstart.md`'s Story 1 section → this is the MVP.
2. User Story 2 → mostly a confirmation task, not new work; the real
   milestone here is external (the dataset arriving), not code.
3. User Story 3 → a small format check, safe to do any time after Story 1.

## Notes

- Every implementation task above should be treated as "verify or build" —
  `/speckit-converge`, run next, determines which of the two it actually is
  for this codebase.
- Commit after each checkpoint, not after each task, to keep the history
  readable per `CONTRIBUTING.md`'s "one logical change per commit" rule.

---

## Phase 6: Convergence

**Generated by `/speckit-converge`** after reading the actual implementation
(`src/models/noise_estimator.py`, `src/models/delete_gate.py`,
`src/models/noise_adaptive_byt5.py`, `src/training/losses.py`,
`src/training/trainer.py`, `src/data/dataset.py`, and both existing test
files). This phase does not modify T001–T016 above; it only adds what those
tasks didn't already anticipate.

**Headline result**: FR-001 through FR-007 are all already correctly
implemented. The gradient-isolation guarantee (FR-006, FR-007) that this
spec cared about most is real in the code, not just claimed in a comment.

- [ ] T017 [P] Reconcile T003 with the existing gradient-isolation test at `tests/test_model_forward.py::test_noise_estimator_is_trained_only_by_l_ne` — decide whether to reference/extend it from `tests/test_noise_estimator.py` or accept a deliberate duplicate, rather than writing an unaware second copy per SC-003 (partial)
- [ ] T018 [P] Review the existing `clamp(min=1.0)` safeguard in `src/models/noise_estimator.py` (line 81) against the all-padding-batch `NEEDS CLARIFICATION` in `spec.md`'s Edge Cases, and formally close that open question in `spec.md` if the existing behavior is intentional (partial)

**Checkpoint**: Once T001, T002, T014, T017, and T018 are done, run
`/speckit-analyze` to check `spec.md`, `plan.md`, and `tasks.md` still agree
with each other before `/speckit-implement` finishes the remainder.
