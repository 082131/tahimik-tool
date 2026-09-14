# Paired Length Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans`
> to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Ensure all model inputs, targets, `n*` labels, and data partitions
refer to the same complete aligned-word pair.

**Architecture:** Add one pure paired-preparation boundary in
`src/data/preprocessing.py`.  Every entry point calls it after loading or
synthetic generation but before label computation and splitting.  The dataset
becomes a validating consumer of prepared pairs rather than a second hidden
truncation policy.

**Tech Stack:** Python 3.11, Hugging Face tokenizer interface, standard-library
`difflib`, PyTorch, pytest.

**Spec:** `specs/023-paired-length-policy/spec.md`

## Global Constraints

- Preserve the configured `max_input_length`, `max_target_length`, and seed 42.
- Measure sequence length with the active tokenizer and `truncation=False`.
- Retain only complete aligned word groups; never split UTF-8 characters/words.
- Compute `n*` only after pair preparation and before splitting.
- Do not commit datasets, checkpoints, generated outputs, or dependency trees.
- All study variants consume the same prepared partitions.

## Constitution Check

- Spec exists before source changes: pass.
- No hyperparameter change: pass.
- Training/data behavior has regression tests: required.
- Full Python suite must pass before review: required.

## Project Structure

```text
src/data/preprocessing.py       paired preparation and audit entities
src/data/dataset.py             prepared-pair length validation
scripts/train.py                gold/synthetic preparation before labels/splits
scripts/evaluate.py             test preparation before labels/split
scripts/run_experiment.py       shared study preparation before labels/splits
tests/test_paired_length_policy.py  policy, audit, label, and split tests
specs/023-paired-length-policy/     spec-driven artifacts
```

## Implementation Tasks

### Task 1: Lock paired-policy behavior with failing tests

**Files:**
- Create: `tests/test_paired_length_policy.py`
- Modify: none

**Produces:** Expected public contracts for `prepare_paired_examples` and
`NormalizationDataset` validation.

- [ ] Write a minimal fake byte tokenizer whose encoded length is text UTF-8
  byte count plus one EOS token, and whose call raises if truncation is used.
- [ ] Add a failing test where `aang gandaaaa` / `ang ganda` fits only through
  the first aligned group; assert retained values are `aang` / `ang`, audit
  records one transformed pair, and `n*` equals
  `compute_noise_level("aang", "ang")`.
- [ ] Add a failing split/merge test where `sanaol ngayon` / `sana all ngayon`
  fits through the `sanaol` -> `sana all` edit hunk; assert the whole hunk is
  kept, never a same-word-index prefix.
- [ ] Add failing tests for an overlong first hunk exclusion reason, exact-limit
  retention, audit counts, pre-split disjoint source indices, and a dataset
  item that raises rather than silently truncates.
- [ ] Run `python -m pytest tests/test_paired_length_policy.py -v`; confirm
  each failure is caused by the missing policy/validation behavior.

### Task 2: Implement deterministic paired preparation

**Files:**
- Modify: `src/data/preprocessing.py`
- Test: `tests/test_paired_length_policy.py`

**Consumes:** Noisy/clean lists, tokenizer, configured limits.

**Produces:** `PreparedPairs` containing retained strings, post-policy labels,
source indices, and `PairPreparationAudit`.

- [ ] Add immutable audit/prepared-result data classes and a tokenizer-length
  helper that calls the active tokenizer without padding or truncation.
- [ ] Add an aligned-word-group builder using deterministic word-level
  Levenshtein alignment over whitespace-delimited words; reconstruct source text only through whole-word
  endpoints.
- [ ] Implement candidate accumulation.  Keep a group only when both complete
  candidate prefixes fit their respective limits; stop at the first group that
  does not fit.
- [ ] Exclude a pair only when no first group fits; record its source index and
  `no_complete_aligned_word_group_fits_limit` reason.
- [ ] Compute labels from retained strings after preparation, log audit counts,
  and run the focused test file until it passes.

### Task 3: Remove hidden dataset truncation

**Files:**
- Modify: `src/data/dataset.py`
- Test: `tests/test_paired_length_policy.py`

**Consumes:** Prepared strings that must already fit configured limits.

**Produces:** Dataset items tokenized without silent truncation.

- [ ] Replace `truncation=True` with non-truncating tokenizer calls.
- [ ] Validate encoded input and target lengths against their configured limits
  and raise a value error identifying the offending item and side.
- [ ] Preserve dynamic batch padding and its masks/`-100` labels unchanged.
- [ ] Run focused policy tests and existing dataset-related tests.

### Task 4: Apply the policy at every entry point

**Files:**
- Modify: `scripts/train.py`
- Modify: `scripts/evaluate.py`
- Modify: `scripts/run_experiment.py`
- Test: `tests/test_paired_length_policy.py`

**Consumes:** Data loaded/generated by `DataPipeline` and the already-loaded
active tokenizer.

**Produces:** Prepared gold/synthetic pairs, labels, source indices, and
partitions before any dataset construction.

- [ ] Prepare gold pairs immediately after loading and before any label
  computation/splitting in all three scripts.
- [ ] Prepare synthetic pairs before their labels/splits in training and the
  experiment runner; remove any retained full-string synthetic label path.
- [ ] Split prepared lists only, preserving source-index metadata for the audit
  and ensuring no pair can occur in multiple partitions.
- [ ] Add or extend entry-point tests with monkeypatched tokenizer/pipeline to
  prove the policy runs before split construction.
- [ ] Run the focused policy test file and all affected existing tests.

### Task 5: Verify and record the policy

**Files:**
- Modify: `specs/023-paired-length-policy/tasks.md`
- Test: all Python tests

- [ ] Mark the completed tasks with exact verification commands/results.
- [ ] Run `python -m pytest tests/ -v` and inspect all failures before any
  commit decision.
- [ ] Run `git diff --check` and inspect staged paths; ensure no dataset,
  checkpoint, or generated audit output is staged.

## Post-Design Constitution Check

The feature changes preprocessing behavior but not hyperparameters.  Its
shared preparation function runs before all splits, preserving fairness under
the fixed seed and making transformed/excluded counts auditable.
