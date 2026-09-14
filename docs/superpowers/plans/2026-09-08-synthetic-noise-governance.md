# Synthetic Noise Governance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Stage 1 synthetic data manifest-driven, reproducible, and auditable.

**Architecture:** Strict manifest validation is isolated in `noise_policy.py`; generator behavior consumes only validated category probabilities. The pipeline owns pair generation, rejection, diagnostics, and lineage export, while command scripts own manifest-file loading.

**Tech Stack:** Python, dataclasses, JSON, PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-09-08-synthetic-noise-governance-design.md`

## Global Constraints

- Never use validation/test labels, network requests, guessed bounds, or fallback production probabilities.
- Preserve source/target text separately from lineage metadata.
- Treat absent reviewed resources as an eligibility failure.

---

### Task 1: Strict manifest validation

**Files:** `src/data/noise_policy.py`, `tests/test_noise_policy.py`

- [ ] Write failing tests for invalid bounds, unknown categories/groups, missing resources, tampered ID, and missing readiness.
- [ ] Implement canonical content validation, known-category validation, bounds checks, explicit readiness, and training-label fingerprinting.
- [ ] Run `python -m pytest tests/test_noise_policy.py -q`.

### Task 2: Manifest-driven pair generation

**Files:** `src/data/noise_generator.py`, `src/data/preprocessing.py`, `tests/test_noise_policy.py`

- [ ] Write failing tests proving no-manifest construction fails for reporting generation, one-character inputs cannot produce copy pairs, and every output has lineage.
- [ ] Remove probability/resource fallbacks; generate with `generate_pair`; retry/reject unchanged pairs; return saved diagnostics and lineage.
- [ ] Run `python -m pytest tests/test_noise_policy.py tests/test_methodology_compliance.py -q`.

### Task 3: Enforce command interface and artifacts

**Files:** `scripts/train.py`, `scripts/run_experiment.py`, `scripts/build_noise_manifest.py`, tests

- [ ] Write failing parser/pipeline tests for missing `--noise-manifest` during Stage 1.
- [ ] Load manifest JSON, pass it to the pipeline, write lineage/diagnostic JSON, and accurately mark ineligible runs.
- [ ] Run focused tests and commit `fix(data): enforce auditable synthetic noise generation`.
