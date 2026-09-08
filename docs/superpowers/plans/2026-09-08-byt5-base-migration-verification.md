# ByT5-Base Migration Verification Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify and harden the shared ByT5-Base experiment configuration.

**Architecture:** Existing migration code remains the implementation authority; this task adds only missing validation/tests identified by the audit.

**Tech Stack:** Python, Transformers, pytest.

**Spec:** `docs/superpowers/specs/2026-09-08-byt5-base-migration-verification-design.md`

---

### Task 1: Verify Base-only configuration and checkpoint contract

**Files:** `configs/base.py`, `src/training/trainer.py`, `scripts/preflight_base.py`, `tests/test_model_configuration.py`, `tests/test_checkpoint_compatibility.py`, `tests/test_base_preflight.py`

- [ ] Add failing tests for incomplete architecture identity and all variant/tokenizer Base resolution.
- [ ] Fail closed on missing architecture fields required for a reporting checkpoint.
- [ ] Run `python -m pytest tests/test_model_configuration.py tests/test_checkpoint_compatibility.py tests/test_base_preflight.py -q` and commit `fix(base): harden ByT5-Base checkpoint identity`.
