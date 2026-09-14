# Padding and Runtime Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct independent padding alignment and latency timing.

**Architecture:** The collator independently calculates dimensions for encoder and labels. The trainer selects alignment based on device/precision; the API preserves a single batched model call while synchronizing CUDA timing.

**Tech Stack:** Python, PyTorch, FastAPI, pytest.

**Spec:** `docs/superpowers/specs/2026-09-08-padding-and-runtime-measurement-design.md`

---

### Task 1: Independent padding and truthful timing

**Files:** `src/data/dataset.py`, `src/training/trainer.py`, `backend/app.py`, `tests/test_dataset.py`, `tests/test_backend.py`

- [ ] Write failing tests for unequal input/target lengths with multiple-of-eight padding and CUDA synchronization around batch generation.
- [ ] Implement independent label alignment, mixed-precision collator selection, and guarded CUDA synchronize calls.
- [ ] Run `python -m pytest tests/test_dataset.py tests/test_backend.py tests/test_efficiency.py -q` and commit `fix(runtime): align dynamic padding and GPU batch timing`.
