# Checkpoint Lineage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist complete best-checkpoint state and Stage 1 parentage for Stage 2.

**Architecture:** The trainer owns checkpoint creation and handoff state, with a small checkpoint-ID helper. Tests inspect real saved payloads rather than logger output.

**Tech Stack:** Python, PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-09-08-checkpoint-lineage-design.md`

---

### Task 1: Persist complete training state

**Files:** `src/training/trainer.py`, `tests/test_checkpoint_handoff.py`

- [ ] Write failing payload-shape and parentage tests.
- [ ] Save optimizer/scheduler/scaler state, canonical checkpoint ID, parent ID, and resolved configuration.
- [ ] Restore model weights before Stage 2 optimizer construction and attach handoff data to Stage 2 payloads.
- [ ] Run `python -m pytest tests/test_checkpoint_handoff.py tests/test_checkpoint_compatibility.py -q` and commit `fix(training): persist complete checkpoint handoff lineage`.
