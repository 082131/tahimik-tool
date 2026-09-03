# Experiment Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Give every result and checkpoint one canonical, deterministic provenance record and an explicit Chapter 3 eligibility decision.

**Architecture:** Expand `src/utils/reproducibility.py` into reusable collectors, canonical serializers, fingerprinting, and eligibility evaluation. All four executable paths and trainer checkpoints call the same builder rather than assembling ad hoc metadata.

**Tech Stack:** Python platform/importlib/subprocess, PyTorch CUDA APIs, JSON/SHA-256, pytest.

**Spec:** `specs/013-experiment-provenance/spec.md`

## Global Constraints

- Never include raw dataset text, secrets, or environment dumps.
- Record resolved configuration, not only config class names.
- Dirty and CPU runs remain debuggable but are ineligible where required.
- Canonical fingerprints must be stable across dictionary ordering.

## Technical Context

Collectors cover Git, Python/packages, OS, PyTorch, CUDA build/runtime, cuDNN, GPU identity/capability, precision, deterministic settings, operational arguments, data fingerprints, resource/manifest IDs, and checkpoint lineage.

## Constitution Check

This centralizes required reproducibility metadata, records deterministic settings, supports exact artifact auditing, and has test-first schema enforcement.

## Project Structure

```text
src/utils/reproducibility.py
src/training/trainer.py
scripts/train.py
scripts/run_experiment.py
scripts/evaluate.py
scripts/benchmark.py
tests/test_provenance.py
```

## Implementation Phases

1. Define the canonical schema and eligibility reason vocabulary.
2. Test serialization/fingerprints and mocked CPU/CUDA collection.
3. Implement the shared builder and deterministic configuration.
4. Integrate all commands and checkpoints.
5. Verify schema equality and ineligibility reasons across artifact types.

## Post-Design Constitution Check

No exception is required.
