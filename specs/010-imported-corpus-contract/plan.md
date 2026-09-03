# Imported Corpus Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Validate the two external annotation CSV exports and stop an ineligible Chapter 3 run before model construction.

**Architecture:** Add pure CSV contract/validation objects under `src/data`, keep orchestration in `DataPipeline`, and make `run_experiment.py` the strict entry point. Validation returns identifiers, counts, fingerprints, and reason codes; it never logs rejected text.

**Tech Stack:** Python 3, standard `csv`/`hashlib`, dataclasses, pytest.

**Spec:** `specs/010-imported-corpus-contract/spec.md`

## Global Constraints

- Do not implement gathering, annotation UI, or semantic classification in this repository.
- Do not commit imported data.
- Validate before tokenizer/model construction.
- Preserve deterministic splitting and text-free rejection reports.

## Technical Context

| Area | Decision |
|---|---|
| Inputs | UTF-8 CSV; gold pairs and long-format reliability labels |
| Full-run counts | 15,000 gold; 12,000/1,500/1,500 split; 3,000 reliability sentences; 1,000,000 synthetic pairs |
| Identity | Stable external IDs; SHA-256 fingerprints over canonical records |
| Split | Seeded deterministic assignment after validation |
| Error model | Collect stable reason codes, then fail strict orchestration |

## Constitution Check

- Configuration controls seed and paths; scientific constants remain explicit contract rules.
- Tests precede implementation and include boundary/failure behavior.
- Reports contain provenance and no raw private text.
- Raw corpora remain outside Git.

## Project Structure

```text
src/data/import_contracts.py
src/data/preprocessing.py
scripts/run_experiment.py
tests/fixtures/import_contracts/
tests/test_import_contracts.py
specs/010-imported-corpus-contract/contracts/
```

## Implementation Phases

1. Define schemas, canonicalization, reason codes, and report entities.
2. Build failing fixture-based unit tests, then CSV validators.
3. Add deterministic split and reliability coverage checks.
4. Integrate strict preflight into the full experiment before model construction.
5. Verify privacy, determinism, exact boundaries, and CLI failure behavior.

## Post-Design Constitution Check

No exception is required: the design is test-first, deterministic, configuration-aware, and excludes raw data from version control.
