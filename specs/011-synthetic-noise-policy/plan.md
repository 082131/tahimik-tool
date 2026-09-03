# Synthetic Noise Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Derive auditable training-only noise probabilities and generate pairs that distinguish preserved language features from input-only errors.

**Architecture:** Introduce versioned manifest and lexicon-provider modules. Build a shared augmented target first, then corrupt only its input copy. `DataPipeline` passes the resolved manifest into `TagalogNoiseGenerator`; full orchestration rejects provisional/missing bounds or resources.

**Tech Stack:** Python 3, dataclasses/JSON/CSV, NumPy random generator, pytest.

**Spec:** `specs/011-synthetic-noise-policy/spec.md`

## Global Constraints

- Compute prevalence from the gold training split only.
- Never scrape KWF or depend on network access.
- Do not invent final category bounds.
- Full experiment fails closed until final bounds and reviewed resources exist.

## Technical Context

| Category group | Application |
|---|---|
| Preserved augmentations | slang, emoji, code-switching, Taglish morphology; applied identically to input and target |
| Correctable noise | spelling/orthographic, abbreviation, elongation, capitalization, punctuation, vowel omission, character errors; applied to input only |
| Probability | observed train prevalence clamped to approved per-category bounds |
| Resources | locally supplied, reviewed, versioned CSV through a provider interface |

## Constitution Check

- Output-changing probabilities and resource versions are fully resolved and recorded.
- Deterministic seeds govern all transformations.
- Tests precede behavior changes.
- No external corpus or generated million-pair dataset is committed.

## Project Structure

```text
src/data/noise_policy.py
src/data/lexicon_provider.py
src/data/noise_generator.py
src/data/preprocessing.py
configs/base.py
tests/test_noise_policy.py
tests/test_noise_generator_contract.py
```

## Implementation Phases

1. Define manifest and reviewed-lexicon contracts.
2. Test and implement training-only prevalence resolution.
3. Refactor generation into shared augmentation followed by input-only corruption.
4. Wire resolved configuration through `DataPipeline`.
5. Add strict readiness and reproducibility checks.

## Post-Design Constitution Check

The only blocked input is scientific policy supplied after annotation: approved numeric bounds. Implementation may proceed, but a full eligible run must remain impossible until they are present.
