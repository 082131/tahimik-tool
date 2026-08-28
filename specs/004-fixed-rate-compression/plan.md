# Implementation Plan: Fixed-Rate Compression (MrT5 Baseline)

**Branch**: `feat/004-fixed-rate-compression` | **Date**: 2026-08-27 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/004-fixed-rate-compression/spec.md`

## Summary

Reproduce MrT5's learned delete gate at encoder layer 3, compressing every input
toward one fixed rate regardless of content. Soft deletion during training so
the gate is trainable; hard deletion at inference where the speedup is actually
earned. This is the efficiency baseline TAHIMIK's adaptive compression is
measured against, and it must stay a faithful replication.

The implementation exists and is substantially correct, including the subtle
log-space attention-bias requirement. The real work is one unresolved
manuscript/code disagreement and a small amount of cleanup.

## Technical Context

**Language/Version**: Python, CI on 3.10 and 3.11.

**Primary Dependencies**: `torch>=2.1.0`, `transformers>=4.36.0`. No new
dependency.

**Storage**: N/A.

**Testing**: `pytest`. `tests/test_delete_gate.py` and
`tests/test_model_forward.py` already exercise this variant extensively — the
plan reuses them rather than duplicating.

**Target Platform**: CPU in CI. The compression *saving* is only observable on
the inference path; measuring it is spec `007`'s job.

**Project Type**: Single project.

**Performance Goals**: None as a target — this variant's inference cost is a
measurement for RQ2, not something to optimize.

**Constraints**: MUST remain a faithful MrT5 replication (see the open item in
`research.md`). MUST gate at the same layer as TAHIMIK. MUST NOT condition on
noise in any way — that is the difference this baseline exists to isolate.

**Scale/Scope**: Two files — the variant wrapper and the fixed-rate mode of the
shared delete gate. No external interface, so no `contracts/`.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Check | Result |
|---|---|---|
| I. Branch Discipline | Carried on the shared spec-retrofit branch rather than `feat/004-` | **DEVIATION** — see Complexity Tracking |
| II. Spec-Driven Development | Spec written from the manuscript and the MrT5 paper, not from reading `fixed_compression_byt5.py`. Class B provenance, recorded in `specs/PROVENANCE.md` | PASS |
| III. Reproducibility | `delete_gate_layer`, `gate_k`, `fixed_deletion_target`, `w_rate`, `w_attn_reg` all live in `configs/mrt5_config.py`, none inline | PASS |
| IV. Test-First & CI Gate | `src/models/` change → test-first mandatory. Existing coverage is substantial; new tests only where genuinely uncovered | PASS |
| V. Transparent AI Assistance | `Co-Authored-By:` on commits; provenance recorded | PASS |

**One finding sits at the edge of Principle II** rather than failing it: the
`L_attn_reg` substitution means code and manuscript disagree. Principle II
forbids *inventing* an answer, and this plan does not — it records the
disagreement and routes it to the author. Leaving it unrecorded would have been
the violation.

## Project Structure

### Documentation (this feature)

```text
specs/004-fixed-rate-compression/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 — six decisions, one open item
├── data-model.md        # Phase 1
├── quickstart.md        # Phase 1
└── tasks.md             # Phase 2
# No contracts/ — internal component.
```

### Source Code (repository root)

```text
src/models/
├── fixed_compression_byt5.py   # The variant wrapper
└── delete_gate.py              # Shared with 005; this spec covers
                                #   noise_adaptive=False mode only

src/training/
└── losses.py                   # L_rate and L_attn_reg — the open item lives here

configs/
└── mrt5_config.py              # Gate layer, k, fixed target, loss weights

tests/
├── test_delete_gate.py         # Existing — gate and losses in isolation
└── test_model_forward.py       # Existing — full forward/backward passes
```

**Structure Decision**: Single project, existing paths. No new files expected
unless T003 finds a genuine coverage gap.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Shares the spec-retrofit branch instead of `feat/004-` | Five specs produced in one pass before a deadline; the findings interlock (the `byt5-small` divergence spans 003, 004, 005) and are more useful reviewed together than split across five PRs. | One branch per feature is the constitutional default and is correct for implementation. Rejected only for this documentation pass. Implementing these tasks MUST return to one branch per feature. |
| The delete gate has no spec directory of its own | Its behaviour genuinely differs between the two modes, and each mode belongs to the variant that uses it. A third spec would duplicate both. | A dedicated `delete-gate` spec was the obvious alternative. Rejected because the fixed-rate and noise-adaptive modes have different requirements, different acceptance criteria, and different owners — splitting by mode is the more honest decomposition. |

## Phase 0: Outline & Research

Complete. See [research.md](./research.md) — six decisions settled from the
manuscript and the MrT5 paper, one item deliberately left open for the author.

## Phase 1: Design & Contracts

See [data-model.md](./data-model.md) and [quickstart.md](./quickstart.md).

No `contracts/`: internal component, constructed and called directly by the
trainer and evaluator.

## Constitution Check — post-design re-check

Phase 1 adds no dependency, no storage, no hyperparameter. The two deviations
above are unchanged. The `L_attn_reg` item remains open and routed to the
author rather than resolved by inference.
