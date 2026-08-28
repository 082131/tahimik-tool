# Implementation Plan: Noise-Adaptive ByT5 (TAHIMIK)

**Branch**: `feat/005-noise-adaptive-byt5` | **Date**: 2026-08-27 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/005-noise-adaptive-byt5/spec.md`

> **This is the study's contribution.** Everything in `002`, `003`, `004`,
> `006`, and `007` exists to make this variant's claim measurable.

## Summary

Condition byte-level compression on a per-sentence noise estimate, so clean
sentences compress hard for speed and noisy sentences compress lightly to
preserve the detail needed to reconstruct them. Two mechanisms carry this: a
learned shift on the gate scores, `cn·(n − navg)`, and a per-sentence deletion
target, `d_max·(1 − n)`. The noise score is detached at both points so the
estimator trains only from `L_NE`, keeping the encoder a genuine control
variable across all three variants.

The implementation exists and satisfies every functional requirement checked,
including both detach points. The remaining work is two unspecified constants,
one unguarded failure mode found during grilling, and test coverage for two
criteria nothing currently exercises.

## Technical Context

**Language/Version**: Python, CI on 3.10 and 3.11.

**Primary Dependencies**: `torch>=2.1.0`, `transformers>=4.36.0`. No new
dependency.

**Storage**: N/A.

**Testing**: `pytest`, tiny-T5 pattern. Existing coverage in
`tests/test_model_forward.py` and `tests/test_delete_gate.py` is substantial and
already includes the gradient-isolation guarantee.

**Target Platform**: CPU in CI. The efficiency half of the claim needs a GPU and
belongs to spec `007`.

**Project Type**: Single project.

**Performance Goals**: Not a target here — the whole point is that efficiency is
*measured*, in `007`, against two baselines. Optimizing this variant in
isolation would corrupt the comparison.

**Constraints**:
- MUST gate at the same layer as MrT5, or gate placement confounds the result.
- MUST detach `n` at both consumption points.
- MUST NOT let `L_NE` reach the shared encoder.
- MUST reuse the noise estimator specified in `002` rather than re-implementing.

**Scale/Scope**: The variant wrapper plus the noise-adaptive mode of the shared
delete gate. No external interface, so no `contracts/`.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Check | Result |
|---|---|---|
| I. Branch Discipline | Carried on the shared spec-retrofit branch rather than `feat/005-` | **DEVIATION** — see Complexity Tracking |
| II. Spec-Driven Development | Written from the manuscript without treating the implementation as the source of intent. Two questions the manuscript does not answer are marked `NEEDS CLARIFICATION` rather than inferred. Class B provenance | PASS |
| III. Reproducibility | `d_max`, `noise_avg_momentum`, `gate_k`, `noise_estimator_hidden_dim`, loss weights all in `configs/tahimik_config.py` | PASS |
| IV. Test-First & CI Gate | `src/models/` → test-first mandatory. Two success criteria (SC-004, SC-005) have no existing coverage | PASS, with obligation: write those tests first |
| V. Transparent AI Assistance | `Co-Authored-By:`; provenance recorded per-spec | PASS |

**Principle II is the one to watch here.** Two constants in the contribution —
`navg`'s initialisation and the gate-shift clamp — are decisions the code makes
that the manuscript does not appear to specify. Inferring a justification for
them would be exactly the invention Principle II forbids, so both are left open.

## Project Structure

### Documentation (this feature)

```text
specs/005-noise-adaptive-byt5/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 — six decisions, two open, one new risk
├── data-model.md        # Phase 1
├── quickstart.md        # Phase 1
└── tasks.md             # Phase 2
```

### Source Code (repository root)

```text
src/models/
├── noise_adaptive_byt5.py   # The variant — the contribution's wiring
├── delete_gate.py           # Shared with 004; this spec covers
│                            #   noise_adaptive=True mode only
└── noise_estimator.py       # Specified in 002; consumed, not re-specified

src/training/
└── losses.py                # L = L_CE + w_rate·L_rate + w_attn_reg·L_attn_reg + L_NE

configs/
└── tahimik_config.py        # d_max, momentum, gate k, estimator width, weights

tests/
├── test_model_forward.py    # Existing — includes gradient isolation
└── test_delete_gate.py      # Existing — includes clean-vs-noisy rate comparison
```

**Structure Decision**: Single project, existing paths. No new source file. New
tests only where SC-004 and SC-005 are genuinely uncovered.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Shares the spec-retrofit branch instead of `feat/005-` | Five specs in one pass before a deadline; findings interlock across them. | One branch per feature is the default and is correct for implementation work. Implementing these tasks MUST return to it. |
| Two mechanisms (gate shift **and** deletion target) rather than one | `research.md` establishes they act at different times on different objects. Target alone vanishes at inference; shift alone is fought by the rate loss pulling toward a shared target. | Using only the deletion target is the obvious simplification. Rejected because the adaptation would then exist during training and disappear at inference — precisely where the efficiency claim is made. |

## Phase 0: Outline & Research

Complete. See [research.md](./research.md) — six decisions, **two open
`NEEDS CLARIFICATION` items**, and one previously unrecorded risk (`cn` sign).

Per Constitution Principle II, the open items do not block this plan. They block
only the specific tasks that depend on them.

## Phase 1: Design & Contracts

See [data-model.md](./data-model.md) and [quickstart.md](./quickstart.md).

No `contracts/` — internal component.

## Constitution Check — post-design re-check

No new dependency, storage, or hyperparameter introduced. The two open items
remain open and are correctly scoped so they block their dependent tasks rather
than the feature.
