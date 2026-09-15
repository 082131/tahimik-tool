# Plan: ByT5 Base Migration

> **Status:** Superseded by `specs/017-small-model-migration/`. This plan is an
> architecture-history record and must not be executed against the current system.

**Component**: Backbone and Training Infrastructure
**Branch**: `feat/015-byt5-base-migration`
**Spec**: [spec.md](spec.md)

---

## 1. Architectural Strategy

1. **Single Authority Backbone**: `BaseConfig.model_name = "google/byt5-base"` governs model configuration and tokenizer instantiation across all scripts and variants.
2. **Dynamic Dimension Conformance**: Modules derive shapes exclusively from `model.config.d_model` (1,536 for Base, 1,472 for Small).
3. **Effective Batch Invariant**: Physical microbatches (2) with accumulation (8 for Stage 1, 4 for Stage 2) preserve effective batch sizes 16 and 8. Loss is scaled by $1 / \text{accum\_steps}$.
4. **Memory Management**: Mixed precision (`torch.amp.autocast`, `GradScaler`) and gradient checkpointing (`model.gradient_checkpointing_enable()`) ensure 582M parameter backbones fit commercial GPU memory.
5. **Checkpoint Architecture Gate**: Checkpoint dictionaries store an `architecture` block validated before `load_state_dict`. Mismatches trigger immediate execution termination.
6. **Hardware Preflight**: `scripts/preflight_base.py` enables offline metadata verification and online single-forward GPU smoke testing before launching full experiments.

---

## 2. Invariant Controls

| Variable | ByT5 Baseline | ByT5 + Fixed Delete Gate | TAHIMIK (Adaptive) |
|---|---|---|---|
| Backbone | `google/byt5-base` | `google/byt5-base` | `google/byt5-base` |
| Tokenizer | Shared ByT5 UTF-8 bytes | Shared ByT5 UTF-8 bytes | Shared ByT5 UTF-8 bytes |
| Delete Gate Layer | N/A | Layer 3 | Layer 3 |
| Delete Rate Target | 0.0 (No deletion) | 0.5 (Fixed) | $\delta(n)$ (Adaptive) |
| Stage 1 Effective Batch | 16 (2 × 8) | 16 (2 × 8) | 16 (2 × 8) |
| Stage 2 Effective Batch | 8 (2 × 4) | 8 (2 × 4) | 8 (2 × 4) |
| Optimizer / Seed | AdamW / 42 | AdamW / 42 | AdamW / 42 |

---

## 3. Testing Architecture

- Offline mock configurations for unit tests (zero model downloads).
- Dynamic shape parameterization (`tests/test_model_forward.py`).
- Step count verification under partial microbatch accumulation (`tests/test_gradient_accumulation.py`).
- Synthetic checkpoint compatibility fixtures (`tests/test_checkpoint_compatibility.py`).
