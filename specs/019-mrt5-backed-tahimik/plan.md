# MrT5-Backed TAHIMIK Implementation Plan

**Branch**: `feat/019-mrt5-backed-tahimik` | **Date**: 2026-09-13 | **Spec**: `specs/019-mrt5-backed-tahimik/spec.md`

## Summary

Initialize TAHIMIK from the same Stanford MrT5 Small model/gate as its comparator, then layer the noise estimator, adaptive target, and centered gate shift on that common foundation.

## Technical Context

**Language/Version**: Python 3.10–3.11
**Dependencies**: PyTorch, Transformers
**Testing**: pytest

## Constitution Check

- Comparison changes only the intended adaptive mechanism: pass.
- Hyperparameters remain config-owned: pass.
- Gradient isolation and gate loading are tested: pass.

## Project Structure

```text
configs/tahimik_config.py
src/models/noise_adaptive.py
src/models/noise_estimator.py
src/models/delete_gate.py
src/training/losses.py
tests/test_mrt5_model_loading.py
tests/test_model_forward.py
tests/test_training_diagnostics.py
```

## Implementation Sequence

1. Load and transfer the same Stanford source/gate as MrT5.
2. Disable embedded gate execution.
3. Attach the estimator and adaptive gate inputs.
4. Preserve detach boundaries and loss ownership.
5. Verify shared initialization, adaptive direction, finite loss, diagnostics, and gradients.
