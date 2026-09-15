# Small Model Migration Implementation Plan

**Branch**: `feat/017-small-model-migration` | **Date**: 2026-09-13 | **Spec**: `specs/017-small-model-migration/spec.md`

## Summary

Assign the Google Small source to ByT5 and the Stanford MrT5 Small source to both compressed variants. Preserve effective batches and enforce checkpoint source/shape compatibility.

## Technical Context

**Language/Version**: Python 3.10–3.11
**Primary Dependencies**: PyTorch, Hugging Face Transformers
**Testing**: pytest
**Project Type**: research training and inference system

## Constitution Check

- Model-output settings remain in `configs/`: pass.
- Seed 42 and experiment controls remain fixed: pass.
- Architecture identity is persisted and validated: pass.
- Model/training behavior has regression coverage: pass.

## Project Structure

```text
configs/base.py
configs/byt5_config.py
configs/mrt5_config.py
configs/tahimik_config.py
src/training/trainer.py
scripts/preflight_base.py
tests/test_experiment_configuration.py
tests/test_model_configuration.py
tests/test_checkpoint_compatibility.py
tests/test_gradient_accumulation.py
```

## Implementation Sequence

1. Resolve each variant to its designated Small source.
2. Restore physical microbatches while preserving effective batches.
3. Persist source and architecture metadata in checkpoints.
4. Reject Base/Small and cross-source checkpoint mismatches.
5. Verify configuration, compatibility, accumulation, and full-suite behavior.
