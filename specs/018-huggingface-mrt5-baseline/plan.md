# Hugging Face MrT5 Baseline Implementation Plan

**Branch**: `feat/018-huggingface-mrt5-baseline` | **Date**: 2026-09-13 | **Spec**: `specs/018-huggingface-mrt5-baseline/spec.md`

## Summary

Load Stanford MrT5 Small with its custom code, transfer the pretrained embedded gate into the project's single wrapper gate, and reject incomplete construction.

## Technical Context

**Language/Version**: Python 3.10–3.11
**Dependencies**: PyTorch, Transformers, huggingface-hub, safetensors
**Testing**: pytest

## Constitution Check

- Model source is config-owned and checkpoint-traceable: pass.
- Gate semantics and failure behavior have focused tests: pass.
- Third-party code/weights are attributed: pass.

## Project Structure

```text
configs/base.py
configs/mrt5_config.py
src/models/fixed_compression.py
src/models/delete_gate.py
docs/project/attributions.md
tests/test_mrt5_model_loading.py
tests/test_delete_gate.py
tests/test_model_forward.py
```

## Implementation Sequence

1. Replace the plain T5 loader with the trusted custom MrT5 loader.
2. Capture and disable the embedded gate.
3. Map the pretrained gate into the local compatible interface.
4. Fail closed if transfer is unavailable.
5. Verify loading, gate math, one-gate execution, forward, and generation behavior.
