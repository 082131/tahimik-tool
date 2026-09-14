# ByT5-Base Migration Verification Design

## Goal

Verify all three experiment conditions consistently use pretrained ByT5-Base and reject incompatible Small checkpoints.

## Design

`BaseConfig.model_name` remains the sole model/tokenizer authority. The baseline and both compression variants derive dimensions from the loaded model configuration. Checkpoint validation requires complete Base architecture identity. The preflight script supplies a hardware/configuration report before real training.

## Acceptance criteria

- All variants and tokenizer resolution use `google/byt5-base`.
- Small or incomplete checkpoints are rejected for reporting runs.
- The Base preflight and configuration tests pass without downloading model weights in metadata-only mode.
