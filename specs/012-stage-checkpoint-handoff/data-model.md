# Data Model

## StageCheckpoint

Required fields: schema version, checkpoint ID, stage, epoch, validation loss, model state, optimizer state, scheduler state, resolved configuration, provenance, and optional parent checkpoint ID.

## HandoffRecord

Required fields: source checkpoint ID/path, source stage/epoch/validation loss, target stage, restoration timestamp, restoration success, and model-state fingerprint after restoration.

## Invariants

- Stage 1 best validation loss is tracked separately from Stage 2.
- A Stage 2 checkpoint created after pretraining has exactly one Stage 1 parent ID.
- A skipped Stage 1 has no fabricated handoff record.
- Failed validation produces no Stage 2 optimizer or checkpoint.
