# Checkpoint Lineage Design

## Goal

Make the Stage 1-to-Stage 2 handoff recoverable and auditable from checkpoint files alone.

## Design

Each best checkpoint stores a stable checkpoint ID, model, optimizer, scheduler, scaler, resolved configuration, provenance, and optional parent ID. Stage 2 receives the restored Stage 1 checkpoint identity and serializes a complete handoff record into its best checkpoint. Optimizer and scheduler state are saved for reproducibility, while Stage 2 still creates fresh optimization state after restoring only the best Stage 1 model parameters.

## Acceptance criteria

- The first Stage 2 weights equal the best Stage 1 weights.
- A Stage 2 checkpoint identifies exactly one Stage 1 parent when pretraining ran.
- Checkpoints contain all required state and reject malformed/missing architecture metadata.
