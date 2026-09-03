# Stage Checkpoint Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Initialize Stage 2 from the best Stage 1 validation checkpoint and preserve an auditable handoff record.

**Architecture:** Make stage checkpoint selection independent, return the saved best path/metadata from `train_stage`, restore model state after Stage 1 and before constructing the Stage 2 optimizer/scheduler, and persist parentage in Stage 2 checkpoints.

**Tech Stack:** PyTorch checkpoints, pathlib, dataclasses/dicts, pytest.

**Spec:** `specs/012-stage-checkpoint-handoff/spec.md`

## Global Constraints

- Best means minimum Stage 1 validation loss.
- Restore before Stage 2 optimizer/scheduler creation.
- Fail closed when requested Stage 1 has no valid best checkpoint.
- Do not overwrite unrelated user edits in model/loss files.

## Technical Context

Checkpoint payloads include stage, epoch, validation loss, model state, optimizer/scheduler state, configuration, provenance, and checkpoint ID. Stage 2 records the Stage 1 checkpoint ID as its parent.

## Constitution Check

The design is deterministic, checkpoint provenance is explicit, output-changing state is serialized, and behavioral tests precede trainer changes.

## Project Structure

```text
src/training/trainer.py
src/utils/checkpointing.py
tests/test_checkpoint_handoff.py
specs/012-stage-checkpoint-handoff/contracts/checkpoint-handoff.schema.json
```

## Implementation Phases

1. Define validation and handoff schema.
2. Test best-epoch selection and malformed/missing checkpoint behavior.
3. Implement atomic checkpoint save/load and Stage 2 restoration order.
4. Verify Stage 2 parentage and independent best-state tracking.

## Post-Design Constitution Check

No exception is required.
