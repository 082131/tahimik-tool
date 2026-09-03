# Research and Decisions

## Restoration point

**Decision:** Restore the Stage 1 best model after Stage 1 completes and before the Stage 2 optimizer and scheduler are created.

**Rationale:** Optimizer construction must observe the intended parameters, and Stage 1 optimization state must not leak into fine-tuning.

## Payload scope

**Decision:** Save a complete checkpoint but restore only compatible state according to the transition: model weights for Stage 2 initialization; optimizer/scheduler only for same-stage resume.

## Identity and integrity

**Decision:** Compute a checkpoint identity from canonical metadata plus the saved artifact digest, validate required keys and stage compatibility, and write atomically.

**Alternatives rejected:** using in-memory last-epoch weights or guessing `best_stage1.pt` without validating its contents.
