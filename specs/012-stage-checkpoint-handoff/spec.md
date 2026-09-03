# Feature Specification: Stage Checkpoint Handoff

**Feature Branch**: `feat/009-methodology-compliance`

**Created**: 2026-09-03

**Status**: Draft

**Input**: Stage 2 must begin from the best Stage 1 validation checkpoint, not merely the last epoch.

## User Scenarios & Testing

### User Story 1 - Continue from the best pretraining state (Priority: P1)

A researcher runs two-stage training and can prove that Stage 2 begins with the Stage 1 epoch having the lowest validation loss.

**Independent Test**: Simulate several Stage 1 epochs with a non-final best epoch and compare Stage 2's initial weights with the saved best state.

### Edge Cases

- Stage 1 may be intentionally skipped for development.
- A missing or malformed best checkpoint stops Stage 2 when Stage 1 was requested.
- Stage 2 maintains its own independent best-validation state.

## Requirements

### Functional Requirements

- **FR-001**: Stage 1 MUST track the minimum validation loss and save its corresponding complete training state.
- **FR-002**: Stage 2 MUST restore the best Stage 1 model state before creating its optimizer and scheduler.
- **FR-003**: The handoff MUST record source epoch, validation loss, checkpoint identity, and restoration success.
- **FR-004**: Stage 2 checkpoint selection MUST be independent from Stage 1 checkpoint selection.
- **FR-005**: A requested Stage 1 without a restorable best checkpoint MUST fail before Stage 2 begins.

### Key Entities

- **Stage Checkpoint**: Model state, stage, epoch, validation loss, configuration, and provenance.
- **Handoff Record**: Identity of the Stage 1 state used to initialize Stage 2.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Stage 2 initial weights exactly match the best Stage 1 checkpoint.
- **SC-002**: A later but worse Stage 1 epoch is never used for the handoff.
- **SC-003**: The handoff is recoverable from saved metadata without reading logs.

## Assumptions

- Stage 1 and Stage 2 use the same compatible model architecture.
