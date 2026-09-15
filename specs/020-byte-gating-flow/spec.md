# Feature Specification: Byte-Gating Flow

**Feature Branch**: `feat/020-byte-gating-flow`

**Created**: 2026-09-15

**Status**: Implemented

**Input**: Explain TAHIMIK's compression as an ordered noise-estimation and per-byte gating flow without implying that whole words are deleted.

## User Scenarios & Testing

### User Story 1 - Follow the adaptive pipeline (Priority: P1)

A viewer can follow three stages: estimate sentence noise, score each byte, and inspect per-byte retain/remove decisions.

**Independent Test**: Render the Engine with TAHIMIK telemetry and verify all three stage labels and byte counts.

### User Story 2 - Distinguish fixed and adaptive models (Priority: P1)

A viewer sees that MrT5 uses a fixed 50% target with noise input disabled, while TAHIMIK derives a target from estimated noise.

**Independent Test**: Switch the telemetry model and verify the explanatory content changes.

### User Story 3 - Avoid fabricated byte decisions (Priority: P1)

When exact byte-position telemetry is unavailable, the interface states that decisions are unavailable instead of synthesizing a pattern.

**Independent Test**: Render data without `bytePrunedPositions` and verify the unavailable message.

### Edge Cases

- The byte map appears only for TAHIMIK when position telemetry exists and its byte count matches the input length.
- Word outlines are reading guides only; state is assigned to byte positions.
- Kept/removed totals come from supplied positions when the map is available.

## Requirements

- **FR-001**: The inspector MUST present noise estimation, per-byte scoring, and per-byte decisions in order.
- **FR-002**: TAHIMIK MUST explain that noise determines a deletion target while the gate decides individual bytes.
- **FR-003**: MrT5 MUST display a fixed 0.50 target and disabled noise input.
- **FR-004**: Exact decisions MUST come from `bytePrunedPositions`; no deterministic-looking substitute may be generated.
- **FR-005**: The byte map MUST require TAHIMIK telemetry and a position list matching input length.
- **FR-006**: The inspector MUST report retained, removed, and total byte counts.
- **FR-007**: The UI MUST explicitly say word outlines do not represent word-level deletion.

## Success Criteria

- **SC-001**: A viewer can identify all three pipeline stages from one card.
- **SC-002**: Every displayed decision corresponds to supplied byte-position data.
- **SC-003**: Missing telemetry produces an honest unavailable state.
- **SC-004**: Frontend type-check and production build pass.

## Assumptions

- `SentenceData` is the Engine's presentation contract.
- Live byte decisions are displayed only when the backend supplies compatible telemetry.
