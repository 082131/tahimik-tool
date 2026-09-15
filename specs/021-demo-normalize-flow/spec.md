# Feature Specification: Demo Normalize Flow

**Feature Branch**: `feat/021-demo-normalize-flow`

**Created**: 2026-09-15

**Status**: Implemented

**Input**: Make Normalize deterministic in presentation mode while preserving the real backend workflow in live mode.

## User Scenarios & Testing

### User Story 1 - Restore the presentation scenario (Priority: P1)

In the default demo mode, pressing Normalize restores the complete configured scenario and its matching model results and telemetry.

**Independent Test**: Alter the input/state, invoke Normalize, and verify the demo sentence, single-row state, selected row, and summary state are restored without an API request.

### User Story 2 - Use real inference explicitly (Priority: P1)

When `VITE_TAHIMIK_DATA_MODE=live`, Normalize sends the entered text to the backend and displays its response/error state.

**Independent Test**: Build/run in live mode with a mocked backend and verify the request path executes.

### Edge Cases

- Blank trimmed input causes no action in either mode.
- Demo mode clears stale API status messages and exits batch/summary views.
- Backend availability checks do not convert demo Normalize into a live request.

## Requirements

- **FR-001**: Data mode MUST resolve from `VITE_TAHIMIK_DATA_MODE` and default to `demo`.
- **FR-002**: Only the exact value `live` MUST enable backend normalization.
- **FR-003**: Demo Normalize MUST restore `DEMO_SENTENCE`, single mode, row zero selection, hidden summary, and no API status message.
- **FR-004**: Demo Normalize MUST NOT call the backend.
- **FR-005**: Live Normalize MUST preserve the existing backend request and error-handling path.
- **FR-006**: Empty trimmed input MUST remain a no-op.

## Success Criteria

- **SC-001**: Repeated demo normalization produces the same complete presentation state.
- **SC-002**: Live mode reaches the backend and demo mode does not.
- **SC-003**: Frontend build succeeds in both configuration modes.

## Assumptions

- Presentation mode is the safe default because trained checkpoints may be unavailable.
- Demo values are illustrative and are never represented as measured experiment results.
