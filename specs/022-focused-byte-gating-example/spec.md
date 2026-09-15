# Feature Specification: Focused Byte-Gating Example

**Feature Branch**: `feat/022-focused-byte-gating-example`

**Created**: 2026-09-15

**Status**: Implemented

**Input**: Use one short example whose repeated characters make byte-level removal easy to inspect during a presentation.

## User Scenarios & Testing

### User Story 1 - Read the example quickly (Priority: P1)

A viewer can understand the noisy input, normalized output, and six removed byte positions without scanning a long sentence.

**Independent Test**: Render the default scenario and verify input, three outputs, noise/deletion values, and byte map.

### User Story 2 - Keep displayed arithmetic consistent (Priority: P1)

The example's 19-byte ASCII input and six unique removed positions display as 32% after rounding.

**Independent Test**: Count input bytes and removed positions and compare the rounded percentage.

### Edge Cases

- Position indices must be unique and within the input.
- Because the example is ASCII, character positions and UTF-8 byte positions coincide.
- Model outputs and latency values are presentation data, not study results.

## Requirements

- **FR-001**: The demo input MUST be `aang gandaaaa mooo!`.
- **FR-002**: ByT5, MrT5, and TAHIMIK demo outputs MUST each be `Ang ganda mo!`.
- **FR-003**: The displayed noise value MUST be 0.36 and deletion percentage MUST be 32.
- **FR-004**: Removed positions MUST be `[0, 10, 11, 12, 16, 17]`.
- **FR-005**: The token/word reading groups MUST be `aang`, `gandaaaa`, and `mooo!`.
- **FR-006**: The scenario MUST remain marked as demo data.

## Success Criteria

- **SC-001**: Six of 19 positions are removed and round to 32%.
- **SC-002**: Every removed position highlights a concrete redundant character in the displayed input.
- **SC-003**: The example fits without horizontal dependence on a long sentence.
- **SC-004**: Frontend type-check and build pass.

## Assumptions

- The example is optimized for explaining byte gating, not for representing corpus frequency or model accuracy.
