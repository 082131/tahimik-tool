# Feature Specification: Experiment Provenance

**Feature Branch**: `feat/009-methodology-compliance`

**Created**: 2026-09-03

**Status**: Draft

**Input**: Every generated result and checkpoint must show exactly which committed code, configuration, data contract, and runtime produced it.

## User Scenarios & Testing

### User Story 1 - Audit any artifact (Priority: P1)

A researcher can inspect a result or checkpoint and reconstruct its code revision, resolved configuration, seed, software versions, device, CUDA runtime, dataset eligibility, and input manifests.

**Independent Test**: Generate metadata under mocked CPU and CUDA environments and validate the complete schema and serialization.

### User Story 2 - Prevent ineligible reporting (Priority: P2)

Development outputs clearly state why they cannot be used as thesis evidence.

**Independent Test**: Produce metadata for dirty, CPU, undersized, and fully eligible conditions and verify the decision and reason codes.

### Edge Cases

- Git information may be unavailable outside a checkout.
- CUDA may be unavailable or partially initialized.
- Configuration values may contain paths, devices, tuples, or other non-primitive values.

## Requirements

### Functional Requirements

- **FR-001**: Training, full experiment, standalone evaluation, standalone benchmark, and checkpoints MUST use one provenance schema.
- **FR-002**: Provenance MUST contain Git SHA, dirty status, resolved model configuration, operational arguments, seed, Python and package versions, platform, PyTorch, CUDA build/runtime, cuDNN, device name/capability, and precision.
- **FR-003**: Provenance MUST include dataset eligibility, split fingerprints, probability manifest identity, lexicon resource versions, and checkpoint parentage when applicable.
- **FR-004**: Values MUST be serialized deterministically without stringifying whole nested records accidentally.
- **FR-005**: Ineligible outputs MUST list machine-readable reason codes.
- **FR-006**: Dirty-tree outputs remain usable for debugging but MUST be marked ineligible.
- **FR-007**: CPU efficiency results MUST be marked ineligible for Chapter 3 GPU claims.
- **FR-008**: Deterministic execution settings MUST be recorded and enabled consistently by every executable command.

### Key Entities

- **Run Provenance**: Code, configuration, runtime, data, and eligibility record.
- **Artifact Lineage**: Parent checkpoint and input-manifest relationships.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Every JSON result and checkpoint contains the same required provenance keys.
- **SC-002**: All metadata records serialize and deserialize without information loss.
- **SC-003**: Every ineligible fixture returns at least one precise reason code.
- **SC-004**: Two equivalent resolved configurations produce identical canonical fingerprints.

## Assumptions

- Raw datasets and checkpoints remain excluded from Git.
