# Feature Specification: Imported Corpus Contract

**Feature Branch**: `feat/009-methodology-compliance`

**Created**: 2026-09-03

**Status**: Draft

**Input**: Finalized CSV exports from the separate gathering and annotation systems must be eligible for the Chapter 3 experiment.

## User Scenarios & Testing

### User Story 1 - Validate final exports (Priority: P1)

A researcher imports finalized gold pairs and reliability labels and receives a deterministic eligibility report before training begins.

**Independent Test**: Validate accepted and rejected CSV fixtures without constructing a model.

**Acceptance Scenarios**:

1. **Given** correctly shaped approved exports, **When** validation runs, **Then** every accepted row and count is reported.
2. **Given** an unapproved, non-anonymized, short, oversized, duplicate, conflicting, or malformed row, **When** validation runs, **Then** it is rejected with a stable reason code.

### User Story 2 - Enforce the full experiment contract (Priority: P2)

The full experiment begins only with exactly 15,000 eligible gold pairs, a deterministic 12,000/1,500/1,500 split, a valid 3,000-sentence reliability subset, and 1,000,000 eligible synthetic pairs.

**Independent Test**: Verify boundary counts stop before model construction.

### Edge Cases

- Exact duplicate pairs collapse before the fixed-count check; conflicting targets always fail.
- Missing annotator/category combinations are reported explicitly.
- Semantic language, spam, advertising, and sensitive-context decisions are trusted from approved external review fields rather than guessed locally.

## Requirements

### Functional Requirements

- **FR-001**: The system MUST accept separate gold-pair and long-format reliability-label CSV exports.
- **FR-002**: Gold rows MUST include stable identity, noisy text, clean text, approval status, and anonymization status.
- **FR-003**: Reliability rows MUST identify sentence, annotator, category, and a binary present/absent value.
- **FR-004**: Accepted sentences MUST contain at least four words and neither side may exceed 1,024 UTF-8 bytes.
- **FR-005**: The validator MUST reject missing approvals, incomplete anonymization, malformed labels, duplicates, and conflicting targets.
- **FR-006**: The full experiment MUST require exactly 15,000 accepted gold pairs and produce exactly 12,000 training, 1,500 validation, and 1,500 test pairs.
- **FR-007**: The full experiment MUST require exactly 1,000,000 accepted synthetic pairs.
- **FR-008**: Validation MUST finish before model or tokenizer construction.
- **FR-009**: Reports MUST expose counts and stable reason codes without copying private raw text into logs.

### Key Entities

- **Gold Pair**: Stable identity, noisy text, clean target, approval, and anonymization status.
- **Reliability Label**: Sentence, annotator, noise category, and binary judgment.
- **Eligibility Report**: Counts, reason codes, split sizes, and eligible/ineligible decision.

## Success Criteria

### Measurable Outcomes

- **SC-001**: One invalid row prevents the full experiment from constructing a model.
- **SC-002**: An eligible export produces exactly 12,000/1,500/1,500 gold splits.
- **SC-003**: Re-running validation with identical exports and seed produces identical membership and reports.
- **SC-004**: No rejected-row report contains source sentence text.

## Assumptions

- Gathering, semantic filtering, consent/privacy review, and annotation happen outside this repository.
- External systems assign stable sentence and annotator identifiers.
