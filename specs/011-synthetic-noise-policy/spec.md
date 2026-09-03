# Feature Specification: Synthetic Noise Policy

**Feature Branch**: `feat/009-methodology-compliance`

**Created**: 2026-09-03

**Status**: Draft — blocked on finalized category bounds and reviewed lexicon export

**Input**: Synthetic generation must reflect training-only annotations while distinguishing meaning-preserving augmentation from correctable corruption.

## User Scenarios & Testing

### User Story 1 - Resolve auditable probabilities (Priority: P1)

A researcher derives category prevalence from approved gold training labels only and saves both observed and bounded probabilities.

**Independent Test**: Use a labeled fixture whose train/validation/test frequencies differ and prove only training labels affect the manifest.

### User Story 2 - Preserve meaningful features (Priority: P1)

Slang, emoji, code-switching, and Taglish morphology may be injected, but the injected feature appears identically in both source and target before correctable noise affects the source.

**Independent Test**: Generate deterministic pairs and assert preserved features match across both sides.

### Edge Cases

- Missing category bounds make the full experiment ineligible.
- Missing reviewed lexicon resources disable the dependent augmentation and make the full experiment ineligible rather than triggering website scraping.
- Validation and test labels never affect probabilities.

## Requirements

### Functional Requirements

- **FR-001**: A manifest MUST separate `preserved_augmentations` from `correctable_noise`.
- **FR-002**: Observed rates MUST be computed from the gold training split only.
- **FR-003**: Each category MUST declare its own approved lower and upper bound before the full experiment.
- **FR-004**: The manifest MUST retain raw observed rates, resolved probabilities, source split identity, seed, and resource versions.
- **FR-005**: Preserved augmentation MUST modify both synthetic input and target identically.
- **FR-006**: Correctable noise MUST modify only the synthetic input.
- **FR-007**: Reviewed lexicon CSV resources MUST be versioned and validated before use.
- **FR-008**: The system MUST NOT scrape KWF; future API access requires documented authorization and a separate provider.
- **FR-009**: Identical seeds, manifests, sentences, and resources MUST reproduce identical pairs.

### Key Entities

- **Probability Manifest**: Observed and resolved rates, bounds, split provenance, and resource versions.
- **Preserved Augmentation**: Meaningful feature shared by input and target.
- **Correctable Noise**: Input-only corruption.
- **Lexicon Resource**: Reviewed mapping or template data with a version identifier.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Changing validation/test labels changes zero manifest probabilities.
- **SC-002**: Every injected preserved feature appears unchanged in the paired target.
- **SC-003**: Every generated pair records the manifest and resource versions that produced it.
- **SC-004**: No network request is needed to generate synthetic data.

## Assumptions

- Category-specific bounds and reviewed lexicon CSV data will be supplied after annotation and reliability work finishes.
- The manuscript document remains unchanged.
