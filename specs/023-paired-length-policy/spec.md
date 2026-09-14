# Feature Specification: Paired Length Policy

**Feature Branch**: `feat/023-paired-length-policy`

**Created**: 2026-09-15

**Status**: Design confirmed

**Input**: Make byte-level noise labels, tokenized model inputs, targets, and
data splits describe the same complete-word noisy-clean example.

## Plain-language summary

Before a noisy-clean pair can receive `n*` or enter a split, it must fit the
configured input and target limits.  A pair that is too long is transformed,
not silently token-truncated: the system retains the longest prefix made from
complete aligned word groups that fits both sides.  If even the first aligned
group cannot fit, the pair is excluded and its reason is recorded.

## Confirmed decisions

- Limits are measured with the active ByT5 tokenizer, including its special
  tokens, because that is the sequence the model receives.
- The retained strings end at complete whitespace-delimited word boundaries;
  no UTF-8 character or word is partially retained.
- Noisy and clean words are aligned into ordered edit groups.  A replacement,
  insertion, or deletion group is kept or omitted as one unit, so split/merge
  normalizations such as `sanaol` -> `sana all` remain valid.
- When the next aligned group exceeds either limit, that group and all later
  groups are omitted from both retained strings.
- A pair whose first aligned group cannot fit is excluded with the auditable
  reason `no_complete_aligned_word_group_fits_limit`.
- `n*` is computed only after this policy and only from retained strings.
- Filtering/transformation occurs before every train/validation/test split and
  is shared by gold, synthetic, training, evaluation, and experiment paths.
- `NormalizationDataset` must not silently truncate a pair that reached it.

## User Scenarios & Testing

### User Story 1 - Train from coherent paired examples (Priority: P1)

A researcher trains any study variant knowing that its input, target, and
noise label refer to the same retained pair.

**Independent Test**: A long pair whose next aligned group exceeds a tiny
fixture limit retains only the preceding complete groups, and its returned
`n*` equals the byte edit-distance label of those retained strings.

**Acceptance Scenarios**:

1. **Given** `aang gandaaaa` and `ang ganda`, **when** the second aligned
   group would exceed a limit, **then** both retained strings stop before
   `gandaaaa`/`ganda`.
2. **Given** `sanaol ngayon` and `sana all ngayon`, **when** the split/merge
   group fits, **then** both sides retain the entire aligned group.
3. **Given** a pair whose first aligned group does not fit, **when** the
   policy runs, **then** the pair is excluded with its documented reason.

---

### User Story 2 - Evaluate every variant fairly (Priority: P1)

A researcher obtains train, validation, and test partitions with no pair
silently changed after its label or partition assignment.

**Independent Test**: Apply the policy before `split_data`, then prove all
retained source indices appear in exactly one partition and every dataset item
fits without tokenizer truncation.

**Acceptance Scenarios**:

1. **Given** retained pairs, **when** partitions are created, **then** no
   retained source index appears in more than one partition.
2. **Given** an overlength pair that bypasses preprocessing, **when**
   `NormalizationDataset` accesses it, **then** it raises a clear validation
   error rather than silently truncating it.

---

### User Story 3 - Audit future overlength data (Priority: P2)

A researcher can state how many pairs were unchanged, transformed, or
excluded for a run without storing raw datasets in version control.

**Independent Test**: A mixed fixture produces exact unchanged, transformed,
and excluded counts and per-source-index exclusion reasons.

### Edge Cases

- A pair exactly at either configured tokenizer limit is retained unchanged.
- Unicode words are retained only as whole Python strings and therefore remain
  valid UTF-8; tokenized length is still the acceptance criterion.
- Empty/whitespace-only inputs remain invalid under existing gold-data
  validation and do not become a retained empty prefix.
- A split/merge edit hunk is atomic even when it contains unequal word counts.

## Requirements

### Functional Requirements

- **FR-001**: The system MUST expose one reusable paired-length preparation
  function that accepts noisy strings, clean strings, an active tokenizer, and
  separate input/target limits.
- **FR-002**: The function MUST measure candidate lengths with the active
  tokenizer without tokenizer truncation.
- **FR-003**: The function MUST retain the longest ordered prefix of complete
  aligned word groups that fits both limits.
- **FR-004**: The function MUST calculate `n*` only from retained pairs and
  return an audit summary containing unchanged, transformed, and excluded
  counts plus exclusion source indices/reasons.
- **FR-005**: Training, offline evaluation, and experiment entry points MUST
  prepare gold and synthetic pairs before label computation and splitting.
- **FR-006**: `NormalizationDataset` MUST tokenize without silent truncation
  and reject a pair that violates either configured length limit.
- **FR-007**: All three model variants MUST receive the same retained split
  data under seed 42.
- **FR-008**: Tests MUST cover a transformed long pair, exact resulting label,
  no silent dataset truncation, audit counts, and disjoint partitions.

### Key Entities

- **Aligned Word Group**: One ordered equality or contiguous edit hunk between
  noisy and clean whitespace-delimited word sequences.
- **Prepared Pair**: A retained noisy-clean prefix plus its post-policy `n*`.
- **Length Policy Audit**: Counts and source-indexed reasons describing the
  policy's action on a supplied collection.

## Success Criteria

- **SC-001**: No `n*` used by a dataset is calculated from discarded text.
- **SC-002**: Every dataset item tokenizes within both configured limits with
  `truncation=False`.
- **SC-003**: A fixture with an overflowing `gandaaaa`/`ganda` aligned group
  retains neither word and calculates its label from the prior complete group.
- **SC-004**: Test partitions contain disjoint retained source indices.
- **SC-005**: `python -m pytest tests/ -v` passes before review.

## Assumptions

- The existing ByT5 tokenizer remains the authority for model sequence length.
- The current dataset has no pair exceeding the limits; its audit should report
  zero transformed and zero excluded pairs.
- Results from a future transformed dataset are a distinct preprocessing state
  and must be recorded with their audit report.
