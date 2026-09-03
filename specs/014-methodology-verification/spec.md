# Feature Specification: Methodology Verification

**Feature Branch**: `feat/009-methodology-compliance`

**Created**: 2026-09-03

**Status**: Draft

**Input**: Chapter 3 metric, reliability, efficiency, and statistical claims must be implemented exactly and protected by regression tests.

## User Scenarios & Testing

### User Story 1 - Trust RQ1 accuracy values (Priority: P1)

A researcher obtains source-aware GLEU+, chrF, corpus ERR, and alpha-word accuracy values that match hand-computed examples.

**Independent Test**: Verify perfect, unchanged-clean, corrected, incorrectly preserved, empty, and unequal-error-weight examples.

### User Story 2 - Trust RQ3 and RQ4 inference (Priority: P1)

A researcher receives paired bootstrap, Wilcoxon, rank-biserial, confidence interval, and Holm results using the exact declared comparison families.

**Independent Test**: Compare every formula with a hand-computed fixture, including ties and stopped Holm sequences.

### User Story 3 - Verify annotation reliability (Priority: P1)

A researcher recomputes nominal Krippendorff alpha separately for every binary noise category from three-annotator long-format labels.

**Independent Test**: Check perfect agreement, disagreement, missing values, malformed labels, and threshold failure.

### User Story 4 - Trust efficiency measurements (Priority: P2)

A researcher receives per-sentence latency and 20 independent peak-memory observations after five warm-ups, with the requested descriptive summaries.

**Independent Test**: Use mocked clocks/CUDA counters to verify reset/read order, vector shape, mean, standard deviation, median difference, and IQR.

### Edge Cases

- A clean input identical to its reference and prediction receives a perfect score rather than a source-copy penalty.
- Corpus ERR aggregates edit-distance totals; per-sentence ERR remains available only for paired resampling.
- Zero Wilcoxon differences and tied ranks remain well-defined.
- CPU runs never invent GPU-memory values.

## Requirements

### Functional Requirements

- **FR-001**: GLEU+ MUST use generated, reference, and noisy-source n-grams; penalize only incorrectly preserved source material; aggregate orders with declared weights and brevity penalty; and score a perfect unchanged-clean output as perfect.
- **FR-002**: Reported corpus ERR MUST aggregate error totals before division, while bootstrap inputs remain paired per sentence.
- **FR-003**: Alpha-word accuracy MUST use one shared Unicode-aware definition across corpus and per-sentence outputs.
- **FR-004**: Paired bootstrap MUST use 1,000 resamples, the two-tailed add-one formula, percentile 95% interval, direction-aware differences, and adjusted-p plus interval significance.
- **FR-005**: Peak memory MUST use two-sided Wilcoxon on 20 paired runs and report mean/standard deviation per model, median and IQR of paired differences in GB, and matched-pairs rank-biserial correlation based on signed-rank sums.
- **FR-006**: Holm correction MUST use step-down stopping and monotonic adjusted p-values.
- **FR-007**: Each efficiency comparison MUST form its own two-test family: latency and memory.
- **FR-008**: Accuracy-family composition MUST be recorded explicitly in results and consistently applied.
- **FR-009**: Reliability MUST use nominal binary alpha separately per category and enforce the 0.80 threshold.
- **FR-010**: Benchmarking MUST perform five discarded warm-ups, reset peak memory before each of 20 timed runs, and read it immediately after each run.
- **FR-011**: Automated tests MUST cover every requirement above and must not claim completion merely by checking that output keys exist.

### Key Entities

- **Sentence Metric Vector**: Paired values in held-out test order.
- **Memory Run Vector**: Twenty paired peak-memory values.
- **Comparison Family**: Named p-values sharing one Holm correction.
- **Reliability Category Result**: Alpha, counts, threshold, and decision.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Every metric and test matches hand computation within declared numerical tolerance.
- **SC-002**: A perfect unchanged-clean triple receives the maximum GLEU+ value.
- **SC-003**: Corpus ERR differs correctly from mean sentence ERR on unequal-denominator fixtures.
- **SC-004**: Every efficiency comparison contains exactly two Holm-adjusted tests.
- **SC-005**: All methodology requirements have at least one behavioral regression test.

## Assumptions

- The equations and prose in the current Chapter 3 manuscript are authoritative.
- The manuscript DOCX will not be edited by this feature.
