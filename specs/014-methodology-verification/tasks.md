# Tasks: Methodology Verification

## Phase 1 — Exact metric behavior

- [ ] T001 [US1] Add failing hand-computed GLEU+ fixtures for perfect unchanged-clean, corrected, copied-noise, empty, and n-gram-order cases in `tests/test_metrics_exact.py`.
- [ ] T002 [US1] Correct source-aware GLEU+ weighting, penalty, and brevity behavior in `src/evaluation/metrics.py` until T001 passes.
- [ ] T003 [US1] Add failing unequal-denominator corpus ERR and shared Unicode alpha-word tests in `tests/test_metrics_exact.py`.
- [ ] T004 [US1] Implement corpus-total ERR and one shared word definition in `src/evaluation/metrics.py` until T003 passes.

## Phase 2 — Statistical inference

- [ ] T005 [US2] Add failing paired-bootstrap tests for paired indices, seed, exactly 1,000 full-run resamples, add-one p, percentile CI, direction, and CI-gated significance in `tests/test_statistics_exact.py`.
- [ ] T006 [US2] Implement the bootstrap contract in `src/evaluation/statistical_tests.py` until T005 passes.
- [ ] T007 [US2] Add failing Wilcoxon fixtures for ties, zeros, rank sums, model summaries, paired median/IQR GB, and rank-biserial in `tests/test_statistics_exact.py`.
- [ ] T008 [US2] Implement correct paired Wilcoxon summaries/effect size in `src/evaluation/statistical_tests.py` until T007 passes.
- [ ] T009 [US2] Add failing Holm tests for monotonic adjusted p-values, step-down stopping, explicit accuracy membership, and two-member per-comparison efficiency families in `tests/test_statistics_exact.py`.
- [ ] T010 [US2] Implement Holm and family assembly in `src/evaluation/statistical_tests.py` and `scripts/run_experiment.py` until T009 passes.

## Phase 3 — Annotation reliability

- [ ] T011 [US3] Add failing nominal binary per-category alpha tests for agreement, disagreement, missing cells, malformed labels, and threshold failure in `tests/test_annotation_reliability.py`.
- [ ] T012 [US3] Implement long-form pivot, nominal alpha, and 0.80 full-run gate in `src/evaluation/annotation.py` and `scripts/run_experiment.py` until T011 passes.

## Phase 4 — Efficiency protocol

- [ ] T013 [US4] Add failing mocked call-order/count tests for five warm-ups, synchronization, 20 peak resets/reads, and CPU unavailable behavior in `tests/test_efficiency_protocol.py`.
- [ ] T014 [US4] Implement independent latency/memory observations and correct summaries in `src/evaluation/efficiency.py` until T013 passes.
- [ ] T015 [US4] Persist vectors, units, and summary fields through `scripts/benchmark.py` and `scripts/run_experiment.py`.

## Phase 5 — Integration and traceability

- [ ] T016 Add a failing end-to-end results-contract and FR-to-test traceability test in `tests/test_methodology_integration.py`.
- [ ] T017 Integrate the statistical results contract and provenance in `scripts/run_experiment.py` until T016 passes.
- [ ] T018 Replace insufficient output-key assertions in `tests/test_methodology_compliance.py` with behavioral assertions or map them to the exact suites.
- [ ] T019 Run all five focused suites and `python -m pytest -q`; update `docs/CHAPTER3_CODE_MAPPING.md` and `README.md` with verified behavior only.

## Dependencies and parallel work

Metric tasks T001–T004, statistical tasks T005–T010, reliability tasks T011–T012, and efficiency tasks T013–T015 can proceed independently. Every implementation follows its failing test. T016–T019 depend on all four streams and on specs 010 and 013.
