# Tasks: Synthetic Noise Policy

## Phase 1 — Contracts

- [ ] T001 Add manifest and lexicon contract fixtures in `tests/fixtures/noise_policy/`.
- [ ] T002 [P] Define manifest/category dataclasses and canonical hashing in `src/data/noise_policy.py`.
- [ ] T003 [P] Define the provider protocol and reviewed CSV implementation in `src/data/lexicon_provider.py`.

## Phase 2 — Training-only probability resolution

- [ ] T004 [US1] Add failing leakage, clamping, missing-bound, invalid-bound, and fingerprint tests in `tests/test_noise_policy.py`.
- [ ] T005 [US1] Implement training-ID filtering and bounded probability resolution in `src/data/noise_policy.py` until T004 passes.
- [ ] T006 [US1] Add failing lexicon validation and no-network tests in `tests/test_noise_policy.py`.
- [ ] T007 [US1] Implement reviewed lexicon loading/version validation in `src/data/lexicon_provider.py` until T006 passes.

## Phase 3 — Pair semantics

- [ ] T008 [US2] Add failing deterministic tests proving preserved features occur on both sides and correctable noise only on input in `tests/test_noise_generator_contract.py`.
- [ ] T009 [US2] Refactor `src/data/noise_generator.py` into preserved-augmentation and correctable-corruption passes until T008 passes.
- [ ] T010 [US2] Add failing tests that configuration probabilities and resources reach the generator in `tests/test_noise_generator_contract.py`.
- [ ] T011 [US2] Wire manifests/providers through `configs/base.py` and `src/data/preprocessing.py` until T010 passes.

## Phase 4 — Strict readiness and verification

- [ ] T012 Add full-run refusal for missing/unapproved bounds or resources in `scripts/run_experiment.py`; final numeric bounds remain blocked until supplied by the research team.
- [ ] T013 Save pair lineage and manifest identity without committing generated text in `src/data/preprocessing.py`.
- [ ] T014 Run `python -m pytest tests/test_noise_policy.py tests/test_noise_generator_contract.py -q` and reproduce a fixture twice with the same seed.

## Dependencies and parallel work

T002 and T003 may run in parallel. T004 precedes T005, T006 precedes T007, and T008 precedes T009. T011 depends on T005, T007, and T009. T012 cannot become eligible until approved numeric bounds are supplied.
