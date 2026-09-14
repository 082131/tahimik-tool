# Tasks: Live Evaluation Dashboard

**Input**: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md),
[data-model.md](data-model.md), and [dashboard-api.md](contracts/dashboard-api.md)

## Phase 1: Telemetry foundation

- [ ] T001 Add failing telemetry invariants to `tests/test_model_forward.py`.
- [ ] T002 Implement baseline `generate_with_telemetry` in `src/models/byt5_baseline.py`.
- [ ] T003 Implement fixed/adaptive telemetry and exact byte-position mapping in compressed wrappers.
- [ ] T004 Run model telemetry and regression tests.

## Phase 2: API comparison foundation

- [ ] T005 Add compare route contract tests to `tests/test_backend.py`.
- [ ] T006 Add shared `run_model`/telemetry serialization helpers in `backend/app.py`.
- [ ] T007 Add atomic `POST /compare` and `POST /compare/batch` routes.
- [ ] T008 Route existing normalization endpoints through the shared helper and run backend tests.

## Phase 3: Combined persisted evaluation job

- [ ] T009 Add job schema/storage/status contract tests to `tests/test_backend.py`.
- [ ] T010 Add `EvaluationJob` schemas and ignored JSON storage under `outputs/dashboard-jobs/`.
- [ ] T011 Add request validation, 2,000-row cap, availability preflight, and one-active-job guard.
- [ ] T012 Add the asynchronous accuracy, profiling, and statistics execution pipeline.
- [ ] T013 Add `POST /evaluate` and `GET /evaluate/{job_id}` and verify persistence/recovery behavior.

## Phase 4: Typed frontend integration

- [ ] T014 Create `frontend/src/api/types.ts` and `frontend/src/api/client.ts`.
- [ ] T015 Replace Engine direct fetches/local live estimates with typed compare data.
- [ ] T016 Keep unlabelled single/batch comparison inference-only, then add the separate **Normalization Evaluation** labelled-test-set review, explicit run action, one-job polling, and progress/error UI.
- [ ] T017 Bind completed job metrics, efficiency, and statistics to Benchmark, Efficiency, and Validation; keep those panels unavailable without a completed labelled evaluation.
- [ ] T018 Render returned byte telemetry in the Engine inspector.

## Phase 5: Verification and handoff

- [ ] T019 Add dashboard route/job documentation.
- [ ] T020 Run the full Python suite, frontend type-check, and production build.
- [ ] T021 Verify no output, checkpoint, or dataset artifacts are staged; do not commit or push.
