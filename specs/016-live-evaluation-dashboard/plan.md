# Live Evaluation Dashboard Implementation Plan

**Branch**: `feat/016-live-evaluation-dashboard` | **Date**: 2026-09-13 | **Spec**: [spec.md](spec.md)

**Goal:** Deliver one auditable API-backed workflow with unlabelled single or
batch inference, plus a separate labelled-test-set evaluation of all three
study variants.

**Architecture:** Extend the existing FastAPI server with shared inference and
telemetry helpers, atomic compare routes, and a persisted background evaluation
job. The React dashboard consumes only a typed client and renders the one job's
outputs across Engine, Benchmark, Efficiency, and Validation.

**Tech Stack:** Python 3.11, FastAPI, Pydantic, PyTorch, existing evaluation
modules, pytest, React 18, TypeScript, Vite.

## Global Constraints

- Keep the existing normalization routes backward-compatible.
- Do not change training code or hyperparameters.
- Use real inference telemetry; never browser heuristics for live results.
- Require all three checkpoints before compare/evaluation work begins.
- Limit jobs to 2,000 examples and one active job.
- Persist ignored runtime artifacts only under `outputs/dashboard-jobs/`.
- Do not commit, push, or add datasets/checkpoints/output artifacts.

## File Structure

```text
backend/app.py                         FastAPI schemas/routes and shared services
src/models/byt5_baseline.py            baseline telemetry generation
src/models/fixed_compression_byt5.py   fixed deletion telemetry generation
src/models/noise_adaptive_byt5.py      adaptive telemetry generation
src/evaluation/efficiency.py           fixed-protocol reusable profiling interface
tests/test_backend.py                  API/job contract tests
tests/test_model_forward.py            telemetry invariants
frontend/src/api/types.ts              API domain types
frontend/src/api/client.ts             one fetch/error/polling client
frontend/src/app/pages/Engine.tsx      live compare and evaluation trigger
frontend/src/app/pages/Benchmark.tsx   API aggregate accuracy values
frontend/src/app/pages/Efficiency.tsx  API profiling values
frontend/src/app/pages/Validation.tsx  API statistical values
```

## Constitution Check

- Hyperparameters remain in `configs/`: pass.
- Seed remains 42: pass.
- Metrics/statistics continue using existing methodology modules: pass.
- Runtime output is ignored and no datasets/checkpoints are committed: pass.
- Full pytest and frontend checks are required before handoff: pass conditionally
  on implementation verification.

## Implementation Sequence

### Task 1: Define telemetry at the model boundary

**Files:** modify model wrappers; modify `tests/test_model_forward.py`.

1. Write failing tests for baseline no-deletion telemetry, fixed deletion byte
   positions, adaptive noise/target telemetry, and equal decoded output between
   ordinary generation and telemetry generation.
2. Add `generate_with_telemetry(input_ids, attention_mask, max_length,
   num_beams)` to every wrapper.
3. Ensure compressed wrappers derive positions from the same hard-deletion mask
   used to construct the compressed encoder representation.
4. Run telemetry tests and existing model-forward tests.

### Task 2: Build shared API inference and comparison routes

**Files:** modify `backend/app.py`; modify `tests/test_backend.py`.

1. Write failing compare tests for three results, 503 atomic failure, batch
   order preservation, and telemetry serialization.
2. Add a typed internal `run_model` helper which loads, tokenizes, generates,
   decodes, measures latency, and maps model telemetry once.
3. Implement `/compare` and `/compare/batch` exclusively through that helper.
4. Retain `/normalize` routes through the same helper to prevent drift.
5. Run backend tests.

### Task 3: Implement persisted combined evaluation jobs

**Files:** modify `backend/app.py`; modify `src/evaluation/efficiency.py`;
modify `tests/test_backend.py`.

1. Write failing contract tests for validation, 202 creation, 409 active-job
   rejection, polling progression, restart recovery, and persisted completion.
2. Define Pydantic job/result schemas and JSON file storage rooted at
   `outputs/dashboard-jobs/`.
3. Validate all input/reference examples and all checkpoint availability before
   task launch.
4. Run accuracy generation for all three variants; preserve outputs and
   per-sentence vectors.
5. Invoke the existing fixed protocol profiler for each variant, then invoke
   existing statistical comparison code with aligned vectors.
6. Persist every state transition and expose `GET /evaluate/{job_id}`.
7. Run backend tests with mocked models/clocks/CUDA.

### Task 4: Replace direct frontend fetches with a typed client

**Files:** create `frontend/src/api/types.ts`, `frontend/src/api/client.ts`;
modify `Engine.tsx`.

1. Define exact TypeScript types from `contracts/dashboard-api.md`.
2. Implement a single request helper which decodes FastAPI `detail` failures
   and exposes compare/evaluate/poll functions.
3. Replace direct `fetch` calls and heuristic live gating values with compare
   response data.
4. Preserve static presentation fallback only when live API/checkpoints are
   unavailable; never merge it with a successful live result.
5. Type-check the frontend.

### Task 5: Bind completed job results to evaluation panels

**Files:** modify `Engine.tsx`, `Benchmark.tsx`, `Efficiency.tsx`,
`Validation.tsx`, and `Root.tsx`.

1. Keep the Engine's single-input and unlabelled-batch paths inference-only:
   show **Model Outputs** and **Compression & Gating Information** from
   `/compare` or `/compare/batch`, without attempting evaluation work.
2. Add a separate **Normalization Evaluation** tab/workflow where an evaluator
   uploads and reviews a paired held-out test set (`input`, `reference`) and
   explicitly selects **Run Evaluation & Profile**.
3. Poll one job ID; display stage/progress/error without claiming results early.
4. Map completed-job `metrics`, `efficiency`, and `statistics` fields to their
   respective panels. Before completion, evaluation panels are unavailable and
   disabled; do not request evaluation, profiling, or statistics for
   unlabelled inference.
5. Render byte telemetry as byte-level inspection evidence.
5. Type-check and production-build the frontend.

### Task 6: Regression verification and documentation

**Files:** modify `README.md` or `docs/backend-dashboard-additions.md`; tests.

1. Document routes, status codes, job storage, required labelled fields, and
   checkpoint prerequisites.
2. Run `python -m pytest tests/ -v`, `npm run typecheck`, and `npm run build`.
3. Inspect generated output only; do not add it to Git.
