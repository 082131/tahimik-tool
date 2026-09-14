# Feature Specification: Live Evaluation Dashboard

**Feature Branch**: `feat/016-live-evaluation-dashboard`

**Created**: 2026-09-13

**Status**: Design confirmed; implementation not started

**Input**: Replace dashboard-only inference approximations with a live, auditable
evaluation workflow for ByT5, MrT5, and TAHIMIK.

## Plain-language summary

The dashboard will continue to normalize one sentence interactively and accept
unlabelled batches for model comparison. Those inference paths show each
model's output and observed compression/gating telemetry only. A separate
**Normalization Evaluation** workflow accepts a labelled held-out test set
(`input` plus clean `reference`), then explicitly starts one background job.
That job evaluates all three study variants, profiles their runtime under the
Chapter 3 protocol, performs the required statistical tests, and saves a
resumable result. The frontend shows benchmark values only from that completed
job; it does not invent compression, accuracy, or profiling values.

## Confirmed decisions

- Comparison and evaluation always include ByT5, MrT5, and TAHIMIK. A missing
  requested checkpoint fails the whole request with HTTP 503.
- `POST /evaluate` starts one combined background run; `GET /evaluate/{job_id}`
  reports its state and final result. Profiling is an internal service, not a
  second public route.
- Evaluation accepts up to 2,000 uploaded `{input, reference}` examples. Only
  one run may be active at a time.
- A completed run persists as ignored local files under `outputs/dashboard-jobs/`.
- The run performs four accuracy metrics, five warm-up passes, 20 measured
  profiling passes at internal batch size 1, and the configured statistical
  comparisons.
- The API returns aggregate values and per-example outputs/scores for all three
  models. Telemetry includes observed aggregate values and byte positions kept
  or deleted by compressed variants.
- The frontend uses one typed API client. Presentation fallback values may
  remain until real data/checkpoints exist, but live results must always come
  from the API.
- Unlabelled single and batch inference never starts evaluation, profiling, or
  statistics. It still returns Model Outputs and Compression & Gating
  Information. Benchmark, Runtime Efficiency, and Statistical Validation are
  unavailable until a labelled evaluation job completes.
- No extra thesis-eligibility badge is required in the UI.

## User Scenarios & Testing

### User Story 1 - Compare one sentence with live model output (Priority: P1)

An evaluator enters one informal sentence and receives the normalized output
from all three trained variants plus telemetry from the exact inference pass.

**Why this priority**: This is the interactive demonstration path and verifies
the model/API/frontend connection before a full evaluation is attempted.

**Independent Test**: Mock all three loaded models, call `POST /compare`, and
verify each decoded output, latency, and telemetry field reaches the typed
frontend client unchanged.

**Acceptance Scenarios**:

1. **Given** all three checkpoints are available, **When** a user compares one
   sentence, **Then** the response contains one result for each model and
   observed telemetry for the compressed models.
2. **Given** one requested checkpoint is unavailable, **When** comparison is
   requested, **Then** the API returns HTTP 503 and no partial comparison.
3. **Given** one unlabelled input is active, **When** the user views the
   dashboard, **Then** Model Outputs and Compression & Gating Information show
   the observed inference response, while all evaluation panels remain
   unavailable and no evaluation job is requested.

---

### User Story 2 - Evaluate one uploaded labelled batch (Priority: P1)

An evaluator uploads up to 2,000 rows containing noisy input and a reference
normalization, reviews the parsed batch, then explicitly starts one combined
evaluation run.

**Why this priority**: A labelled batch is required to calculate the study's
accuracy metrics and creates the common input set for every model comparison.

**Independent Test**: Submit a small fixture batch to `POST /evaluate`, poll
the returned job ID, and assert that the saved result contains all three models,
aggregate metrics, and per-example outputs.

**Acceptance Scenarios**:

1. **Given** a valid labelled batch of at most 2,000 rows and no active job,
   **When** the user starts evaluation, **Then** the API returns HTTP 202 with
   a unique job ID and a queued/running state.
2. **Given** a row without a non-empty reference, **When** evaluation is
   requested, **Then** the API rejects the request before model inference.
3. **Given** an active job, **When** another evaluation is requested, **Then**
   the API returns HTTP 409 with the active job ID.
4. **Given** a completed job, **When** the frontend requests its status,
   **Then** it receives aggregate metrics and each row's ByT5, MrT5, and
   TAHIMIK outputs/scores.

---

### User Story 3 - Obtain controlled efficiency and statistical results (Priority: P1)

The same job profiles every variant using the uploaded inputs and produces the
accuracy and efficiency comparisons required by the study.

**Why this priority**: The thesis claims require more than individual outputs;
they need repeatable latency/memory measurements and valid comparisons.

**Independent Test**: Use mock clocks, mocked CUDA memory counters, and known
metric vectors to prove five warm-ups, 20 measured runs, and the expected
statistical-result payload.

**Acceptance Scenarios**:

1. **Given** a run begins profiling, **When** timing starts, **Then** five
   warm-up passes have completed and are excluded from reported timing.
2. **Given** profiling is on CUDA, **When** each measured pass runs, **Then**
   peak memory is reset before the pass and recorded immediately after it.
3. **Given** all stages complete, **When** results are returned, **Then** the
   validation data contains the ByT5-vs-TAHIMIK and MrT5-vs-TAHIMIK families.
4. **Given** CPU execution, **When** profiling completes, **Then** latency is
   returned and GPU-memory values are explicitly unavailable rather than made
   up.

### Edge Cases

- A duplicate `input/reference` pair is evaluated once; conflicting references
  for the same input are rejected using the existing gold-data validation rule.
- A job failure records its failed state and safe error detail to disk; polling
  never reports it as complete.
- A server restart reloads completed/failed job metadata from disk; an
  interrupted running job becomes failed with an interruption reason.
- Byte-position telemetry is omitted for ByT5 because it deletes no bytes.
- Frontend file parsing may accept several formats, but evaluation requires a
  non-empty reference for every submitted row.

## Requirements

### Functional Requirements

- **FR-001**: The API MUST retain `GET /health`, `POST /normalize`, and
  `POST /normalize/batch` without duplicated handler logic.
- **FR-002**: `POST /compare` MUST run all three variants for one input and
  fail with HTTP 503 if any variant is unavailable.
- **FR-003**: `POST /compare/batch` MUST run all three variants for one to 50
  non-empty inputs and fail atomically when any variant is unavailable.
- **FR-004**: All live model results MUST originate from one shared inference
  helper that decodes generated output and attaches observed telemetry.
- **FR-005**: Compressed-model telemetry MUST include compression mode, input,
  retained, and deleted byte counts, deletion rate, byte positions, latency,
  and adaptive noise/target values where applicable.
- **FR-006**: `POST /evaluate` MUST accept 1–2,000 labelled examples, reject
  missing/blank references, and create one combined background job.
- **FR-007**: Only one evaluation job MAY be active. A concurrent request MUST
  return HTTP 409; a missing model MUST return HTTP 503 before the job starts.
- **FR-008**: `GET /evaluate/{job_id}` MUST expose queued, running, completed,
  or failed status, stage progress, and final results when completed.
- **FR-009**: One completed job MUST calculate all four accuracy metrics and
  return aggregate values, per-example outputs, and per-example score vectors.
- **FR-010**: One completed job MUST profile all variants on the uploaded input
  set using batch size 1, five warm-ups, and 20 measured runs.
- **FR-011**: One completed job MUST produce the existing two model-pair
  statistical families from its accuracy and efficiency vectors.
- **FR-012**: Job state/results MUST be written below `outputs/dashboard-jobs/`
  and MUST remain excluded from version control.
- **FR-013**: The frontend MUST call APIs through one typed client and MUST NOT
  compute live normalization, noise, deletion, accuracy, or efficiency values
  in browser code.
- **FR-014**: The frontend MUST permit unlabelled single-sentence and batch
  inference, render their Model Outputs and Compression & Gating Information,
  and MUST NOT request evaluation, profiling, or statistics for those inputs.
- **FR-015**: The frontend MUST provide a separate **Normalization Evaluation**
  workflow where a user reviews a parsed labelled held-out test set and
  explicitly starts evaluation/profiling; it MUST poll a single job ID and map
  completed-job values to Benchmark, Efficiency, and Validation panels.

### Key Entities

- **EvaluationExample**: One uploaded `input` plus its required `reference`.
- **ModelTelemetry**: Observed inference compression/noise facts for one model
  and one sentence.
- **EvaluationJob**: Persisted unit of work with ID, state, input snapshot,
  progress, timestamps, and either result or failure detail.
- **EvaluationResult**: Combined outputs from accuracy, profiling, and
  statistical comparison stages.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Mocked API contract tests prove compare and evaluate routes do
  not return partial three-model results when a requested model is unavailable.
- **SC-002**: A fixture batch completes through one job ID and produces all
  required accuracy, profiling, and validation payload sections.
- **SC-003**: Profiling tests verify exactly five warm-ups and 20 measured
  passes with batch size 1.
- **SC-004**: Frontend tests prove unlabelled single/batch inputs render Model
  Outputs and telemetry without requesting aggregate work, and evaluation
  panels render only completed-job API values.
- **SC-005**: `python -m pytest tests/ -v`, `npm run typecheck`, and
  `npm run build` pass after implementation.

## Assumptions

- The existing evaluation metric/statistical implementations remain the
  authority for formulas and comparison families.
- The final held-out dataset will fit within the selected 2,000-row dashboard
  limit; larger official experiments remain supported by offline scripts.
- The local API server has write permission for its ignored output directory.
- Checkpoints and the tokenizer are available before any live inference run.
