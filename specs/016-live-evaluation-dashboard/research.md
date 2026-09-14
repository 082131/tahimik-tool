# Research: Live Evaluation Dashboard

## Decision: One combined persisted evaluation job

**Decision**: `POST /evaluate` creates one background job that runs accuracy,
profiling, and statistical validation in sequence. `GET /evaluate/{job_id}` is
the only progress/result read route.

**Rationale**: One uploaded batch must remain the common source for SOP/RQ1–4.
One job ID prevents the frontend from combining accuracy from one upload with
profiling from another. Persistence prevents a long result being lost on a
local server restart.

**Alternatives rejected**:

- Independent public `/evaluate` and `/profile` jobs: exposes two IDs and lets
  frontend sequencing accidentally mix inputs.
- Synchronous evaluation: unsuitable for up to 2,000 rows and 20 profiling
  passes.

## Decision: Strict all-three-model comparisons

**Decision**: Compare/evaluate routes require ByT5, MrT5, and TAHIMIK. Any
unavailable model returns HTTP 503 before partial results are produced.

**Rationale**: The dashboard represents a three-variant study. A partial
comparison may look complete during QA or presentation.

## Decision: Uploaded labelled data, bounded to 2,000 rows

**Decision**: The dashboard sends `{input, reference}` examples in its request;
the backend accepts at most 2,000 rows and one active job.

**Rationale**: Uploaded data supports the interactive workflow without exposing
server paths. The limit covers the expected held-out split while protecting a
local GPU from concurrent long jobs.

## Decision: Preserve Chapter 3 profiling protocol

**Decision**: Profiling iterates the uploaded dataset with internal batch size
1, discards five complete warm-up passes, then executes 20 measured passes.

**Rationale**: Batch size 1 means one sentence is processed at a time; it does
not mean the user must upload one sentence or repeat an upload 20 times. The
backend performs the repetitions automatically.

## Decision: Byte-accurate telemetry

**Decision**: Telemetry includes compression summaries plus exact original byte
positions retained/deleted by MrT5 and TAHIMIK.

**Rationale**: The frontend must not estimate noise or pruning in JavaScript.
Byte positions are sufficient to create an auditable inspection view and avoid
calling byte-level evidence word-level deletion.

## Decision: No eligibility badge

**Decision**: Do not add a new dashboard eligibility badge.

**Rationale**: The existing presentation UI already provides sufficient context.
This decision does not change provenance recorded in saved results.
