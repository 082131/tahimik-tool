# Live Dashboard Integration Design

## Goal

Replace the mock-only `thesis_UI_v2` dashboard with a maintainable frontend in
`frontend/` whose normalization, all-model comparison, batch processing, and
compression inspector are backed by the TAHIMIK FastAPI service. The work must
not alter training behavior, model hyperparameters, checkpoints, or results.

## Constraints

- Do not commit or push any work.
- Preserve the existing public `POST /normalize`, `POST /normalize/batch`, and
  `GET /health` contracts for existing callers.
- The UI must not show invented outputs, compression values, benchmark values,
  accuracy scores, or statistical conclusions.
- A missing checkpoint remains a clear, actionable `503` response.
- Runtime telemetry must come from the same hard-deletion inference path used
  to create the displayed output.
- Existing uncommitted work outside this feature remains untouched.

## Architecture

The React application will use one typed API client. It will request health
data on load, submit a single sentence to a comparison route, and submit a
list of sentences to a batch comparison route. The dashboard components will
render only the returned data and will use a dedicated unavailable/empty state
when the API cannot produce data.

The FastAPI service will retain its existing single-model routes and add a
comparison service layer. That layer loads each requested available variant,
runs its real `generate` path, and returns normalized text, measured latency,
and inference telemetry. A small model-facing telemetry interface will keep
the existing `generate` method stable: each wrapper receives a
`generate_with_telemetry` method used only by the API. Its output sequence is
identical to `generate`; it adds only values observed while preparing the
encoder state.

Evaluation is separate from interactive inference. The API accepts caller
provided pairs of noisy input and reference normalization and uses the existing
metric implementations. It reports no aggregate result until references are
provided. This keeps the UI useful without falsely presenting the dashboard's
former sample data as experiment evidence.

## API Contract

### Existing routes retained

- `GET /health`: exposes device and the loaded/available state of ByT5, MrT5,
  and TAHIMIK.
- `POST /normalize`: performs one model's normalization.
- `POST /normalize/batch`: performs a true batch for one model.

The two normalization responses gain an optional `telemetry` object without
renaming or removing existing fields.

### `POST /compare`

Input:

```json
{
  "text": "Sanaol nlng tlga sa inyo.",
  "models": ["byt5", "mrt5", "tahimik"],
  "max_length": 512,
  "num_beams": 4
}
```

Response contains the input and one result per requested model. Each result
contains `model`, `label`, `normalized`, `inference_time_ms`, and telemetry.
Unavailable models are returned as a per-model unavailable result rather than
silently omitted; malformed model names remain a `400` request error.

### `POST /compare/batch`

Accepts up to 50 non-empty texts and the same model/generation options. It
runs one true batch per model and returns rows indexed to the submitted texts.
The frontend can parse pasted lines or a local text/CSV file, but file content
never leaves the browser except as these text values.

### `POST /evaluate`

Accepts up to 50 `{ "input": string, "reference": string }` examples and
one or more models. For each available model it returns the actual normalized
outputs, GLEU+, chrF, error-reduction rate, alphabetic-word accuracy, and
per-example inference time. It returns a clear validation error when no
reference is supplied. It does not claim a statistical significance result:
that requires a controlled dataset and the repository's full experiment
protocol, not an ad-hoc dashboard request.

## Telemetry Contract

Every inference result reports:

- `input_token_count`: non-padding ByT5 input positions supplied to the model;
- `kept_token_count`, `deleted_token_count`, and `deletion_rate`;
- `input_utf8_byte_count`: user-visible UTF-8 bytes in the submitted string.

For ByT5, all input tokens are kept and deletion rate is zero. For MrT5, the
result also identifies `compression_mode: "fixed"`. For TAHIMIK, the result
identifies `compression_mode: "adaptive"` and adds the predicted
`noise_score`, `target_deletion_rate`, `adaptive_coefficient`, and
`noise_average`. Values unrelated to a model are absent rather than shown as
made-up values.

The API will not call a second forward pass merely to collect telemetry. The
compressed model wrappers will share one preparation helper between
`generate` and `generate_with_telemetry`, preventing output/telemetry drift.

## Frontend Design

The replacement keeps the new UI's visual direction and responsive navigation
but removes its Figma export structure and hard-coded `BATCH` fixture.

- Landing and dashboard shell become focused app components.
- A normalization workspace accepts one input, lets the user choose one or all
  models, and shows returned outputs and API state.
- Pasted lines or an optional local file become batch input. The frontend calls
  the batch comparison route and renders only returned rows.
- Compression cards use returned telemetry; selecting a result shows details
  only if that model supports them.
- The benchmark/efficiency/validation area is replaced by an evaluation form.
  It accepts reference-labeled examples and shows real metrics after an
  evaluation response; otherwise it explains what is required.
- The frontend retains explanatory project/about content, corrected to ByT5
  terminology and actual Group 9 data already present in the repository.
- A typed API module, domain types, and small presentational components replace
  the current multi-hundred-line page and unused generic component export.

## Error Handling

The client maps API errors to four clear states: API unreachable, invalid
input, model unavailable/no checkpoint, and evaluation reference required.
Individual unavailable comparison variants remain visible so users can tell
the difference between a missing checkpoint and an absent result. Aborting or
replacing an in-flight request prevents stale responses from overwriting newer
input.

## Verification

- Backend unit tests cover telemetry for every model type, single comparison,
  true-batch comparison, unavailable model handling, validation, and backward
  compatibility of the existing routes.
- Model tests prove `generate_with_telemetry` retains the same generated
  sequence as `generate` using the existing lightweight model doubles.
- Frontend typecheck and production build pass.
- A development run is checked against the API health state. With no
  checkpoints, the resulting unavailable UI is the expected outcome.

## Documentation Deliverable

After implementation, `docs/backend-dashboard-additions.md` documents every
new backend route, request and response shapes, telemetry definitions,
availability behavior, and the frontend component that consumes each field.
