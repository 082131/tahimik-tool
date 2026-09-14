# Dashboard API Contract

Base URL is `VITE_API_URL`, defaulting to `http://localhost:8100`.

## `POST /compare`

Request:

```json
{ "text": "Sanaol nlng", "num_beams": 4 }
```

Success (200):

```json
{
  "input": "Sanaol nlng",
  "results": [
    {"model":"byt5","label":"ByT5","normalized":"...","inference_time_ms":0.0,"telemetry":{"compression_mode":"none","input_byte_count":12,"retained_byte_count":12,"deleted_byte_count":0,"deletion_rate":0.0,"retained_byte_positions":[],"deleted_byte_positions":[],"noise_score":null,"target_deletion_rate":null}},
    {"model":"mrt5","label":"MrT5","normalized":"...","inference_time_ms":0.0,"telemetry":{}},
    {"model":"tahimik","label":"TAHIMIK","normalized":"...","inference_time_ms":0.0,"telemetry":{}}
  ]
}
```

Errors: 422 invalid text; 503 any required checkpoint unavailable.

## `POST /compare/batch`

Request: `{ "texts": ["first", "second"], "num_beams": 4 }`. The API
accepts 1–50 non-empty texts. Success returns one comparison object per input.
Errors mirror `/compare`; no partial result is returned.

## `POST /evaluate`

Request:

```json
{
  "examples": [
    {"input":"Sanaol nlng", "reference":"Sana all na lang"}
  ],
  "num_beams": 4
}
```

Success (202):

```json
{"job_id":"uuid", "status":"queued", "stage":"validation", "total_examples":1}
```

Errors: 409 another evaluation job active; 422 invalid/missing reference or
more than 2,000 rows; 503 any model checkpoint unavailable.

## `GET /evaluate/{job_id}`

Returns job state/progress. A completed response includes:

```json
{
  "job_id":"uuid",
  "status":"completed",
  "stage":"complete",
  "completed_examples":1,
  "total_examples":1,
  "result": {
    "examples": [{"input":"...","reference":"...","outputs":{"byt5":"...","mrt5":"...","tahimik":"..."},"scores":{"byt5":{},"mrt5":{},"tahimik":{}}}],
    "metrics": {"byt5":{},"mrt5":{},"tahimik":{}},
    "efficiency": {"byt5":{},"mrt5":{},"tahimik":{}},
    "statistics": {}
  }
}
```

Returns 404 for an unknown job ID. Failed jobs return 200 with `status: failed`
and a safe `error` field so the frontend can render the failure without treating
it as successful data.
