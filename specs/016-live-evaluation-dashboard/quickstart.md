# Quickstart: Live Evaluation Dashboard

1. Start the API after all three checkpoints are available:

```powershell
uvicorn backend.app:app --host 0.0.0.0 --port 8100
```

2. Start the frontend with `VITE_API_URL=http://localhost:8100`.

3. In Text Normalization, use one sentence for live comparison only.

4. Upload a CSV, TSV, JSON, or JSONL file with an input column and a reference
column. Confirm the parser detected both before selecting **Run Evaluation &
Profile**.

5. Poll the job status until `completed`. Display its API values in Benchmark,
Runtime Efficiency, and Statistical Validation.

6. Inspect the generated ignored JSON below `outputs/dashboard-jobs/` when QA
requires an artifact outside the browser.

Expected failure states: 503 means a required checkpoint is unavailable; 409
means another job is active; 422 means uploaded data is invalid.
