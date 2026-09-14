# Live Dashboard Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the mock dashboard with a cleaned React interface backed by live TAHIMIK inference, comparison, batch, and evaluation APIs.

**Architecture:** Keep existing single-model endpoints stable while introducing a backend inference service that returns generated text and observed encoder-compression telemetry. The frontend uses a typed API client and focused dashboard components, never static model results. Evaluation remains request-driven and only calculates accuracy when the caller provides reference normalizations.

**Tech Stack:** Python, FastAPI, Pydantic, PyTorch, React 18, TypeScript, Vite, Tailwind CSS.

**Spec:** `docs/superpowers/specs/2026-09-11-live-dashboard-integration-design.md`

## Global Constraints

- Do not commit or push any work.
- Keep `POST /normalize`, `POST /normalize/batch`, and `GET /health` backwards compatible.
- Do not change training behavior, model hyperparameters, or checkpoint files.
- Do not present static sample outputs or fabricated evaluation data.
- Report compression telemetry from the exact hard-deletion path used for inference.
- Preserve existing unrelated working-tree changes.

---

### Task 1: Extract Model Inference Telemetry

**Files:**
- Modify: `src/models/byt5_baseline.py`
- Modify: `src/models/fixed_compression_byt5.py`
- Modify: `src/models/noise_adaptive_byt5.py`
- Test: `tests/test_model_forward.py`

**Interfaces:**
- Produces: `generate_with_telemetry(input_ids, attention_mask, max_length=1024, num_beams=4) -> tuple[torch.Tensor, list[dict[str, float | int | str]]]` on all three wrappers.
- Consumes: existing wrapper `generate` semantics and `DeleteGate` hard-deletion masks.

- [ ] **Step 1: Write failing model telemetry tests**

```python
def test_baseline_generate_with_telemetry_reports_no_deletion(model, inputs):
    output, telemetry = model.generate_with_telemetry(**inputs)
    assert output.shape[0] == inputs["input_ids"].shape[0]
    assert telemetry[0]["compression_mode"] == "none"
    assert telemetry[0]["deletion_rate"] == 0.0
    assert telemetry[0]["kept_token_count"] == telemetry[0]["input_token_count"]

def test_adaptive_generate_with_telemetry_reports_gate_values(model, inputs):
    _, telemetry = model.generate_with_telemetry(**inputs)
    assert telemetry[0]["compression_mode"] == "adaptive"
    assert 0.0 <= telemetry[0]["noise_score"] <= 1.0
    assert "target_deletion_rate" in telemetry[0]
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `python -m pytest tests/test_model_forward.py -k telemetry -v`

Expected: FAIL because `generate_with_telemetry` is undefined.

- [ ] **Step 3: Add baseline telemetry without changing baseline generation**

```python
@torch.no_grad()
def generate_with_telemetry(self, input_ids, attention_mask, max_length=1024, num_beams=4, **kwargs):
    output = self.generate(input_ids, attention_mask, max_length=max_length, num_beams=num_beams, **kwargs)
    counts = attention_mask.sum(dim=1).tolist()
    return output, [
        {"compression_mode": "none", "input_token_count": int(count),
         "kept_token_count": int(count), "deleted_token_count": 0,
         "deletion_rate": 0.0}
        for count in counts
    ]
```

- [ ] **Step 4: Refactor the compressed wrappers around one encoder-preparation helper**

```python
def _prepare_compressed_encoder(self, input_ids, attention_mask):
    # Execute the existing pre-gate layers, gate, hard deletion, post-gate
    # layers, and final layer norm exactly once.
    # Return BaseModelOutput, compressed attention mask, and batch telemetry.
    ...

def generate(self, input_ids, attention_mask, max_length=1024, num_beams=4, **kwargs):
    encoder_outputs, compressed_mask, _ = self._prepare_compressed_encoder(input_ids, attention_mask)
    return self.model.generate(encoder_outputs=encoder_outputs, attention_mask=compressed_mask,
                               max_length=max_length, num_beams=num_beams, early_stopping=True)

def generate_with_telemetry(self, input_ids, attention_mask, max_length=1024, num_beams=4, **kwargs):
    encoder_outputs, compressed_mask, telemetry = self._prepare_compressed_encoder(input_ids, attention_mask)
    output = self.model.generate(encoder_outputs=encoder_outputs, attention_mask=compressed_mask,
                                 max_length=max_length, num_beams=num_beams, early_stopping=True)
    return output, telemetry
```

The MrT5 helper emits `compression_mode: "fixed"`; TAHIMIK emits
`compression_mode: "adaptive"`, `noise_score`, `target_deletion_rate`,
`adaptive_coefficient`, and `noise_average` per item. Counts are derived from
the input attention mask and the hard-deletion compressed mask.

- [ ] **Step 5: Run model telemetry and regression tests**

Run: `python -m pytest tests/test_model_forward.py tests/test_delete_gate.py -v`

Expected: PASS.

### Task 2: Build Comparison and Evaluation API Services

**Files:**
- Create: `backend/services/inference.py`
- Create: `backend/services/evaluation.py`
- Modify: `backend/app.py`
- Modify: `tests/test_backend.py`

**Interfaces:**
- Consumes: model `generate_with_telemetry`, `NormalizationMetrics`, `get_model`, and `VARIANTS`.
- Produces: `run_model`, `compare_texts`, `evaluate_examples`, and the FastAPI response schemas/routes.

- [ ] **Step 1: Write failing route tests**

```python
def test_compare_returns_all_requested_models(client, mock_model_loader):
    response = client.post("/compare", json={"text": "raw", "models": ["byt5", "tahimik"]})
    assert response.status_code == 200
    assert [item["model"] for item in response.json()["results"]] == ["byt5", "tahimik"]
    assert response.json()["results"][0]["telemetry"]["deletion_rate"] == 0.0

def test_compare_preserves_unavailable_model(client, monkeypatch):
    monkeypatch.setattr("backend.app.is_available", lambda name: False)
    response = client.post("/compare", json={"text": "raw", "models": ["tahimik"]})
    assert response.json()["results"][0]["status"] == "unavailable"

def test_evaluate_requires_reference(client):
    response = client.post("/evaluate", json={"examples": [{"input": "raw"}], "models": ["byt5"]})
    assert response.status_code == 422
```

- [ ] **Step 2: Run the focused route tests and confirm failure**

Run: `python -m pytest tests/test_backend.py -k 'compare or evaluate' -v`

Expected: FAIL because the routes do not exist.

- [ ] **Step 3: Create the inference service and schemas**

```python
@dataclass
class ModelRun:
    model: str
    label: str
    normalized: str
    inference_time_ms: float
    telemetry: dict[str, Any]

def run_model(texts: list[str], model_name: str, max_length: int, num_beams: int) -> list[ModelRun]:
    tokenizer, model = get_model(model_name)
    encoded = tokenizer(texts, return_tensors="pt", max_length=1024, truncation=True, padding=True)
    outputs, telemetry = model.generate_with_telemetry(**move_to_device(encoded), max_length=max_length, num_beams=num_beams)
    # Decode once, merge the matching telemetry with UTF-8 byte count, and return rows.
```

`POST /compare` accepts one non-empty text and a deduplicated requested-model
list. `POST /compare/batch` accepts one to 50 non-empty texts. Unavailable
variants become result objects with `status: "unavailable"` and an actionable
message; invalid variant names produce a 400 error.

- [ ] **Step 4: Add caller-supplied evaluation**

```python
class EvaluationExample(BaseModel):
    input: str = Field(..., min_length=1, max_length=2048)
    reference: str = Field(..., min_length=1, max_length=2048)

def evaluate_examples(examples: list[EvaluationExample], models: list[str]) -> list[EvaluationResult]:
    # Run each available model, then call NormalizationMetrics.compute_all
    # with predictions, references, and noisy inputs.
    ...
```

Return per-model output rows, aggregate accuracy metrics, and average
inference time. Return no bootstrap significance or benchmark claim. Keep the
existing `/normalize` and `/normalize/batch` routes and add telemetry to their
result objects as optional fields.

- [ ] **Step 5: Run API regression tests**

Run: `python -m pytest tests/test_backend.py -v`

Expected: PASS, including existing single and true-batch tests.

### Task 3: Replace the Mock Frontend with Typed Live Components

**Files:**
- Modify: `frontend/package.json`
- Modify: `frontend/src/main.tsx`
- Modify: `frontend/src/styles/index.css`
- Create: `frontend/src/api/client.ts`
- Create: `frontend/src/api/types.ts`
- Create: `frontend/src/app/Dashboard.tsx`
- Create: `frontend/src/app/components/NormalizationWorkspace.tsx`
- Create: `frontend/src/app/components/ComparisonTable.tsx`
- Create: `frontend/src/app/components/CompressionInspector.tsx`
- Create: `frontend/src/app/components/EvaluationPanel.tsx`
- Create: `frontend/src/app/components/AppStatus.tsx`
- Delete after migration: old mock page/export files and unused generated UI files.

**Interfaces:**
- Consumes: `GET /health`, `POST /compare`, `POST /compare/batch`, and `POST /evaluate`.
- Produces: an accessible responsive dashboard that displays only API data.

- [ ] **Step 1: Replace static frontend types with API domain types**

```ts
export type ModelName = "byt5" | "mrt5" | "tahimik";
export type Telemetry = {
  compression_mode: "none" | "fixed" | "adaptive";
  input_token_count: number;
  kept_token_count: number;
  deleted_token_count: number;
  deletion_rate: number;
  input_utf8_byte_count: number;
  noise_score?: number;
  target_deletion_rate?: number;
  adaptive_coefficient?: number;
  noise_average?: number;
};
```

- [ ] **Step 2: Implement a single API client with explicit errors**

```ts
const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8100";

export async function compare(text: string, models: ModelName[], signal?: AbortSignal) {
  return request<CompareResponse>("/compare", { text, models }, signal);
}
```

The client reads FastAPI `detail` values, distinguishes network failures from
HTTP failures, and accepts an abort signal so stale requests cannot update the
screen.

- [ ] **Step 3: Build the live normalization and comparison workspace**

Use one controlled text area, model selection for one/all variants, a disabled
submit button while loading, and response-driven output cards. Render explicit
unavailable results, keep model order stable, and include a keyboard
Ctrl/Cmd+Enter shortcut.

- [ ] **Step 4: Build batch and compression views from returned data**

Parse pasted newline text and an optional local plain-text/CSV file into no
more than 50 non-empty rows. Submit only text values to `/compare/batch`.
Compute no compression values in React: pass the selected result's telemetry
to `CompressionInspector`, which renders a neutral ByT5 state, fixed MrT5
details, or adaptive TAHIMIK gate details.

- [ ] **Step 5: Replace static evaluation sections with the live evaluation form**

Accept CSV/text entries containing noisy input and reference normalization.
Do not render metrics before a successful `/evaluate` response. Explain that
formal statistical tests require the controlled experiment workflow and are
not an interactive dashboard result.

- [ ] **Step 6: Clean dependencies and generated artifacts**

Keep only dependencies imported by the new frontend. Remove unused Figma
imports, redundant UI components, placeholder member records, mock `BATCH`
data, and duplicate app/router entrypoints. Move retained images into
`frontend/src/assets/` with descriptive names.

- [ ] **Step 7: Run frontend verification**

Run: `npm run typecheck`

Expected: PASS.

Run: `npm run build`

Expected: PASS.

### Task 4: Document the Backend Additions

**Files:**
- Create: `docs/backend-dashboard-additions.md`
- Modify: `frontend/README.md`

**Interfaces:**
- Consumes: final FastAPI schemas and frontend API client.
- Produces: a beginner-readable integration guide.

- [ ] **Step 1: Write route and telemetry documentation**

Document the retained routes plus `/compare`, `/compare/batch`, and
`/evaluate`, including complete request/response examples, model availability
states, each telemetry definition, and a table mapping every frontend panel to
its source API field.

- [ ] **Step 2: Document local startup**

Add commands for running the FastAPI server and Vite app together, using
`VITE_API_URL`, handling a missing checkpoint, and preparing reference-labeled
evaluation examples.

- [ ] **Step 3: Run documentation link and final diff checks**

Run: `rg -n "TODO|TBD|BATCH|mock" frontend/src backend docs/backend-dashboard-additions.md`

Expected: no mock data or unresolved placeholders in production paths.

Run: `git diff --check`

Expected: no whitespace errors.

### Task 5: Full Verification

**Files:**
- Test: `tests/test_backend.py`
- Test: `tests/test_model_forward.py`

- [ ] **Step 1: Run the complete Python suite**

Run: `python -m pytest tests/ -v`

Expected: PASS.

- [ ] **Step 2: Re-run frontend checks from the frontend directory**

Run: `npm run typecheck`

Expected: PASS.

Run: `npm run build`

Expected: PASS.

- [ ] **Step 3: Verify scope and non-push constraint**

Run: `git status --short`

Expected: only feature files plus pre-existing user changes; do not stage,
commit, or push any file.
