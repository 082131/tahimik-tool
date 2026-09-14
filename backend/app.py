# =============================================================================
# TAHIMIK Backend API — Normalization Inference Server
#
# Serves the three variants the study compares under identical control
# variables:
#
#   byt5     — ByT5 baseline, no compression      (accuracy ceiling)
#   mrt5     — MrT5, fixed-rate byte deletion     (efficiency baseline)
#   tahimik  — Proposed: noise-adaptive deletion  (our contribution)
#
# Each is loaded lazily from the checkpoint scripts/train.py writes, so the
# server starts even when nothing has been trained yet and reports exactly
# which checkpoints are missing.
#
# Usage:
#   python backend/app.py
#   # or:
#   uvicorn backend.app:app --host 0.0.0.0 --port 8100 --reload
#
# Endpoints:
#   GET  /health          — Which variants are available / loaded
#   POST /normalize       — Normalize a single sentence
#   POST /normalize/batch — Normalize multiple sentences
# =============================================================================

import importlib
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Make `configs` and `src` importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import NormalizationDataset
from src.evaluation.efficiency import EfficiencyBenchmark
from src.evaluation.metrics import NormalizationMetrics
from src.evaluation.statistical_tests import StatisticalAnalysis
from src.training.trainer import validate_checkpoint_architecture


# ── Variant registry ────────────────────────────────────────────────────
# Maps the name the frontend sends to the config and model class that
# implement that variant.
VARIANTS: Dict[str, Dict[str, str]] = {
    "byt5": {
        "label": "ByT5",
        "config_module": "configs.byt5_config",
        "config_class": "ByT5Config",
        "model_module": "src.models.byt5_baseline",
        "model_class": "ByT5Baseline",
    },
    "mrt5": {
        "label": "MrT5",
        "config_module": "configs.mrt5_config",
        "config_class": "MrT5Config",
        "model_module": "src.models.fixed_compression",
        "model_class": "FixedCompressionByT5",
    },
    "tahimik": {
        "label": "TAHIMIK",
        "config_module": "configs.tahimik_config",
        "config_class": "TAHIMIKConfig",
        "model_module": "src.models.noise_adaptive",
        "model_class": "NoiseAdaptiveByT5",
    },
}

DEFAULT_MODEL = "tahimik"

# Loaded on first use, cached for the process lifetime.
_loaded: Dict[str, Tuple[Any, Any]] = {}
device: Optional[torch.device] = None
EVALUATION_JOBS: Dict[str, Dict[str, Any]] = {}
JOB_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dashboard-jobs"


def load_config(name: str):
    """Instantiate the dataclass config for a variant."""
    entry = VARIANTS[name]
    module = importlib.import_module(entry["config_module"])
    return getattr(module, entry["config_class"])()


def checkpoint_path(name: str) -> Path:
    """
    Resolve which checkpoint a variant should load. Stage 2 (gold
    fine-tuning) is preferred; stage 1 is accepted so a partially trained
    model can still be demonstrated.

    Override per variant with e.g. TAHIMIK_CHECKPOINT=/path/to/model.pt
    """
    override = os.environ.get(f"{name.upper()}_CHECKPOINT")
    if override:
        return Path(override)

    config = load_config(name)
    base = Path(config.checkpoint_dir)
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    variant_dir = base / config.variant_name

    for stage in ("best_stage2.pt", "best_stage1.pt"):
        candidate = variant_dir / stage
        if candidate.is_file():
            return candidate

    # Nothing on disk — return the stage 2 path so the error can name it.
    return variant_dir / "best_stage2.pt"


def is_available(name: str) -> bool:
    """True when this variant has a checkpoint that can be loaded."""
    return name in VARIANTS and checkpoint_path(name).is_file()


def get_model(name: str) -> Tuple[Any, Any]:
    """
    Return (tokenizer, model), loading on first use. Raises HTTPException
    with an actionable message when the variant cannot be served.
    """
    if name not in VARIANTS:
        raise HTTPException(
            400, f"Unknown model '{name}'. Available: {', '.join(VARIANTS)}."
        )

    if name in _loaded:
        return _loaded[name]

    entry = VARIANTS[name]
    path = checkpoint_path(name)

    if not path.is_file():
        raise HTTPException(
            503,
            f"{entry['label']} has not been trained yet — no checkpoint at "
            f"{path}. Train it with: python scripts/train.py "
            f"--variant {name} --gold_data data/gold.csv",
        )

    print(f"Loading '{name}' ({entry['label']})")
    start = time.time()

    config = load_config(name)

    # ByT5 uses a byte-level tokenizer, but transformers still resolves it
    # through the hub on first use.
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    except Exception as exc:
        raise HTTPException(
            503,
            f"Could not load the '{config.model_name}' tokenizer. It is "
            f"fetched from HuggingFace on first use, so this machine needs "
            f"network access once, or a warm HF cache. ({exc})",
        ) from exc

    module = importlib.import_module(entry["model_module"])
    model = getattr(module, entry["model_class"])(config)

    checkpoint = torch.load(path, map_location="cpu")
    try:
        validate_checkpoint_architecture(
            checkpoint,
            model,
            expected_model_name=config.model_name,
        )
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            503,
            f"Checkpoint for {entry['label']} is incompatible with "
            f"'{config.model_name}': {exc}",
        ) from exc
    model.to(device)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(
        f"Loaded '{name}' in {time.time() - start:.1f}s "
        f"({params:,} parameters, stage={checkpoint.get('stage', 'unknown')})"
    )

    _loaded[name] = (tokenizer, model)
    return _loaded[name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ready = [n for n in VARIANTS if is_available(n)]
    missing = [n for n in VARIANTS if n not in ready]
    print(f"Trained and available: {', '.join(ready) if ready else 'none'}")
    if missing:
        print(f"Not yet trained: {', '.join(missing)}")

    # Warm the default variant so the first request is not slowed by a load.
    if is_available(DEFAULT_MODEL):
        try:
            get_model(DEFAULT_MODEL)
        except HTTPException as exc:
            print(f"Preload of '{DEFAULT_MODEL}' skipped: {exc.detail}")

    yield


# ── FastAPI app ─────────────────────────────────────────────────────────
app = FastAPI(
    title="TAHIMIK Normalization API",
    description="Filipino/Taglish social media text normalization",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ─────────────────────────────────────────────────────────────
class NormalizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2048)
    model: str = Field(DEFAULT_MODEL, description="byt5 | mrt5 | tahimik")
    max_length: int = Field(512, ge=16, le=2048)
    num_beams: int = Field(4, ge=1, le=10)


class NormalizeResponse(BaseModel):
    input: str
    normalized: str
    model: str
    inference_time_ms: float


class BatchNormalizeRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=50)
    model: str = Field(DEFAULT_MODEL)
    max_length: int = Field(512, ge=16, le=2048)
    num_beams: int = Field(4, ge=1, le=10)


class BatchNormalizeResponse(BaseModel):
    results: List[NormalizeResponse]
    total_time_ms: float


class CompareRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2048)
    max_length: int = Field(512, ge=16, le=2048)
    num_beams: int = Field(4, ge=1, le=10)


class CompareResult(BaseModel):
    model: str
    label: str
    normalized: str
    inference_time_ms: float
    telemetry: Dict[str, Any]


class CompareResponse(BaseModel):
    input: str
    results: List[CompareResult]


class CompareBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=50)
    max_length: int = Field(512, ge=16, le=2048)
    num_beams: int = Field(4, ge=1, le=10)


class CompareBatchResponse(BaseModel):
    results: List[CompareResponse]


class EvaluationExample(BaseModel):
    input: str = Field(..., min_length=1, max_length=2048)
    reference: str = Field(..., min_length=1, max_length=2048)


class EvaluateRequest(BaseModel):
    examples: List[EvaluationExample] = Field(..., min_length=1, max_length=2000)
    max_length: int = Field(512, ge=16, le=2048)
    num_beams: int = Field(4, ge=1, le=10)


class EvaluationJobResponse(BaseModel):
    job_id: str
    status: str
    stage: str
    total_examples: int


# ── Inference ───────────────────────────────────────────────────────────
@torch.no_grad()
def normalize_texts(
    texts: List[str],
    model_name: str,
    max_length: int = 512,
    num_beams: int = 4,
) -> Tuple[List[str], float]:
    """Normalize a batch of sentences. Returns (normalized_texts, elapsed_ms)."""
    if not texts:
        raise HTTPException(status_code=400, detail="Text list cannot be empty")

    tokenizer, model = get_model(model_name)

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding=True,
    )
    if device is not None:
        inputs = {
            k: v.to(device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

    start = time.perf_counter()
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_beams=num_beams,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded, elapsed_ms


@torch.no_grad()
def normalize_text(
    text: str,
    model_name: str,
    max_length: int = 512,
    num_beams: int = 4,
) -> Tuple[str, float]:
    """Normalize one sentence. Returns (normalized_text, elapsed_ms)."""
    decoded_list, elapsed_ms = normalize_texts(
        [text],
        model_name=model_name,
        max_length=max_length,
        num_beams=num_beams,
    )
    return decoded_list[0], elapsed_ms


@torch.no_grad()
def run_model_with_telemetry(
    text: str,
    model_name: str,
    max_length: int = 512,
    num_beams: int = 4,
) -> Tuple[str, Dict[str, Any], float]:
    """Run one variant and return its decoded output with observed telemetry."""
    tokenizer, model = get_model(model_name)
    inputs = tokenizer(
        [text],
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding=True,
    )
    if device is not None:
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    start = time.perf_counter()
    generated, model_telemetry = model.generate_with_telemetry(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_beams=num_beams,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    if len(decoded) != 1 or len(model_telemetry) != 1:
        raise HTTPException(
            status_code=502,
            detail=f"Model '{model_name}' returned an invalid single-input result.",
        )

    telemetry = dict(model_telemetry[0])
    byte_count = len(text.encode("utf-8"))
    telemetry["input_byte_count"] = byte_count
    if telemetry.get("compression_mode") == "none":
        telemetry["retained_byte_count"] = byte_count
        telemetry["deleted_byte_count"] = 0
        telemetry["retained_byte_positions"] = list(range(byte_count))
        telemetry["deleted_byte_positions"] = []
    return decoded[0], telemetry, elapsed_ms


def require_all_variants_available() -> None:
    """Reject study comparisons unless every configured variant is available."""
    missing = [name for name in VARIANTS if not is_available(name)]
    if missing:
        labels = ", ".join(VARIANTS[name]["label"] for name in missing)
        raise HTTPException(
            status_code=503,
            detail=f"Comparison requires all three checkpoints; unavailable: {labels}.",
        )


def persist_evaluation_job(job: Dict[str, Any]) -> None:
    JOB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (JOB_OUTPUT_DIR / f"{job['job_id']}.json").write_text(json.dumps(job, indent=2), encoding="utf-8")


def execute_evaluation_job(job_id: str) -> None:
    """Run the complete labelled accuracy, profiling, and statistics workflow."""
    job = EVALUATION_JOBS[job_id]
    try:
        job.update(status="running", stage="evaluation")
        persist_evaluation_job(job)
        inputs = [example["input"] for example in job["examples"]]
        references = [example["reference"] for example in job["examples"]]
        metrics = NormalizationMetrics()
        outputs_by_model: Dict[str, List[str]] = {}
        score_vectors: Dict[str, Dict[str, List[float]]] = {}
        efficiency: Dict[str, Dict[str, Any]] = {}

        for name in VARIANTS:
            outputs = [run_model_with_telemetry(text, name, job["max_length"], job["num_beams"])[0] for text in inputs]
            outputs_by_model[name] = outputs
            score_vectors[name] = metrics.compute_per_sentence(outputs, references, inputs)
            job["completed_examples"] = len(inputs)
            persist_evaluation_job(job)

        job["stage"] = "profiling"
        persist_evaluation_job(job)
        runtime_device = device or torch.device("cpu")
        for name in VARIANTS:
            tokenizer, model = get_model(name)
            dataset = NormalizationDataset(inputs, references, tokenizer)
            profile = EfficiencyBenchmark(model, tokenizer, runtime_device, num_beams=job["num_beams"]).benchmark(dataset, batch_size=1)
            efficiency[name] = profile
            score_vectors[name]["inference_time"] = profile["per_sentence_time_seconds"]

        job["stage"] = "statistics"
        persist_evaluation_job(job)
        statistics = StatisticalAnalysis().run_full_comparison(
            score_vectors,
            {name: efficiency[name]["peak_gpu_memory_runs_mb"] for name in VARIANTS},
        )
        job["result"] = {
            "examples": [
                {"input": source, "reference": reference, "outputs": {name: outputs_by_model[name][index] for name in VARIANTS}, "scores": {name: {metric: score_vectors[name][metric][index] for metric in ("gleu_plus", "chrf", "err", "alpha_word_accuracy")} for name in VARIANTS}}
                for index, (source, reference) in enumerate(zip(inputs, references))
            ],
            "metrics": {name: metrics.compute_all(outputs_by_model[name], references, inputs) for name in VARIANTS},
            "efficiency": efficiency,
            "statistics": statistics,
        }
        job.update(status="completed", stage="complete")
    except Exception as exc:
        job.update(status="failed", stage="failed", error=f"Evaluation failed: {exc}")
    persist_evaluation_job(job)


# ── Endpoints ───────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device) if device else None,
        "default_model": DEFAULT_MODEL,
        "models": {
            name: {
                "label": entry["label"],
                "checkpoint": str(checkpoint_path(name)),
                "available": is_available(name),
                "loaded": name in _loaded,
            }
            for name, entry in VARIANTS.items()
        },
    }


@app.post("/normalize", response_model=NormalizeResponse)
def normalize(req: NormalizeRequest):
    normalized, time_ms = normalize_text(
        req.text,
        model_name=req.model,
        max_length=req.max_length,
        num_beams=req.num_beams,
    )
    return NormalizeResponse(
        input=req.text,
        normalized=normalized,
        model=req.model,
        inference_time_ms=round(time_ms, 2),
    )


@app.post("/normalize/batch", response_model=BatchNormalizeResponse)
def normalize_batch(req: BatchNormalizeRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="Text list cannot be empty")

    normalized_list, total_ms = normalize_texts(
        req.texts,
        model_name=req.model,
        max_length=req.max_length,
        num_beams=req.num_beams,
    )
    per_sample_ms = round(total_ms / len(req.texts), 2) if req.texts else 0.0
    results = [
        NormalizeResponse(
            input=text,
            normalized=norm,
            model=req.model,
            inference_time_ms=per_sample_ms,
        )
        for text, norm in zip(req.texts, normalized_list)
    ]
    return BatchNormalizeResponse(results=results, total_time_ms=round(total_ms, 2))


@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    """Compare all three study variants for one unlabelled inference input."""
    require_all_variants_available()
    results = []
    for name, entry in VARIANTS.items():
        normalized, telemetry, elapsed_ms = run_model_with_telemetry(
            req.text,
            name,
            max_length=req.max_length,
            num_beams=req.num_beams,
        )
        results.append(
            CompareResult(
                model=name,
                label=entry["label"],
                normalized=normalized,
                inference_time_ms=round(elapsed_ms, 2),
                telemetry=telemetry,
            )
        )
    return CompareResponse(input=req.text, results=results)


@app.post("/compare/batch", response_model=CompareBatchResponse)
def compare_batch(req: CompareBatchRequest):
    """Compare all variants for each unlabelled input without evaluation work."""
    require_all_variants_available()
    comparisons = []
    for text in req.texts:
        results = []
        for name, entry in VARIANTS.items():
            normalized, telemetry, elapsed_ms = run_model_with_telemetry(
                text,
                name,
                max_length=req.max_length,
                num_beams=req.num_beams,
            )
            results.append(
                CompareResult(
                    model=name,
                    label=entry["label"],
                    normalized=normalized,
                    inference_time_ms=round(elapsed_ms, 2),
                    telemetry=telemetry,
                )
            )
        comparisons.append(CompareResponse(input=text, results=results))
    return CompareBatchResponse(results=comparisons)


@app.post("/evaluate", response_model=EvaluationJobResponse, status_code=202)
def evaluate(req: EvaluateRequest, background_tasks: BackgroundTasks):
    active = next((job for job in EVALUATION_JOBS.values() if job["status"] in {"queued", "running"}), None)
    if active:
        raise HTTPException(status_code=409, detail=f"Evaluation job '{active['job_id']}' is already active.")
    require_all_variants_available()
    job = {
        "job_id": str(uuid.uuid4()), "status": "queued", "stage": "queued",
        "total_examples": len(req.examples), "completed_examples": 0,
        "examples": [example.model_dump() for example in req.examples],
        "max_length": req.max_length, "num_beams": req.num_beams,
    }
    EVALUATION_JOBS[job["job_id"]] = job
    persist_evaluation_job(job)
    background_tasks.add_task(execute_evaluation_job, job["job_id"])
    return EvaluationJobResponse(job_id=job["job_id"], status="queued", stage="queued", total_examples=job["total_examples"])


@app.get("/evaluate/{job_id}")
def evaluate_status(job_id: str):
    job = EVALUATION_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Evaluation job '{job_id}' was not found.")
    return job


# ── Run directly ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    # Port 8000 is often taken by other local services, so default to 8100.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8100)))
