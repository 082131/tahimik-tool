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
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Make `configs` and `src` importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
        "model_module": "src.models.fixed_compression_byt5",
        "model_class": "FixedCompressionByT5",
    },
    "tahimik": {
        "label": "TAHIMIK",
        "config_module": "configs.tahimik_config",
        "config_class": "TAHIMIKConfig",
        "model_module": "src.models.noise_adaptive_byt5",
        "model_class": "NoiseAdaptiveByT5",
    },
}

DEFAULT_MODEL = "tahimik"

# Loaded on first use, cached for the process lifetime.
_loaded: Dict[str, Tuple[Any, Any]] = {}
device: Optional[torch.device] = None


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
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
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


# ── Inference ───────────────────────────────────────────────────────────
@torch.no_grad()
def normalize_text(
    text: str,
    model_name: str,
    max_length: int = 512,
    num_beams: int = 4,
) -> Tuple[str, float]:
    """Normalize one sentence. Returns (normalized_text, elapsed_ms)."""
    tokenizer, model = get_model(model_name)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding=True,
    ).to(device)

    start = time.perf_counter()
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_beams=num_beams,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    return tokenizer.decode(outputs[0], skip_special_tokens=True), elapsed_ms


# ── Endpoints ───────────────────────────────────────────────────────────
@app.get("/health")
async def health():
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
async def normalize(req: NormalizeRequest):
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
async def normalize_batch(req: BatchNormalizeRequest):
    results = []
    total_start = time.perf_counter()

    for text in req.texts:
        normalized, time_ms = normalize_text(
            text,
            model_name=req.model,
            max_length=req.max_length,
            num_beams=req.num_beams,
        )
        results.append(
            NormalizeResponse(
                input=text,
                normalized=normalized,
                model=req.model,
                inference_time_ms=round(time_ms, 2),
            )
        )

    total_ms = (time.perf_counter() - total_start) * 1000
    return BatchNormalizeResponse(results=results, total_time_ms=round(total_ms, 2))


# ── Run directly ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    # Port 8000 is often taken by other local services, so default to 8100.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8100)))
