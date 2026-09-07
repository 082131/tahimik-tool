import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer

# Make project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.byt5_config import ByT5Config
from configs.mrt5_config import MrT5Config
from configs.tahimik_config import TAHIMIKConfig
from src.models.byt5_baseline import ByT5Baseline
from src.models.fixed_compression_byt5 import FixedCompressionByT5
from src.models.noise_adaptive_byt5 import NoiseAdaptiveByT5

VARIANTS = {
    "byt5": (ByT5Config, ByT5Baseline),
    "mrt5": (MrT5Config, FixedCompressionByT5),
    "tahimik": (TAHIMIKConfig, NoiseAdaptiveByT5),
}


def load_variant_model(name: str, config, device: str) -> Tuple[Any, Any]:
    """Load model and tokenizer for preflight verification."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    _, model_cls = VARIANTS[name]
    model = model_cls(config)
    model.to(device)
    model.eval()
    return tokenizer, model


def run_preflight(
    device: str = "cuda",
    max_input_length: int = 1024,
    metadata_only: bool = False,
) -> Dict[str, Any]:
    """
    Executes hardware and configuration preflight for ByT5-Base.

    Returns:
        Dict reporting model identity, effective batch sizes, parameter counts,
        forward/generate validation, peak memory allocation, and recommendations.
    """
    resolved_device = device if torch.cuda.is_available() else "cpu"
    result: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "requested_device": device,
        "resolved_device": resolved_device,
        "cuda_available": torch.cuda.is_available(),
        "metadata_only": metadata_only,
        "all_models_base": True,
        "variants": {},
    }

    # Verify configs
    for name, (config_cls, _) in VARIANTS.items():
        cfg = config_cls()
        is_base = (cfg.model_name == "google/byt5-base")
        if not is_base:
            result["all_models_base"] = False

        s1_eff = cfg.stage1_batch_size * cfg.stage1_gradient_accumulation_steps
        s2_eff = cfg.stage2_batch_size * cfg.stage2_gradient_accumulation_steps

        result["variants"][name] = {
            "model_name": cfg.model_name,
            "is_byt5_base": is_base,
            "stage1_physical_batch": cfg.stage1_batch_size,
            "stage1_accumulation_steps": cfg.stage1_gradient_accumulation_steps,
            "stage1_effective_batch": s1_eff,
            "stage2_physical_batch": cfg.stage2_batch_size,
            "stage2_accumulation_steps": cfg.stage2_gradient_accumulation_steps,
            "stage2_effective_batch": s2_eff,
            "precision": getattr(cfg, "precision", "fp16"),
            "gradient_checkpointing": getattr(cfg, "gradient_checkpointing", True),
        }

    if not result["all_models_base"]:
        result["success"] = False
        result["error"] = "Not all model variants resolve to 'google/byt5-base'."
        return result

    if metadata_only:
        result["success"] = True
        return result

    # Execute runtime generation check
    sample_text = "kumusta po kayo lahat sa araw na ito"
    all_passed = True

    try:
        for name, (config_cls, _) in VARIANTS.items():
            cfg = config_cls()
            cfg.device = resolved_device
            tokenizer, model = load_variant_model(name, cfg, resolved_device)

            param_count = sum(p.numel() for p in model.parameters())
            result["variants"][name]["parameter_count"] = param_count

            inputs = tokenizer(
                sample_text,
                return_tensors="pt",
                max_length=max_input_length,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(resolved_device) for k, v in inputs.items()}

            if resolved_device == "cuda":
                torch.cuda.reset_peak_memory_stats(resolved_device)

            start = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=128,
                    num_beams=cfg.num_beams,
                )
            elapsed_ms = (time.perf_counter() - start) * 1000

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            result["variants"][name]["generate_success"] = True
            result["variants"][name]["sample_output"] = decoded
            result["variants"][name]["latency_ms"] = round(elapsed_ms, 2)

            if resolved_device == "cuda":
                peak_bytes = torch.cuda.max_memory_allocated(resolved_device)
                result["variants"][name]["peak_memory_mb"] = round(peak_bytes / (1024 * 1024), 2)

            del model, inputs, outputs
            if resolved_device == "cuda":
                torch.cuda.empty_cache()

    except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as oom:
        if resolved_device == "cuda":
            torch.cuda.empty_cache()
        result["success"] = False
        result["oom_detected"] = True
        result["error"] = str(oom)
        result["recommendation"] = (
            "CUDA Out of Memory: Lower physical batch size (e.g. to 1 or 2) and "
            "increase gradient accumulation steps proportionally to preserve effective "
            "batch sizes (16 for Stage 1, 8 for Stage 2)."
        )
        return result
    except Exception as exc:
        if "out of memory" in str(exc).lower():
            if resolved_device == "cuda":
                torch.cuda.empty_cache()
            result["success"] = False
            result["oom_detected"] = True
            result["error"] = str(exc)
            result["recommendation"] = (
                "CUDA Out of Memory: Lower physical batch size (e.g. to 1 or 2) and "
                "increase gradient accumulation steps proportionally to preserve effective "
                "batch sizes (16 for Stage 1, 8 for Stage 2)."
            )
            return result
        result["success"] = False
        result["error"] = str(exc)
        return result

    result["success"] = all_passed
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="ByT5 Base hardware and configuration preflight")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--metadata-only", action="store_true", help="Inspect config metadata without downloading/running weights")
    parser.add_argument("--output", type=str, default=None, help="Optional path to write JSON results")
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_preflight(
        device=args.device,
        max_input_length=args.max_input_length,
        metadata_only=args.metadata_only,
    )

    formatted = json.dumps(result, indent=2)
    print(formatted)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(formatted)

    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
