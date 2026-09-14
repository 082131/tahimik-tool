# =============================================================================
# Efficiency Benchmarking Script for TAHIMIK
#
# Measures inference time and peak GPU memory for a trained model.
# Follows the manuscript protocol: 5 warmup passes + 20 timed runs.
#
# Usage:
#   python scripts/benchmark.py \
#       --variant tahimik \
#       --checkpoint outputs/checkpoints/tahimik_noise_adaptive/best_stage2.pt \
#       --gold_data data/gold.csv \
#       --output_file outputs/efficiency_tahimik.json
# =============================================================================

import sys
import os
import json
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

from configs.byt5_config import ByT5Config
from configs.mrt5_config import MrT5Config
from configs.tahimik_config import TAHIMIKConfig

from src.models.byt5_baseline import ByT5Baseline
from src.models.fixed_compression import FixedCompressionByT5
from src.models.noise_adaptive import NoiseAdaptiveByT5

from src.data.preprocessing import DataPipeline, prepare_paired_examples
from src.data.dataset import NormalizationDataset
from src.evaluation.efficiency import EfficiencyBenchmark
from src.utils.logging_utils import setup_logger

logger = setup_logger("tahimik.scripts.benchmark")

VARIANT_MAP = {
    "byt5": (ByT5Config, ByT5Baseline),
    "mrt5": (MrT5Config, FixedCompressionByT5),
    "tahimik": (TAHIMIKConfig, NoiseAdaptiveByT5),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark TAHIMIK efficiency")

    parser.add_argument("--variant", type=str, required=True, choices=["byt5", "mrt5", "tahimik"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--gold_data", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup_passes", type=int, default=5)
    parser.add_argument("--inference_runs", type=int, default=20)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    ConfigClass, ModelClass = VARIANT_MAP[args.variant]
    config = ConfigClass()
    config.device = args.device
    config.seed = args.seed

    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # ── Load test data ──────────────────────────────────────────────────
    pipeline = DataPipeline(config, seed=config.seed)
    gold_noisy, gold_clean = pipeline.load_gold_standard(args.gold_data)
    gold_prepared = prepare_paired_examples(
        gold_noisy, gold_clean, tokenizer,
        config.max_input_length, config.max_target_length,
    )
    gold_splits = pipeline.split_data(
        gold_prepared.noisy_texts, gold_prepared.clean_texts, gold_prepared.noise_levels,
        train_ratio=config.gold_train_ratio,
        val_ratio=config.gold_val_ratio,
        test_ratio=config.gold_test_ratio,
    )
    test_noisy, test_clean, test_noise = gold_splits["test"]

    test_dataset = NormalizationDataset(
        test_noisy, test_clean, tokenizer,
        max_input_length=config.max_input_length,
        max_target_length=config.max_target_length,
        precomputed_noise_levels=test_noise,
    )

    # ── Load model ──────────────────────────────────────────────────────
    model = ModelClass(config)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Benchmarking {args.variant} on {len(test_dataset)} sentences")

    # ── Run benchmark ───────────────────────────────────────────────────
    bench = EfficiencyBenchmark(
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_beams=args.num_beams,
        warmup_passes=args.warmup_passes,
        inference_runs=args.inference_runs,
    )

    results = bench.benchmark(test_dataset, batch_size=1)

    # ── Save results ────────────────────────────────────────────────────
    output = {
        "variant": args.variant,
        "checkpoint": args.checkpoint,
        "efficiency": results,
        "config": {
            "warmup_passes": args.warmup_passes,
            "inference_runs": args.inference_runs,
            "num_beams": args.num_beams,
            "device": args.device,
        },
    }

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")

    return results


if __name__ == "__main__":
    main()
