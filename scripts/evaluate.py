# =============================================================================
# Evaluation Entry Point for TAHIMIK
#
# Loads a trained checkpoint and evaluates it on the gold test set using
# all four normalization metrics: GLEU+, chrF, ERR, Alpha-word Accuracy.
#
# Usage:
#   python scripts/evaluate.py \
#       --variant tahimik \
#       --checkpoint outputs/checkpoints/tahimik_noise_adaptive/best_stage2.pt \
#       --gold_data data/gold.csv \
#       --output_file outputs/results_tahimik.json
# =============================================================================

import sys
import os
import json
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

from configs.byt5_config import ByT5Config
from configs.mrt5_config import MrT5Config
from configs.tahimik_config import TAHIMIKConfig

from src.models.byt5_baseline import ByT5Baseline
from src.models.fixed_compression_byt5 import FixedCompressionByT5
from src.models.noise_adaptive_byt5 import NoiseAdaptiveByT5

from src.data.preprocessing import DataPipeline
from src.data.dataset import NormalizationDataset, NormalizationCollator, collate_fn
from src.data.noise_label import compute_noise_level

from src.evaluation.metrics import NormalizationMetrics
from src.utils.logging_utils import setup_logger

logger = setup_logger("tahimik.scripts.evaluate")

VARIANT_MAP = {
    "byt5": (ByT5Config, ByT5Baseline),
    "mrt5": (MrT5Config, FixedCompressionByT5),
    "tahimik": (TAHIMIKConfig, NoiseAdaptiveByT5),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a TAHIMIK model")

    parser.add_argument("--variant", type=str, required=True, choices=["byt5", "mrt5", "tahimik"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--gold_data", type=str, required=True, help="Path to gold standard CSV/JSON")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Setup ───────────────────────────────────────────────────────────
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

    gold_noise_levels = [
        compute_noise_level(n, c) for n, c in zip(gold_noisy, gold_clean)
    ]

    # Use only the test split (80/10/10 — last 10%)
    gold_splits = pipeline.split_data(
        gold_noisy, gold_clean, gold_noise_levels,
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=NormalizationCollator(pad_token_id=tokenizer.pad_token_id),
    )


    # ── Load model ──────────────────────────────────────────────────────
    model = ModelClass(config)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Loaded checkpoint: {args.checkpoint}")
    logger.info(f"  Stage: {checkpoint.get('stage', 'unknown')}")
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"  Val loss: {checkpoint.get('val_loss', 'unknown')}")

    # ── Generate predictions ────────────────────────────────────────────
    predictions = []

    logger.info(f"Generating predictions for {len(test_noisy)} test sentences...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config.max_target_length,
                num_beams=args.num_beams,
            )

            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predictions.extend(decoded)

    # ── Compute metrics ─────────────────────────────────────────────────
    metrics = NormalizationMetrics()
    results = metrics.compute_all(predictions, test_clean, test_noisy)

    logger.info("Evaluation Results:")
    logger.info(f"  GLEU+:              {results['gleu_plus']:.2f}")
    logger.info(f"  chrF:               {results['chrf']:.2f}")
    logger.info(f"  ERR:                {results['err']:.4f}")
    logger.info(f"  Alpha-word Accuracy: {results['alpha_word_accuracy']:.4f}")

    # ── Save results ────────────────────────────────────────────────────
    output = {
        "variant": args.variant,
        "checkpoint": args.checkpoint,
        "num_test_sentences": len(test_noisy),
        "metrics": results,
        # Sample predictions for manual inspection
        "samples": [
            {"noisy": n, "predicted": p, "reference": r}
            for n, p, r in zip(test_noisy[:10], predictions[:10], test_clean[:10])
        ],
    }

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output_file}")

    return results


if __name__ == "__main__":
    main()
