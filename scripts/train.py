# =============================================================================
# Training Entry Point for TAHIMIK
#
# Trains one model variant (ByT5 / MrT5 / TAHIMIK) through the full
# two-stage pipeline. Reads data from CSV/JSON files, prepares datasets,
# and runs Stage 1 (synthetic) + Stage 2 (gold standard) training.
#
# Usage:
#   python scripts/train.py --variant tahimik --gold_data data/gold.csv
#   python scripts/train.py --variant byt5 --gold_data data/gold.csv
#   python scripts/train.py --variant mrt5 --gold_data data/gold.csv
#
# For Stage 1, provide --clean_corpus to generate synthetic pairs.
# For Stage 2 only (debugging), omit --clean_corpus.
# =============================================================================

import sys
import os
import argparse
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

from configs.byt5_config import ByT5Config
from configs.mrt5_config import MrT5Config
from configs.tahimik_config import TAHIMIKConfig

from src.models.byt5_baseline import ByT5Baseline
from src.models.fixed_compression_byt5 import FixedCompressionByT5
from src.models.noise_adaptive_byt5 import NoiseAdaptiveByT5

from src.training.losses import TAHIMIKLoss
from src.training.trainer import TAHIMIKTrainer

from src.data.preprocessing import DataPipeline
from src.data.dataset import NormalizationDataset
from src.utils.logging_utils import setup_logger

logger = setup_logger("tahimik.scripts.train")

VARIANT_MAP = {
    "byt5": (ByT5Config, ByT5Baseline),
    "mrt5": (MrT5Config, FixedCompressionByT5),
    "tahimik": (TAHIMIKConfig, NoiseAdaptiveByT5),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a TAHIMIK model variant")

    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["byt5", "mrt5", "tahimik"],
        help="Model variant to train",
    )
    parser.add_argument(
        "--gold_data",
        type=str,
        required=True,
        help="Path to gold standard CSV/JSON (noisy/clean columns)",
    )
    parser.add_argument(
        "--clean_corpus",
        type=str,
        default=None,
        help="Path to clean corpus for synthetic noise generation (Stage 1). "
             "If omitted, Stage 1 is skipped.",
    )
    parser.add_argument(
        "--synthetic_size",
        type=int,
        default=1_000_000,
        help="Target number of synthetic pairs for Stage 1 (default: 1M)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory for checkpoints and logs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Configuration ───────────────────────────────────────────────────
    ConfigClass, ModelClass = VARIANT_MAP[args.variant]
    config = ConfigClass()
    config.device = args.device
    config.output_dir = args.output_dir
    config.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    config.seed = args.seed

    logger.info(f"Training variant: {args.variant} ({config.variant_name})")
    logger.info(f"Device: {config.device}")

    # ── Reproducibility ─────────────────────────────────────────────────
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # ── Tokenizer ───────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # ── Data Pipeline ───────────────────────────────────────────────────
    pipeline = DataPipeline(config, seed=config.seed)

    # Load gold standard
    gold_noisy, gold_clean = pipeline.load_gold_standard(args.gold_data)

    # Compute noise levels for gold data
    from src.data.noise_label import compute_noise_level
    gold_noise_levels = [
        compute_noise_level(n, c) for n, c in zip(gold_noisy, gold_clean)
    ]

    # Split gold data (80/10/10)
    gold_splits = pipeline.split_data(
        gold_noisy, gold_clean, gold_noise_levels,
        train_ratio=config.gold_train_ratio,
        val_ratio=config.gold_val_ratio,
        test_ratio=config.gold_test_ratio,
    )

    # Gold datasets
    gold_train_noisy, gold_train_clean, gold_train_noise = gold_splits["train"]
    gold_val_noisy, gold_val_clean, gold_val_noise = gold_splits["val"]

    stage2_train = NormalizationDataset(
        gold_train_noisy, gold_train_clean, tokenizer,
        max_input_length=config.max_input_length,
        max_target_length=config.max_target_length,
        precomputed_noise_levels=gold_train_noise,
    )
    stage2_val = NormalizationDataset(
        gold_val_noisy, gold_val_clean, tokenizer,
        max_input_length=config.max_input_length,
        max_target_length=config.max_target_length,
        precomputed_noise_levels=gold_val_noise,
    )

    # ── Stage 1: Synthetic data (optional) ──────────────────────────────
    stage1_train = None
    stage1_val = None

    if args.clean_corpus:
        clean_sentences = pipeline.load_clean_corpus(args.clean_corpus)
        syn_noisy, syn_clean, syn_noise = pipeline.generate_synthetic_pairs(
            clean_sentences, target_size=args.synthetic_size
        )
        syn_splits = pipeline.split_data(
            syn_noisy, syn_clean, syn_noise,
            train_ratio=config.synthetic_train_ratio,
            val_ratio=config.synthetic_val_ratio,
        )
        syn_train_noisy, syn_train_clean, syn_train_noise = syn_splits["train"]
        syn_val_noisy, syn_val_clean, syn_val_noise = syn_splits["val"]

        stage1_train = NormalizationDataset(
            syn_train_noisy, syn_train_clean, tokenizer,
            max_input_length=config.max_input_length,
            max_target_length=config.max_target_length,
            precomputed_noise_levels=syn_train_noise,
        )
        stage1_val = NormalizationDataset(
            syn_val_noisy, syn_val_clean, tokenizer,
            max_input_length=config.max_input_length,
            max_target_length=config.max_target_length,
            precomputed_noise_levels=syn_val_noise,
        )

    # ── Model ───────────────────────────────────────────────────────────
    model = ModelClass(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss Function ───────────────────────────────────────────────────
    use_compression = getattr(config, "use_compression", False)
    noise_adaptive = getattr(config, "noise_adaptive", False)

    loss_fn = TAHIMIKLoss(
        w_rate=getattr(config, "w_rate", 1.0),
        w_attn_reg=getattr(config, "w_attn_reg", 0.01),
        use_compression=use_compression,
        noise_adaptive=noise_adaptive,
    )

    # ── Trainer ─────────────────────────────────────────────────────────
    trainer = TAHIMIKTrainer(model, config, loss_fn)

    # ── Train ───────────────────────────────────────────────────────────
    history = trainer.train(
        stage1_train=stage1_train,
        stage1_val=stage1_val,
        stage2_train=stage2_train,
        stage2_val=stage2_val,
    )

    logger.info("Training complete!")
    logger.info(f"Checkpoints saved to: {config.checkpoint_dir}")


if __name__ == "__main__":
    main()
