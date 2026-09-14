# =============================================================================
# Full Experiment Pipeline for TAHIMIK
#
# Runs the complete experiment: trains all three model variants, evaluates
# each on the gold test set, runs efficiency benchmarks, and performs
# statistical significance testing across all pairwise comparisons.
#
# This script reproduces the full experiment described in the manuscript.
#
# Usage:
#   python scripts/run_experiment.py \
#       --gold_data data/gold.csv \
#       --clean_corpus data/clean_corpus.txt \
#       --output_dir outputs/experiment
#
# For a quick test run (Stage 2 only, no synthetic data):
#   python scripts/run_experiment.py \
#       --gold_data data/gold.csv \
#       --output_dir outputs/experiment
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

from src.training.losses import TAHIMIKLoss
from src.training.trainer import TAHIMIKTrainer

from src.data.preprocessing import DataPipeline, prepare_paired_examples
from src.data.noise_policy import load_probability_manifest
from src.data.dataset import NormalizationDataset, collate_fn
from src.evaluation.metrics import NormalizationMetrics
from src.evaluation.efficiency import EfficiencyBenchmark
from src.evaluation.statistical_tests import StatisticalAnalysis
from src.utils.logging_utils import setup_logger
from src.utils.reproducibility import collect_run_metadata, configure_determinism

from tqdm import tqdm
from torch.utils.data import DataLoader

logger = setup_logger("tahimik.experiment")

VARIANTS = {
    "byt5": (ByT5Config, ByT5Baseline, False, False),
    "mrt5": (MrT5Config, FixedCompressionByT5, True, False),
    "tahimik": (TAHIMIKConfig, NoiseAdaptiveByT5, True, True),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run full TAHIMIK experiment")

    parser.add_argument("--gold_data", type=str, required=True)
    parser.add_argument("--clean_corpus", type=str, default=None)
    parser.add_argument("--synthetic_size", type=int, default=1_000_000)
    parser.add_argument("--noise_manifest", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/experiment")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["fp32", "fp16", "bf16"],
        help="Precision mode (default: from config)",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable model gradient checkpointing",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.clean_corpus and not args.noise_manifest:
        raise ValueError("--noise_manifest is required when --clean_corpus enables Stage 1")
    manifest = load_probability_manifest(args.noise_manifest) if args.clean_corpus else None
    configure_determinism(args.seed)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Shared configuration and tokenizer --------------------------------------
    base_config = ByT5Config()
    base_config.seed = args.seed
    tokenizer = AutoTokenizer.from_pretrained(base_config.model_name)
    pipeline = DataPipeline(base_config, seed=args.seed, noise_manifest=manifest)


    gold_noisy, gold_clean = pipeline.load_gold_standard(args.gold_data)
    gold_prepared = prepare_paired_examples(
        gold_noisy, gold_clean, tokenizer,
        base_config.max_input_length, base_config.max_target_length,
    )
    gold_splits = pipeline.split_data(
        gold_prepared.noisy_texts, gold_prepared.clean_texts, gold_prepared.noise_levels,
        train_ratio=base_config.gold_train_ratio,
        val_ratio=base_config.gold_val_ratio,
        test_ratio=base_config.gold_test_ratio,
    )

    train_noisy, train_clean, train_noise = gold_splits["train"]
    val_noisy, val_clean, val_noise = gold_splits["val"]
    test_noisy, test_clean, test_noise = gold_splits["test"]

    stage2_train_ds = NormalizationDataset(
        train_noisy, train_clean, tokenizer,
        max_input_length=base_config.max_input_length,
        max_target_length=base_config.max_target_length,
        precomputed_noise_levels=train_noise,
    )
    stage2_val_ds = NormalizationDataset(
        val_noisy, val_clean, tokenizer,
        max_input_length=base_config.max_input_length,
        max_target_length=base_config.max_target_length,
        precomputed_noise_levels=val_noise,
    )

    test_dataset = NormalizationDataset(
        test_noisy, test_clean, tokenizer,
        max_input_length=base_config.max_input_length,
        max_target_length=base_config.max_target_length,
        precomputed_noise_levels=test_noise,
    )

    # Synthetic data (optional)
    stage1_train_ds = None
    stage1_val_ds = None
    if args.clean_corpus:
        clean_sentences = pipeline.load_clean_corpus(args.clean_corpus)
        syn_noisy, syn_clean, syn_noise = pipeline.generate_synthetic_pairs(
            clean_sentences, tokenizer, base_config.max_input_length,
            base_config.max_target_length, target_size=args.synthetic_size,
        )
        with open(os.path.join(args.output_dir, "synthetic_lineage.json"), "w", encoding="utf-8") as handle:
            json.dump([item.to_dict() for item in pipeline.synthetic_lineage], handle, indent=2)
        with open(os.path.join(args.output_dir, "synthetic_diagnostics.json"), "w", encoding="utf-8") as handle:
            json.dump(pipeline.synthetic_diagnostics, handle, indent=2)
        syn_splits = pipeline.split_data(
            syn_noisy, syn_clean, syn_noise,
            train_ratio=base_config.synthetic_train_ratio,
            val_ratio=base_config.synthetic_val_ratio,
        )
        syn_train_noisy, syn_train_clean, syn_train_noise = syn_splits["train"]
        syn_val_noisy, syn_val_clean, syn_val_noise = syn_splits["val"]

        stage1_train_ds = NormalizationDataset(
            syn_train_noisy, syn_train_clean, tokenizer,
            max_input_length=base_config.max_input_length,
            max_target_length=base_config.max_target_length,
            precomputed_noise_levels=syn_train_noise,
        )
        stage1_val_ds = NormalizationDataset(
            syn_val_noisy, syn_val_clean, tokenizer,
            max_input_length=base_config.max_input_length,
            max_target_length=base_config.max_target_length,
            precomputed_noise_levels=syn_val_noise,
        )

    # =========================================================================
    # Train + Evaluate each variant
    # ========================================================================= 
    all_metrics = {}
    all_per_sentence = {}
    all_gpu_memory = {}
    all_predictions = {}

    metrics_calculator = NormalizationMetrics()

    for variant_name, (ConfigClass, ModelClass, use_comp, noise_adapt) in VARIANTS.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"  VARIANT: {variant_name}")
        logger.info(f"{'='*70}")

        config = ConfigClass()
        config.device = args.device
        config.seed = args.seed
        config.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        if args.precision is not None:
            config.precision = args.precision
        if args.gradient_checkpointing is not None:
            config.gradient_checkpointing = args.gradient_checkpointing

        # --- Train -----------------------------------------------------------
        model = ModelClass(config)

        loss_fn = TAHIMIKLoss(
            w_rate=getattr(config, "w_rate", 1.0),
            w_attn_reg=getattr(config, "w_attn_reg", 0.01),
            use_compression=use_comp,
            noise_adaptive=noise_adapt,
        )

        trainer = TAHIMIKTrainer(model, config, loss_fn)
        trainer.train(
            stage1_train=stage1_train_ds,
            stage1_val=stage1_val_ds,
            stage2_train=stage2_train_ds,
            stage2_val=stage2_val_ds,
        )

        # --- Evaluate --------------------------------------------------------
        model.eval()
        model.to(device)

        test_loader = DataLoader(
            test_dataset, batch_size=base_config.eval_batch_size,
            shuffle=False, collate_fn=collate_fn,
        )

        predictions = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Eval {variant_name}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                gen_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=config.max_target_length,
                    num_beams=config.num_beams,
                )
                decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                predictions.extend(decoded)

        all_predictions[variant_name] = predictions

        # Compute corpus-level metrics
        results = metrics_calculator.compute_all(predictions, test_clean, test_noisy)
        all_metrics[variant_name] = results
        logger.info(f"  {variant_name} results: {results}")

        # Compute the exact shared per-sentence metric definitions.
        per_sentence = metrics_calculator.compute_per_sentence(predictions, test_clean, test_noisy)

        # Efficiency Benchmark
        bench = EfficiencyBenchmark(
            model=model, tokenizer=tokenizer, device=device,
            num_beams=config.num_beams,
            warmup_passes=config.warmup_passes,
            inference_runs=config.inference_runs,
        )
        eff_results = bench.benchmark(test_dataset, batch_size=1)
        all_metrics[variant_name]["efficiency"] = eff_results

        # Collect independent per-run GPU memory observations; CPU yields an empty list.
        all_gpu_memory[variant_name] = eff_results.get("peak_gpu_memory_runs_mb", [])
        per_sentence["inference_time"] = eff_results.get("per_sentence_time_seconds", [])
        all_per_sentence[variant_name] = per_sentence

    # =========================================================================
    # Statistical Testing
    # ========================================================================= 
    logger.info(f"\n{'='*70}")
    logger.info("  STATISTICAL ANALYSIS")
    logger.info(f"{'='*70}")

    stats_analyzer = StatisticalAnalysis(
        alpha=0.05, n_bootstrap=1000, seed=args.seed
    )
    stat_results = stats_analyzer.run_full_comparison(
        per_sentence_scores=all_per_sentence,
        gpu_memory_runs=all_gpu_memory,
    )

    # --- Save everything -----------------------------------------------------
    final_output = {
        "metrics": {k: v for k, v in all_metrics.items()},
        "statistical_tests": {
            k: (v if not isinstance(v, list) else
                [{kk: str(vv) if not isinstance(vv, (int, float, bool, type(None))) else vv
                  for kk, vv in item.items()} for item in v])
            for k, v in stat_results.items()
        },
        "config": {
            "seed": args.seed,
            "device": args.device,
            "gold_data": args.gold_data,
            "clean_corpus": args.clean_corpus,
        },
    }

    output_path = os.path.join(args.output_dir, "full_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, default=str)

    logger.info(f"\nAll results saved to {output_path}")
    logger.info("Experiment complete!")


if __name__ == "__main__":
    main()




