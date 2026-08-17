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
from src.models.fixed_compression_byt5 import FixedCompressionByT5
from src.models.noise_adaptive_byt5 import NoiseAdaptiveByT5

from src.training.losses import TAHIMIKLoss
from src.training.trainer import TAHIMIKTrainer

from src.data.preprocessing import DataPipeline
from src.data.dataset import NormalizationDataset, collate_fn
from src.data.noise_label import compute_noise_level
from src.evaluation.metrics import NormalizationMetrics
from src.evaluation.efficiency import EfficiencyBenchmark
from src.evaluation.statistical_tests import StatisticalAnalysis
from src.utils.logging_utils import setup_logger

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
    parser.add_argument("--output_dir", type=str, default="outputs/experiment")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Shared tokenizer ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

    # ── Prepare data (shared across all variants) ───────────────────────
    base_config = ByT5Config()
    base_config.seed = args.seed
    pipeline = DataPipeline(base_config, seed=args.seed)

    gold_noisy, gold_clean = pipeline.load_gold_standard(args.gold_data)
    gold_noise_levels = [
        compute_noise_level(n, c) for n, c in zip(gold_noisy, gold_clean)
    ]
    gold_splits = pipeline.split_data(
        gold_noisy, gold_clean, gold_noise_levels,
        train_ratio=base_config.gold_train_ratio,
        val_ratio=base_config.gold_val_ratio,
        test_ratio=base_config.gold_test_ratio,
    )

    stage2_train_ds = NormalizationDataset(
        *gold_splits["train"], tokenizer,
        max_input_length=base_config.max_input_length,
        max_target_length=base_config.max_target_length,
    )
    stage2_val_ds = NormalizationDataset(
        *gold_splits["val"], tokenizer,
        max_input_length=base_config.max_input_length,
        max_target_length=base_config.max_target_length,
    )

    test_noisy, test_clean, test_noise = gold_splits["test"]
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
            clean_sentences, target_size=args.synthetic_size
        )
        syn_splits = pipeline.split_data(
            syn_noisy, syn_clean, syn_noise,
            train_ratio=base_config.synthetic_train_ratio,
            val_ratio=base_config.synthetic_val_ratio,
        )
        stage1_train_ds = NormalizationDataset(
            *syn_splits["train"], tokenizer,
            max_input_length=base_config.max_input_length,
            max_target_length=base_config.max_target_length,
        )
        stage1_val_ds = NormalizationDataset(
            *syn_splits["val"], tokenizer,
            max_input_length=base_config.max_input_length,
            max_target_length=base_config.max_target_length,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Train + Evaluate each variant
    # ══════════════════════════════════════════════════════════════════════
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

        # ── Train ───────────────────────────────────────────────────────
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

        # ── Evaluate ────────────────────────────────────────────────────
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

        # Compute per-sentence scores for bootstrap testing
        per_sentence = {
            "gleu_plus": [],
            "chrf": [],
            "err": [],
            "alpha_word_accuracy": [],
        }

        bleu_scorer = metrics_calculator.bleu_scorer
        import editdistance
        import re
        alpha_pat = re.compile(r"^[a-zA-Z]+$")

        for pred, ref, noisy in zip(predictions, test_clean, test_noisy):
            per_sentence["gleu_plus"].append(
                bleu_scorer.sentence_score(pred, [ref]).score
            )
            per_sentence["chrf"].append(
                metrics_calculator.chrf_scorer.sentence_score(pred, [ref]).score
            )
            eb = editdistance.eval(noisy, ref)
            ea = editdistance.eval(pred, ref)
            per_sentence["err"].append(
                (eb - ea) / eb if eb > 0 else (1.0 if ea == 0 else 0.0)
            )

            ref_words = ref.split()
            pred_words = pred.split()
            correct = total = 0
            for i, rw in enumerate(ref_words):
                if alpha_pat.match(rw):
                    total += 1
                    if i < len(pred_words) and pred_words[i].lower() == rw.lower():
                        correct += 1
            per_sentence["alpha_word_accuracy"].append(
                correct / total if total > 0 else 1.0
            )

        all_per_sentence[variant_name] = per_sentence

        # ── Efficiency Benchmark ────────────────────────────────────────
        bench = EfficiencyBenchmark(
            model=model, tokenizer=tokenizer, device=device,
            num_beams=config.num_beams,
            warmup_passes=config.warmup_passes,
            inference_runs=config.inference_runs,
        )
        eff_results = bench.benchmark(test_dataset, batch_size=1)
        all_metrics[variant_name]["efficiency"] = eff_results

        # Collect per-run GPU memory for Wilcoxon test
        all_gpu_memory[variant_name] = [
            eff_results["peak_gpu_memory_mb"]
        ] * config.inference_runs

    # ══════════════════════════════════════════════════════════════════════
    # Statistical Testing
    # ══════════════════════════════════════════════════════════════════════
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

    # ── Save everything ─────────────────────────────────────────────────
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
