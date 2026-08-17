# =============================================================================
# Statistical Testing Suite for TAHIMIK
#
# From the manuscript (Statistical Treatment of Data section):
#
#   Paired Bootstrap Resampling — Primary significance test for
#       normalization metrics (GLEU+, chrF, ERR, Alpha-word Accuracy).
#       1000 bootstrap samples with replacement; computes p-value as
#       the fraction of samples where the baseline >= the proposed model.
#
#   Wilcoxon Signed-Rank Test — Non-parametric test for GPU memory
#       measurements (20 paired observations per model). Appropriate
#       because GPU memory measurements are not normally distributed.
#
#   Holm-Bonferroni Correction — Multiple comparison correction applied
#       to all p-values across the three pairwise model comparisons
#       (ByT5 vs MrT5, ByT5 vs TAHIMIK, MrT5 vs TAHIMIK).
# =============================================================================

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple
from src.utils.logging_utils import setup_logger

logger = setup_logger("tahimik.stats")


class StatisticalAnalysis:
    """
    Statistical testing for TAHIMIK model comparisons.

    Implements all three tests specified in the manuscript to determine
    whether performance differences between model variants are
    statistically significant at alpha = 0.05.

    Args:
        alpha: Significance level (default 0.05).
        n_bootstrap: Number of bootstrap resampling iterations.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_bootstrap: int = 1000,
        seed: int = 42,
    ):
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.RandomState(seed)

    def paired_bootstrap(
        self,
        scores_a: List[float],
        scores_b: List[float],
        metric_name: str = "metric",
    ) -> Dict[str, float]:
        """
        Paired bootstrap resampling test.

        Tests whether model B significantly outperforms model A on a
        per-sentence metric. The null hypothesis is that A >= B.

        Args:
            scores_a: Per-sentence scores for model A (baseline).
            scores_b: Per-sentence scores for model B (proposed).
            metric_name: Name for logging.

        Returns:
            Dict with 'p_value', 'mean_a', 'mean_b', 'delta',
            'significant' (bool), and 'ci_lower'/'ci_upper' (95% CI
            of the difference).
        """
        a = np.array(scores_a)
        b = np.array(scores_b)
        n = len(a)

        observed_delta = b.mean() - a.mean()

        # Bootstrap
        bootstrap_deltas = []
        wins_a = 0

        for _ in range(self.n_bootstrap):
            indices = self.rng.randint(0, n, size=n)
            boot_a = a[indices].mean()
            boot_b = b[indices].mean()
            delta = boot_b - boot_a
            bootstrap_deltas.append(delta)

            if boot_a >= boot_b:
                wins_a += 1

        p_value = wins_a / self.n_bootstrap

        # 95% confidence interval of the difference
        bootstrap_deltas = np.array(bootstrap_deltas)
        ci_lower = np.percentile(bootstrap_deltas, 2.5)
        ci_upper = np.percentile(bootstrap_deltas, 97.5)

        result = {
            "p_value": p_value,
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
            "delta": float(observed_delta),
            "significant": p_value < self.alpha,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
        }

        logger.info(
            f"  Bootstrap [{metric_name}]: "
            f"A={result['mean_a']:.4f}, B={result['mean_b']:.4f}, "
            f"delta={result['delta']:.4f}, p={result['p_value']:.4f} "
            f"{'*' if result['significant'] else 'ns'}"
        )

        return result

    def wilcoxon_test(
        self,
        measurements_a: List[float],
        measurements_b: List[float],
        metric_name: str = "gpu_memory",
    ) -> Dict[str, float]:
        """
        Wilcoxon signed-rank test for paired measurements.

        Used for GPU memory comparisons where the sample size is small
        (20 measurements per model) and normality cannot be assumed.

        Args:
            measurements_a: Measurements from model A.
            measurements_b: Measurements from model B.
            metric_name: Name for logging.

        Returns:
            Dict with 'statistic', 'p_value', 'significant',
            'mean_a', 'mean_b'.
        """
        a = np.array(measurements_a)
        b = np.array(measurements_b)

        # Wilcoxon requires non-zero differences
        differences = b - a
        if np.all(differences == 0):
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "mean_a": float(a.mean()),
                "mean_b": float(b.mean()),
            }

        stat, p_value = stats.wilcoxon(a, b, alternative="two-sided")

        result = {
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
        }

        logger.info(
            f"  Wilcoxon [{metric_name}]: "
            f"A={result['mean_a']:.2f}, B={result['mean_b']:.2f}, "
            f"W={result['statistic']:.1f}, p={result['p_value']:.4f} "
            f"{'*' if result['significant'] else 'ns'}"
        )

        return result

    @staticmethod
    def holm_bonferroni(
        p_values: List[Tuple[str, float]],
        alpha: float = 0.05,
    ) -> List[Dict[str, any]]:
        """
        Holm-Bonferroni correction for multiple comparisons.

        Controls the family-wise error rate (FWER) across all pairwise
        model comparisons.

        Args:
            p_values: List of (comparison_name, raw_p_value) tuples.
            alpha: Family-wise significance level.

        Returns:
            List of dicts with 'comparison', 'raw_p', 'adjusted_alpha',
            'significant_corrected', and 'rank'.
        """
        m = len(p_values)

        # Sort by p-value (ascending)
        sorted_pairs = sorted(p_values, key=lambda x: x[1])

        results = []
        for rank, (name, p) in enumerate(sorted_pairs, start=1):
            adjusted_alpha = alpha / (m - rank + 1)
            significant = p < adjusted_alpha

            results.append({
                "comparison": name,
                "raw_p": p,
                "rank": rank,
                "adjusted_alpha": adjusted_alpha,
                "significant_corrected": significant,
            })

            logger.info(
                f"  Holm-Bonferroni rank {rank}: {name} "
                f"p={p:.4f} < {adjusted_alpha:.4f}? "
                f"{'YES' if significant else 'NO'}"
            )

        return results

    def run_full_comparison(
        self,
        per_sentence_scores: Dict[str, Dict[str, List[float]]],
        gpu_memory_runs: Dict[str, List[float]],
    ) -> Dict[str, any]:
        """
        Run the complete statistical analysis across all three model pairs.

        Args:
            per_sentence_scores: Nested dict of
                {model_name: {metric_name: [per_sentence_scores]}}.
                Expected model names: "byt5", "mrt5", "tahimik".
                Expected metrics: "gleu_plus", "chrf", "err", "alpha_word_accuracy".
            gpu_memory_runs: {model_name: [memory_measurements_per_run]}.

        Returns:
            Comprehensive results dict with all tests and corrections.
        """
        models = ["byt5", "mrt5", "tahimik"]
        pairs = [
            ("byt5", "mrt5"),
            ("byt5", "tahimik"),
            ("mrt5", "tahimik"),
        ]

        metrics = ["gleu_plus", "chrf", "err", "alpha_word_accuracy"]

        all_results = {"bootstrap": {}, "wilcoxon": {}, "holm_bonferroni": {}}
        all_p_values = []

        # ── Bootstrap tests for normalization metrics ───────────────────
        logger.info("Running paired bootstrap resampling tests...")
        for model_a, model_b in pairs:
            pair_key = f"{model_a}_vs_{model_b}"
            all_results["bootstrap"][pair_key] = {}

            for metric in metrics:
                scores_a = per_sentence_scores[model_a][metric]
                scores_b = per_sentence_scores[model_b][metric]

                result = self.paired_bootstrap(
                    scores_a, scores_b,
                    metric_name=f"{pair_key}/{metric}",
                )
                all_results["bootstrap"][pair_key][metric] = result
                all_p_values.append(
                    (f"{pair_key}/{metric}", result["p_value"])
                )

        # ── Wilcoxon tests for GPU memory ───────────────────────────────
        logger.info("Running Wilcoxon signed-rank tests for GPU memory...")
        for model_a, model_b in pairs:
            pair_key = f"{model_a}_vs_{model_b}"
            mem_a = gpu_memory_runs[model_a]
            mem_b = gpu_memory_runs[model_b]

            result = self.wilcoxon_test(
                mem_a, mem_b,
                metric_name=f"{pair_key}/gpu_memory",
            )
            all_results["wilcoxon"][pair_key] = result
            all_p_values.append(
                (f"{pair_key}/gpu_memory", result["p_value"])
            )

        # ── Holm-Bonferroni correction ──────────────────────────────────
        logger.info("Applying Holm-Bonferroni correction...")
        all_results["holm_bonferroni"] = self.holm_bonferroni(
            all_p_values, alpha=self.alpha
        )

        return all_results
