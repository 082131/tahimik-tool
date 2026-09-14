"""Statistical hypothesis testing suite for text normalization and efficiency benchmarks.

Features:
  1. Paired bootstrap resampling (1,000 iterations) with add-one smoothing,
     two-tailed empirical p-values, and 95% percentile confidence intervals.
  2. Wilcoxon signed-rank test for paired GPU memory observations with median,
     interquartile range (IQR), and rank-biserial correlation effect size.
  3. Holm-Bonferroni step-down family-wise error rate (FWER) correction.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats


class StatisticalAnalysis:
    """Statistical testing suite for comparing model variants."""

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
        higher_is_better: bool = True,
    ) -> Dict[str, Any]:
        """Paired bootstrap test comparing model A (baseline) and model B (TAHIMIK).

        Resamples paired sentence observations with replacement to determine if
        the performance delta between model B and model A is statistically
        distinguishable from zero.

        Sign convention:
          - For higher-is-better metrics (accuracy, GLEU+, chrF, ERR):
              delta = B - A  (positive means B/TAHIMIK is better)
          - For lower-is-better metrics (inference latency):
              delta = A - B  (positive means B/TAHIMIK is faster)
        """
        a = np.asarray(scores_a, dtype=float)
        b = np.asarray(scores_b, dtype=float)

        if len(a) != len(b) or len(a) == 0:
            raise ValueError("Paired scores must be non-empty and of equal length.")

        # Observed difference on the actual sample
        if higher_is_better:
            observed_delta = b.mean() - a.mean()
        else:
            observed_delta = a.mean() - b.mean()

        # Resample sentence indices with replacement 1,000 times
        deltas = []
        n = len(a)

        for _ in range(self.n_bootstrap):
            # Sample paired indices together so the sentence alignment is preserved
            idx = self.rng.randint(0, n, size=n)
            resampled_a = a[idx]
            resampled_b = b[idx]

            if higher_is_better:
                boot_delta = resampled_b.mean() - resampled_a.mean()
            else:
                boot_delta = resampled_a.mean() - resampled_b.mean()

            deltas.append(boot_delta)

        deltas = np.asarray(deltas)

        # Two-tailed empirical p-value with add-one smoothing (Davison & Hinkley, 1997).
        # Add-one smoothing ensures p-value is never exactly 0.0 under finite resampling.
        lower_tail = (1.0 + np.sum(deltas <= 0)) / (self.n_bootstrap + 1.0)
        upper_tail = (1.0 + np.sum(deltas >= 0)) / (self.n_bootstrap + 1.0)
        p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))

        # 95% Percentile Confidence Interval (2.5th and 97.5th percentiles)
        ci_lower, ci_upper = np.percentile(deltas, [2.5, 97.5])

        # Dual significance rule: must have p < alpha AND the 95% CI must exclude zero
        is_significant = bool(p_value < self.alpha and (ci_lower > 0 or ci_upper < 0))

        return {
            "metric": metric_name,
            "p_value": float(p_value),
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
            "delta": float(observed_delta),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "significant": is_significant,
            "higher_is_better": higher_is_better,
        }

    def wilcoxon_test(
        self,
        measurements_a: List[float],
        measurements_b: List[float],
        metric_name: str = "gpu_memory",
    ) -> Dict[str, Any]:
        """Paired two-sided Wilcoxon signed-rank test for independent GPU memory runs.

        Reports test statistic, p-value, median, IQR for both variants, and the
        matched-pairs rank-biserial correlation effect size.
        """
        a = np.asarray(measurements_a, dtype=float)
        b = np.asarray(measurements_b, dtype=float)

        if len(a) != len(b) or len(a) == 0:
            raise ValueError("Paired measurements must be non-empty and of equal length.")

        differences = b - a

        if np.all(differences == 0):
            statistic = 0.0
            p_value = 1.0
        else:
            stat_res = stats.wilcoxon(a, b, alternative="two-sided")
            statistic = float(stat_res.statistic)
            p_value = float(stat_res.pvalue)

        # Matched-pairs rank-biserial correlation uses signed rank sums.
        nonzero_diffs = differences[differences != 0]
        if len(nonzero_diffs) > 0:
            ranks = stats.rankdata(np.abs(nonzero_diffs), method="average")
            positive_ranks = float(ranks[nonzero_diffs > 0].sum())
            negative_ranks = float(ranks[nonzero_diffs < 0].sum())
            rank_total = len(nonzero_diffs) * (len(nonzero_diffs) + 1) / 2
            rank_biserial = (positive_ranks - negative_ranks) / rank_total
        else:
            rank_biserial = 0.0

        # Per-model summaries remain in MB; paired practical differences are GB.
        q25_a, q75_a = np.percentile(a, [25, 75])
        q25_b, q75_b = np.percentile(b, [25, 75])
        iqr_a = float(q75_a - q25_a)
        iqr_b = float(q75_b - q25_b)
        differences_gb = differences / 1024.0
        q25_diff, q75_diff = np.percentile(differences_gb, [25, 75])

        return {
            "metric": metric_name,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": bool(p_value < self.alpha),
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
            "std_a": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "std_b": float(b.std(ddof=1)) if len(b) > 1 else 0.0,
            "median_a": float(np.median(a)),
            "median_b": float(np.median(b)),
            "iqr_a": iqr_a,
            "iqr_b": iqr_b,
            "median_difference_gb": float(np.median(differences_gb)),
            "iqr_difference_gb": float(q75_diff - q25_diff),
            "rank_biserial": float(rank_biserial),
        }

    @staticmethod
    def holm_bonferroni(
        p_values: List[Tuple[str, float]],
        alpha: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """Applies step-down Holm-Bonferroni correction to control Family-Wise Error Rate (FWER).

        Sorts p-values in ascending order, evaluates each against alpha / (m - rank + 1),
        and enforces step-down early stopping (if test k fails to reject, all subsequent tests
        fail as well). Adjusted p-values are cumulative maxima ensuring monotonicity.

        Args:
            p_values: List of (comparison_name, raw_p_value) tuples.
            alpha: Overall family significance threshold (default 0.05).

        Returns:
            List of result dicts sorted by rank, containing raw and adjusted p-values.
        """
        m = len(p_values)
        ordered_comparisons = sorted(p_values, key=lambda item: item[1])

        results = []
        still_rejecting = True
        previous_adjusted = 0.0

        for rank, (name, raw_p) in enumerate(ordered_comparisons, start=1):
            # Step-down threshold for this rank
            divisor = m - rank + 1
            threshold = alpha / divisor

            # Step-down rule: reject only if current test meets threshold AND all prior tests were rejected
            reject = still_rejecting and (raw_p < threshold)
            still_rejecting = reject

            # Monotonic adjusted p-value: p_adj = min(1.0, max(prev_adj, raw_p * divisor))
            adjusted_p = min(1.0, max(previous_adjusted, raw_p * divisor))
            previous_adjusted = adjusted_p

            results.append({
                "comparison": name,
                "raw_p": float(raw_p),
                "adjusted_p": float(adjusted_p),
                "rank": rank,
                "adjusted_alpha": float(threshold),
                "significant_corrected": bool(reject),
            })

        return results

    def run_full_comparison(
        self,
        per_sentence_scores: Dict[str, Dict[str, List[float]]],
        gpu_memory_runs: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """Runs paired bootstrap, Wilcoxon signed-rank, and Holm-Bonferroni tests.

        Compares:
          1. ByT5 vs. TAHIMIK
          2. MrT5 vs. TAHIMIK

        Across:
          - Accuracy family: 4 metrics x 2 model pairs = 8 tests
          - Efficiency family: Latency and GPU memory x 2 model pairs = up to 4 tests
        """
        pairs = [("byt5", "tahimik"), ("mrt5", "tahimik")]
        accuracy_metrics = ["gleu_plus", "chrf", "err", "alpha_word_accuracy"]

        result: Dict[str, Any] = {
            "bootstrap": {},
            "wilcoxon": {},
            "holm_bonferroni": {
                "accuracy": [],
                "efficiency": {},
            },
        }

        accuracy_p_values: List[Tuple[str, float]] = []
        for model_a, model_b in pairs:
            pair_key = f"{model_a}_vs_{model_b}"
            result["bootstrap"][pair_key] = {}
            pair_efficiency_p_values: List[Tuple[str, float]] = []

            # 1. Accuracy metrics via paired bootstrap
            for metric in accuracy_metrics:
                boot_res = self.paired_bootstrap(
                    per_sentence_scores[model_a][metric],
                    per_sentence_scores[model_b][metric],
                    metric_name=metric,
                    higher_is_better=True,
                )
                result["bootstrap"][pair_key][metric] = boot_res
                accuracy_p_values.append((f"{pair_key}:{metric}", boot_res["p_value"]))

            # 2. Inference latency via paired bootstrap (lower is better)
            if (
                "inference_time" in per_sentence_scores.get(model_a, {})
                and "inference_time" in per_sentence_scores.get(model_b, {})
            ):
                boot_time = self.paired_bootstrap(
                    per_sentence_scores[model_a]["inference_time"],
                    per_sentence_scores[model_b]["inference_time"],
                    metric_name="inference_time",
                    higher_is_better=False,
                )
                result["bootstrap"][pair_key]["inference_time"] = boot_time
                pair_efficiency_p_values.append(("inference_time", boot_time["p_value"]))

            # 3. Peak GPU memory via Wilcoxon signed-rank test
            mem_a = gpu_memory_runs.get(model_a, [])
            mem_b = gpu_memory_runs.get(model_b, [])

            if mem_a and mem_b:
                wilc_res = self.wilcoxon_test(mem_a, mem_b, metric_name="gpu_memory")
                result["wilcoxon"][pair_key] = wilc_res
                pair_efficiency_p_values.append(("gpu_memory", wilc_res["p_value"]))

            if pair_efficiency_p_values:
                corrections = self.holm_bonferroni(pair_efficiency_p_values, alpha=self.alpha)
                result["holm_bonferroni"]["efficiency"][pair_key] = corrections
                for correction in corrections:
                    metric = correction["comparison"]
                    target = (
                        result["bootstrap"][pair_key][metric]
                        if metric == "inference_time"
                        else result["wilcoxon"][pair_key]
                    )
                    target["adjusted_p"] = correction["adjusted_p"]
                    target["significant_corrected"] = correction["significant_corrected"]

        # 4. Apply Holm-Bonferroni correction within each declared family
        result["holm_bonferroni"]["accuracy"] = self.holm_bonferroni(
            accuracy_p_values,
            alpha=self.alpha,
        )

        return result
