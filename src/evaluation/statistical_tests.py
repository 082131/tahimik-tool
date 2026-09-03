import numpy as np
from scipy import stats
from typing import Dict, List, Tuple

class StatisticalAnalysis:
    def __init__(self, alpha: float = 0.05, n_bootstrap: int = 1000, seed: int = 42):
        self.alpha, self.n_bootstrap = alpha, n_bootstrap
        self.rng = np.random.RandomState(seed)

    def paired_bootstrap(self, scores_a, scores_b, metric_name="metric", higher_is_better=True):
        a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
        if len(a) != len(b) or not len(a): raise ValueError("paired scores must be non-empty and equal length")
        observed = (b.mean() - a.mean()) if higher_is_better else (a.mean() - b.mean())
        deltas = []
        for _ in range(self.n_bootstrap):
            idx = self.rng.randint(0, len(a), len(a))
            deltas.append((b[idx].mean() - a[idx].mean()) if higher_is_better else (a[idx].mean() - b[idx].mean()))
        deltas = np.asarray(deltas)
        lower_tail = (1 + np.sum(deltas <= 0)) / (self.n_bootstrap + 1)
        upper_tail = (1 + np.sum(deltas >= 0)) / (self.n_bootstrap + 1)
        p = min(1.0, 2 * min(lower_tail, upper_tail))
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        return {"metric": metric_name, "p_value": float(p), "mean_a": float(a.mean()), "mean_b": float(b.mean()), "delta": float(observed), "ci_lower": float(lo), "ci_upper": float(hi), "significant": bool(p < self.alpha and (lo > 0 or hi < 0)), "higher_is_better": higher_is_better}

    def wilcoxon_test(self, measurements_a, measurements_b, metric_name="gpu_memory"):
        a, b = np.asarray(measurements_a, dtype=float), np.asarray(measurements_b, dtype=float)
        if len(a) != len(b) or not len(a): raise ValueError("paired measurements must be non-empty and equal length")
        d = b - a
        stat, p = (0.0, 1.0) if np.all(d == 0) else stats.wilcoxon(a, b, alternative="two-sided")
        nonzero = d[d != 0]
        rank_biserial = float((np.sum(nonzero > 0) - np.sum(nonzero < 0)) / len(nonzero)) if len(nonzero) else 0.0
        qa = np.percentile(a, [25, 75]); qb = np.percentile(b, [25, 75])
        return {"metric": metric_name, "statistic": float(stat), "p_value": float(p), "significant": bool(p < self.alpha), "mean_a": float(a.mean()), "mean_b": float(b.mean()), "median_a": float(np.median(a)), "median_b": float(np.median(b)), "iqr_a": float(qa[1]-qa[0]), "iqr_b": float(qb[1]-qb[0]), "rank_biserial": rank_biserial}

    @staticmethod
    def holm_bonferroni(p_values: List[Tuple[str, float]], alpha=0.05):
        m = len(p_values); ordered = sorted(p_values, key=lambda x: x[1]); results=[]; keep=True; previous=0.0
        for rank, (name, raw) in enumerate(ordered, 1):
            threshold = alpha / (m-rank+1); reject = keep and raw < threshold; keep = reject
            adjusted = min(1.0, max(previous, raw * (m-rank+1))); previous = adjusted
            results.append({"comparison": name, "raw_p": float(raw), "adjusted_p": float(adjusted), "rank": rank, "adjusted_alpha": threshold, "significant_corrected": bool(reject)})
        return results

    def run_full_comparison(self, per_sentence_scores: Dict[str, Dict[str, List[float]]], gpu_memory_runs: Dict[str, List[float]]):
        pairs = [("byt5", "tahimik"), ("mrt5", "tahimik")]; metrics = ["gleu_plus", "chrf", "err", "alpha_word_accuracy"]
        result = {"bootstrap": {}, "wilcoxon": {}, "holm_bonferroni": {"accuracy": [], "efficiency": []}}; accuracy_p=[]; efficiency_p=[]
        for a,b in pairs:
            key=f"{a}_vs_{b}"; result["bootstrap"][key]={}
            for metric in metrics:
                r=self.paired_bootstrap(per_sentence_scores[a][metric], per_sentence_scores[b][metric], metric); result["bootstrap"][key][metric]=r; accuracy_p.append((f"{key}:{metric}", r["p_value"]))
            if "inference_time" in per_sentence_scores[a] and "inference_time" in per_sentence_scores[b]:
                r=self.paired_bootstrap(per_sentence_scores[a]["inference_time"], per_sentence_scores[b]["inference_time"], "inference_time", higher_is_better=False); result["bootstrap"][key]["inference_time"]=r; efficiency_p.append((f"{key}:inference_time", r["p_value"]))
            ma,mb=gpu_memory_runs.get(a,[]),gpu_memory_runs.get(b,[])
            if ma and mb: result["wilcoxon"][key]=self.wilcoxon_test(ma,mb); efficiency_p.append((f"{key}:gpu_memory",result["wilcoxon"][key]["p_value"]))
        result["holm_bonferroni"]["accuracy"]=self.holm_bonferroni(accuracy_p,self.alpha); result["holm_bonferroni"]["efficiency"]=self.holm_bonferroni(efficiency_p,self.alpha) if efficiency_p else []
        return result
