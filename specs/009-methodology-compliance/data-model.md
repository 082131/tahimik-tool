# Data Model

`Metrics` stores corpus means and per-sentence arrays for `gleu_plus`, `chrf`, `err`, and `alpha_word_accuracy`. `EfficiencyResult` stores `per_sentence_time_seconds`, `run_mean_time_seconds`, `peak_gpu_memory_runs_mb`, and an explicit `gpu_memory_available` flag.

Each statistical record stores comparison name, metric, direction, observed delta, p-value, CI bounds, raw significance, and Holm-adjusted significance. GPU records additionally store Wilcoxon statistic, median/IQR, and rank-biserial effect size.

`RunMetadata` stores timestamp, git SHA, dirty flag, resolved configuration, seed, Python/package versions, torch/CUDA/device details, and validation counts.
