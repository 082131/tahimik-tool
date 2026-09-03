# Data Model

## MetricResult

Contains corpus `gleu_plus`, `chrf`, `err`, and `alpha_word_accuracy`, plus aligned per-sentence vectors and declared metric direction.

## BootstrapResult

Contains comparison/metric names, seed, resample count, observed oriented difference, 95% interval, raw p-value, adjusted p-value, family ID, and final significance decision.

## WilcoxonResult

Contains 20-value paired vectors or their artifact reference, statistic, raw/adjusted p, model means/standard deviations, paired-difference median/IQR in GB, positive/negative rank sums, rank-biserial effect, and zero/tie policy.

## ComparisonFamily

Contains stable family ID, purpose (`accuracy` or `efficiency`), ordered member test IDs, alpha, Holm ordering, adjusted p-values, and rejection decisions. Each efficiency family has exactly two members.

## ReliabilityCategoryResult

Contains category, alpha, sentence/annotator/valid-cell counts, threshold 0.80, pass/fail, and missing/malformed reason codes.

## EfficiencyResult

Contains five warm-up count, 20 latency observations, 20 peak-memory observations when CUDA is available, units, summaries, and device/provenance identity.
