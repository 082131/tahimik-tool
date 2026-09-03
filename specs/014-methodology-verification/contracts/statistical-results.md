# Statistical Results Contract

The saved result JSON must include:

- `metrics`: corpus scores, aligned per-sentence vectors, units, and directions.
- `reliability.by_category`: nominal alpha and threshold decision for every configured category.
- `bootstrap`: one entry per declared accuracy/latency comparison with observed difference, 1,000 resamples, raw p, 95% CI, family ID, adjusted p, and significance.
- `wilcoxon`: one entry per memory comparison with 20 paired observations, rank sums, rank-biserial, raw p, summaries, family ID, adjusted p, and significance.
- `holm_families`: explicit member IDs, ordered raw p-values, monotonic adjusted p-values, and step-down decisions.
- `efficiency`: five warm-ups, 20 runs, latency vectors, memory vectors or an explicit unavailable reason, and units.
- `provenance`: the complete spec 013 record.

No significance claim is valid unless the test belongs to a recorded family and satisfies its procedure-specific p-value and confidence-interval rules.
