# Phase 1 Data Model: Evaluation and Statistical Testing

The quantities this feature produces, and the distinction between per-sentence
and per-run measurement that determines which statistical test applies.

## The measurement-level distinction

This is the design's load-bearing idea, and getting it wrong invalidates the
statistics:

| Level | Quantities | Test | Why |
|-------|-----------|------|-----|
| **Per sentence** | GLEU+, chrF, ERR, alpha-word accuracy, inference time | Paired bootstrap | One value per test sentence, so resampling sentences is meaningful |
| **Per run** | Peak GPU memory | Wilcoxon signed-rank | One value per inference run. There is nothing to resample |

Bootstrapping a per-run quantity is not a suboptimal choice, it is meaningless —
you would be resampling a single number. The manuscript states this explicitly.

## Accuracy metric set

Produced by `NormalizationMetrics.compute_all()`.

| Key | Range | Source | Notes |
|-----|-------|--------|-------|
| `gleu_plus` | `[0, 1]` | `nltk` | RQ1.1 |
| `chrf` | `[0, 100]` | `sacrebleu`, `char_order=6, word_order=0` | RQ1.2. Note the different scale from the others |
| `err` | `(−∞, 1]` | `editdistance` | RQ1.3. **The only metric that can be negative**, when the model makes text worse. Degenerate case (`errors_before = 0`) resolves to `1.0` if perfect, `0.0` otherwise |
| `alpha_word_accuracy` | `[0, 1]` | in-repo | RQ1.4 |

**Inputs required**: predictions, references, and — uniquely for ERR — the
original **noisy** input, since ERR measures error *reduction* and needs the
starting error count.

## Efficiency measurement set

Produced by `EfficiencyBenchmark.benchmark()`.

| Key | Unit | Notes |
|-----|------|-------|
| `avg_inference_time` | seconds per sentence | RQ2.1. Per-sentence, so bootstrappable |
| `peak_gpu_memory_mb` | MB | RQ2.2. Per-run. **Returns `0.0` on CPU**, which is a sentinel, not a measurement — a CI run reports zero memory and that number is meaningless |

**Protocol**: 5 warmup passes discarded → peak counter reset → 20 timed runs.
The reset must follow warmup or warmup allocations inflate every reported peak.

**Manuscript reporting requirement**: peak memory is reported in **GB**;
the code stores MB. Conversion is a presentation step, but the manuscript's
"median difference (in GB)" requirement means the statistics layer must convert,
not just the write-up.

## Bootstrap comparison result

Per metric, per comparison.

| Field | Notes |
|-------|-------|
| `delta` | Observed difference. Sign convention differs by metric direction: higher-is-better uses `M_A − M_F`; lower-is-better (time) uses `M_F − M_A`, so a positive delta always means TAHIMIK won |
| `p_value` | **MUST be two-tailed with add-one smoothing.** Currently one-tailed without it — see `research.md` |
| `ci_lower`, `ci_upper` | 2.5th and 97.5th percentiles of the bootstrap distribution |
| `significant` | **MUST require `p < α` AND the CI excluding zero.** Currently checks the p-value only |

The sign convention is worth noting: it means every reported delta reads the
same direction regardless of whether the metric is "higher is better", which
prevents a whole class of transcription error when writing results up.

## Wilcoxon comparison result (peak memory only)

**Currently returned**: `statistic`, `p_value`, `significant`, `mean_a`,
`mean_b`.

**Required by the manuscript but absent**:

| Field | Why it matters |
|-------|----------------|
| median difference (GB) | The manuscript specifies medians; the code reports means. For 20 measurements with any outlier, these differ |
| interquartile range | Conveys practical magnitude alongside significance |
| matched-pairs rank-biserial `r` | Effect size in `[−1, 1]`. Without it, a significant result has no stated magnitude |

## Holm-Bonferroni family

**Input**: a list of `(comparison_name, raw_p)` pairs.

**Output**: per comparison — `raw_p`, `adjusted_alpha`, `significant`.

**Family composition**: for computational efficiency the manuscript defines the
family as two tests — inference time (bootstrap) and peak memory (Wilcoxon).

**Both raw and adjusted p-values are reported**, per the manuscript's
transparency requirement. Verified present.

## Results file

Written by `scripts/evaluate.py` and `scripts/benchmark.py` via `--output_file`.

**Currently contains**: metric values and comparison results.

**Required by Constitution Principle III but absent**: git SHA, dirty-tree flag,
resolved config. Shared gap with spec `006`. Until fixed, no reported number can
be traced to the code that produced it.
