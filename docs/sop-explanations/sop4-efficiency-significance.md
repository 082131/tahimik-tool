# SOP 4: Statistical Comparison of Computational Efficiency

SOP 4 asks whether TAHIMIK's computational efficiency is significantly different from ByT5 and MrT5. It uses the two SOP 2 metrics:

- inference time per sentence
- peak GPU memory during inference

The comparisons are:

1. TAHIMIK versus ByT5
2. TAHIMIK versus MrT5

Efficiency requires special care because “better” means lower time or lower memory, unlike accuracy where higher is better.

## What data enters SOP 4?

Use the underlying repeated observations, not only the final summary values.

### Inference time

For each test sentence, retain the latency measured for each model. The current benchmark averages repeated runs for each sentence and returns `per_sentence_time_seconds`. These per-sentence arrays are the input to the paired bootstrap.

```text
Sentence 1: ByT5 0.20 s, TAHIMIK 0.15 s
Sentence 2: ByT5 0.30 s, TAHIMIK 0.22 s
Sentence 3: ByT5 0.25 s, TAHIMIK 0.24 s
```

Sentence $i$ must be paired: the ByT5 and TAHIMIK values must refer to the same input sentence.

### GPU memory

For each repeated benchmark run, retain the run-level peak memory for each model:

```text
Run 1: ByT5 2,430 MB, TAHIMIK 1,760 MB
Run 2: ByT5 2,421 MB, TAHIMIK 1,770 MB
Run 3: ByT5 2,435 MB, TAHIMIK 1,765 MB
```

Run 1 for ByT5 must be paired with Run 1 for TAHIMIK under the same hardware and benchmark conditions. Do not use only one final maximum per model for a significance test; one value per model cannot establish variability.

## 1. Inference-time comparison: paired bootstrap

### Why it is paired

The same sentences are timed for both models. Sentence length and difficulty affect generation time, so pairing removes much of this shared variation. The bootstrap resamples sentence indices together.

### Direction of the difference

Lower latency is better. The current code therefore defines the efficiency improvement as:

$$
\Delta_{time}
=\overline{t}_{\text{baseline}}
-\overline{t}_{\text{TAHIMIK}}
$$

A positive value means TAHIMIK is faster. A negative value means TAHIMIK is slower.

### Procedure

1. Calculate the observed difference in mean seconds per sentence.
2. Sample the same sentence indices with replacement for both models.
3. Recalculate the baseline-minus-TAHIMIK mean difference.
4. Repeat 1,000 times.
5. Use the bootstrap distribution for the 95% percentile CI and two-sided empirical p-value.

The same p-value and CI rules described in SOP 3 apply. The code requires p < 0.05 and a CI that excludes zero before the unadjusted result is called significant.

### Wilcoxon ranking formula and worked process

Suppose five matched runs produce these memory values:

| Run | ByT5 (MB) | TAHIMIK (MB) | Difference $d_r$ = TAHIMIK − ByT5 |
|---:|---:|---:|---:|
| 1 | 2,400 | 2,350 | −50 |
| 2 | 2,410 | 2,430 | +20 |
| 3 | 2,405 | 2,365 | −40 |
| 4 | 2,415 | 2,415 | 0 |
| 5 | 2,420 | 2,360 | −60 |

Remove the zero difference, rank the absolute nonzero differences, and restore their signs:

```text
Positive ranks: +1
Negative ranks: −2, −3, −4
```

$$
W^+=1,\qquad W^-=2+3+4=9
$$

$$
W=\min(W^+,W^-)=\min(1,9)=1
$$

A very small $W$ means most paired differences point in one direction. The software converts this statistic into a p-value using the signed-rank distribution.

### Example

```text
ByT5 mean latency:    250 ms/sentence
TAHIMIK mean latency: 145 ms/sentence
```

Because lower is better:

$$
\Delta_{time}=250-145=105\text{ ms/sentence}
$$

The positive 105 means TAHIMIK saves 105 milliseconds per sentence on average. The bootstrap CI and p-value determine whether that observed saving is statistically reliable.

## 2. GPU-memory comparison: paired Wilcoxon signed-rank test

### Why Wilcoxon is used

Peak-memory observations can be non-normal, tied, or based on a relatively small number of repeated runs. The paired Wilcoxon signed-rank test does not require normally distributed differences. It tests whether the paired memory differences are systematically away from zero.

The test is paired because each pair contains the same run number and hardware conditions for two models.

In simpler terms, Wilcoxon is appropriate when we have a small set of matched measurements and do not want to assume that the differences follow a bell-shaped normal distribution. GPU peaks can be tied or affected by allocation behavior, so comparing only the two maximum values would discard useful repeated-run information.

### Difference and interpretation

The code computes differences as:

$$
d_r=m_{r,\text{TAHIMIK}}-m_{r,\text{baseline}}
$$

For memory, a negative difference favors TAHIMIK because it uses less memory. The Wilcoxon test itself is two-sided: it can detect either lower or higher TAHIMIK memory.

### Procedure

1. Pair the memory peaks by run number.
2. Compute the difference for each pair.
3. Remove zero differences for ranking.
4. Rank the absolute nonzero differences.
5. Restore the signs and calculate the signed-rank statistic.
6. Obtain the two-sided p-value.

The null hypothesis is that there is no systematic paired memory difference. The alternative hypothesis is that the typical paired difference is not zero.

### Example

```text
ByT5:    2,430, 2,421, 2,435 MB
TAHIMIK: 1,760, 1,770, 1,765 MB
```

The paired differences (TAHIMIK minus ByT5) are:

```text
-670, -651, -670 MB
```

All differences favor TAHIMIK. The Wilcoxon p-value determines whether the pattern is statistically significant.

## Why the Wilcoxon p-value is two-sided

The code calls SciPy with `alternative="two-sided"`. This tests both possible directions:

```text
TAHIMIK may use significantly less memory than the baseline,
or TAHIMIK may use significantly more memory than the baseline.
```

The hypotheses are:

$$
H_0:\text{the paired memory difference is zero}
$$

$$
H_1:\text{the paired memory difference is not zero}
$$

A two-sided test is appropriate because the comparison should detect an unexpected memory increase as well as a memory reduction. A one-sided test should be used only if a direction was specified before examining the results.

### Descriptive statistics and effect size

The code reports the mean, median, and interquartile range (IQR) for each model:

$$
IQR=Q_{75}-Q_{25}
$$

The IQR describes the middle 50% of memory observations and is less sensitive to an unusually high run than the mean.

These are called descriptive statistics because they summarize what was observed; they do not test significance:

| Statistic | What it tells you |
|---|---|
| Mean | Average memory requirement across runs. |
| Median | Typical middle run, less affected by an unusually large peak. |
| IQR | Spread of the middle half of the runs. A smaller IQR means more consistent memory usage. |
| Wilcoxon p-value | Whether the paired differences provide evidence of a systematic difference. |

For example, two models can have similar medians but one can have a much wider IQR, meaning its memory use is less predictable. Conversely, a statistically significant p-value can occur with a small practical difference, so the descriptive values must be reported alongside the test result.

The code also reports rank-biserial correlation:

$$
r_{rb}=\frac{\text{number of positive nonzero differences}
-\text{number of negative nonzero differences}}
{\text{number of nonzero differences}}
$$

Its sign follows the code's difference direction. A strongly negative value means most nonzero differences favor TAHIMIK's lower memory; a strongly positive value means most favor the baseline. This is an effect-size description, not a substitute for the p-value.

The implementation uses the manuscript's signed-rank-sum rank-biserial
correlation, $(W^+ - W^-)/(n(n+1)/2)$, and reports the median paired memory
difference and its IQR in GB.

## Holm–Bonferroni correction for SOP 4

Each SOP 4 baseline comparison has its own two-test family:

| Comparison | Inference time | GPU memory |
|---|---|---|
| TAHIMIK vs ByT5 | test 1 | test 2 |
| TAHIMIK vs MrT5 | test 3 | test 4 |

Latency and GPU memory for ByT5 versus TAHIMIK form one family; latency and GPU
memory for MrT5 versus TAHIMIK form the other. They are corrected separately.

For each baseline comparison, Holm–Bonferroni sorts its two raw p-values and
compares the value at rank $k$ with:

$$
\alpha_k=\frac{0.05}{2-k+1}
$$

The thresholds are 0.025 then 0.05. The smallest p-value faces the strictest
threshold. The procedure stops declaring later ordered tests significant after
the first failure. Report both raw and adjusted p-values.

## SOP 4 decision workflow

```text
SOP 2 benchmark
       ↓
Per-sentence latency + per-run memory observations
       ↓
Latency: paired bootstrap
Memory: paired Wilcoxon signed-rank
       ↓
Correct the two p-values within each baseline comparison
       ↓
Holm–Bonferroni correction within efficiency family
       ↓
Report means/medians, differences, CI or IQR, raw p, adjusted p, decision
```

## How to report SOP 4

For latency, report average ms per sentence, the signed baseline-minus-TAHIMIK difference, 95% bootstrap CI, raw p-value, adjusted p-value, and significance decision.

For memory, report median and IQR (and optionally mean), the direction of the paired difference, Wilcoxon statistic, raw p-value, adjusted p-value, and rank-biserial correlation.

Example wording:

> TAHIMIK reduced mean inference time by 105 ms per sentence relative to ByT5. The paired-bootstrap confidence interval excluded zero, and the Holm-adjusted p-value was below 0.05, indicating a statistically significant latency reduction.

> TAHIMIK used less peak GPU memory than ByT5 across the matched runs. The Wilcoxon signed-rank test and Holm-adjusted p-value determine whether this consistent reduction is statistically significant.

The final efficiency claim should consider both practical size and statistical evidence. A tiny statistically significant saving may have little operational value, while a large observed saving with a wide CI should be described as uncertain rather than definitive.
