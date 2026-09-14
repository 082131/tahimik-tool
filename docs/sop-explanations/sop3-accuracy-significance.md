# SOP 3: Statistical Comparison of Accuracy

SOP 3 asks whether TAHIMIK's accuracy is significantly different from the accuracy of ByT5 and MrT5. The four accuracy metrics come from SOP 1:

- GLEU+
- chrF
- ERR
- Alpha-word Accuracy

The comparisons are planned as:

1. TAHIMIK versus ByT5
2. TAHIMIK versus MrT5

Because the same test sentences are evaluated by every model, the observations are paired.

## What data enters SOP 3?

Do not use only the three final dataset averages. For each sentence, save each model's score:

| Sentence | ByT5 GLEU+ | TAHIMIK GLEU+ | ByT5 chrF | TAHIMIK chrF |
|---|---:|---:|---:|---:|
| 1 | ... | ... | ... | ... |
| 2 | ... | ... | ... | ... |
| ... | ... | ... | ... | ... |

The same arrangement is made for ERR and Alpha-word Accuracy. Sentence $i$ must remain matched across models; never independently shuffle one model's scores.

Use unrounded per-sentence values. Round only the final means, confidence intervals, and p-values shown in the results table.

## The complete SOP 3 flow

SOP 3 is not a second accuracy calculation. SOP 1 already calculates each model's metric score for each sentence. SOP 3 uses those saved scores to test whether the observed model differences are reliable.

```text
Same test sentences
        ↓
SOP 1: calculate GLEU+, chrF, ERR, and Alpha-word Accuracy per sentence
        ↓
Keep sentence i paired across ByT5, MrT5, and TAHIMIK
        ↓
For each metric and model comparison:
  calculate the observed mean difference
  resample paired sentence indices 1,000 times
  calculate one difference per resample
  form the bootstrap CI and raw p-value
        ↓
Collect the eight accuracy raw p-values
        ↓
Apply Holm–Bonferroni to the accuracy family
        ↓
Use the adjusted p-value for the final significance decision
```

### What comes from the manuscript and what comes from the code

| Item | Manuscript requirement | Code implementation |
|---|---|---|
| Research question | Test whether TAHIMIK differs from ByT5 and MrT5 in four accuracy metrics. | `run_full_comparison()` uses those two pairs and four metric names. |
| Test type | Paired bootstrap for accuracy comparisons. | `paired_bootstrap()` resamples paired sentence scores. |
| Multiple testing | Holm correction is required. | `holm_bonferroni()` corrects the eight accuracy p-values together. |
| Number of resamples | Not specified in the manuscript. | Operational setting: 1,000. |
| Confidence interval | Not specified in the manuscript. | Operational setting: 95% percentile interval. |
| Empirical p-value details | Not specified in the manuscript. | Two-sided tail count with add-one smoothing. |
| Final decision threshold | Statistical significance is tested. | Code uses alpha = 0.05; final reporting should use Holm-adjusted p-values. |

The code therefore follows the manuscript's requested method and adds reproducible details that the manuscript leaves open. Those details should be stated in the methodology so another researcher can repeat the analysis.

## Code-input requirement

The arrays passed to `paired_bootstrap()` must be the **per-sentence outputs from SOP 1**, not the single dataset averages:

```python
analysis.paired_bootstrap(
    per_sentence_scores["byt5"]["chrf"],
    per_sentence_scores["tahimik"]["chrf"],
    metric_name="chrf",
    higher_is_better=True,
)
```

For this call, both lists have one chrF value for every identical test sentence. If one model's list is reordered or has a different length, the pairing is invalid; the code correctly raises an error for unequal lengths but cannot detect an incorrect manual reordering.

## Why paired bootstrap is used

The paired bootstrap is suitable because every model receives the same sentence. A difficult sentence may produce low scores for all models, while an easy sentence may produce high scores for all models. Pairing controls for this shared sentence difficulty.

It also avoids relying on a normal-distribution assumption for the metric scores. This is useful because normalization scores can be bounded, skewed, or contain many tied values.

## The comparison quantity

For a higher-is-better accuracy metric, the code defines the observed difference as:

$$
\Delta=\overline{X}_{\text{TAHIMIK}}-\overline{X}_{\text{baseline}}
$$

where the baseline is either ByT5 or MrT5. A positive $Δ$ means TAHIMIK has the higher average score; a negative $Δ$ means the baseline is higher.

### Formula elements

| Element | Meaning |
|---|---|
| $X_{\text{TAHIMIK},i}$ | TAHIMIK's score for test sentence $i$. |
| $X_{\text{baseline},i}$ | ByT5's or MrT5's score for the same test sentence $i$. |
| $N$ | Number of paired test sentences. |
| $\overline{X}$ | Mean score across the $N$ sentences. |
| $\Delta$ | Difference in mean scores, using TAHIMIK minus the baseline. |

The observed difference can also be written as the average of the paired sentence differences:

$$
\Delta
=\frac{1}{N}\sum_{i=1}^{N}
\left(X_{\text{TAHIMIK},i}-X_{\text{baseline},i}\right)
$$

For accuracy metrics, a positive result favors TAHIMIK. This is used for GLEU+, chrF, ERR, and Alpha-word Accuracy.

## How paired bootstrap works

Assume there are $N$ test sentences and a pair of scores for each sentence.

1. Calculate the observed mean difference using all $N$ paired sentences.
2. Randomly select $N$ sentence indices with replacement. The same indices are used for both models.
3. Calculate the mean difference for this resampled paired dataset.
4. Repeat the resampling 1,000 times, as configured in the code.
5. The 1,000 resampled differences form the bootstrap distribution.

Sampling with replacement means some sentences can appear more than once in a bootstrap sample and others may be absent. This simulates how the observed difference would vary across comparable samples from the test population.

### Small conceptual example

Suppose the paired differences are:

```text
TAHIMIK - ByT5:  +4, +2, -1, +3
```

The observed mean difference is:

$$
\Delta=\frac{4+2-1+3}{4}=2
$$

A resample might select sentences 1, 1, 3, and 4, giving:

```text
+4, +4, -1, +3
```

Its bootstrap difference is 2.5. Repeating this many times shows how stable the estimated improvement is.

### Complete test-set example

Consider four paired test sentences with Alpha-word Accuracy scores:

| Sentence | ByT5 | TAHIMIK | Difference |
|---|---:|---:|---:|
| 1 | 70 | 75 | +5 |
| 2 | 80 | 78 | −2 |
| 3 | 60 | 65 | +5 |
| 4 | 90 | 95 | +5 |

The observed difference uses all four original test sentences:

$$
\Delta=\frac{(75-70)+(78-80)+(65-60)+(95-90)}{4}=3.25
$$

For one bootstrap repetition, suppose the randomly selected indices are $[1,1,3,4]$. The same indices are used for both models:

```text
ByT5:    70, 70, 60, 90
TAHIMIK: 75, 75, 65, 95
```

The resampled difference is:

$$
\Delta^{*}_1
=\frac{75+75+65+95}{4}-\frac{70+70+60+90}{4}
=78.75-72.50=6.25
$$

Another repetition might select $[2,2,2,3]$, producing a different difference of $-0.25$. After 1,000 repetitions, the 1,000 resampled differences form the bootstrap distribution.

## Exact code implementation

The implementation is in `src/evaluation/statistical_tests.py`, inside `StatisticalAnalysis.paired_bootstrap()`.

| Code operation | Statistical meaning |
|---|---|
| `a = np.asarray(scores_a)` | Converts the baseline's per-sentence test scores into a numeric array. |
| `b = np.asarray(scores_b)` | Converts TAHIMIK's per-sentence test scores into a numeric array. |
| Equal-length check | Confirms both models were evaluated on the same test sentences. |
| `observed_delta = b.mean() - a.mean()` | Computes the original observed mean difference. |
| `idx = self.rng.randint(0, n, size=n)` | Selects $N$ sentence indices with replacement. |
| `a[idx]`, `b[idx]` | Applies the same selected sentences to both models, preserving pairing. |
| `deltas.append(boot_delta)` | Stores one bootstrap mean difference. |
| `n_bootstrap=1000` | Repeats the resampling 1,000 times. |
| `np.percentile(deltas, [2.5, 97.5])` | Produces the 95% percentile confidence interval. |

The `run_full_comparison()` method calls `paired_bootstrap()` for four metrics in each of two comparisons (ByT5 versus TAHIMIK and MrT5 versus TAHIMIK). This produces eight raw accuracy p-values, which are passed to `holm_bonferroni()` as one accuracy family.

## Confidence interval and p-value

The code uses the 2.5th and 97.5th percentiles of the bootstrap differences as the 95% percentile confidence interval:

$$
CI_{95\%}=[\text{2.5th percentile},\ \text{97.5th percentile}]
$$

This interval is a range of plausible values for the population-level model difference. It is not a range of individual sentence scores. For example, an interval of $[1.2, 6.8]$ percentage points is entirely positive, while $[-1.5, 6.8]$ includes zero and does not establish which model is better.

The code also calculates a two-sided empirical p-value. It counts how often bootstrap differences are at or below zero and how often they are at or above zero. Add-one smoothing uses:

$$
\text{tail proportion}=\frac{1+\text{tail count}}{1+B}
$$

where $B=1,000$ is the number of bootstrap samples. The extra 1 prevents a finite bootstrap run from reporting an impossible-looking p-value of exactly 0.

In the implementation, the two-sided p-value is:

$$
p=\min\left(1,\ 2\min\left[
\frac{1+\#(\Delta^*\leq0)}{B+1},
\frac{1+\#(\Delta^*\geq0)}{B+1}
\right]\right)
$$

The first tail counts bootstrap differences at or below zero; the second counts differences at or above zero. Zero is the null value because it represents no difference between the models.

The implementation declares an unadjusted result significant only when both conditions hold:

```text
p-value < 0.05
and the 95% confidence interval does not include zero
```

The confidence interval answers “how large and stable is the difference?” The p-value answers “how incompatible is a zero difference with the resampling distribution?” They provide related but different evidence.

## Eight accuracy tests and Holm–Bonferroni correction

After the eight p-values are corrected, the Holm result is the decision used for SOP 3. The unadjusted `significant` field returned by `paired_bootstrap()` is an intermediate diagnostic; use `significant_corrected` from `holm_bonferroni()` for the final claim.

SOP 3 contains eight planned tests:

| Comparison | GLEU+ | chrF | ERR | Alpha-word Accuracy |
|---|---|---|---|---|
| TAHIMIK vs ByT5 | test 1 | test 2 | test 3 | test 4 |
| TAHIMIK vs MrT5 | test 5 | test 6 | test 7 | test 8 |

If each test uses 0.05 without adjustment, the chance of at least one false positive across the family is greater than 5%. This is the multiple-comparisons problem.

Holm–Bonferroni controls the family-wise error rate while retaining more power than the ordinary Bonferroni procedure.

### Step-down procedure

1. Collect the eight raw p-values.
2. Sort them from smallest to largest.
3. Let $m=8$. Compare the p-value at rank $k$ with:

$$
\alpha_k=\frac{0.05}{m-k+1}
$$

4. Start with the smallest p-value. Continue rejecting while each p-value meets its rank-specific threshold.
5. Once one ordered test fails, later tests are not declared significant by the step-down rule.
6. Report adjusted p-values as well as raw p-values.

For eight tests, the first thresholds are 0.00625, 0.00714, 0.00833, and so on. The exact threshold depends on the p-value's rank after sorting, not on which metric name it has.

### Why adjust p-values instead of only changing the threshold?

An adjusted p-value lets the reader compare every result with the same 0.05 decision rule. It incorporates the number of tests and their ordering. A result is Holm-significant when its adjusted p-value is below 0.05, subject to the step-down procedure.

The code keeps accuracy and efficiency as separate correction families. Therefore, the eight SOP 3 accuracy p-values are corrected together, not mixed with SOP 4 efficiency p-values.

## How to report SOP 3

For every metric and comparison, report:

- ByT5 or MrT5 mean
- TAHIMIK mean
- signed difference (TAHIMIK minus baseline)
- 95% bootstrap CI
- raw p-value
- Holm-adjusted p-value
- final corrected significance decision

Example wording:

> TAHIMIK scored 85.71% Alpha-word Accuracy compared with 80.00% for ByT5, an improvement of 5.71 percentage points. The paired-bootstrap 95% CI was [2.10, 9.34] percentage points. After Holm–Bonferroni correction, the adjusted p-value was below 0.05, indicating a statistically significant difference.

Statistical significance does not automatically mean the improvement is practically important. Always show the effect size (the difference and CI) beside the p-value.
