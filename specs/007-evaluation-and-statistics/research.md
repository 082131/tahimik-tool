# Phase 0 Research: Evaluation and Statistical Testing

The manuscript is unusually precise in this section — it gives formulas, not
descriptions. That precision is why the divergences found here are unambiguous
rather than matters of interpretation.

No design question is left open. Every decision below is settled by the
manuscript; the gaps are code that disagrees with it.

## Decision: six metrics, fixed by the research questions

**Decision**: Four accuracy metrics (GLEU+, chrF, ERR, alpha-word accuracy) and
two efficiency metrics (average inference time per sentence, peak GPU memory).

**Rationale**: These are not selected, they are the research questions. RQ1 names
the four accuracy metrics by subsection (1.1–1.4); RQ2 names the two efficiency
metrics (2.1–2.2). Dropping one leaves a stated research question unanswered.

**Alternatives considered**: Adding BLEU or word error rate for familiarity.
Rejected — extra metrics enlarge the multiple-comparison family, which makes the
comparisons the study actually asks about harder to pass. There is a real
statistical cost to answering questions nobody asked.

## Decision: ERR's degenerate case gets a defined convention

**Decision**: When `errors_before = 0`, return `1.0` if the prediction is
perfect and `0.0` otherwise.

**Rationale**: `ERR = (errors_before − errors_after) / errors_before` is
undefined when the noisy input was already clean. The convention at
`metrics.py:109-112` is defensible: if there was nothing to fix, leaving the
sentence alone is complete success, and altering it is complete failure.

**Worth confirming with the author**, but it is a real convention rather than a
fudge, and the alternative — excluding such sentences from the average —
silently changes the denominator in a way readers would not see.

**Related property**: ERR is the only metric here that can go **negative**, when
the model makes text worse than it found it. That is a feature. A metric unable
to express harm would hide it.

## Decision: peak GPU memory uses Wilcoxon, never bootstrap

**Decision**: Paired bootstrap for accuracy metrics and inference time; Wilcoxon
signed-rank for peak GPU memory.

**Rationale**: Manuscript — *"Peak GPU memory cannot be resampled at the
sentence level because it is measured once per inference run. Therefore, the 20
paired peak-memory measurements for each model comparison will be analyzed using
the Wilcoxon signed-rank test instead of paired bootstrap resampling."*

Bootstrap works by resampling sentences. Accuracy and per-sentence timing have a
value per sentence, so resampling means something. Peak memory is one number per
*run* — there is nothing to resample. Bootstrapping it would be statistically
meaningless, not merely suboptimal.

**Alternatives considered**: Bootstrapping memory for methodological uniformity.
Rejected for the reason above.

## Decision: the bootstrap p-value is TWO-TAILED

**Decision**: Two-tailed, with add-one smoothing, per the manuscript's explicit
formula.

**Rationale**: H₀₁ and H₀₂ both state *"there is no significant difference"*.
"No difference" is a two-sided null — a difference in either direction refutes
it. The manuscript then removes any ambiguity by giving the formula:

```
p = 2 × min( (1 + Σ I(Δb ≤ 0))/(B+1),  (1 + Σ I(Δb ≥ 0))/(B+1) )
```

The leading `2 ×` is the two-tailed correction. The `+1` terms are add-one
smoothing, which prevents reporting an impossible `p = 0` — with B = 1,000 the
floor becomes roughly 0.002.

**This is the highest-severity divergence in the repository.**
`statistical_tests.py:94` computes `p_value = wins_a / self.n_bootstrap` —
one-tailed, no smoothing. Its own docstring states the intent: *"The null
hypothesis is that A >= B."* That is a one-sided null, which contradicts the
manuscript's hypotheses.

A one-tailed test is roughly twice as easy to pass. Any result currently
reported as significant may not survive the method the manuscript describes.

**Corroborating evidence this is an oversight, not a choice**: `wilcoxon_test`
in the same class passes `alternative="two-sided"` (`statistical_tests.py:154`).
The two significance tests in one file disagree on tail count. A deliberate
methodological decision would have been applied consistently.

## Decision: significance requires BOTH p and CI

**Decision**: A comparison is significant only if the adjusted p-value is below
α **and** the confidence interval excludes zero.

**Rationale**: Manuscript — *"A comparison will be considered statistically
significant if the adjusted p-value is less than the significance level **and**
the confidence interval does not include zero."*

They are not redundant; they answer different questions. The p-value asks how
often the difference reversed sign under resampling. The CI asks what range of
true differences is plausible. A CI spanning `[−0.001, 0.42]` says the effect
might be nothing at all, even if resampling rarely flipped its sign. Requiring
both guards against a technically-significant but practically-empty result.

**Divergence**: `statistical_tests.py:106` sets
`significant = p_value < self.alpha` only. The CI is computed two lines earlier
(`:98-99`) and then never consulted.

## Decision: Holm-Bonferroni, not plain Bonferroni

**Decision**: Holm-Bonferroni correction, reporting both raw and adjusted
p-values.

**Rationale**: Running several tests at α = 0.05 means something will look
significant by chance. Plain Bonferroni divides α by the number of tests — safe
but conservative, and it costs real statistical power on a study that already
has limited data.

Holm sorts the p-values and applies progressively less severe thresholds
(`α/m`, `α/(m−1)`, …), achieving the same family-wise error control with more
power. The manuscript specifies it, along with reporting both unadjusted and
adjusted values *"for transparency"*.

**Verified present**: `statistical_tests.py:175`, returning both.

## Decision: 5 warmup passes before 20 timed runs

**Decision**: Discard 5 warmup passes, reset the peak-memory counter, then run
20 timed iterations.

**Rationale**: The first CUDA calls on a fresh context are dramatically slower
than steady state — kernel compilation, memory-pool initialisation, autotuning.
Timing them would measure setup rather than inference, and would systematically
disadvantage whichever model ran first.

The counter reset must happen **after** warmup, or warmup's allocations inflate
every reported peak. `efficiency.py:135` does this correctly.

## Decision: two comparisons only

**Decision**: ByT5-vs-TAHIMIK and MrT5-vs-TAHIMIK. Not ByT5-vs-MrT5.

**Rationale**: RQ3 and RQ4 ask about *"the proposed noise-adaptive ByT5 model
compared to (a) ByT5 and (b) MrT5"*. The baseline-versus-baseline comparison is
not asked, and is already established by MrT5's own paper.

**Additional reason not to add it**: a third comparison enlarges the
multiple-comparison family, raising the correction burden and making the two
comparisons the study does care about harder to pass. There is a measurable cost
to testing a question you did not ask.

## Gap: Wilcoxon reporting is incomplete

**Status**: Verified gap, not an open question.

The manuscript requires, alongside the p-value: the **median** difference in GB,
its **interquartile range**, and the **matched-pairs rank-biserial correlation**
as an effect size.

`wilcoxon_test` (`statistical_tests.py:120-171`) returns `statistic`, `p_value`,
`significant`, `mean_a`, and `mean_b`. It reports **means where the manuscript
asks for medians**, and computes no effect size or IQR at all.

The test itself is correct and correctly two-sided. Only the reporting is
incomplete.
