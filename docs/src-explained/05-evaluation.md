# `src/evaluation/` — scoring the trained models (exhaustive line-by-line)

**Flow position:** the **final** stage. After [`src/training`](04-training.md)
produces a checkpoint, these files measure how good it is: accuracy, speed, and
whether differences between models are statistically real.

**Files (order results are produced):**
1. `metrics.py` — normalization accuracy (4 metrics)
2. `efficiency.py` — inference speed + GPU memory
3. `statistical_tests.py` — is the difference significant?
4. `annotation.py` — inter-annotator agreement (validates the *dataset*)
5. `__init__.py` — re-exports

**How they connect:**
```
trained model ─► generate predictions ─► metrics.py (accuracy per model)
                                       └► efficiency.py (time + memory per model)
                                                     └► statistical_tests.py (compare models)
annotation.py stands apart: it scores the gold dataset's reliability, during collection.
```

---

## Terms & abbreviations used across this folder
| Term | Full form / plain meaning |
|---|---|
| metric | a formula turning predictions into a score |
| ground truth / reference | the correct answer to compare against |
| prediction | the model's output |
| numpy (`np`) | the standard library for fast numeric arrays + math |
| corpus-level | one score computed over all sentences together |
| sentence-level | a score per sentence, then averaged |
| n-gram | a run of *n* consecutive items (words or characters) |
| significance | confidence that a measured difference is real, not luck |
| p-value | the probability the difference happened by chance |
| CI | confidence interval — a plausible range for the true value |
| bootstrap | re-measuring by resampling the data many times |
| null hypothesis | the "no real difference" assumption a test tries to reject |
| IAA | inter-annotator agreement |
| α (alpha) | Krippendorff's alpha — the IAA statistic (and the significance level) |

> **Format reminder:** each block starts with **▸ What this block does**, then
> breaks down every line, variable, and technical term.

---

## `metrics.py` — accuracy

**Role:** Compute four normalization-accuracy scores comparing predictions to the
human references.

- **Input:** three parallel lists — `predictions`, `references` (clean gold),
  `noisy_inputs`
- **Output:** a dict `{gleu_plus, chrf, err, alpha_word_accuracy}`
- **Used by:** `scripts/evaluate.py`, `scripts/run_experiment.py`
- **Connects to:** `sacrebleu` (BLEU/chrF), `editdistance`

### Where this fits in TAHIMIK
This answers **Research Question 1: does the model actually normalize well?** It
turns raw model outputs into the four accuracy numbers that fill the results table
— the evidence that any of this works. Every variant is scored the same way, so the
numbers are directly comparable.

**What breaks without it:** you'd have a trained model but no way to say how good it
is. No accuracy column in the thesis.

**Example**
```
metrics.compute_all(predictions, references, noisy_inputs)
# {
#   "gleu_plus": 78.4,   ← 0–100, higher better
#   "chrf": 82.1,        ← 0–100, character overlap
#   "err": 0.63,         ← 63% of the input's errors were fixed
#   "alpha_word_accuracy": 0.71,
# }
```

**Terms & abbreviations in this file**
| Term | Full form / plain meaning |
|---|---|
| BLEU | Bilingual Evaluation Understudy — n-gram overlap accuracy metric |
| GLEU+ | Google-BLEU (plus) — a sentence-level BLEU variant, 0–100 |
| chrF | character n-gram F-score — character-level accuracy, 0–100 |
| ERR | error reduction rate — fraction of the input's errors fixed |
| alpha-word | letters-only word (no digits/punctuation/emoji) |
| smoothing | a fix that avoids zero scores on short sentences |
| edit distance | minimum single-character edits between two strings |
| regex `^[a-zA-Z]+$` | pattern meaning "only letters, start to end" |

### Constructor

```python
    def __init__(self):
        self.bleu_scorer = BLEU(smooth_method="add-k", smooth_value=1)
        self.chrf_scorer = CHRF(char_order=6, word_order=2)
```
**▸ What this block does:** build the two scorers from `sacrebleu`.
- `BLEU(smooth_method="add-k", smooth_value=1)` — a BLEU scorer with **add-one
  smoothing** (adds 1 to counts so short sentences don't score 0).
- `CHRF(char_order=6, word_order=2)` — a chrF scorer using 6-character grams and
  2-word grams.

### Method `compute_gleu_plus`

```python
    def compute_gleu_plus(self, predictions, references):
        scores = []
        for pred, ref in zip(predictions, references):
            score = self.bleu_scorer.sentence_score(pred, [ref])
            scores.append(score.score)
        return sum(scores) / max(len(scores), 1)
```
**▸ What this block does:** score each prediction against its reference, then
average.
- `zip(predictions, references)` — walk both lists together.
- `sentence_score(pred, [ref])` — BLEU for one sentence. `[ref]` is a list because
  BLEU allows multiple references; here there's one. `.score` is the numeric value.
- `sum(scores) / max(len(scores), 1)` — the average (`max(..., 1)` avoids /0).

### Method `compute_chrf`

```python
    def compute_chrf(self, predictions, references):
        result = self.chrf_scorer.corpus_score(predictions, [references])
        return result.score
```
**▸ What this block does:** compute chrF once over the whole corpus.
- `corpus_score(predictions, [references])` — one score for all sentences together.
  `[references]` wraps the reference list as "reference set #1."

### Method `compute_err`

**▸ What this method does (whole function):** measure the fraction of the input's
errors the model actually fixed, using edit distance.

```python
        for pred, ref, noisy in zip(predictions, references, noisy_inputs):
            errors_before = editdistance.eval(noisy, ref)
            errors_after = editdistance.eval(pred, ref)
            if errors_before == 0:
                err = 1.0 if errors_after == 0 else 0.0
            else:
                err = (errors_before - errors_after) / errors_before
            err_scores.append(err)
        return sum(err_scores) / max(len(err_scores), 1)
```
**▸ line notes:**
- `errors_before` — edit distance from the **noisy input** to the gold reference
  (how wrong the input was).
- `errors_after` — edit distance from the **model's output** to the gold (how wrong
  the output is).
- `(errors_before - errors_after) / errors_before` — the fraction of errors
  removed: 1.0 = fixed everything, 0 = no improvement, negative = made it worse.
- `if errors_before == 0:` — the input was already clean; then ERR is 1.0 if the
  output stayed clean, else 0.0 (a `1.0 if ... else 0.0` **conditional expression**).
- The final line averages across sentences.

### Method `compute_alpha_word_accuracy`

**▸ What this method does (whole function):** among letters-only reference words,
count how many the prediction matches at the same position (case-insensitive).

```python
        alpha_pattern = re.compile(r"^[a-zA-Z]+$")
        for pred, ref in zip(predictions, references):
            ref_words = ref.split(); pred_words = pred.split()
            for i, ref_word in enumerate(ref_words):
                if not alpha_pattern.match(ref_word):
                    continue
                total_words += 1
                if i < len(pred_words) and pred_words[i].lower() == ref_word.lower():
                    correct_words += 1
        return correct_words / max(total_words, 1)
```
**▸ line notes:**
- `re.compile(r"^[a-zA-Z]+$")` — precompile a regex meaning "only letters from
  start (`^`) to end (`$`)"; skips words with digits/punctuation/emoji.
- `.split()` — words of each sentence.
- `enumerate(ref_words)` — index + word, so we can compare by position `i`.
- `if not alpha_pattern.match(ref_word): continue` — skip non-letter words.
- `total_words += 1` — count eligible words.
- `i < len(pred_words) and pred_words[i].lower() == ref_word.lower()` — the
  prediction has a word at that position **and** it matches (lowercased for a
  case-insensitive compare).
- Returns the fraction correct.

### Method `compute_all`

```python
    def compute_all(self, predictions, references, noisy_inputs):
        return {"gleu_plus": ..., "chrf": ..., "err": ..., "alpha_word_accuracy": ...}
```
**▸ What this block does:** run all four metrics and return them in one dict —
the single call the scripts use.

---

## `efficiency.py` — speed and memory

**Role:** Measure inference speed and peak GPU memory, following the manuscript's
exact protocol.

- **Input:** a trained model, tokenizer, device, and the test dataset
- **Output:** `{avg_time_per_sentence, std_time_per_sentence, peak_gpu_memory_mb, num_sentences}`
- **Used by:** `scripts/benchmark.py`, `scripts/run_experiment.py`
- **Connects to:** `dataset.py` (test data + `collate_fn`)

### Where this fits in TAHIMIK
This answers **Research Question 2: is the compression actually worth it?** The
entire point of deleting bytes is efficiency, so this measures the payoff — speed
and memory. It's what lets you say "TAHIMIK runs X× faster / uses Y% less memory
than the full ByT5 ceiling." Same hardware and batch size for all three variants,
so the comparison is fair.

**What breaks without it:** the efficiency half of the thesis is unsupported — you
could claim compression is faster but couldn't prove it.

**Example**
```
bench.benchmark(test_dataset, batch_size=1)
# {
#   "avg_time_per_sentence": 0.042,   ← seconds
#   "std_time_per_sentence": 0.003,
#   "peak_gpu_memory_mb": 1180.5,
#   "num_sentences": 1500,
# }
```

**Terms & abbreviations in this file**
| Term | Full form / plain meaning |
|---|---|
| warmup | throwaway first runs (the GPU is misleadingly slow at first) |
| synchronize | wait for the GPU to actually finish (it works asynchronously) |
| peak memory | the maximum memory used during a run |
| `perf_counter` | a high-precision timer |
| std | standard deviation — how spread out the timings are |
| MB | megabyte (1 MB = 1024×1024 bytes) |

### Method `_run_inference`

```python
    @torch.no_grad()
    def _run_inference(self, dataloader):
        self.model.eval()
        for batch in dataloader:
            ...
            if self.device.type == "cuda": torch.cuda.synchronize()
            start = time.perf_counter()
            self.model.generate(input_ids=..., attention_mask=..., max_length=1024, num_beams=...)
            if self.device.type == "cuda": torch.cuda.synchronize()
            total_time += time.perf_counter() - start
        return total_time
```
**▸ What this block does:** run one full pass over the test set and time only the
actual GPU work.
- `@torch.no_grad()` / `self.model.eval()` — inference mode, no gradients.
- `torch.cuda.synchronize()` — the GPU runs work **asynchronously** (it returns
  before finishing), so this waits for it to truly finish. Placed **before** the
  timer starts and **after** generation ends, so the timer brackets only the real
  work.
- `time.perf_counter()` — a high-precision clock; `end - start` = elapsed seconds.
- `self.model.generate(...)` — the actual inference being measured.

### Method `benchmark`

```python
        for _ in range(self.warmup_passes):        # 5 warmup runs (discarded)
            self._run_inference(dataloader)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        for run in range(self.inference_runs):      # 20 timed runs
            run_time = self._run_inference(dataloader)
            per_run_times.append(run_time / max(num_sentences, 1))
        avg_time = times_tensor.mean().item()
        std_time = times_tensor.std().item()
        peak_memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024*1024)
```
**▸ What this block does:** the manuscript protocol — 5 warmup passes (discarded),
reset the memory counter, 20 timed passes, then report averages.
- `for _ in range(self.warmup_passes):` — run 5 times without recording (the first
  GPU calls are slow/misleading).
- `torch.cuda.reset_peak_memory_stats(...)` — zero the memory counter **after**
  warmup so warmup allocations don't inflate the peak.
- `run_time / max(num_sentences, 1)` — convert total time into **per-sentence**
  time.
- `.mean().item()` / `.std().item()` — average and standard deviation of the 20
  runs, as plain numbers (`.item()` extracts a Python float from a size-1 tensor).
- `torch.cuda.max_memory_allocated(...) / (1024*1024)` — peak GPU memory in bytes,
  converted to **MB**.

---

## `statistical_tests.py` — is the difference real?

**Role:** Decide whether one model genuinely beats another, or the gap is random
noise. Three tests + one orchestrator.

- **Input:** per-sentence scores per model, and GPU-memory runs per model
- **Output:** a nested dict of p-values, CIs, and corrected significance per pair
- **Used by:** `scripts/run_experiment.py`
- **Connects to:** `numpy`, `scipy.stats`

### Where this fits in TAHIMIK
This turns **"model A scored higher" into "model A is *significantly* better (or
not)."** It's the difference between a claim and a defensible claim. An examiner
*will* ask "is that gap real or just noise?" — this file is the answer, and it's why
the study can state its conclusions with confidence rather than hand-waving.

**What breaks without it:** every comparison would be anecdotal. A 2-point GLEU+
difference might be luck; without significance testing you couldn't tell, and the
thesis's conclusions would be unsupported.

**Example**
```
paired_bootstrap(byt5_scores, tahimik_scores)
# {
#   "mean_a": 76.1, "mean_b": 78.4, "delta": 2.3,
#   "p_value": 0.012, "ci_lower": 0.9, "ci_upper": 3.6,
#   "significant": True,   ← p < 0.05 (see FINDINGS #1–2 on the CI/one-tailed caveat)
# }
```

**Terms & abbreviations in this file**
| Term | Full form / plain meaning |
|---|---|
| p-value | probability the difference happened by luck; small (<0.05) = "probably real" |
| CI | confidence interval — a plausible range for the true difference |
| alpha (`self.alpha`) | the significance threshold (0.05) |
| bootstrap | resampling the data many times to estimate reliability |
| with replacement | resampling where the same item can be picked more than once |
| percentile | the value below which a given % of the data falls |
| Wilcoxon signed-rank | a test for paired data that doesn't assume a normal distribution |
| non-parametric | making no assumption about the data's distribution shape |
| Holm-Bonferroni | a correction for running many tests at once |
| FWER | family-wise error rate — the chance of *any* false positive across all tests |
| `np.array` | a numpy numeric array (fast math) |

### Method `paired_bootstrap`

**▸ What this method does (whole function):** the main accuracy test — resample the
per-sentence scores 1000 times to estimate how reliably model B beats model A,
producing a p-value and a 95% confidence interval of the difference.

**Variables at a glance**
| Variable | Type | Holds |
|---|---|---|
| `scores_a`, `scores_b` (params) | lists of floats | Per-sentence scores for two models |
| `a`, `b` | `np.array` | Those scores as numpy arrays |
| `n` | `int` | Number of sentences |
| `observed_delta` | `float` | B's mean minus A's mean (the real gap) |
| `wins_a` | `int` | How many resamples A won |
| `bootstrap_deltas` | list | The gap in each resample |
| `p_value`, `ci_lower`, `ci_upper` | `float` | The results |

```python
        a = np.array(scores_a); b = np.array(scores_b); n = len(a)
        observed_delta = b.mean() - a.mean()
```
**▸ What this block does:** turn the scores into numpy arrays and compute the real
observed difference.
- `np.array(...)` — a fast numeric array. `.mean()` — the average.
- `observed_delta` — how much B beats A on the real data.

```python
        for _ in range(self.n_bootstrap):
            indices = self.rng.randint(0, n, size=n)
            boot_a = a[indices].mean(); boot_b = b[indices].mean()
            bootstrap_deltas.append(boot_b - boot_a)
            if boot_a >= boot_b: wins_a += 1
```
**▸ What this block does:** 1000 times, resample the sentences **with replacement**
and recompute each model's mean, tracking the gap and how often A wins.
- `self.rng.randint(0, n, size=n)` — `n` random position numbers in `[0, n)`,
  **with replacement** (a position can repeat). This is the bootstrap resample.
- `a[indices]` — the scores at those positions; `.mean()` — their average.
- `if boot_a >= boot_b: wins_a += 1` — count resamples where A did at least as well
  as B.

```python
        p_value = wins_a / self.n_bootstrap
        ci_lower = np.percentile(bootstrap_deltas, 2.5)
        ci_upper = np.percentile(bootstrap_deltas, 97.5)
        result = {..., "significant": p_value < self.alpha, "ci_lower": ..., "ci_upper": ...}
```
**▸ What this block does:** turn the counts into a p-value and a 95% CI.
- `wins_a / self.n_bootstrap` — the fraction of resamples A won = the p-value.
- `np.percentile(bootstrap_deltas, 2.5)` / `97.5` — the 2.5th and 97.5th
  percentiles of the gaps = the **95% confidence interval**.
- `"significant": p_value < self.alpha` — flagged significant if p < 0.05.

> ⚠️ **Known issue (`FINDINGS.md` #1–2):** this counts one-sided wins and flags
> `significant` from the p-value alone. The manuscript specifies a **two-tailed**
> test **and** requires the CI to exclude 0. Fix before reporting significance.

### Method `wilcoxon_test`

```python
        differences = b - a
        if np.all(differences == 0):
            return {..., "p_value": 1.0, "significant": False, ...}
        stat, p_value = stats.wilcoxon(a, b, alternative="two-sided")
```
**▸ What this block does:** compare GPU-memory measurements (few samples, not
normally distributed) with a non-parametric test.
- `b - a` — element-wise differences. `np.all(differences == 0)` — if every pair is
  identical, there's nothing to test; return "not significant."
- `stats.wilcoxon(a, b, alternative="two-sided")` — the **Wilcoxon signed-rank
  test**, correctly **two-sided**. Returns a statistic and a p-value.

### Method `holm_bonferroni`

```python
        sorted_pairs = sorted(p_values, key=lambda x: x[1])
        for rank, (name, p) in enumerate(sorted_pairs, start=1):
            adjusted_alpha = alpha / (m - rank + 1)
            significant = p < adjusted_alpha
```
**▸ What this block does:** correct for running many tests at once, so you don't
get a false "significant" by luck.
- `sorted(p_values, key=lambda x: x[1])` — sort the `(name, p)` pairs by p-value
  ascending. `lambda x: x[1]` is a tiny inline function returning the p-value to
  sort on.
- `enumerate(..., start=1)` — rank from 1.
- `adjusted_alpha = alpha / (m - rank + 1)` — the **Holm-Bonferroni** step: the
  smallest p-value faces the strictest threshold, loosening by rank. `m` = number
  of tests.
- `significant = p < adjusted_alpha` — pass only if below the adjusted threshold.
  This controls the **family-wise error rate** (chance of any false positive).

### Method `run_full_comparison`

```python
        pairs = [("byt5","mrt5"), ("byt5","tahimik"), ("mrt5","tahimik")]
        metrics = ["gleu_plus","chrf","err","alpha_word_accuracy"]
        ... bootstrap each pair×metric, wilcoxon each pair on memory, then holm_bonferroni ...
```
**▸ What this block does:** the orchestrator — run the bootstrap on every
model-pair × accuracy-metric, the Wilcoxon test on every pair's memory, collect all
the p-values, then apply Holm-Bonferroni across them. Returns one big results dict.

---

## `annotation.py` — inter-annotator agreement

**Role:** Measure how consistently the human annotators agreed when building the
gold dataset. Validates the *data*, not a model — conceptually part of data
collection, but placed here with the other statistics.

- **Input:** a table of annotations (rows = annotators, cols = items), or raw
  annotator outputs + a reference
- **Output:** `{alpha, interpretation, num_annotators, num_items}`
- **Used by:** the reliability analysis for the gold dataset
- **Connects to:** `numpy`, the optional `krippendorff` package, `editdistance`

### Where this fits in TAHIMIK
This validates **the gold dataset itself, before any model touches it.** If three
human annotators normalize the same sentence very differently, the "gold standard"
isn't gold — and every result trained or tested on it is suspect. Krippendorff's α
is the number that lets you say "our annotators agreed reliably (α ≥ 0.80), so the
dataset is trustworthy." It belongs to *data collection* (the annotation platform),
not model scoring, but lives here with the other statistics.

**What breaks without it:** you couldn't defend the quality of your ground truth —
a question a methodology-focused examiner will absolutely ask.

**Example**
```
iaa.compute(reliability_data)     # rows = annotators, cols = sentences
# {
#   "alpha": 0.83,
#   "interpretation": "Reliable agreement (alpha >= 0.80)",
#   "num_annotators": 3, "num_items": 500,
# }
```

**Terms & abbreviations in this file**
| Term | Full form / plain meaning |
|---|---|
| IAA | inter-annotator agreement |
| Krippendorff's α | a 0–1 agreement statistic (≥0.80 reliable, ≥0.67 tentative) |
| reliability data | a table: rows = annotators, columns = items |
| `np.nan` | "not a number" — marks a missing value (annotator didn't rate this item) |
| `try/except` | run code, and handle an error instead of crashing |
| `ImportError` | the error raised when an optional package isn't installed |
| level of measurement | the data type for α ("ratio" for continuous values) |

### Optional-import guard

```python
try:
    import krippendorff as krippendorff_lib
    HAS_KRIPPENDORFF = True
except ImportError:
    HAS_KRIPPENDORFF = False
```
**▸ What this block does:** try to import the optional `krippendorff` package; if
it's not installed, set a flag instead of crashing.
- `try: ... except ImportError:` — attempt the import; catch the specific error
  raised when the package is missing.
- `HAS_KRIPPENDORFF` — a boolean flag checked later.

### Method `compute`

```python
        if not HAS_KRIPPENDORFF:
            return {"alpha": None, "interpretation": "Package not installed", ...}
        alpha = krippendorff_lib.alpha(reliability_data=reliability_data,
                                       level_of_measurement=level_of_measurement)
        if alpha >= 0.80: interpretation = "Reliable agreement (alpha >= 0.80)"
        elif alpha >= 0.67: interpretation = "Tentative agreement ..."
        else: interpretation = "Unreliable agreement ..."
        return {"alpha": float(alpha), "interpretation": interpretation, ...}
```
**▸ What this block does:** compute α and translate the number into a plain-English
verdict (or report the package is missing).
- `if not HAS_KRIPPENDORFF:` — degrade gracefully if the library is absent.
- `krippendorff_lib.alpha(...)` — compute the statistic from the table.
- `reliability_data` — the annotator×item table (`np.nan` marks unrated cells).
- `level_of_measurement="ratio"` — treat the values as continuous ratios.
- The `if/elif/else` maps the number to the reliability bands from the manuscript.

### Method `compute_from_normalizations`

```python
        dist = editdistance.eval(output, ref)
        reliability_data[a_idx, i_idx] = dist / max(len(output), len(ref), 1)
        return self.compute(reliability_data, level_of_measurement="ratio")
```
**▸ What this block does:** a convenience path — convert each annotator's raw text
into an edit-distance ratio against the consensus reference (same idea as n\*),
fill the table, and compute α.
- `editdistance.eval(output, ref)` — edits between an annotator's output and the
  reference.
- `dist / max(len(output), len(ref), 1)` — normalize into [0,1] (the `1` avoids /0).
- `reliability_data[a_idx, i_idx] = ...` — store it at row = annotator, column =
  item.

---

## `__init__.py`

```python
from src.evaluation.metrics import NormalizationMetrics
from src.evaluation.efficiency import EfficiencyBenchmark
from src.evaluation.statistical_tests import StatisticalAnalysis
from src.evaluation.annotation import InterAnnotatorAgreement
```
**▸ What this block does:** re-exports the four public classes so other code can do
`from src.evaluation import NormalizationMetrics`, etc.

- **Inputs:** none (runs on import)
- **Outputs:** the four names above, importable from `src.evaluation`

---

## You've now covered all of `src/` at full depth

| Folder | Doc | Role in the flow |
|---|---|---|
| `utils` | [01](01-utils.md) | shared toolbox |
| `data` | [02](02-data.md) | make + tensorize training pairs |
| `models` | [03](03-models.md) | the three networks |
| `training` | [04](04-training.md) | loss + loop → checkpoint |
| `evaluation` | [05](05-evaluation.md) | score accuracy, speed, significance |

Full loop: **data → models → training → evaluation**, with **utils** underneath.
Abbreviations and technical terms are in each file's box and the
[README glossary](README.md).
