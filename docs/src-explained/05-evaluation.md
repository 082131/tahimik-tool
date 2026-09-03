# `src/evaluation/` — scoring the trained models (exhaustive, every line)

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

> **Format:** every line of the real source is shown and explained (including
> comments, `logger.info`, and `return`s). Each block opens with **▸ What this
> block does**.

---

## Current methodology implementation (2026-09-03)

This section supersedes the older method bodies below. It explains the code merged
in `3dda429` and separates what works now from the remaining spec-014 corrections.

### `metrics.py`: current scoring flow

```python
def _sentence_gleu_plus(self, prediction: str, reference: str, noisy: str = "") -> float:
    pred, ref, source = prediction.split(), reference.split(), noisy.split()
    if not pred or not ref:
        return 100.0 if pred == ref else 0.0
    scores = []
    for order in range(1, 5):
        pc, rc = self._ngrams(pred, order), self._ngrams(ref, order)
        if not pc:
            continue
        overlap = sum((pc & rc).values())
        precision = overlap / max(sum(pc.values()), 1)
        recall = overlap / max(sum(rc.values()), 1)
        source_overlap = sum((pc & self._ngrams(source, order)).values()) if source else 0
        penalty = 1.0 - source_overlap / max(sum(pc.values()), 1) if source else 1.0
        scores.append(min(precision, recall) * max(penalty, 0.0))
    return 100.0 * sum(scores) / max(len(scores), 1)
```

**▸ Line-by-line calculation:**

- `.split()` turns the prediction, clean reference, and noisy source into word lists.
- An empty prediction/reference pair scores `100`; only one empty side scores `0`.
- `range(1, 5)` evaluates 1-, 2-, 3-, and 4-word n-grams.
- `_ngrams` returns `Counter` objects. `pc & rc` keeps the minimum count shared
  by prediction and reference.
- Precision divides overlap by predicted n-grams; recall divides by reference
  n-grams. `max(..., 1)` prevents division by zero.
- `source_overlap` counts prediction n-grams also found in the noisy source.
- The penalty subtracts that source-copy fraction, then the method multiplies the
  smaller of precision/recall by the nonnegative penalty.
- The final line averages available n-gram orders and scales to 0–100.

**Known correctness gap:** this code penalizes *all* source overlap, including text
that was already correct. For noisy = reference = prediction, overlap is valid but
the penalty becomes zero. Spec 014 requires penalizing only incorrectly preserved
source material and giving this clean unchanged triple a perfect score.

```python
def compute_per_sentence(self, predictions, references, noisy_inputs):
    return {
        "gleu_plus": self.compute_gleu_plus(predictions, references, noisy_inputs),
        "chrf": [self.chrf_scorer.sentence_score(p, [r]).score for p, r in zip(predictions, references)],
        "err": [self._err(p, r, n) for p, r, n in zip(predictions, references, noisy_inputs)],
        "alpha_word_accuracy": [self._alpha(p, r) for p, r in zip(predictions, references)],
    }

def compute_all(self, predictions, references, noisy_inputs):
    per_sentence = self.compute_per_sentence(predictions, references, noisy_inputs)
    return {key: float(sum(values) / max(len(values), 1)) for key, values in per_sentence.items()}
```

`zip` preserves pairing by sentence. Each list comprehension produces one score
per held-out item, which is what paired bootstrap needs. `compute_all` then averages
each vector.

**ERR example:** if sentence A has 1 source error and fixes it, its ERR is `1.0`.
If sentence B has 9 source errors and fixes none, its ERR is `0.0`. The present
reported mean is `(1 + 0)/2 = 0.5`. Corpus ERR should instead aggregate errors:
`(10 - 9)/10 = 0.1`. Sentence ERR is valid for paired resampling, but spec 014
must correct the reported corpus aggregate.

`_alpha` uses one Unicode-aware letters-only regex and is shared by corpus and
sentence paths. A reference word counts only when it contains letters without
numbers, underscore, punctuation, or emoji; matching is case-insensitive and
position-based.

### `efficiency.py`: five warm-ups and repeated measurements

```python
def benchmark(self, test_dataset, batch_size=1):
    if batch_size != 1: raise ValueError("Chapter 3 per-sentence latency requires batch_size=1")
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    n = len(test_dataset)
    for _ in range(self.warmup_passes): self._run_inference(loader)
    timed = []; memory_runs = []
    for _ in range(self.inference_runs):
        if self.device.type == "cuda": torch.cuda.reset_peak_memory_stats(self.device)
        durations = self._run_inference(loader); timed.append(durations)
        if self.device.type == "cuda": memory_runs.append(torch.cuda.max_memory_allocated(self.device)/(1024*1024))
```

- Batch size is forced to one so each duration represents one sentence.
- `shuffle=False` preserves the same test order for every model.
- The first loop runs warm-ups and discards their values. Configuration currently
  supplies five warm-ups.
- The second loop creates independent repeated vectors; configuration currently
  supplies 20 runs.
- On CUDA, peak allocation is reset before each full pass and read immediately
  afterward. Division by `1024*1024` converts bytes to MiB.
- `_run_inference` synchronizes CUDA before starting and after generation, so the
  CPU timer includes completed GPU work rather than only asynchronous launch time.

```python
per_sentence = [sum(row[i] for row in timed)/len(timed) for i in range(n)] if n else []
run_means = [sum(row)/max(len(row),1) for row in timed]
```

The first comprehension averages each sentence's latency across runs. The second
averages all sentences within each run. CPU execution returns an empty memory list
and `gpu_memory_available=False`, which avoids inventing GPU measurements.

**Remaining gap:** memory is summarized as maximum MiB per model. Chapter 3 also
needs model mean/standard deviation and paired-difference median/IQR in GB, tested
against exact reset/read order. Spec 014 owns those corrections.

### `statistical_tests.py`: bootstrap

```python
observed = (b.mean() - a.mean()) if higher_is_better else (a.mean() - b.mean())
for _ in range(self.n_bootstrap):
    idx = self.rng.randint(0, len(a), len(a))
    deltas.append((b[idx].mean() - a[idx].mean()) if higher_is_better else (a[idx].mean() - b[idx].mean()))
lower_tail = (1 + np.sum(deltas <= 0)) / (self.n_bootstrap + 1)
upper_tail = (1 + np.sum(deltas >= 0)) / (self.n_bootstrap + 1)
p = min(1.0, 2 * min(lower_tail, upper_tail))
lo, hi = np.percentile(deltas, [2.5, 97.5])
```

- Arrays `a` and `b` must be non-empty and equal length.
- Positive `observed` always favors model B: for accuracy it calculates B−A; for
  latency (lower is better) it calculates A−B.
- One random index vector selects the same resampled sentences from both models,
  which preserves pairing.
- The add-one numerator/denominator prevents a literal zero p-value.
- Twice the smaller tail is a two-sided p-value, capped at one.
- The 2.5th and 97.5th percentiles form the bootstrap 95% interval.
- The returned `significant` is true only when raw `p < alpha` and the interval
  excludes zero. The full experiment constructs this class with 1,000 resamples.

### Wilcoxon and the current effect-size gap

```python
d = b - a
stat, p = (0.0, 1.0) if np.all(d == 0) else stats.wilcoxon(a, b, alternative="two-sided")
nonzero = d[d != 0]
rank_biserial = float((np.sum(nonzero > 0) - np.sum(nonzero < 0)) / len(nonzero)) if len(nonzero) else 0.0
```

The two-sided Wilcoxon call correctly treats the measurements as paired and avoids
SciPy's all-zero failure by returning `p=1`. But the variable named
`rank_biserial` only compares counts of positive and negative differences. The
matched-pairs rank-biserial required by Chapter 3 must rank absolute nonzero
differences, sum positive and negative ranks, then compute
`(R_positive - R_negative)/(R_positive + R_negative)`. The current code also
reports each model's median/IQR rather than the median/IQR of paired differences.

**Tiny example:** differences `[1, 2, -100]` give the present sign-count effect
`(2-1)/3 = 0.333`. Signed ranks are `+1`, `+2`, and `-3`, so the required effect is
`(3-3)/6 = 0`. Magnitude ranks change the conclusion.

### Holm–Bonferroni

```python
m = len(p_values); ordered = sorted(p_values, key=lambda x: x[1])
results = []; keep = True; previous = 0.0
for rank, (name, raw) in enumerate(ordered, 1):
    threshold = alpha / (m-rank+1)
    reject = keep and raw < threshold
    keep = reject
    adjusted = min(1.0, max(previous, raw * (m-rank+1)))
    previous = adjusted
```

- Tests are sorted from smallest p-value to largest.
- The threshold becomes less strict as fewer hypotheses remain.
- `keep` implements step-down stopping: after the first failure, all later tests
  remain not significant.
- `max(previous, ...)` makes adjusted p-values monotonic; `min(1.0, ...)` keeps
  them within the probability range.

`run_full_comparison` currently puts eight accuracy tests (two TAHIMIK comparisons
× four metrics) in one accuracy family. It puts all latency and memory tests for
both comparisons into one global efficiency family. Spec 014 requires a separate
two-test efficiency family—latency plus memory—for each model comparison.

### `annotation.py`: what is and is not connected

`compute(...)` can call the `krippendorff` library with a chosen measurement
level, but its default is `"ratio"`. `compute_from_normalizations(...)` converts
whole normalized strings into continuous edit-distance ratios and calls ratio
alpha. The full experiment does not currently load the external long-format
binary-label CSV or compute nominal alpha separately for each noise category.
Spec 010 defines that CSV; spec 014 requires three-annotator category matrices and
a hard full-run stop when any category has alpha below `0.80`.

### Easy defense summary

- **Bootstrap:** “Resample the same sentences for both models 1,000 times and ask
  whether the improvement consistently stays away from zero.”
- **Wilcoxon:** “Rank the sizes of 20 paired memory differences and test whether
  their positive and negative ranks are balanced.”
- **Holm:** “Correct the p-values from smallest upward so multiple tests do not
  create false discoveries.”
- **Current honesty statement:** “The pipeline runs and several statistical pieces
  are present, but exact GLEU+, corpus ERR, rank-biserial, family composition, and
  nominal per-category alpha remain explicitly unclaimed until spec 014 is done.”
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
| α (alpha) | Krippendorff's alpha (and the significance level) |
| `try/except` | run code, and handle an error instead of crashing |

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
turns raw model outputs into the four accuracy numbers that fill the results table.
Every variant is scored the same way, so the numbers are directly comparable.

**What breaks without it:** a trained model but no way to say how good it is.

**Example**
```
metrics.compute_all(predictions, references, noisy_inputs)
# {"gleu_plus": 78.4, "chrf": 82.1, "err": 0.63, "alpha_word_accuracy": 0.71}
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
| `re.compile` | pre-build a regex pattern for reuse |

### Imports + constructor

```python
import re
import editdistance
from typing import List, Dict
from sacrebleu.metrics import BLEU, CHRF
```
**▸ What this block does:** import the regex module, the edit-distance library, type
hints, and SacreBLEU's `BLEU` and `CHRF` scorer classes.

```python
class NormalizationMetrics:
    def __init__(self):
        # SacreBLEU BLEU scorer with sentence-level smoothing
        self.bleu_scorer = BLEU(smooth_method="add-k", smooth_value=1)
        # chrF scorer with character 6-grams and word 2-grams
        self.chrf_scorer = CHRF(char_order=6, word_order=2)
```
**▸ What this block does:** build the two scorers once, on construction.
- `BLEU(smooth_method="add-k", smooth_value=1)` — a BLEU scorer with **add-one
  smoothing** (adds 1 to counts so short sentences don't score 0).
- `CHRF(char_order=6, word_order=2)` — a chrF scorer using 6-character grams and
  2-word grams.

### Method `compute_gleu_plus`

```python
    def compute_gleu_plus(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        scores = []
        for pred, ref in zip(predictions, references):
            score = self.bleu_scorer.sentence_score(pred, [ref])
            scores.append(score.score)

        return sum(scores) / max(len(scores), 1)
```
**▸ line by line:**
- `scores = []` — collect per-sentence scores.
- `for pred, ref in zip(predictions, references):` — walk the two lists together.
- `self.bleu_scorer.sentence_score(pred, [ref])` — BLEU for one sentence; `[ref]` is
  a list because BLEU allows multiple references (here just one).
- `scores.append(score.score)` — `.score` is the numeric value; collect it.
- `return sum(scores) / max(len(scores), 1)` — the average (`max(..., 1)` avoids
  divide-by-zero).

### Method `compute_chrf`

```python
    def compute_chrf(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        result = self.chrf_scorer.corpus_score(predictions, [references])
        return result.score
```
**▸ line by line:**
- `self.chrf_scorer.corpus_score(predictions, [references])` — compute chrF once over
  the whole corpus. `[references]` wraps the reference list as "reference set #1."
- `return result.score` — return the numeric score.

### Method `compute_err`

**▸ What this method does (whole function):** measure the fraction of the input's
errors the model actually fixed, using edit distance.

```python
    def compute_err(
        self,
        predictions: List[str],
        references: List[str],
        noisy_inputs: List[str],
    ) -> float:
        err_scores = []

        for pred, ref, noisy in zip(predictions, references, noisy_inputs):
            errors_before = editdistance.eval(noisy, ref)
            errors_after = editdistance.eval(pred, ref)
```
**▸ What this block does:** for each triple, measure how wrong the input was vs how
wrong the output is.
- `err_scores = []` — collect per-sentence ERR.
- `zip(predictions, references, noisy_inputs)` — walk all three lists together.
- `errors_before = editdistance.eval(noisy, ref)` — edit distance from the noisy
  input to the gold (how wrong the input was).
- `errors_after = editdistance.eval(pred, ref)` — edit distance from the model's
  output to the gold (how wrong the output is).

```python
            if errors_before == 0:
                # Input was already clean — ERR is 1.0 if output is also
                # clean, otherwise penalize
                err = 1.0 if errors_after == 0 else 0.0
            else:
                err = (errors_before - errors_after) / errors_before

            err_scores.append(err)

        return sum(err_scores) / max(len(err_scores), 1)
```
**▸ What this block does:** compute ERR per sentence, handling the already-clean case,
then average.
- `if errors_before == 0:` — the input was already clean; then `err = 1.0 if
  errors_after == 0 else 0.0` (a **conditional expression**: 1.0 if the output stayed
  clean, else 0.0).
- `else: err = (errors_before - errors_after) / errors_before` — the fraction of
  errors removed: 1.0 = fixed everything, 0 = no help, negative = made it worse.
- `err_scores.append(err)` — collect it.
- `return sum(err_scores) / max(len(err_scores), 1)` — average across sentences.

### Method `compute_alpha_word_accuracy`

**▸ What this method does (whole function):** among letters-only reference words,
count how many the prediction matches at the same position (case-insensitive).

```python
    def compute_alpha_word_accuracy(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        total_words = 0
        correct_words = 0

        alpha_pattern = re.compile(r"^[a-zA-Z]+$")
```
**▸ What this block does:** initialize counters and precompile the "letters only"
regex.
- `total_words`, `correct_words` — running counts.
- `re.compile(r"^[a-zA-Z]+$")` — a pattern meaning "only letters from start (`^`) to
  end (`$`)"; precompiled for speed.

```python
        for pred, ref in zip(predictions, references):
            ref_words = ref.split()
            pred_words = pred.split()

            for i, ref_word in enumerate(ref_words):
                if not alpha_pattern.match(ref_word):
                    continue

                total_words += 1
                if i < len(pred_words):
                    if pred_words[i].lower() == ref_word.lower():
                        correct_words += 1

        return correct_words / max(total_words, 1)
```
**▸ What this block does:** compare word-by-word by position and tally matches.
- `ref.split()` / `pred.split()` — the words of each sentence.
- `for i, ref_word in enumerate(ref_words):` — index + word, so we compare by
  position `i`.
- `if not alpha_pattern.match(ref_word): continue` — skip words with
  digits/punctuation/emoji.
- `total_words += 1` — count eligible reference words.
- `if i < len(pred_words):` — the prediction has a word at that position.
- `if pred_words[i].lower() == ref_word.lower():` — case-insensitive match →
  `correct_words += 1`.
- `return correct_words / max(total_words, 1)` — the fraction correct.

### Method `compute_all`

```python
    def compute_all(
        self,
        predictions: List[str],
        references: List[str],
        noisy_inputs: List[str],
    ) -> Dict[str, float]:
        return {
            "gleu_plus": self.compute_gleu_plus(predictions, references),
            "chrf": self.compute_chrf(predictions, references),
            "err": self.compute_err(predictions, references, noisy_inputs),
            "alpha_word_accuracy": self.compute_alpha_word_accuracy(
                predictions, references
            ),
        }
```
**▸ What this block does:** run all four metrics and return them in one dict — the
single call the scripts use. Each entry calls one of the methods above.

---

## `efficiency.py` — speed and memory

**Role:** Measure inference speed and peak GPU memory, following the manuscript's
exact protocol.

- **Input:** a trained model, tokenizer, device, and the test dataset
- **Output:** `{avg_time_per_sentence, std_time_per_sentence, peak_gpu_memory_mb, num_sentences}`
- **Used by:** `scripts/benchmark.py`, `scripts/run_experiment.py`
- **Connects to:** `dataset.py` (test data + `collate_fn`)

### Where this fits in TAHIMIK
This answers **Research Question 2: is the compression actually worth it?** The whole
point of deleting bytes is efficiency, so this measures the payoff (speed + memory).
Same hardware and batch size for all three variants → fair comparison.

**What breaks without it:** the efficiency half of the thesis is unsupported.

**Example**
```
bench.benchmark(test_dataset, batch_size=1)
# {"avg_time_per_sentence": 0.042, "std_time_per_sentence": 0.003,
#  "peak_gpu_memory_mb": 1180.5, "num_sentences": 1500}
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

### Imports + constructor

```python
import time
import torch
from typing import Dict, List
from torch.utils.data import DataLoader

from src.data.dataset import NormalizationDataset, collate_fn
from src.utils.logging_utils import setup_logger

logger = setup_logger("tahimik.efficiency")
```
**▸ What this block does:** import timing, PyTorch, type hints, the DataLoader, the
Dataset + `collate_fn` (connection to `dataset.py`), and the logger factory; then
build this module's logger named `tahimik.efficiency`.

```python
class EfficiencyBenchmark:
    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        num_beams: int = 4,
        warmup_passes: int = 5,
        inference_runs: int = 20,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_beams = num_beams
        self.warmup_passes = warmup_passes
        self.inference_runs = inference_runs
```
**▸ What this block does:** store the model, tokenizer, device, and the protocol
settings — beam width (4), warmup passes (5), timed runs (20).

### Method `_run_inference`

```python
    @torch.no_grad()
    def _run_inference(
        self,
        dataloader: DataLoader,
    ) -> float:
        """Run one full inference pass and return total time in seconds."""
        self.model.eval()
        total_time = 0.0
```
**▸ What this block does:** declare a no-gradient inference pass; put the model in eval
mode; start a time accumulator.
- `@torch.no_grad()` — turn off gradient tracking for the whole method.

```python
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
```
**▸ What this block does:** for each batch, move the inputs to the device, wait for the
GPU to be idle, then start the timer.
- `torch.cuda.synchronize()` — the GPU works **asynchronously** (returns before
  finishing); this waits so the timer starts clean.
- `time.perf_counter()` — a high-precision clock.

```python
            self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=1024,
                num_beams=self.num_beams,
            )

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            total_time += time.perf_counter() - start

        return total_time
```
**▸ What this block does:** run generation (the work being measured), wait for the GPU
to finish, add the elapsed time, and after all batches return the total.
- The second `synchronize()` ensures the timer stops only after the GPU is truly done.
- `total_time += time.perf_counter() - start` — accumulate elapsed seconds.

### Method `benchmark`

**▸ What this method does (whole function):** run the 5-warmup / 20-timed protocol and
report per-sentence time (mean + std) and peak GPU memory.

```python
    def benchmark(
        self,
        test_dataset: NormalizationDataset,
        batch_size: int = 1,
    ) -> Dict[str, float]:
        dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        num_sentences = len(test_dataset)
```
**▸ What this block does:** build a DataLoader (`batch_size=1` for true per-sentence
timing, no shuffle) and count the sentences.

```python
        # ── Warmup passes ───────────────────────────────────────────────
        logger.info(f"Running {self.warmup_passes} warmup passes...")
        for _ in range(self.warmup_passes):
            self._run_inference(dataloader)
```
**▸ What this block does:** log and run 5 warmup passes whose times are **discarded**
(the first GPU runs are misleadingly slow).

```python
        # Reset memory stats after warmup
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
```
**▸ What this block does:** zero the GPU peak-memory counter **after** warmup, so
warmup allocations don't inflate the reported peak.

```python
        # ── Timed runs ──────────────────────────────────────────────────
        logger.info(f"Running {self.inference_runs} timed inference passes...")
        per_run_times: List[float] = []

        for run in range(self.inference_runs):
            run_time = self._run_inference(dataloader)
            per_sentence_time = run_time / max(num_sentences, 1)
            per_run_times.append(per_sentence_time)
```
**▸ What this block does:** run 20 timed passes, converting each pass's total time into
a per-sentence time.
- `run_time / max(num_sentences, 1)` — total pass time ÷ sentence count.
- `per_run_times.append(...)` — collect the 20 per-sentence times.

```python
        # ── Compute statistics ──────────────────────────────────────────
        times_tensor = torch.tensor(per_run_times)
        avg_time = times_tensor.mean().item()
        std_time = times_tensor.std().item()
```
**▸ What this block does:** turn the 20 measurements into a tensor and compute their
mean and standard deviation.
- `.item()` — pull a plain Python float out of the size-1 result tensor.

```python
        # Peak GPU memory
        if self.device.type == "cuda":
            peak_memory_bytes = torch.cuda.max_memory_allocated(self.device)
            peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        else:
            peak_memory_mb = 0.0
```
**▸ What this block does:** read the peak GPU memory in bytes and convert to MB (or 0
on CPU).
- `torch.cuda.max_memory_allocated(...)` — the max memory used since the reset.
- `/ (1024 * 1024)` — bytes → megabytes.

```python
        results = {
            "avg_time_per_sentence": avg_time,
            "std_time_per_sentence": std_time,
            "peak_gpu_memory_mb": peak_memory_mb,
            "num_sentences": num_sentences,
        }

        logger.info(f"Efficiency results:")
        logger.info(f"  Avg time/sentence: {avg_time*1000:.2f} ms (+/- {std_time*1000:.2f})")
        logger.info(f"  Peak GPU memory:   {peak_memory_mb:.1f} MB")

        return results
```
**▸ What this block does:** assemble the results dict, log a summary (times shown in
milliseconds), and return it.
- `avg_time*1000` — seconds → ms for readability.

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
not)."** An examiner *will* ask "is that gap real or just noise?" — this file is the
answer, and why the study can state conclusions with confidence.

**What breaks without it:** every comparison would be anecdotal.

**Example**
```
paired_bootstrap(byt5_scores, tahimik_scores)
# {"mean_a": 76.1, "mean_b": 78.4, "delta": 2.3,
#  "p_value": 0.012, "ci_lower": 0.9, "ci_upper": 3.6, "significant": True}
```
> ⚠️ **FINDINGS #1–2:** the bootstrap below is one-tailed and marks `significant`
> from the p-value alone; the manuscript wants a two-tailed test **and** a CI
> excluding 0. Fix before reporting significance.

**Terms & abbreviations in this file**
| Term | Full form / plain meaning |
|---|---|
| p-value | probability the difference happened by luck; small (<0.05) = "probably real" |
| CI | confidence interval — a plausible range for the true difference |
| alpha (`self.alpha`) | the significance threshold (0.05) |
| bootstrap | resampling the data many times to estimate reliability |
| with replacement | resampling where the same item can be picked more than once |
| percentile | the value below which a given % of the data falls |
| Wilcoxon signed-rank | a paired test that doesn't assume a normal distribution |
| Holm-Bonferroni | a correction for running many tests at once |
| FWER | family-wise error rate — the chance of *any* false positive across all tests |
| `np.array` / `np.percentile` | numpy array / the value at a given percentile |

### Imports + constructor

```python
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple
from src.utils.logging_utils import setup_logger

logger = setup_logger("tahimik.stats")
```
**▸ What this block does:** import numpy (fast arrays), SciPy's `stats` (for Wilcoxon),
type hints, and the logger factory; build the `tahimik.stats` logger.

```python
class StatisticalAnalysis:
    def __init__(
        self,
        alpha: float = 0.05,
        n_bootstrap: int = 1000,
        seed: int = 42,
    ):
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.RandomState(seed)
```
**▸ What this block does:** store the significance level (0.05), the number of
bootstrap resamples (1000), and a seeded numpy RNG for reproducible resampling.
- `np.random.RandomState(seed)` — numpy's seeded random generator.

### Method `paired_bootstrap`

**▸ What this method does (whole function):** the main accuracy test — resample the
per-sentence scores 1000 times to estimate how reliably model B beats model A,
producing a p-value and a 95% confidence interval of the difference.

```python
    def paired_bootstrap(
        self,
        scores_a: List[float],
        scores_b: List[float],
        metric_name: str = "metric",
    ) -> Dict[str, float]:
        a = np.array(scores_a)
        b = np.array(scores_b)
        n = len(a)

        observed_delta = b.mean() - a.mean()
```
**▸ What this block does:** turn the two score lists into numpy arrays and compute the
real observed difference.
- `np.array(...)` — a fast numeric array. `n = len(a)` — number of sentences.
- `observed_delta = b.mean() - a.mean()` — how much B beats A on the real data.

```python
        # Bootstrap
        bootstrap_deltas = []
        wins_a = 0

        for _ in range(self.n_bootstrap):
            indices = self.rng.randint(0, n, size=n)
            boot_a = a[indices].mean()
            boot_b = b[indices].mean()
            delta = boot_b - boot_a
            bootstrap_deltas.append(delta)

            if boot_a >= boot_b:
                wins_a += 1
```
**▸ What this block does:** 1000 times, resample the sentences **with replacement** and
recompute each model's mean, tracking the gap and how often A wins.
- `bootstrap_deltas = []`, `wins_a = 0` — collectors.
- `self.rng.randint(0, n, size=n)` — `n` random positions in `[0, n)`, with
  replacement (a position can repeat) — the bootstrap resample.
- `a[indices].mean()` / `b[indices].mean()` — each model's mean on that resample.
- `delta = boot_b - boot_a`; `bootstrap_deltas.append(delta)` — record the gap.
- `if boot_a >= boot_b: wins_a += 1` — count resamples where A did at least as well.

```python
        p_value = wins_a / self.n_bootstrap
```
**▸ What this block does:** the p-value = fraction of resamples A won. *(This is the
one-tailed form flagged in FINDINGS #1.)*

```python
        # 95% confidence interval of the difference
        bootstrap_deltas = np.array(bootstrap_deltas)
        ci_lower = np.percentile(bootstrap_deltas, 2.5)
        ci_upper = np.percentile(bootstrap_deltas, 97.5)
```
**▸ What this block does:** compute the 95% CI of the difference from the resample
gaps.
- `np.percentile(..., 2.5)` / `97.5` — the 2.5th and 97.5th percentiles bracket the
  middle 95%.

```python
        result = {
            "p_value": p_value,
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
            "delta": float(observed_delta),
            "significant": p_value < self.alpha,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
        }
```
**▸ What this block does:** package the results. `"significant": p_value < self.alpha`
— flagged significant if p < 0.05. *(FINDINGS #2: this ignores the CI, which the
manuscript requires to also exclude 0.)* `float(...)` converts numpy numbers to plain
Python floats (JSON-friendly).

```python
        logger.info(
            f"  Bootstrap [{metric_name}]: "
            f"A={result['mean_a']:.4f}, B={result['mean_b']:.4f}, "
            f"delta={result['delta']:.4f}, p={result['p_value']:.4f} "
            f"{'*' if result['significant'] else 'ns'}"
        )

        return result
```
**▸ What this block does:** log a one-line summary (`*` = significant, `ns` = not) and
return the result dict.

### Method `wilcoxon_test`

**▸ What this method does (whole function):** compare GPU-memory measurements (few
samples, not normally distributed) with a non-parametric paired test.

```python
    def wilcoxon_test(
        self,
        measurements_a: List[float],
        measurements_b: List[float],
        metric_name: str = "gpu_memory",
    ) -> Dict[str, float]:
        a = np.array(measurements_a)
        b = np.array(measurements_b)

        # Wilcoxon requires non-zero differences
        differences = b - a
        if np.all(differences == 0):
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "mean_a": float(a.mean()),
                "mean_b": float(b.mean()),
            }
```
**▸ What this block does:** convert to arrays, compute element-wise differences, and
short-circuit if they're all zero (nothing to test → "not significant").
- `np.all(differences == 0)` — True only if every paired difference is zero.

```python
        stat, p_value = stats.wilcoxon(a, b, alternative="two-sided")
```
**▸ What this block does:** run the **Wilcoxon signed-rank test**, correctly
**two-sided**, returning a test statistic and a p-value.

```python
        result = {
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
        }

        logger.info(
            f"  Wilcoxon [{metric_name}]: "
            f"A={result['mean_a']:.2f}, B={result['mean_b']:.2f}, "
            f"W={result['statistic']:.1f}, p={result['p_value']:.4f} "
            f"{'*' if result['significant'] else 'ns'}"
        )

        return result
```
**▸ What this block does:** package the result, log a one-line summary, and return it.

### Method `holm_bonferroni`

```python
    @staticmethod
    def holm_bonferroni(
        p_values: List[Tuple[str, float]],
        alpha: float = 0.05,
    ) -> List[Dict[str, any]]:
        m = len(p_values)

        # Sort by p-value (ascending)
        sorted_pairs = sorted(p_values, key=lambda x: x[1])

        results = []
        for rank, (name, p) in enumerate(sorted_pairs, start=1):
            adjusted_alpha = alpha / (m - rank + 1)
            significant = p < adjusted_alpha

            results.append({
                "comparison": name,
                "raw_p": p,
                "rank": rank,
                "adjusted_alpha": adjusted_alpha,
                "significant_corrected": significant,
            })

            logger.info(
                f"  Holm-Bonferroni rank {rank}: {name} "
                f"p={p:.4f} < {adjusted_alpha:.4f}? "
                f"{'YES' if significant else 'NO'}"
            )

        return results
```
**▸ line by line:**
- `@staticmethod` — no `self`; a pure utility.
- `m = len(p_values)` — how many tests are being corrected.
- `sorted(p_values, key=lambda x: x[1])` — sort the `(name, p)` pairs by p-value
  ascending; `lambda x: x[1]` is a tiny inline function returning the p-value to sort
  on.
- `for rank, (name, p) in enumerate(sorted_pairs, start=1):` — rank from 1 (smallest
  p first).
- `adjusted_alpha = alpha / (m - rank + 1)` — the **Holm-Bonferroni** step: the
  smallest p faces the strictest threshold, loosening by rank.
- `significant = p < adjusted_alpha` — pass only if below the adjusted threshold.
- `results.append({...})` — record the comparison, raw p, rank, threshold, and
  verdict.
- `logger.info(...)` — log each decision.
- `return results` — the corrected verdicts (controls the family-wise error rate).

### Method `run_full_comparison`

**▸ What this method does (whole function):** run the bootstrap on every model-pair ×
accuracy-metric, the Wilcoxon test on every pair's memory, collect all p-values, then
apply Holm-Bonferroni.

```python
    def run_full_comparison(
        self,
        per_sentence_scores: Dict[str, Dict[str, List[float]]],
        gpu_memory_runs: Dict[str, List[float]],
    ) -> Dict[str, any]:
        models = ["byt5", "mrt5", "tahimik"]
        pairs = [
            ("byt5", "mrt5"),
            ("byt5", "tahimik"),
            ("mrt5", "tahimik"),
        ]

        metrics = ["gleu_plus", "chrf", "err", "alpha_word_accuracy"]

        all_results = {"bootstrap": {}, "wilcoxon": {}, "holm_bonferroni": {}}
        all_p_values = []
```
**▸ What this block does:** set up the model list, the three pairwise comparisons, the
four metrics, and the containers for results and collected p-values.

```python
        # ── Bootstrap tests for normalization metrics ───────────────────
        logger.info("Running paired bootstrap resampling tests...")
        for model_a, model_b in pairs:
            pair_key = f"{model_a}_vs_{model_b}"
            all_results["bootstrap"][pair_key] = {}

            for metric in metrics:
                scores_a = per_sentence_scores[model_a][metric]
                scores_b = per_sentence_scores[model_b][metric]

                result = self.paired_bootstrap(
                    scores_a, scores_b,
                    metric_name=f"{pair_key}/{metric}",
                )
                all_results["bootstrap"][pair_key][metric] = result
                all_p_values.append(
                    (f"{pair_key}/{metric}", result["p_value"])
                )
```
**▸ What this block does:** for each pair × each accuracy metric, run the bootstrap and
store the result + its p-value.
- `pair_key = f"{model_a}_vs_{model_b}"` — e.g. `"byt5_vs_tahimik"`.
- The inner loop pulls each model's per-sentence scores for the metric and runs
  `paired_bootstrap`, appending `(name, p_value)` to `all_p_values`.

```python
        # ── Wilcoxon tests for GPU memory ───────────────────────────────
        logger.info("Running Wilcoxon signed-rank tests for GPU memory...")
        for model_a, model_b in pairs:
            pair_key = f"{model_a}_vs_{model_b}"
            mem_a = gpu_memory_runs[model_a]
            mem_b = gpu_memory_runs[model_b]

            result = self.wilcoxon_test(
                mem_a, mem_b,
                metric_name=f"{pair_key}/gpu_memory",
            )
            all_results["wilcoxon"][pair_key] = result
            all_p_values.append(
                (f"{pair_key}/gpu_memory", result["p_value"])
            )
```
**▸ What this block does:** for each pair, run the Wilcoxon memory test and store the
result + its p-value.

```python
        # ── Holm-Bonferroni correction ──────────────────────────────────
        logger.info("Applying Holm-Bonferroni correction...")
        all_results["holm_bonferroni"] = self.holm_bonferroni(
            all_p_values, alpha=self.alpha
        )

        return all_results
```
**▸ What this block does:** apply the multiple-comparison correction across every
collected p-value, store it, and return the full nested results dict.

---

## `annotation.py` — inter-annotator agreement

**Role:** Measure how consistently the human annotators agreed when building the gold
dataset. Validates the *data*, not a model.

- **Input:** a table of annotations (rows = annotators, cols = items), or raw
  annotator outputs + a reference
- **Output:** `{alpha, interpretation, num_annotators, num_items}`
- **Used by:** the reliability analysis for the gold dataset
- **Connects to:** `numpy`, the optional `krippendorff` package, `editdistance`

### Where this fits in TAHIMIK
This validates **the gold dataset itself, before any model touches it.** If three
annotators normalize the same sentence very differently, the "gold standard" isn't
gold. Krippendorff's α is the number that says "our annotators agreed reliably (α ≥
0.80)."

**What breaks without it:** you couldn't defend the quality of your ground truth.

**Example**
```
iaa.compute(reliability_data)     # rows = annotators, cols = sentences
# {"alpha": 0.83, "interpretation": "Reliable agreement (alpha >= 0.80)",
#  "num_annotators": 3, "num_items": 500}
```

**Terms & abbreviations in this file**
| Term | Full form / plain meaning |
|---|---|
| IAA | inter-annotator agreement |
| Krippendorff's α | a 0–1 agreement statistic (≥0.80 reliable, ≥0.67 tentative) |
| reliability data | a table: rows = annotators, columns = items |
| `np.nan` | "not a number" — marks a missing value |
| `try/except` | run code, and handle an error instead of crashing |
| `ImportError` | the error raised when an optional package isn't installed |
| level of measurement | the data type for α ("ratio" for continuous values) |

### Imports + optional-import guard

```python
import numpy as np
from typing import List, Dict, Optional

try:
    import krippendorff as krippendorff_lib
    HAS_KRIPPENDORFF = True
except ImportError:
    HAS_KRIPPENDORFF = False

from src.utils.logging_utils import setup_logger

logger = setup_logger("tahimik.annotation")
```
**▸ What this block does:** import numpy + type hints, try to import the optional
`krippendorff` package, and set up the logger.
- `try: import krippendorff ... HAS_KRIPPENDORFF = True` — attempt the import and flag
  success.
- `except ImportError: HAS_KRIPPENDORFF = False` — if it's not installed, set the flag
  False instead of crashing.
- `logger = setup_logger("tahimik.annotation")` — this module's logger.

### Method `compute`

```python
    def compute(
        self,
        reliability_data: np.ndarray,
        level_of_measurement: str = "ratio",
    ) -> Dict[str, float]:
        if not HAS_KRIPPENDORFF:
            logger.warning(
                "krippendorff package not installed. "
                "Install it with: pip install krippendorff"
            )
            return {
                "alpha": None,
                "interpretation": "Package not installed",
                "num_annotators": reliability_data.shape[0],
                "num_items": reliability_data.shape[1],
            }
```
**▸ What this block does:** if the package is missing, warn and return a result saying
so (graceful degradation instead of crashing).
- `reliability_data.shape[0]` / `[1]` — the table's rows (annotators) and columns
  (items).

```python
        alpha = krippendorff_lib.alpha(
            reliability_data=reliability_data,
            level_of_measurement=level_of_measurement,
        )
```
**▸ What this block does:** compute Krippendorff's α from the table.
- `reliability_data` — rows = annotators, columns = items; `np.nan` marks unrated
  cells.
- `level_of_measurement="ratio"` — treat the values as continuous ratios.

```python
        # Interpret per Krippendorff's guidelines
        if alpha >= 0.80:
            interpretation = "Reliable agreement (alpha >= 0.80)"
        elif alpha >= 0.67:
            interpretation = "Tentative agreement (0.67 <= alpha < 0.80)"
        else:
            interpretation = "Unreliable agreement (alpha < 0.67)"
```
**▸ What this block does:** translate the number into a plain-English verdict using the
manuscript's reliability bands.

```python
        result = {
            "alpha": float(alpha),
            "interpretation": interpretation,
            "num_annotators": int(reliability_data.shape[0]),
            "num_items": int(reliability_data.shape[1]),
        }

        logger.info(
            f"Krippendorff's alpha = {alpha:.4f} — {interpretation}"
        )

        return result
```
**▸ What this block does:** package α, the verdict, and the table dimensions; log the
result; return it.

### Method `compute_from_normalizations`

**▸ What this method does (whole function):** convert raw annotator texts into
edit-distance ratios against a consensus reference, build the table, and compute α.

```python
    def compute_from_normalizations(
        self,
        annotator_outputs: List[List[str]],
        reference: List[str],
    ) -> Dict[str, float]:
        import editdistance

        num_annotators = len(annotator_outputs)
        num_items = len(reference)

        reliability_data = np.full(
            (num_annotators, num_items), np.nan
        )
```
**▸ What this block does:** import `editdistance` (locally), read the table
dimensions, and build a table pre-filled with `np.nan` (so unrated cells stay
missing).
- `np.full((rows, cols), np.nan)` — a rows×cols array where every cell starts as "not
  a number."

```python
        for a_idx, outputs in enumerate(annotator_outputs):
            for i_idx, output in enumerate(outputs):
                if not output:
                    continue
                ref = reference[i_idx]
                dist = editdistance.eval(output, ref)
                max_len = max(len(output), len(ref), 1)
                reliability_data[a_idx, i_idx] = dist / max_len

        return self.compute(reliability_data, level_of_measurement="ratio")
```
**▸ What this block does:** fill the table with each annotator's edit-distance ratio,
then compute α on it.
- `for a_idx, outputs in enumerate(annotator_outputs):` — loop annotators (row index).
- `for i_idx, output in enumerate(outputs):` — loop items (column index).
- `if not output: continue` — skip blank entries (leave them `np.nan`).
- `dist = editdistance.eval(output, ref)` — edits between this annotator's text and
  the reference.
- `max_len = max(len(output), len(ref), 1)` — the normalizer (the `1` avoids /0).
- `reliability_data[a_idx, i_idx] = dist / max_len` — store the ratio (same idea as
  n\*) at row = annotator, column = item.
- `return self.compute(...)` — compute α on the filled table.

---

## `__init__.py`

```python
from src.evaluation.metrics import NormalizationMetrics
from src.evaluation.efficiency import EfficiencyBenchmark
from src.evaluation.statistical_tests import StatisticalAnalysis
from src.evaluation.annotation import InterAnnotatorAgreement
```
**▸ What this block does:** re-export the four public classes so other code can do
`from src.evaluation import NormalizationMetrics`, etc. Each line lifts one class to
the package level.

- **Inputs:** none (runs on import)
- **Outputs:** the four names above, importable from `src.evaluation`

---

## You've now covered all of `src/` at full every-line depth

| Folder | Doc | Role in the flow |
|---|---|---|
| `utils` | [01](01-utils.md) | shared toolbox |
| `data` | [02](02-data.md) | make + tensorize training pairs |
| `models` | [03](03-models.md) | the three networks |
| `training` | [04](04-training.md) | loss + loop → checkpoint |
| `evaluation` | [05](05-evaluation.md) | score accuracy, speed, significance |

Full loop: **data → models → training → evaluation**, with **utils** underneath.
