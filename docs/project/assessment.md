# TAHIMIK — system implementation assessment

**Date:** 2026-08-29
**Scope:** whole repository, with the deepest pass on the TAHIMIK model
(`src/models/`), its training/loss path, and the demo tool
(`backend/` + `frontend/`).
**Method:** read every Python source file; ran the test suite; ran targeted
experiments against the real code (synthetic-noise distribution, gate
sensitivity, HuggingFace encoder equivalence, function-signature binding).
Every claim below is either a file reference or a measured number, and the
command that produced it is shown.

This document does **not** change any code. It is a findings report, in the
same spirit as [`specs/FINDINGS.md`](specs/FINDINGS.md) — which it also
corrects in three places.

---

## 1. Headline verdict

The architecture is genuinely well built. The three variants share one
contract, the configs are clean, the gradient isolation the thesis depends on
is real and tested, and the reasoning is documented at the point of the
decision rather than in a wiki nobody reads. That is better than most
undergraduate thesis code.

But there is a gap between "the code is written" and "the study can be run",
and the repository currently describes itself as being on the right side of
that gap. It is not:

> README: *"The implementation is complete; the study is not."*

More precisely: **the model code is complete; the entry points that would run
it have never executed.** `scripts/train.py` and `scripts/run_experiment.py`
both crash with a `TypeError` before a single model is constructed (§3.1).
No amount of GPU time or gold data fixes that — it is a four-line argument bug
in each file, present since the code was written.

Three further issues would produce numbers that look correct and are not:

1. The two compressed variants silently lose ByT5's relative position bias in
   11 of 12 encoder layers; the baseline keeps it. The comparison is not
   controlled (§3.2).
2. GPU-memory significance is computed from one measurement copied 20 times,
   which makes every memory comparison "significant" by construction (§5.1).
3. On the synthetic data the pipeline actually generates, TAHIMIK's adaptive
   deletion target collapses to within a few points of MrT5's fixed 0.5, so
   Stage 1 cannot demonstrate the contribution (§4.2).

Severity summary:

| # | Finding | Area | Severity |
|---|---------|------|----------|
| 3.1 | `train.py` / `run_experiment.py` crash at startup | scripts | **Blocker** |
| 3.2 | Relative position bias dropped in gated variants only | models | **Blocker (validity)** |
| 5.1 | Wilcoxon runs on one replicated measurement | evaluation | **High** |
| 4.2 | Synthetic data makes the adaptive target ≈ the fixed target | data | **High** |
| 3.3 | Everything padded to 1024 bytes | data / efficiency | **High** |
| 5.2 | Inference-time significance never computed | evaluation | **High** |
| 3.4 | `cn`'s authority is unconstrained *and* unmonitored | models | **High** |
| 5.3 | Holm-Bonferroni missing its step-down rule | evaluation | Medium |
| 4.1 | Stage 2 resumes from last-epoch, not best, weights | training | Medium |
| 6 | Backend blocks its own event loop; batch endpoint isn't batched | tool | Medium |
| 3.5 | `forward()` without `labels` raises in both gated variants | models | Medium |
| 4.3 | Noise probabilities hardcoded outside `configs/` | data | Medium |
| 5.4 | Alpha-word accuracy aligns by position index | evaluation | Medium |
| 7 | Test suite covers the gate well and nothing else | tests | Medium |
| 5.5 | "GLEU+" is smoothed sentence-BLEU, not GLEU | evaluation | Low |
| 8 | `FINDINGS.md` is stale on three items | docs | Low |

---

## 2. What is genuinely right

Stated first, because the rest of this document is a list of problems and the
codebase is better than that list makes it sound.

- **The three-variant contract holds.** One config base, one trainer, one loss
  module, one dataset class. Swapping the compression mechanism really is the
  only intended difference — which is why §3.2 matters so much: it is the one
  place the intent leaks.
- **Gradient isolation is real.** `n` is detached at both consumption points
  (`delete_gate.py:162`, `noise_adaptive_byt5.py:187`) and there is a test that
  fails if either detach is removed
  (`tests/test_model_forward.py::test_noise_estimator_is_trained_only_by_l_ne`).
- **The soft/hard deletion split is correct**, including the subtle part: the
  gate bias must be *added* to the extended attention mask rather than
  multiplied into the binary mask, and the suppression must be applied *after*
  T5's RMS final norm. Both are right, and both are documented at the call site
  with the reason the obvious alternative breaks.
- **Hard deletion is now vectorised** (`delete_gate.py:250-270`,
  `cumsum`/`scatter_add_`/`gather`) and **Gumbel noise is applied at training
  only** (`delete_gate.py:150-152`). `FINDINGS.md` still lists both as pending.
- **The delete gate is well tested** — 15 tests including the failure modes
  that actually bite (rate loss reaching the gate, the regulariser not
  collapsing the gate, padding never counting, determinism at inference).
- **`configs/` is clean.** No stray hyperparameters in the model or training
  code, seed 42 everywhere, per-variant files inheriting one base.
- **The annotation platform is the most production-ready component.** RLS is
  enabled on all six tables with per-table policies, only the anon key is used
  client-side, and `.env.example` explains why the service-role key must never
  appear. No secrets are tracked; no `__pycache__`, checkpoints, or datasets in
  git (197 tracked files, 0 of them build artefacts).
- **`python -m pytest tests/ -v` passes**: 26 passed in 11.5s.

---

## 3. The main tool — the TAHIMIK model

### 3.1 Blocker: the training and experiment entry points crash immediately

`scripts/train.py:145` and its three siblings, plus `scripts/run_experiment.py:105`
and its three siblings, all do this:

```python
stage2_train = NormalizationDataset(
    *gold_splits["train"], tokenizer,
    max_input_length=config.max_input_length,
    ...
)
```

`split_data()` returns a **3-tuple** — `(noisy, clean, noise_levels)`
(`preprocessing.py:246-250`). Unpacking it gives three positional arguments,
so `tokenizer` lands in the fourth position, which is `max_input_length` —
which is then also passed by keyword.

Verified against the real signature:

```bash
python -c "import inspect; from src.data.dataset import NormalizationDataset; inspect.signature(NormalizationDataset.__init__).bind(None, ['a'],['b'],[0.1], 'TOK', max_input_length=1024)"
```

```
TypeError: multiple values for argument 'max_input_length'
```

Eight call sites are affected:

| File | Lines |
|------|-------|
| `scripts/train.py` | 145, 150, 170, 175 |
| `scripts/run_experiment.py` | 105, 110, 137, 142 |

The two that are written correctly (`run_experiment.py:117`,
`benchmark.py:91`) pass `precomputed_noise_levels=` as a keyword — which is
what the other eight should do.

**What this means.** `train.py` fails before the model is even constructed, so
no checkpoint has ever been produced by this repository, and
`run_experiment.py` — the "full pipeline" the README documents — has never run
end to end. This is not a "waiting on the gold dataset" problem: it fails on
any input, including a ten-row CSV.

It also blocks the handoff's top recommendation (train a small Stage-1
checkpoint so the tool answers something at the defense). A second obstacle
sits next to it: `--gold_data` is `required=True` in `train.py`, so there is no
supported way to run Stage 1 alone even after the crash is fixed.

**Fix:** four lines per file, plus a `--stage1_only` path (or make
`--gold_data` optional). Roughly 20 minutes of work. It should be first.

### 3.2 Blocker for validity: the gated variants lose relative position bias

`NoiseAdaptiveByT5._run_encoder_layers` (`noise_adaptive_byt5.py:118`) and its
identical twin in `fixed_compression_byt5.py:98` run encoder blocks in a
manual Python loop:

```python
for i in range(start_layer, end_layer):
    layer = encoder.block[i]
    layer_output = layer(hidden_states, attention_mask=extended_mask)
    hidden_states = layer_output[0]
```

T5 computes its relative position bias **once, in block 0**, and threads it
through every later block. Confirmed on the installed version (transformers
5.15.0):

```
block 0  has_relative_attention_bias = True
block 1+ has_relative_attention_bias = False
```

HuggingFace's own `T5Stack` keeps `position_bias` in a local variable and
passes it into each subsequent block (`modeling_t5.py:722-742`). The loop above
discards it — `layer(...)` returns `(hidden_states, position_bias, attn_weights)`
and only `[0]` is kept. Every block after the first therefore falls into the
`has_relative_attention_bias=False` branch and receives a **zeros** bias
(`modeling_t5.py:335-344`).

Measured divergence from the correct encoder output, same weights, same input,
untrained 4-layer T5:

```
max abs diff: 0.102   (reference max activation: 2.755)
```

~3.7% on random weights, where the learned bias table is still noise. On a
pretrained ByT5, where those biases carry real positional information, the
divergence is far larger.

**Why this is the most serious finding after §3.1.** The ByT5 baseline calls
`self.model(...)` and goes through HuggingFace's correct path
(`byt5_baseline.py:59-63`). Only MrT5 and TAHIMIK use the manual loop. So the
independent variable is not "compression mechanism" — it is "compression
mechanism **and** whether the encoder has positional information in 11 of its
12 layers". Any accuracy gap between the baseline and the two compressed
variants is partly this bug, and there is no way after the fact to say how
much.

**Fix:** thread `position_bias` through the loop, and re-derive it after hard
deletion (sequence length changes, so the cached bias cannot be reused across
the gate). Small change, large consequence — and it is the single thing most
worth being able to say you found yourself at the defense.

**Related:** `_run_encoder_layers` is duplicated verbatim across the two model
files. When you fix this, fix it in one shared place, or you will fix it once.

### 3.3 Everything is padded to 1024 bytes, and it distorts the efficiency claim

`dataset.py:82` and `:91` tokenize with `padding="max_length"` and
`max_length=1024`. Every sample in every batch is 1024 positions long, whether
the sentence is 20 bytes or 900. `collate_fn` then just stacks them — it does
no dynamic padding, despite existing for exactly that purpose.

Two consequences:

1. **Training cost.** Filipino social-media sentences run ~40-120 bytes.
   Padding to 1024 is roughly a 10x waste of compute and memory on Stage 1's
   ~1M pairs, and worse than 10x in the pre-gate layers where attention is
   quadratic. This alone may be why Stage 1 has never been attempted at scale.
2. **The efficiency result becomes unsafe.** At inference the gate deletes
   padding along with everything else (`kept_mask = kept_mask * attention_mask`,
   `delete_gate.py:196`). So the compressed variants shrink 1024 → ~real length,
   while the baseline processes all 1024. `EfficiencyBenchmark` feeds it exactly
   this padded dataset (`efficiency.py:118-123`). The measured speedup would be
   dominated by *removing padding the baseline was never given a chance to skip*
   — not by compressing real bytes. That is RQ2's headline number, measured on
   an artefact.

**Fix:** dynamic padding in `collate_fn` (pad to the longest sequence in the
batch), keeping 1024 as a truncation cap. Then re-reason about the benchmark:
with `batch_size=1` and dynamic padding, all three variants see the true
sentence length and the comparison is honest.

### 3.4 `cn` has unconstrained authority and nothing watches it

`FINDINGS.md` item 7 records that nothing constrains the *sign* of `cn`
(`delete_gate.py:93`), so training could silently invert the adaptivity. That
is correct and remains true. Two things should be added to it.

**Scale.** The shift is added to a gate score living in `[k, 0] = [-30, 0]`:

```python
shift = self.cn * (n_detached - self.noise_avg)     # delete_gate.py:174
gate_outputs = gate_outputs + shift
```

With `cn = 1.0` at init and `n - navg` bounded by 1, the shift can move a
byte's keep probability by at most `1/30 ≈ 3.3%`. Measured on a fresh gate,
sweeping `cn`, with noise scores 0.05 vs 0.95:

| `cn` | deletion rate, clean | deletion rate, noisy | spread |
|------|---------------------|---------------------|--------|
| 1.0 | 0.629 | 0.563 | 0.066 |
| 5.0 | 0.797 | 0.402 | 0.395 |
| 15.0 | 0.965 | 0.059 | 0.906 |
| 30.0 | 1.000 | 0.000 | 1.000 |

At the initial `cn = 1.0` the mechanism moves the deletion rate by ~7
percentage points *under the most extreme noise contrast possible*. Under the
noise spread the synthetic data actually produces (§4.2 — clean vs the 95th
percentile, 0.00 vs 0.31), it moves it by **2.3 points**. The contribution is
therefore entirely dependent on `cn` growing by roughly an order of magnitude
during training, driven only indirectly through `L_rate`. That may happen. It
may not. Nothing currently tells you which.

**Observability.** `cn`, `navg`, and the realised deletion-rate spread across
noise levels are never logged. `trainer.py:245-258` logs `l_ce`, `l_rate`,
`l_attn_reg`, `l_ne` and nothing else. So the one quantity that decides whether
the thesis has a contribution at all is invisible during training and absent
from every checkpoint's metadata.

**Fix (cheap, high value):** log `cn` and `navg` per epoch, and at validation
log mean deletion rate bucketed by `n*` decile. That converts "the model is
adaptive" from an assumption into a plot you can put on a slide. If `cn` goes
negative or stays near 1.0, you want to know in epoch 2, not after the run.

**Also note** the test that guards this claim,
`test_noisier_sentences_are_compressed_less`, asserts `rate_noisy <= rate_clean`
— a non-strict inequality, satisfied by exact equality. It would pass on a
model with no adaptivity whatsoever. Make it strict, and add a case using a
realistic noise spread rather than 0.05 vs 0.95.

### 3.5 `forward()` without `labels` raises in both gated variants

`noise_adaptive_byt5.py:245-250` and `fixed_compression_byt5.py:203-208` call:

```python
decoder_outputs = self.model.decoder(
    encoder_hidden_states=hidden_states,
    encoder_attention_mask=modified_mask,
)
```

with no `decoder_input_ids`. Confirmed against the installed transformers:

```
ValueError: You must specify exactly one of input_ids or inputs_embeds
```

Nothing in the current pipeline hits this branch (training passes labels,
evaluation uses `generate()`), so it is latent rather than active. But it is
exactly the branch a panelist would hit if they asked you to "just call the
model on this sentence" during live debugging. Either fix it or delete it.

### 3.6 Smaller model-layer items

- `fixed_compression_byt5.py:172` — `encoder_outputs = (hidden_states,)` is
  assigned and never used. (Already a `FINDINGS.md` minor.)
- `delete_gate.py:167-171` — the `navg` EMA update is not under `no_grad()`.
  Harmless today because `n_detached` carries no graph, but it also means the
  buffer's dtype can flip to fp16 under autocast.
- `generate()` calls `self.eval()` on both gated variants, mutating module
  state as a side effect of an inference call. Surprising if it is ever called
  mid-training-loop.
- The `L_attn_reg` mean (`losses.py:127`) is taken over **all** positions
  including padding, where `keep_prob` is forced to 0 and contributes 0. With
  1024-padding (§3.3) the regulariser's effective weight is scaled by the
  fraction of real tokens — roughly 5-10% of its nominal `w_attn_reg`, and
  varying batch to batch. Divide by `attention_mask.sum()` instead.

---

## 4. Data and training pipeline

### 4.1 Stage 2 does not start from the best Stage 1 checkpoint

`trainer.py:296-325`: `train()` runs `train_stage("stage1", ...)`, resets
`best_val_loss`, then runs `train_stage("stage2", ...)` on the same in-memory
model. The model at that point holds the weights from the **final** Stage 1
epoch, not the best-validation ones that were written to `best_stage1.pt`.

So `best_stage1.pt` is selected, saved, and then never used. If Stage 1
overfits in its last epoch — plausible at 3 epochs over 1M pairs — Stage 2
inherits the worse model. Either load `best_stage1.pt` before Stage 2, or drop
the pretence that Stage 1 checkpoint selection means anything and say so.

A related point worth deciding deliberately: validation runs under
`model.eval()`, which for the two gated variants means **hard** deletion, while
training used soft deletion. Checkpoint selection is therefore made on a
different computation path than the one being optimised. That is defensible —
it selects for the inference behaviour you actually ship — but it should be a
stated choice, not an accident of `eval()`.

### 4.2 The synthetic data cannot demonstrate the contribution

Measured against the real generator (`TagalogNoiseGenerator(seed=42)`, 4000
samples over 8 representative Tagalog/Taglish sentences):

```
identical pairs (noisy == clean): 1749/4000 = 43.7%
n* mean = 0.082   median = 0.037   sd = 0.109   p95 = 0.314   max = 0.677
n* > 0.3: 5.6% of samples
```

The nine noise categories fire independently with probabilities that leave a
~20% chance none is selected (`noise_generator.py:129-137`), and several that
do fire are no-ops (abbreviation only triggers on words present in
`ABBREVIATION_MAP`). The result is that **44% of Stage 1 is a copy task** and
the noise labels cluster near zero.

Now push that through TAHIMIK's adaptive target,
`d_target = d_max · (1 − n)` with `d_max = 0.5`:

| Input | `n*` | Adaptive target | MrT5's fixed target |
|-------|------|-----------------|---------------------|
| mean sentence | 0.082 | **0.459** | 0.500 |
| 95th percentile noise | 0.314 | **0.343** | 0.500 |
| noisiest observed | 0.677 | 0.161 | 0.500 |

For 95% of Stage 1, TAHIMIK is asked to compress between 0.34 and 0.50, against
MrT5's flat 0.50. The independent variable barely varies. A constant predictor
that always outputs 0.082 achieves an `L_NE` MSE of **0.0119** (RMSE 0.109) —
so the noise estimator can score well while learning nothing useful, and
`navg` settles near 0.08, which further shrinks `(n − navg)`.

This is a *design* finding, not a bug: the generator does what it was written
to do. But it means Stage 1 alone cannot show that adaptivity helps, and if the
gold data has a similar distribution, neither will Stage 2. Worth knowing
before you buy GPU time.

**Options:** raise the per-category probabilities and guarantee at least one
transformation fires; or stratify the synthetic set toward a target `n*`
distribution (e.g. roughly uniform over [0, 0.6]); or keep the distribution and
reconsider `d_max`. Any of these is a design decision for the author, not
something to patch quietly — and it belongs in the spec loop.

### 4.3 Noise probabilities are hyperparameters living outside `configs/`

`noise_generator.py:129-137` hardcodes all nine probabilities in `__init__`.
`CLAUDE.md` and the constitution both say hyperparameters live in `configs/`.
These nine numbers determine the entire synthetic training distribution and,
per §4.2, the whole dynamic range of the contribution. They are the most
consequential undocumented constants in the repository.

---

## 5. Evaluation and statistics

### 5.1 GPU-memory significance is manufactured

`run_experiment.py:267-269`:

```python
all_gpu_memory[variant_name] = [
    eff_results["peak_gpu_memory_mb"]
] * config.inference_runs
```

`EfficiencyBenchmark.benchmark()` returns **one** number — the peak across all
runs (`efficiency.py:148-151`). That single scalar is then replicated 20 times
and handed to `wilcoxon_test` as if it were 20 independent measurements.

The consequence is not noise, it is a guaranteed result: 20 identical non-zero
differences, all the same sign, is the most extreme rank configuration
possible. Wilcoxon will return a vanishingly small p for **every** pair,
always, regardless of whether the memory difference is 4 GB or 4 MB. Three of
the fifteen p-values entering Holm-Bonferroni are fabricated this way, which
also distorts the correction applied to the twelve real ones.

**Fix:** record `torch.cuda.max_memory_allocated()` per run inside the timed
loop (resetting stats before each), and return the list. That is a ~6-line
change to `efficiency.py` and makes the test mean what the manuscript says it
means.

`FINDINGS.md` item 3 already notes the memory *reporting* is missing median,
IQR, and rank-biserial effect size. That still stands, and is downstream of
this: you cannot report a median of one number.

### 5.2 Inference-time significance is never computed

README: *"Paired bootstrap resampling — 1000 samples, two-tailed, for accuracy
metrics **and per-sentence inference time**."*

`EfficiencyBenchmark` never records per-sentence times — `_run_inference`
accumulates a total and divides by sentence count (`efficiency.py:139-141`), so
what survives is 20 per-run averages, and even those are not passed to any
test. `run_full_comparison` accepts only `per_sentence_scores` and
`gpu_memory_runs` (`statistical_tests.py:219-223`). RQ2's speed claim has no
significance test at all.

**Fix:** time each sentence individually in `_run_inference` and return the
vector; feed it to `paired_bootstrap` alongside the four accuracy metrics.

### 5.3 Holm-Bonferroni is missing its step-down stopping rule

`statistical_tests.py:196-208` compares each sorted p-value against its own
`alpha / (m - rank + 1)` independently. Holm requires stopping at the **first**
non-rejection: every hypothesis after it is retained regardless of its own
threshold.

Without the stop, a later comparison can be declared significant after an
earlier one failed. With `m = 15`: rank 14 gets `α/2 = 0.025` and rank 15 gets
`α/1 = 0.05`, so `p₁₄ = 0.030` (fail) followed by `p₁₅ = 0.040` (pass) is
reported as significant — which Holm forbids. It bites exactly at the tail
where p-values sit near α, which is where borderline thesis results live.

Note that `FINDINGS.md` lists *"Holm-Bonferroni is implemented correctly"*
under **What was found correct**. That entry should move.

### 5.4 Alpha-word accuracy aligns by position index

`metrics.py:139-148` compares `pred_words[i]` to `ref_words[i]`. One inserted
or deleted word shifts every subsequent index and zeroes the rest of the
sentence. For a normalization task where the model may legitimately split
`sanaol` into `sana all`, this systematically under-reports accuracy — and
under-reports it *differently* per variant, depending on how often each changes
token count. Consider an edit-distance alignment, or a multiset comparison, and
state which in the manuscript.

The same logic is then re-implemented inline in `run_experiment.py:242-251`
rather than calling `NormalizationMetrics` — with a different empty-sentence
default (`1.0` there, excluded from the denominator in `metrics.py`). Two
implementations of one metric will drift.

### 5.5 "GLEU+" is smoothed sentence-BLEU

`metrics.py:42` uses `sacrebleu.BLEU(smooth_method="add-k", smooth_value=1)`
and averages `sentence_score` across sentences. GLEU as defined for grammatical
error correction (Napoles et al.) is a *source-aware* metric: it rewards
n-grams that correctly changed relative to the input and penalises errors left
uncorrected. This implementation never looks at the noisy input.

If the manuscript defines GLEU+ as "sentence-level BLEU with add-one
smoothing", the code matches the manuscript — but the name will draw a question
from anyone who knows the GEC literature, and the docstring's claim that it
"captures n-gram precision **and recall**" is wrong for BLEU. Decide which
metric you mean and make the three places agree.

---

## 6. The demo tool (backend + frontend)

The registry design is good: lazy loading, per-variant checkpoint resolution
with env overrides, `best_stage1.pt` accepted as a fallback, and a 503 that
names both the missing path and the exact command that produces it. `/health`
reporting availability per variant is the right call. This is thoughtfully
built for its situation.

Issues, roughly in order:

- **The endpoints block the event loop.** `normalize` and `normalize_batch` are
  `async def` (`app.py:296`, `:312`) but call synchronous, CPU-bound
  `model.generate()`. FastAPI runs `async def` handlers directly on the event
  loop, so one request freezes the entire server — including `/health` — for
  its whole duration. Beam search over a few hundred bytes on CPU is seconds to
  minutes. **Fix: drop the `async`.** FastAPI then runs them in a threadpool.
  One-word change, and it is the difference between a demo that degrades and a
  demo that appears hung.
- **The batch endpoint isn't batched.** `normalize_batch` loops one sentence at
  a time (`app.py:314-330`), so it is strictly slower than the client calling
  `/normalize` in a loop, with none of the GPU batching benefit its name
  implies.
- **No timeout anywhere.** The frontend `fetch` (`App.tsx:49`) has no
  `AbortController`, and the backend has no cap. A slow CPU generate leaves the
  UI spinning indefinitely with no way out but a reload. For a live defense
  demo, add a client-side abort (~30s) and a clear timeout message.
- **The frontend never calls `/health`.** The model dropdown offers all three
  variants unconditionally; the user learns a variant is untrained only by
  submitting and getting a 503. `/health` already returns exactly the
  information needed to disable unavailable options and explain why. Doing this
  turns your biggest demo risk into a feature — the panel sees the tool
  correctly reporting its own state instead of appearing to error.
- **`torch.load(path, map_location="cpu")`** (`app.py:164`) omits
  `weights_only=True`, unlike `trainer.py:342` and `evaluate.py:117` which pass
  it. On torch ≥ 2.6 the default is now `True`, so this is currently safe — but
  `requirements.txt` allows `torch>=2.1.0`, where the default is `False` and
  loading a checkpoint executes arbitrary pickle. Pass it explicitly for
  consistency.
- **`allow_origins=["*"]` with `allow_credentials=True`** (`app.py:193-199`) is
  a combination browsers reject outright. Harmless here (no credentials are
  sent) but wrong as written.
- **Silent truncation.** `normalize_text` truncates at 1024 bytes
  (`app.py:262`) with no signal to the user, who sees a short output and no
  explanation.
- Neither `frontend/` nor `annotation-platform/` is covered by CI — the
  workflow runs `pytest tests/` only. `npm run typecheck` exists in
  `frontend/package.json` and nothing ever executes it.

---

## 7. Tests and CI

26 tests, all passing, in 11.5 seconds. The coverage is deep in exactly one
place and absent everywhere else:

| Component | Tests |
|-----------|-------|
| `delete_gate.py` | 15 — thorough, including the right failure modes |
| model forward/generate (all 3 variants) | 11, against a stubbed tokenizer/model |
| `losses.py` | only indirectly, via the gate tests |
| `metrics.py` | **none** |
| `statistical_tests.py` | **none** |
| `noise_generator.py`, `noise_label.py`, `dataset.py`, `preprocessing.py` | **none** |
| `trainer.py` | **none** |
| `scripts/*` | **none** ← §3.1 would have been caught by one import-and-call test |
| `backend/app.py` | **none** — FastAPI's `TestClient` makes `/health` and the 503 path trivial to test |
| both frontends | **none**, and not type-checked in CI |

The gap that cost the most is `scripts/`. A single test that builds a 10-row
CSV and calls `train.py`'s data-preparation path would have caught §3.1 the day
it was written. `statistical_tests.py` is the second: it is pure functions over
arrays, the easiest thing in the repository to test, and it is where three of
the findings above live — spec 007's `quickstart.md` already contains a hand
computation that could be pasted in as an assertion.

---

## 8. Corrections to `specs/FINDINGS.md`

`FINDINGS.md` remains the best artefact in the repository and most of it is
accurate. Three entries are now out of date:

1. **"Stanford adds Gumbel noise to the gate logits during training; this repo
   does not."** — It does now. `delete_gate.py:150-152`, config-gated via
   `use_gumbel_noise`, training-only, with two tests.
2. **"Hard deletion here is a Python loop over the batch."** — No longer.
   `delete_gate.py:250-270` is the vectorised `cumsum`/`scatter_add_`/`gather`
   version, with the reasoning recorded in the comment.
3. **"Holm-Bonferroni is implemented correctly"** (under *What was found
   correct*) — see §5.3. It is missing the step-down rule.

Findings 1, 2, 4, 5, 6, 7, 8, 9 and all four minors are still accurate as
written. The uncommitted edit in the working tree (`src/training/losses.py`,
deleting the words *"Only applies to TAHIMIK."* from a comment) is
comment-only and behaviourally inert.

---

## 9. Recommended order of work

Ordered by (blocks everything else) → (silently corrupts results) → (defense
risk) → (cleanup).

**Before anything else**

1. **Fix the eight `NormalizationDataset(...)` call sites** (§3.1) and add a
   smoke test that runs `train.py`'s data prep on a 10-row CSV. Also make
   `--gold_data` optional so Stage 1 can run alone.
2. **Thread `position_bias` through `_run_encoder_layers`** (§3.2), in one
   shared implementation rather than two copies. Add a test asserting that
   TAHIMIK with the gate disabled matches HuggingFace's encoder output.

**Before any number is recorded**

3. **Per-run GPU memory** in `EfficiencyBenchmark` (§5.1) — the current code
   cannot produce an honest memory result.
4. **Per-sentence inference times** + bootstrap (§5.2).
5. **Dynamic padding in `collate_fn`** (§3.3) — otherwise the speedup measures
   padding removal.
6. Holm step-down rule (§5.3); two-tailed bootstrap with add-one smoothing and
   the CI check (`FINDINGS.md` 1 & 2). These four are the difference between
   results you can defend and results you cannot.

**Before the defense**

7. **Log `cn`, `navg`, and deletion rate by `n*` decile** (§3.4). Cheap, and it
   is the evidence that the contribution exists.
8. **Drop `async` from the two endpoints; add frontend `/health` gating and a
   request timeout** (§6). Small changes; they convert the 503 from an
   embarrassment into a demonstration of correct behaviour.
9. Decide what to do about the synthetic noise distribution (§4.2) — through
   the spec loop, since it is a design question, not a patch.

**Cleanup**

10. Stage 2 resuming from `best_stage1.pt` (§4.1); noise probabilities into
    `configs/` (§4.3); de-duplicate `_run_encoder_layers` and the inline
    per-sentence metrics; fix or delete the label-free `forward()` branch
    (§3.5); mask the `L_attn_reg` mean (§3.6); tests for
    `statistical_tests.py` and `/health`.

---

## 10. How to talk about this at the defense

The honest framing is stronger than it looks, and it is the one the repository
has already committed to.

The claim is not "the implementation is finished". It is: *the architecture is
built and specified, the spec loop found nine divergences before any result was
reported, and a systematic implementation review found several more — including
one that would have invalidated the controlled comparison.* That is what
methodological rigour looks like from the inside, and it is a considerably
better story than a clean-looking repo whose numbers nobody has audited.

Two things to have ready:

- **The position-bias bug (§3.2) is the best thing in this report to own.** It
  is subtle, it is real, it would have quietly biased every accuracy comparison
  against the compressed variants, and it was found by reading the code against
  the framework's source rather than by a test failing. Panelists who write
  models will recognise the class of bug immediately.
- **If asked "does it run?", do not say yes yet.** Say: the model and the test
  harness run; the entry-point scripts have an argument-passing bug that is
  four lines per file; here is the `TypeError`, here is the fix. Being able to
  produce the exact failure and its cause on demand reads as command of the
  system. Guessing does not.
