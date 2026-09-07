# TAHIMIK — Tool Defense Preparation

A practical guide to defending this system in front of an industry panel:
how it's built, how it flows end to end, how to survive a live-debugging moment,
and the questions you will be asked (with answers you can defend).

> **Read this alongside** [`docs/src-explained/`](src-explained/README.md) (the
> line-by-line code docs) and [`specs/FINDINGS.md`](../specs/FINDINGS.md) (the
> known gaps). This file is the *strategy*; those are the *detail*.

---

## 0. First, a reframe of what actually gets tested

You said code quality, coupling, architecture, and variable consistency matter.
**You're right — but rank them correctly for a *tool* defense:**

| What panels actually probe | Why | Your risk |
|---|---|---|
| **1. Can you trace your own flow live?** | They test understanding, not memorization | HIGH — you said you're unfamiliar with the flow |
| **2. Can you justify each design decision with a "why"?** | Industry devs smell cargo-culting | MEDIUM — the "why"s exist in your specs; learn them |
| **3. Does the demo work?** | A tool defense expects a working tool | **CRITICAL — your tool returns HTTP 503; no model is trained** |
| **4. Architecture / coupling / cohesion** | Signals engineering maturity | LOW–MEDIUM — the code is actually fairly clean; know its seams |
| **5. Variable/naming consistency** | Signals care | LOW — mostly consistent; know the 2 traps below |

So: architecture matters, but **the flow, the "why"s, and the demo-that-can't-yet-
normalize are where you win or lose.** This doc front-loads those.

---

## 1. The 30-second and 3-minute pitches (memorize these)

**30-second (the elevator):**
> "TAHIMIK normalizes noisy Filipino/Taglish social-media text — *slmt* back to
> *salamat*. Our contribution is **noise-adaptive compression**: the model
> estimates how noisy each sentence is, then compresses clean sentences
> aggressively for speed and preserves noisy ones for accuracy. We prove it by
> comparing three models — a no-compression ceiling (ByT5), a fixed-compression
> baseline (MrT5), and our adaptive model — under identical conditions."

**3-minute (the architecture story) — say it in this order:**
1. **Problem:** Filipino social text is noisy; normalization helps downstream NLP.
2. **Insight:** not all sentences are equally noisy — compression should adapt.
3. **Backbone:** ByT5, a byte-level model, because the noise is sub-word.
4. **Mechanism:** a *delete gate* (from MrT5) scores each byte keep/delete; we add
   a *noise estimator* that shifts those scores by predicted noise.
5. **Experiment:** three variants, one config, one dataset, one trainer — only the
   compression differs (independent variable).
6. **Evaluation:** accuracy (GLEU+, chrF, ERR, alpha-word), efficiency (time,
   memory), and significance (bootstrap, Wilcoxon, Holm-Bonferroni).
7. **Status (say this proactively):** the implementation and harness are complete
   and tested; the gold dataset is still being annotated, so trained checkpoints
   and final results don't exist yet. The tool runs and reports exactly what's
   missing.

---

## 2. The system, end to end (the flow you must own)

There are **two flows**. Panels conflate them; you should separate them cleanly.

### Flow A — the experiment (offline: how a model is made and scored)

```
 clean corpus (.txt/.csv)                     gold dataset (.csv, being annotated)
        │                                              │
        ▼  preprocessing.load_clean_corpus             ▼  preprocessing.load_gold_standard
 noise_generator.apply_noise  ──►  noisy text          │
        │                                              │
        ▼  noise_label.compute_noise_level (n*)        ▼  compute_noise_level (n*)
 (noisy, clean, n*) synthetic pairs            (noisy, clean, n*) gold pairs
        │                                              │
        └───────────────┬──────────────────────────────┘
                        ▼  preprocessing.split_data (seed 42)
                 train / val / test  →  NormalizationDataset  →  DataLoader (collate_fn)
                        │
                        ▼  trainer.train()
        Stage 1 (synthetic, 3 epochs)  →  Stage 2 (gold, 10 epochs)
                        │   each step: model.forward → TAHIMIKLoss → backward → AdamW
                        ▼
                 best_stage2.pt (checkpoint)
                        │
                        ▼  evaluate.py / benchmark.py / run_experiment.py
        metrics (accuracy) + efficiency (time/mem) + statistical_tests (significance)
                        │
                        ▼
                 full_results.json
```

### Flow B — the live tool (online: a user normalizes one sentence)

```
Browser (frontend/App.tsx)
   │  POST /normalize  { text, model, num_beams }
   ▼
FastAPI (backend/app.py)
   │  get_model(name)  → lazy-load checkpoint  (byt5 / mrt5 / tahimik)
   │     └─ if no checkpoint on disk → HTTP 503 naming the missing file  ◄── happens today
   ▼
model.generate(input_ids, attention_mask, num_beams)
   │  embed → pre-gate layers → (estimator → gate → HARD delete) → post-gate → decoder beam search
   ▼
normalized text + inference_time_ms  →  back to the browser
```

> **Rehearse tracing ONE sentence through Flow B out loud.** e.g. `"grabeeee ang
> init"` → tokenized to byte IDs → encoder layers 0–2 → estimator predicts n≈low
> (it's mildly noisy) → gate deletes some redundant bytes → decoder writes
> `"grabe ang init"`. If you can narrate that, you've handled 50% of the Q&A.

---

## 3. Architecture & code-quality talking points (honest assessment)

Industry panelists will poke the structure. Here's the truthful picture so you can
speak to it confidently — strengths *and* seams.

### 3.1 The core architectural idea: "one contract, three variants"

This is the single most important thing to articulate. Every variant is
interchangeable because they all honor the same **informal contract**:

- Same **config inheritance**: `BaseConfig` → `ByT5Config` / `MrT5Config` /
  `TAHIMIKConfig`. Shared settings live once; each subclass changes only its
  variable. *(This is the "control variables vs independent variable" of your
  experiment, expressed in code.)*
- Same **model interface**: each is an `nn.Module` with `forward(...)` and
  `generate(...)`, returning a **dict** with a common set of keys.
- Same **dispatch pattern**: every script has a `VARIANT_MAP` /`VARIANTS` dict
  mapping `"byt5"/"mrt5"/"tahimik"` → (config class, model class). See
  `scripts/train.py`, `scripts/run_experiment.py`, `backend/app.py`.
- Same **trainer + loss**: `TAHIMIKTrainer` and `TAHIMIKLoss` are generic; they
  switch behavior with flags (`use_compression`, `noise_adaptive`), not by
  special-casing each model.

**Say it like this:** *"The architecture is designed so the three models are drop-in
interchangeable — same config base, same forward/generate interface, same trainer.
That's deliberate: it's what makes the comparison fair, because only the
compression mechanism varies."*

### 3.2 Cohesion (strong) — separation of concerns

Each folder has one job, and they don't bleed into each other:
`data/` (make tensors), `models/` (architecture), `training/` (optimize),
`evaluation/` (score), `utils/` (shared), `backend/` (serve). This is **high
cohesion** and it's genuinely good. Point to it.

### 3.3 Coupling — where it's loose (good) and tight (a risk you should own)

**Loosely coupled (defend as intentional):**
- Models talk to the trainer/loss through a **dict of outputs**, not by type. The
  trainer calls `batch.get("noise_level")` and works for all three without knowing
  which model it holds.
- The delete gate and noise estimator are **standalone `nn.Module`s** reused by the
  variants — composition, not duplication.

**Tightly coupled (name these before they do — it reads as maturity):**
1. **Models reach into Hugging Face internals.** `noise_adaptive_byt5.py` and
   `fixed_compression_byt5.py` call `encoder.block[i]`, `embed_tokens`,
   `get_extended_attention_mask`, `final_layer_norm`, `_shift_right`, `lm_head`.
   This is necessary to split the encoder around the gate, but it **couples you to
   the `transformers` version** — you even have comments handling the 4.x vs 5.x
   `get_extended_attention_mask` signature. *Owning line:* "We accept coupling to
   transformers internals because splitting the encoder mid-stack requires it;
   we've pinned the version and documented the version-sensitive calls."
2. **The loss depends on exact output-dict keys.** `TAHIMIKLoss` reads
   `deletion_rate`, `target_deletion_rate`, `keep_prob`, `fixed_deletion_target`,
   `noise_scores`, `ne_loss`. If a model forgets a key, the loss breaks at runtime.
   This is an **implicit contract** — a place a stricter design would use a typed
   object. *Owning line:* "The model↔loss contract is a dict today; a dataclass
   would make it explicit and type-checked — a fair refactor."

### 3.4 Variable/naming consistency — know these two traps

- **`n` vs `n*` (predicted vs truth).** `noise_scores` = the model's *predicted*
  noise `n`; `noise_level` = the *ground-truth* `n*` (edit-distance ratio). Never
  mix these up on the stand — the whole L_NE story depends on the distinction.
- **Output-key inconsistency (a real wart, `FINDINGS.md` #2).** `byt5_baseline.py`
  returns a slightly different set of dict keys than the two compressed variants
  (it has no gate outputs). *Owning line:* "The baseline's output dict isn't fully
  uniform with the others — a known consistency gap we tracked in FINDINGS."

### 3.5 Reproducibility as an architectural value

Everything is **seed-42, config-driven**. Hyperparameters live in `configs/`, never
inline (a project rule). This is a genuine strength with industry devs — say
"our runs are reproducible from commit + seed + config." *(Caveat you must also
own: seeding is done but full CUDA determinism isn't enabled yet — `FINDINGS.md`
#5.)*

---

## 4. Design decisions you MUST be able to defend (each with its "why")

Panels love "why did you do X and not Y?" Have the *reason*, not just the *what*.

| Decision | The "why" (say this) |
|---|---|
| **Byte-level (ByT5), not word/subword** | Filipino noise is *sub-word* (dropped vowels, elongation, typos). A word tokenizer chokes on `grabeeee`; a byte model just sees more `e` bytes. |
| **Delete gate at encoder layer 3** | Early enough to save compute on most of the encoder, late enough that the first layers built real context to score bytes intelligently. Same layer for both compressed variants → fair comparison. |
| **Soft deletion in training, hard at inference** | Soft = differentiable (gradients can train the gate); hard = physically shorter sequence = the real speed-up. You can't train through a hard cut. |
| **Gate added to attention (not multiplied)** | HF converts a binary mask to `(1-mask)*-1e34`; multiplying a soft 0.99997 keep would become a hard delete and kill the gradient. Documented at the call site. |
| **`.detach()` on the noise score** | Gradient isolation — the estimator must learn *only* from L_NE (its own MSE against n*), not from the gate. This is the integrity of the contribution; a test guards it. |
| **`target = d_max·(1−n)`** | Clean (n≈0) → target ≈ 0.5 (compress hard); noisy (n≈1) → target ≈ 0 (preserve). This one line *is* "noise-adaptive." |
| **Two-stage training** | Stage 1 (synthetic, ~1M) teaches general Filipino noise; Stage 2 (gold, ~15K) calibrates on real distributions. Best-val checkpoint per stage. |
| **EMA for `navg` (momentum 0.99)** | A stable "typical noise level" baseline for the shift `cn·(n−navg)`; a per-batch average would be too jumpy. |
| **Gumbel noise, training only** | Makes the near-discrete keep/delete decision explorable so the gate doesn't lock in early; off at inference so the same sentence compresses identically every time. |
| **Bootstrap + Wilcoxon + Holm-Bonferroni** | Per-sentence resampling for accuracy; Wilcoxon for the once-per-run memory number; Holm-Bonferroni because you run many comparisons and must control false positives. |

---

## 5. Live-debugging readiness

You're unsure whether there's a live debugging session. **Prepare as if there is.**
Industry panels sometimes say "open the file — walk me through what happens if…"
or "this returns the wrong shape, find it."

### 5.1 Your safety net: the tests (know what they prove)

```bash
python -m pytest tests/ -v
```
- `tests/test_delete_gate.py` — unit tests on the gate: deletion rate is a valid
  fraction *and* differentiable; the regularizer is symmetric; the soft mask is
  additive; **noisier sentences are compressed less** (the thesis claim);
  Gumbel-noise is training-only; hard deletion left-packs correctly.
- `tests/test_model_forward.py` — integration tests using a **tiny random T5**
  (built via `monkeypatch`, because a real ByT5 is ~1.2 GB) to prove wiring,
  shapes, gradient flow, and — critically —
  `test_noise_estimator_is_trained_only_by_l_ne` (the gradient isolation).

**If asked "how do you know the gate works without a trained model?"** → "These
tests prove the *mechanism* (gradients, shapes, the adaptive direction) on a tiny
model. Training quality needs the dataset; correctness of the machinery doesn't."

### 5.2 Where to set breakpoints / what to inspect

If they hand you a debugger or a notebook:

| Symptom | Where to look | What to print |
|---|---|---|
| Wrong shapes | `delete_gate.forward`, model `forward` | `.shape` of `hidden_states`, `gate_outputs`, `keep_prob`, `kept_mask` |
| Gate does nothing / rate stuck | `delete_gate.forward` step 5 | `deletion_rate`, `keep_prob.mean()`, and check `rate.grad_fn is not None` |
| Estimator not learning | `losses.forward` | `l_ne`; verify `noise_scores` vs `noise_level` |
| NaN loss | training step | `torch.isfinite(loss)`, drop to fp32, check LR |
| Adaptivity inverted | `delete_gate` step 2 | sign of `cn` and `shift` (see `FINDINGS.md` #7) |
| Device error | trainer | every tensor's `.device` must match |

### 5.3 The "trace a bug" playbook (say your method aloud)

1. **Reproduce minimally** — smallest batch, tiny T5, `model.train()` vs `.eval()`.
2. **Print shapes at each stage** — most bugs are shape/mask mismatches.
3. **Check `grad_fn`** — if a tensor that should be trainable has `grad_fn=None`,
   the gradient path is broken (this is literally what bug-fix tests here guard).
4. **Isolate soft vs hard** — training bug? inspect the soft path; inference bug?
   inspect `apply_hard_deletion`.
5. **Bisect the loss** — call `.backward()` on one term at a time to see which
   reaches the gate/estimator.

### 5.4 Common real failure points in THIS codebase (pre-load these)

- **HTTP 503 on the tool** — *expected*; no checkpoint exists. Not a bug.
- **Tokenizer download** — `AutoTokenizer.from_pretrained` needs network on first
  use (or a warm HF cache). The backend catches this and says so.
- **transformers version drift** — the `get_extended_attention_mask` device/dtype
  positional-arg difference between 4.x and 5.x (commented in the models).
- **fp16 underflow** — handled by `GradScaler`; Gumbel `eps` is loosened under fp16.
- **`generate()` needs `BaseModelOutput`** — passing a bare tuple triggers a
  `bos_token_id` error (commented at the call site).

---

## 6. The demo problem — how to defend a tool that can't normalize yet

**This is your biggest exposure.** A "tool defense" implies a working tool, but
`/normalize` returns **503** because no model is trained (the gold dataset is still
being annotated). Do not get ambushed. Options, best first:

**Option A — Train a tiny demo checkpoint beforehand (recommended if time allows).**
Run Stage-1-only on a small synthetic set with ByT5-base (or tiny backbone fixture for local validation) to produce a real
`best_stage1.pt`. It won't be accurate, but the tool will *respond*, and you can
say "quality needs the gold data; this proves the end-to-end path is live." The
backend already accepts `best_stage1.pt` as a fallback.

**Option B — Demo the system's honesty.** Show `GET /health` listing each variant
and its exact missing checkpoint path; show the 503 message naming the file and the
command that produces it. Frame it as *engineering maturity*: "the tool fails
loudly and precisely instead of pretending."

**Option C — Demo the tests live.** `pytest tests/ -v` in front of them proves the
mechanism works, including the thesis claim (noisier → less compression).

**Option D — Walk the forward pass on the tiny T5** in a notebook, printing the
gate's `deletion_rate` for a clean vs noisy sentence to *show* adaptivity.

**The framing that wins:** *"The scientific contribution is the method and its
validated implementation. Results are blocked on the gold dataset, not on the
code. We separated 'what's verifiable now' from 'what needs data' deliberately —
it's in our specs."* (This is literally your project's stated status.)

---

## 7. Anticipated panel questions + defensible answers

### Architecture / engineering
- **"Isn't splitting the encoder and reaching into HF internals fragile?"** →
  "Yes, it couples us to the transformers version — we pin it and comment the
  version-sensitive calls. The alternative (subclassing the whole model) buys
  isolation at the cost of far more code to maintain and audit for a thesis."
- **"Why a dict contract between model and loss instead of a typed object?"** →
  "Pragmatism; it kept the three variants interchangeable quickly. A dataclass
  would make the contract explicit and catch a missing key at construction — a
  legitimate refactor I'd do for production."
- **"How would this scale to a new variant?"** → "Add a config subclass, a model
  class emitting the same output keys, and one line in each `VARIANT_MAP`. The
  trainer, loss, and backend don't change."

### Machine learning
- **"How does the model know the noise at inference with no clean reference?"** →
  "It doesn't have n*; the *noise estimator* predicts n from the encoder's hidden
  states. n* only exists in training, as the estimator's target (L_NE)."
- **"Why won't the gate just delete everything to minimize the rate loss?"** →
  "L_rate targets a *specific* rate (0.5 or the adaptive target), not zero; and
  L_CE punishes deleting information the decoder needs. The symmetric L_attn_reg
  only discourages indecision, not one direction." *(Note the honest caveat:
  L_attn_reg deviates from MrT5's paper — `FINDINGS.md` #8.)*
- **"What stops the adaptive behavior from inverting?"** → "Nothing currently
  constrains the sign of `cn`; a negative `cn` would invert it silently. We
  flagged it (`FINDINGS.md` #7); the fix is to constrain `cn ≥ 0`."

### Reproducibility / rigor
- **"Are your numbers reproducible?"** → "Seed 42, config-driven, no inline
  hyperparameters. Full CUDA determinism isn't enabled yet — a known gap
  (`FINDINGS.md` #5) — so today it's reproducible up to GPU non-determinism."
- **"Why byt5-small in the config if the paper says base?"** → "Small was an early dev
  default; the codebase migrated authoritatively to `google/byt5-base` across all
  configs, tokenizers, and checkpoint validators (resolved in AD-002). Checkpoint architecture
  validation explicitly rejects legacy small checkpoints (`FINDINGS.md` #4)."

### The gaps (they *will* find at least one — get there first)
- **"Your significance test looks one-tailed."** → "Correct, and it also ignores
  the confidence interval — both are logged in `FINDINGS.md` #1–2 with the fix.
  We found them via our spec-review process before reporting any result."

---

## 8. Own the weaknesses first (turn FINDINGS into a strength)

Industry devs respect engineers who know their own bugs. **Volunteer these**; don't
get caught. Each is in [`specs/FINDINGS.md`](../specs/FINDINGS.md):

| # | Gap | One-line framing |
|---|---|---|
| 1–2 | Bootstrap is one-tailed & ignores the CI | "Found in spec review; fix scoped before we report significance." |
| 4 | Config runs `byt5-base`, legacy small rejected | "Resolved: ByT5-base is the shared backbone; small checkpoints fail closed." |
| 5 | Seeded but not bit-reproducible | "Determinism mode not yet enabled; documented trade-off." |
| 6 | Checkpoints don't record the commit | "Provenance stamping is scoped." |
| 7 | `cn` sign unconstrained (could invert adaptivity) | "The one silent failure; fix is `cn ≥ 0`." |
| 8 | `L_attn_reg` deviates from MrT5 Appendix D | "Deliberate, documented substitute; reconciling with the manuscript." |

**The meta-point to make:** *"These exist because we ran a spec-first review that
deliberately looked for divergences between the manuscript and the code, instead of
letting the code silently become the spec. Finding them is the process working."*

---

## 9. A one-week prep plan

1. **Day 1–2:** Narrate Flow A and Flow B out loud until fluent (Section 2). Trace
   one real sentence through Flow B.
2. **Day 3:** Memorize the design "why" table (Section 4). These are your Q&A wins.
3. **Day 4:** Run `pytest tests/ -v`; read each test's docstring so you can explain
   what it proves (Section 5.1).
4. **Day 5:** Decide your demo strategy (Section 6). If feasible, train a tiny
   Stage-1 checkpoint so the tool responds.
5. **Day 6:** Read `FINDINGS.md` end to end; rehearse owning each gap (Section 8).
6. **Day 7:** Mock defense — have a groupmate ask from Section 7.

---

## 10. Quick reference — file → one line (for the "open X" moment)

| File | If they open it, it does… |
|---|---|
| `configs/base.py` | Shared settings (seed, LR, splits, schedule) |
| `configs/{byt5,mrt5,tahimik}_config.py` | Per-variant overrides (the one changed variable) |
| `src/data/noise_generator.py` | Makes synthetic noise (9 categories) |
| `src/data/noise_label.py` | Computes n* (edit-distance ratio) |
| `src/data/dataset.py` | Text → tensors (`input_ids`, mask, labels, n*) |
| `src/data/preprocessing.py` | Loads, cleans, generates, splits data |
| `src/models/delete_gate.py` | Scores bytes, soft/hard deletion (the engine) |
| `src/models/noise_estimator.py` | Predicts n from hidden states (MLP) |
| `src/models/noise_adaptive_byt5.py` | **TAHIMIK** — MrT5 + estimator (the contribution) |
| `src/models/fixed_compression_byt5.py` | MrT5 baseline (fixed 50%) |
| `src/models/byt5_baseline.py` | Plain ByT5 (accuracy ceiling) |
| `src/training/losses.py` | Combines L_CE + L_rate + L_attn_reg + L_NE |
| `src/training/trainer.py` | Two-stage loop, saves best checkpoint |
| `src/evaluation/*` | Accuracy, efficiency, significance, IAA |
| `backend/app.py` | FastAPI server; lazy-loads checkpoints; 503 if none |
| `scripts/run_experiment.py` | Runs all 3 variants + stats end to end |

**Deeper detail for any file:** [`docs/src-explained/`](src-explained/README.md).
**Every known gap:** [`specs/FINDINGS.md`](../specs/FINDINGS.md).
