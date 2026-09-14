# Findings summary — read this first

Every gap the spec pass found between what the manuscript specifies and what
the code does, ordered by how much it matters. Written to be read in one
sitting and corrected.

**Historical record.** Findings 1–3 below describe the pre-remediation
implementation. The current code implements the two-tailed add-one bootstrap,
CI-gated significance, paired memory-difference summaries in GB, and
signed-rank rank-biserial effect size. The remaining entries preserve the
original audit trail and must not be treated as a current code reading.

---

## Fix before any results are reported

### 1. The bootstrap significance test is one-tailed; the manuscript specifies two-tailed

**Where**: `src/evaluation/statistical_tests.py:94`
**Spec**: [007](007-evaluation-and-statistics/spec.md) F1

The code computes:

```python
p_value = wins_a / self.n_bootstrap
```

The manuscript specifies:

```
p = 2 × min( (1 + Σ I(Δb ≤ 0))/(B+1),  (1 + Σ I(Δb ≥ 0))/(B+1) )
```

Three differences: the code is one-tailed where the manuscript is two-tailed,
it omits the add-one smoothing, and its docstring states a one-sided null
(*"the null hypothesis is that A >= B"*) while your hypotheses H₀₁ and H₀₂ are
written as "no significant difference", which is two-tailed.

**Why it matters**: a one-tailed test is roughly twice as easy to pass. Results
called significant under the current code may not be significant under the
method your manuscript describes.

**Corroboration**: `wilcoxon_test` in the same file *is* correctly two-sided.
The two tests disagree with each other, which suggests oversight rather than a
deliberate choice.

### 2. Significance ignores the confidence interval

**Where**: `src/evaluation/statistical_tests.py:106`
**Spec**: [007](007-evaluation-and-statistics/spec.md) F2

The manuscript requires **both** `p < 0.05` **and** a confidence interval
excluding zero. The code checks only the p-value. The CI is computed correctly
two lines earlier and then never consulted.

### 3. GPU memory reporting is missing three required statistics

**Where**: `src/evaluation/statistical_tests.py:120-171`
**Spec**: [007](007-evaluation-and-statistics/spec.md) F3

The manuscript requires median difference in GB, interquartile range, and
matched-pairs rank-biserial effect size. The code reports means and no effect
size. The test itself is correct; the reporting is incomplete.

---

## Fix before running real experiments
 
### 4. The code runs `byt5-small`; the manuscript specifies `byt5-base` [RESOLVED 2026-09-08]

**Where**: `configs/base.py:25`, inherited by all three variants
**Specs**: [003](003-byt5-baseline/spec.md), [004](004-fixed-rate-compression/spec.md), [005](005-noise-adaptive-byt5/spec.md) F1
**Status**: **RESOLVED** on 2026-09-08 via AD-002 and the ByT5-Base Migration Plan.

Manuscript: *"The base variant of ByT5 will be used, and both MrT5 and the
proposed noise-adaptive model will adopt the base variant."*

Resolution:
1. `BaseConfig.model_name` is set to `"google/byt5-base"` as the single authority across all variants and tokenizer loaders.
2. `scripts/run_experiment.py` uses `base_config.model_name` rather than any hardcoded string.
3. All compression modules (delete gate, noise estimator) dynamically derive shapes from `config.d_model` (1,536 for Base).
4. Physical batch sizes (2) are paired with gradient accumulation steps (8 for Stage 1, 4 for Stage 2) to preserve effective batches of 16 and 8.
5. Checkpoint architecture validation explicitly rejects Small (`d_model=1472`) checkpoints from being loaded into Base models.
6. Checkpoint provenance and hardware preflight script (`scripts/preflight_base.py`) verify model identity before training.

### 5. Runs are seeded but not bit-reproducible

**Where**: `scripts/train.py:117` and siblings
**Spec**: [006](006-two-stage-training/spec.md) F1

Seeding is done correctly at every entry point. What is missing is the rest of
deterministic mode — `torch.use_deterministic_algorithms(True)`, cuDNN
deterministic flags, `CUBLAS_WORKSPACE_CONFIG`. This is the gap your
constitution knowingly opened today when you chose full determinism.

### 6. Checkpoints cannot be traced to a commit

**Where**: `src/training/trainer.py:176-195`
**Spec**: [006](006-two-stage-training/spec.md) F2

Constitution Principle III requires every results file to carry the git SHA, a
dirty flag, and the resolved config. Checkpoints currently record epoch, stage,
and validation loss only.

---

## Found by grilling, not by the first pass

### 7. Nothing constrains the sign of `cn`

**Where**: `src/models/delete_gate.py:77`
**Spec**: [005](005-noise-adaptive-byt5/spec.md) T016

`cn` is the learned coefficient controlling how strongly noise shifts the
keep/delete decision. It initialises to `1.0`, and nothing prevents it crossing
zero during training.

A negative `cn` **inverts the contribution**: noisy sentences would be
compressed *more* rather than less, the exact opposite of the design. The model
would still run, still converge, and still produce plausible numbers. No test
detects it.

This is the only finding here describing a failure with no visible symptom.
Everything else either already works or is visibly incomplete.

---

## Decided, pending implementation

**[AD-001](DECISIONS.md)** — the delete gate will be replaced with Stanford's
official MrT5 implementation, trained from `byt5-small` on this study's data.
Their released checkpoint will **not** be used, because its extra multilingual
pretraining would be exclusive to one variant and would confound the
comparison.

Comparing the two implementations surfaced two differences worth knowing before
that lands:

- **Stanford adds Gumbel noise to the gate logits during training; this repo
  does not.** That is a standard technique for making discrete keep/delete
  decisions trainable, so training dynamics may differ materially.
- **Hard deletion here is a Python loop over the batch**
  (`delete_gate.py:226`); Stanford's is vectorised with `cumsum`/`gather`.
  Hard deletion runs at *inference*, which is exactly what RQ2 measures, and
  the overhead falls on the two compressed variants — the ones whose claim is
  that they are faster. The magnitude is unmeasured, but the direction works
  against the hypothesis.

Nothing has been changed. See [`DECISIONS.md`](DECISIONS.md) for the full
rationale and what it costs.

---

## Needs your decision, not a code change

### 8. `L_attn_reg` deviates from MrT5 Appendix D

**Where**: `src/training/losses.py:106-129`
**Spec**: [004](004-fixed-rate-compression/spec.md) F2

MrT5 Appendix D defines this term over attention weights. The code uses a
gate-commitment term `mean(4p(1-p))` instead, and documents the substitution
honestly in a comment, including why the previous version was worse (it was
minimised by deleting the entire sequence).

The code is honest. The manuscript is silent. Either reconcile the code to the
paper, or document the substitution in the manuscript as deliberate. A reader
comparing the two today would find an unacknowledged mismatch.

### 9. Two unspecified constants in the contribution

**Both are marked `NEEDS CLARIFICATION` and their dependent tasks are BLOCKED**,
per Constitution Principle II. They are not guesses awaiting confirmation; they
are open questions awaiting an answer.

**Where**: `src/models/delete_gate.py:81` and `:151`
**Spec**: [005](005-noise-adaptive-byt5/spec.md) F3

`navg` initialises to `0.5`, and the gate shift is clamped to `[k, 0]`. Neither
value appears in the manuscript text located so far. Both are reasonable, but
they are currently code decisions standing in for design decisions. Marked
`NEEDS CLARIFICATION` rather than guessed.

---

## Minor

| Item | Where | Spec |
|------|-------|------|
| Dead `encoder_outputs` assignment | `fixed_compression_byt5.py:176` | 004 F3 |
| Deprecated `torch.cuda.amp` spelling | `trainer.py:27` | 006 F3 |
| Non-uniform output keys across variants | `byt5_baseline.py` | 003 F2 |
| `navg` EMA update not under `no_grad()` | `delete_gate.py:136` | 005 F4 |

---

## What was found correct

Worth stating, because the list above reads worse than the codebase is:

- **The gradient isolation the whole comparison depends on is real.** `n` is
  detached at both consumption points, exactly as the manuscript states.
  `delete_gate.py:131` and `noise_adaptive_byt5.py:187`.
- The soft/hard deletion split is correct, including the subtle log-space bias
  requirement, which is not only right but documented at the call site with the
  reasoning for why the obvious alternative breaks it.
- All four accuracy metrics and both efficiency metrics are implemented.
- The 5-warmup / 20-run benchmark protocol matches the manuscript exactly.
- Holm-Bonferroni is implemented correctly.
- Two-stage training, per-stage hyperparameters, and the best-validation reset
  between stages are all correct.
- Seeding is done properly at every entry point.

Of roughly 40 functional requirements checked across six specs, the substantial
majority are already satisfied. The findings above are the exceptions.

---

## Resolved Remediations (2026-09-08 Implementation)

The manuscript alignment remediation completed and verified the following key fixes:

1. **Authoritative ByT5-Base Backbone (Finding 4)**: `configs/base.py` and `scripts/run_experiment.py` configure `google/byt5-base` across all three conditions and load tokenizers dynamically.
2. **Non-Negative Adaptive Coefficient (Finding 7)**: `src/models/delete_gate.py` parameterizes $c_n$ via `raw_cn` and `F.softplus(raw_cn) >= 0.0`, eliminating gate-inversion risk. Legacy checkpoint loading automatically adapts scalar `cn`.
3. **EMA Gradient Isolation**: `navg` update is executed under `torch.no_grad()`.
4. **Preserved Relative Position Bias**: Layer index 1 relative position bias is preserved and gathered via `compress_position_bias` in `src/models/encoder_layers.py`.
5. **Dynamic Padding & Collation**: `src/data/dataset.py` tokenizes with a 1,024-byte ceiling without static padding; batches are collated dynamically with `NormalizationCollator`.
6. **Synthetic Noise Lineage & Policy**: Stage 1 requires a validated manifest, records lineage/diagnostics, and rejects copy pairs. Code-switching and Taglish-morphology generation remain externally blocked until reviewed local resources are supplied and implemented.
7. **Best-Stage-1 Checkpoint Handoff**: `src/training/trainer.py` restores `best_stage1.pt` before constructing Stage 2 optimizers and saves checkpoint parentage/state.
8. **Aligned Alpha-Word Accuracy**: Sequence-aligned Levenshtein word matching prevents positional drift penalties.
9. **Batched Non-Blocking Inference API**: `backend/app.py` executes single-call `model.generate()` over batches without blocking the async event loop.

