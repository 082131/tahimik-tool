# TAHIMIK Manuscript Alignment Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove every currently confirmed logic, measurement, and runtime mismatch that could prevent the implementation from faithfully testing the manuscript's claims.

**Architecture:** Keep the three-model controlled experiment intact and repair shared infrastructure before model-specific behavior. A shared encoder runner will preserve relative position bias for both compressed variants; data loading will use a 1,024-position truncation ceiling with dynamic padding; adaptive diagnostics, synthetic-noise provenance, checkpoint handoff, metrics, and inference APIs will then be corrected behind regression tests.

**Tech Stack:** Python 3.11, PyTorch, Hugging Face Transformers/ByT5, FastAPI, pytest, NumPy/SciPy, JSON/CSV provenance artifacts.

**Spec:** `docs/superpowers/specs/2026-09-03-chapter3-gap-remediation-design.md`; `specs/005-noise-adaptive-byt5/spec.md`; `specs/011-synthetic-noise-policy/spec.md`; `specs/012-stage-checkpoint-handoff/spec.md`; `specs/014-methodology-verification/spec.md`

## Global Constraints

- Treat the manuscript equations and declared experimental protocol as authoritative.
- Keep ByT5, fixed-compression, and TAHIMIK data splits, initialization family, decoding settings, and measurement protocol identical unless the compression mechanism requires otherwise.
- Keep `max_input_length=1024` and `max_target_length=1024` as truncation ceilings; never restore per-sample fixed padding.
- Keep gradient isolation: predicted noise is detached at every gate/rate-target consumption point and learns only through `L_NE`.
- Keep five discarded warm-ups, twenty timed runs, and `batch_size=1` for the primary per-sentence efficiency benchmark.
- Do not use the released `stanfordnlp/mrt5-small` checkpoint for one comparator only.
- Do not report development, CPU-efficiency, dirty-tree, undersized-data, or incomplete-manifest outputs as thesis evidence.
- Preserve unrelated working-tree changes. Before each task, inspect `git diff -- <files>` and edit only the task's lines.
- The separate plan `docs/superpowers/plans/2026-09-08-byt5-base-migration.md` owns the Small-to-Base migration.

## Gemini 3.8 execution contract

- Execute one numbered task at a time and stop after its test/commit checkpoint for review.
- Before editing, read every file and spec named by that task and inspect the existing diff for those paths.
- Treat currently modified files as user-owned work; preserve changes outside the task's exact scope.
- Follow the red-green-refactor order exactly: add the behavioral test, observe the expected failure, make the smallest implementation, then rerun focused tests.
- Do not replace behavioral assertions with output-key or shape-only checks when the task specifies a mathematical invariant.
- If the manuscript does not settle a numerical or linguistic policy, implement fail-closed validation and report the external decision needed; do not guess a value.
- After each task, report changed files, the exact commands run, pass/fail totals, and any remaining dependency before proceeding.

## Current-state reconciliation

The September 4 audit includes items that are already present in the working tree. Do not reimplement them; retain and regression-test them:

- two-tailed add-one paired bootstrap and CI gating;
- Wilcoxon median/IQR/rank-biserial reporting;
- Holm step-down behavior;
- deterministic execution and provenance collection;
- Gumbel noise in training only;
- vectorized hard deletion;
- source-aware GLEU+ and per-sentence inference-time storage.

The tasks below cover the remaining confirmed gaps.

---

### Task 1: Preserve relative position bias in both compressed encoders

**Files:**
- Create: `src/models/encoder_layers.py`
- Modify: `src/models/delete_gate.py`
- Modify: `src/models/fixed_compression_byt5.py`
- Modify: `src/models/noise_adaptive_byt5.py`
- Test: `tests/test_encoder_position_bias.py`
- Test: `tests/test_delete_gate.py`

**Interfaces:**
- Produces: `EncoderLayerResult(hidden_states: Tensor, position_bias: Tensor)`.
- Produces: `run_encoder_layers(encoder, hidden_states, attention_mask, start_layer, end_layer, position_bias=None, gate_bias=None) -> EncoderLayerResult`.
- Produces: `HardDeletionResult(hidden_states: Tensor, attention_mask: Tensor, source_positions: Tensor)`.
- Produces: `compress_position_bias(position_bias, source_positions, attention_mask) -> Tensor`.
- Consumes: Hugging Face T5 block output index `1` as the self-attention position bias.

- [ ] **Step 1: Write failing position-bias propagation tests**

Create a tiny T5 encoder and instrument every block after the first to capture its `position_bias` argument. Test both a soft pass and a hard-deletion pass. The assertions must prove that later layers receive a non-`None` tensor and that hard deletion shrinks both query and key dimensions:

```python
def test_shared_runner_threads_position_bias_between_layers(tiny_encoder, states, mask):
    result = run_encoder_layers(tiny_encoder, states, mask, 0, 3)
    assert result.position_bias is not None
    assert result.position_bias.shape[-2:] == (states.size(1), states.size(1))


def test_compressed_position_bias_matches_retained_positions():
    bias = torch.arange(2 * 4 * 4).reshape(1, 2, 4, 4).float()
    source_positions = torch.tensor([[0, 2]])
    mask = torch.tensor([[1, 1]])
    compressed = compress_position_bias(bias, source_positions, mask)
    expected = bias[:, :, [0, 2]][:, :, :, [0, 2]]
    assert torch.equal(compressed, expected)
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python -m pytest tests/test_encoder_position_bias.py -v`

Expected: FAIL because `src.models.encoder_layers` does not exist and the current duplicated loops discard `layer_output[1]`.

- [ ] **Step 3: Implement the shared encoder runner**

Create `src/models/encoder_layers.py` with dataclasses for the two result types. The runner must construct the extended attention mask once, add `gate_bias[:, None, None, :]` when supplied, pass `position_bias` into every block, and update it from `layer_output[1]`:

```python
@dataclass
class EncoderLayerResult:
    hidden_states: torch.Tensor
    position_bias: torch.Tensor


def run_encoder_layers(...):
    extended_mask = encoder.get_extended_attention_mask(
        attention_mask, hidden_states.shape[:2]
    )
    if gate_bias is not None:
        extended_mask = extended_mask + gate_bias[:, None, None, :]
    for index in range(start_layer, end_layer):
        output = encoder.block[index](
            hidden_states,
            attention_mask=extended_mask,
            position_bias=position_bias,
        )
        hidden_states = output[0]
        position_bias = output[1]
    return EncoderLayerResult(hidden_states, position_bias)
```

Implement `compress_position_bias` with two `torch.gather` operations so each batch row retains exactly the query/key positions selected by hard deletion. Zero rows and columns masked as padding.

- [ ] **Step 4: Return source positions from hard deletion**

Replace the two-tuple return from `DeleteGate.apply_hard_deletion` with `HardDeletionResult`. Use the already computed `src_positions` as `source_positions`. Update all tests and call sites explicitly; do not add a compatibility tuple that allows callers to silently ignore positions.

- [ ] **Step 5: Replace both duplicated encoder loops**

Delete `_run_encoder_layers` from both compressed model classes. In training, carry the pre-gate bias into the post-gate runner. In evaluation/generation, compress the pre-gate bias with `source_positions` before the post-gate runner. The fixed and adaptive models must use the same shared functions.

- [ ] **Step 6: Run model and position-bias tests**

Run: `python -m pytest tests/test_encoder_position_bias.py tests/test_delete_gate.py tests/test_model_forward.py -v`

Expected: PASS; a regression that removes `position_bias=` from the shared runner must fail.

- [ ] **Step 7: Commit**

```bash
git add src/models/encoder_layers.py src/models/delete_gate.py src/models/fixed_compression_byt5.py src/models/noise_adaptive_byt5.py tests/test_encoder_position_bias.py tests/test_delete_gate.py tests/test_model_forward.py
git commit -m "fix(models): preserve encoder position bias across deletion"
```

---

### Task 2: Replace 1,024-position fixed padding with dynamic padding

**Files:**
- Modify: `src/data/dataset.py`
- Modify: `src/training/trainer.py`
- Modify: `src/evaluation/efficiency.py`
- Modify: `scripts/evaluate.py`
- Modify: `scripts/run_experiment.py`
- Test: `tests/test_dataset.py`
- Test: `tests/test_efficiency.py`

**Interfaces:**
- Produces: `NormalizationCollator(pad_token_id: int, pad_to_multiple_of: Optional[int] = None)` callable.
- Produces: dynamically padded `input_ids`, `attention_mask`, and `labels`; labels use `-100` padding.
- Consumes: variable-length tensors returned by `NormalizationDataset.__getitem__`.

- [ ] **Step 1: Write failing dataset and collator tests**

Test that two samples of lengths 12 and 31 are stored without 1,024 padding, then collated to length 31. Test that label padding is `-100`, input padding is the configured token ID, and the attention mask is zero only on padding.

```python
def test_dataset_uses_1024_as_ceiling_not_padding_length(tokenizer):
    dataset = NormalizationDataset(["maikli"], ["maikli"], tokenizer)
    item = dataset[0]
    assert item["input_ids"].numel() < 1024


def test_collator_pads_only_to_batch_max(dataset, tokenizer):
    collate = NormalizationCollator(tokenizer.pad_token_id)
    batch = collate([dataset[0], dataset[1]])
    expected = max(dataset[0]["input_ids"].numel(), dataset[1]["input_ids"].numel())
    assert batch["input_ids"].shape[1] == expected
    assert torch.all(batch["labels"][batch["labels"] == -100] == -100)
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest tests/test_dataset.py -v`

Expected: FAIL because each item is currently padded with `padding="max_length"` and the collator only stacks equal shapes.

- [ ] **Step 3: Implement variable-length tokenization and the collator**

Change both tokenizer calls to `padding=False`, preserve `max_length=1024`, `truncation=True`, and use `.squeeze(0)` rather than `.squeeze()`. Implement `NormalizationCollator` with `torch.nn.utils.rnn.pad_sequence`; pad inputs with `pad_token_id`, masks with `0`, and labels with `-100`. If `pad_to_multiple_of` is supplied, right-pad the collated tensors to the next multiple.

- [ ] **Step 4: Update every DataLoader call site**

Instantiate the collator from the tokenizer associated with the run:

```python
collator = NormalizationCollator(tokenizer.pad_token_id)
DataLoader(dataset, ..., collate_fn=collator)
```

Use `pad_to_multiple_of=8` for mixed-precision training only. Use no multiple for the manuscript's `batch_size=1` latency benchmark so the tensor length equals the actual tokenized length.

- [ ] **Step 5: Add an efficiency regression**

Mock model generation and assert the benchmark receives a short sequence for a short sentence. Also assert `batch_size == 1` in the primary benchmark loader.

- [ ] **Step 6: Run data and efficiency tests**

Run: `python -m pytest tests/test_dataset.py tests/test_efficiency.py tests/test_model_forward.py -v`

Expected: PASS, with no sample tensor fixed at 1,024 unless its content reaches the ceiling.

- [ ] **Step 7: Commit**

```bash
git add src/data/dataset.py src/training/trainer.py src/evaluation/efficiency.py scripts/evaluate.py scripts/run_experiment.py tests/test_dataset.py tests/test_efficiency.py
git commit -m "fix(data): use dynamic padding under the 1024-byte ceiling"
```

---

### Task 3: Make the gate regularizer padding-aware

**Files:**
- Modify: `src/models/fixed_compression_byt5.py`
- Modify: `src/models/noise_adaptive_byt5.py`
- Modify: `src/training/losses.py`
- Test: `tests/test_delete_gate.py`

**Interfaces:**
- Produces: model output field `gate_attention_mask: Tensor[batch, gate_sequence]`.
- Consumes: `keep_prob` and `gate_attention_mask` in `TAHIMIKLoss.forward`.

- [ ] **Step 1: Write a failing padding-invariance test**

Construct the same two real keep probabilities once without padding and once with nine zero-padding positions. Assert that `l_attn_reg` is identical.

```python
def test_attention_regularizer_ignores_padding():
    real = torch.tensor([[0.2, 0.8]])
    padded = torch.tensor([[0.2, 0.8, 0.0, 0.0, 0.0]])
    short = compression_outputs(real, torch.tensor([[1, 1]]))
    long = compression_outputs(padded, torch.tensor([[1, 1, 0, 0, 0]]))
    assert loss_fn(short)["l_attn_reg"] == pytest.approx(
        loss_fn(long)["l_attn_reg"]
    )
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_delete_gate.py::test_attention_regularizer_ignores_padding -v`

Expected: FAIL because `.mean()` currently includes padded positions.

- [ ] **Step 3: Implement masked reduction**

Return the original gate mask from both compression models and compute:

```python
mask = model_outputs["gate_attention_mask"].to(keep_prob.dtype)
penalty = 4.0 * keep_prob * (1.0 - keep_prob)
l_attn_reg = (penalty * mask).sum() / mask.sum().clamp_min(1.0)
```

Do not use the post-deletion mask: `keep_prob` belongs to the pre-deletion gate sequence.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_delete_gate.py tests/test_model_forward.py -v`

```bash
git add src/models/fixed_compression_byt5.py src/models/noise_adaptive_byt5.py src/training/losses.py tests/test_delete_gate.py tests/test_model_forward.py
git commit -m "fix(loss): exclude padding from gate regularization"
```

---

### Task 4: Constrain and expose the adaptive coefficient

**Files:**
- Modify: `src/models/delete_gate.py`
- Modify: `src/models/noise_adaptive_byt5.py`
- Modify: `src/training/trainer.py`
- Test: `tests/test_delete_gate.py`
- Create: `tests/test_training_diagnostics.py`

**Interfaces:**
- Produces: `DeleteGate.adaptive_coefficient -> Tensor`, always non-negative.
- Produces: TAHIMIK output fields `adaptive_coefficient` and `noise_average`.
- Produces: epoch diagnostics for coefficient, EMA, mean noise, mean deletion rate, and deletion rate by fixed noise bands `[0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1]`.

- [ ] **Step 1: Write failing coefficient tests**

Test that optimization can drive the underlying parameter negative while the exposed coefficient remains non-negative, and that a higher noise score never creates a lower shift than a lower score when all other inputs are equal.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_delete_gate.py -k "coefficient or noisier" -v`

Expected: FAIL because `cn` is an unconstrained parameter.

- [ ] **Step 3: Parameterize the coefficient with softplus**

Replace `self.cn` with `self.raw_cn`, initialized so `softplus(raw_cn) == 1.0`, and expose:

```python
@property
def adaptive_coefficient(self) -> torch.Tensor:
    return torch.nn.functional.softplus(self.raw_cn)
```

Use `self.adaptive_coefficient` in the shift. Update the EMA with in-place operations inside `torch.no_grad()`:

```python
with torch.no_grad():
    self.noise_avg.mul_(self.noise_avg_momentum).add_(
        batch_avg, alpha=1.0 - self.noise_avg_momentum
    )
```

- [ ] **Step 4: Emit and aggregate diagnostics**

Add scalar outputs to TAHIMIK's forward result. In the trainer, aggregate sample-weighted noise and deletion values, plus fixed-band counts/sums. Log `null`/`N/A` for empty bands; never fabricate zero observations.

- [ ] **Step 5: Add checkpoint migration behavior**

When loading an older checkpoint containing `delete_gate.cn`, convert it with inverse softplus and store it as `delete_gate.raw_cn`. Reject negative legacy values with an explicit message because they represent inverted adaptivity.

- [ ] **Step 6: Run tests and commit**

Run: `python -m pytest tests/test_delete_gate.py tests/test_training_diagnostics.py tests/test_model_forward.py -v`

```bash
git add src/models/delete_gate.py src/models/noise_adaptive_byt5.py src/training/trainer.py tests/test_delete_gate.py tests/test_training_diagnostics.py tests/test_model_forward.py
git commit -m "fix(adaptive): constrain and monitor the noise gate coefficient"
```

---

### Task 5: Replace hardcoded synthetic probabilities with a fail-closed manifest

**Files:**
- Create: `src/data/noise_policy.py`
- Create: `scripts/build_noise_manifest.py`
- Modify: `src/data/noise_generator.py`
- Modify: `src/data/preprocessing.py`
- Modify: `scripts/train.py`
- Modify: `scripts/run_experiment.py`
- Create: `tests/test_noise_policy.py`
- Modify: `tests/test_methodology_compliance.py`

**Interfaces:**
- Produces: `build_probability_manifest(training_labels, bounds, seed, resource_versions) -> ProbabilityManifest`.
- Produces: `ProbabilityManifest.require_ready() -> None`, raising `ValueError` if bounds/resources are unapproved.
- Changes: `TagalogNoiseGenerator(seed, manifest)`; remove silent hardcoded category probabilities.
- Produces: lineage containing manifest ID, applied categories, and resource versions for every synthetic pair.

- [ ] **Step 1: Write failing manifest tests**

Use fixtures with deliberately different train/validation/test frequencies. Assert that only training labels affect resolved probabilities, each probability is clamped to its approved category bounds, missing bounds fail readiness, and equivalent canonical content yields the same SHA-256 manifest ID.

- [ ] **Step 2: Write failing generation tests**

Assert that identical seeds/manifests/resources reproduce identical pairs; preserved augmentation changes source and target identically; correctable corruption changes only the source; generated training pairs never have `noisy == clean`; and the abbreviation alternatives do not contain their source form.

- [ ] **Step 3: Verify failure**

Run: `python -m pytest tests/test_noise_policy.py tests/test_methodology_compliance.py -v`

Expected: FAIL because current defaults are embedded in `TagalogNoiseGenerator` and slang/emoji are forced to zero.

- [ ] **Step 4: Implement the manifest model and builder**

Use dataclasses matching `specs/011-synthetic-noise-policy/data-model.md`. Canonicalize JSON with sorted keys and compact separators before hashing. The CLI must require explicit `--training-labels`, `--bounds`, and `--output`; it must not read validation/test labels or access the network.

- [ ] **Step 5: Refactor generation around the manifest**

Separate the pipeline:

```text
clean base
  -> preserved augmentation applied to source and target
  -> copy augmented target to source
  -> correctable corruption applied to source only
  -> pair plus lineage
```

Remove identity alternatives such as `"lang" -> "lang"` and `"totoo" -> "totoo"`. Validate every reviewed lexicon row and reject unapproved rows.

- [ ] **Step 6: Make thesis runs fail closed**

Require `--noise-manifest` for Stage 1 in `train.py` and `run_experiment.py`. Permit a clearly marked development manifest only when the resulting provenance is ineligible for reporting. Do not insert guessed production bounds into the repository.

- [ ] **Step 7: Export distribution diagnostics**

For each generated corpus, save observed `n*` mean, median, quantiles, zero-noise fraction, per-category counts, and manifest ID. Add an eligibility failure when any copy pair remains or the requested synthetic pair count cannot be generated.

- [ ] **Step 8: Run tests and commit**

Run: `python -m pytest tests/test_noise_policy.py tests/test_methodology_compliance.py -v`

```bash
git add src/data/noise_policy.py scripts/build_noise_manifest.py src/data/noise_generator.py src/data/preprocessing.py scripts/train.py scripts/run_experiment.py tests/test_noise_policy.py tests/test_methodology_compliance.py
git commit -m "fix(data): derive synthetic noise from an auditable manifest"
```

---

### Task 6: Restore the best Stage 1 checkpoint before Stage 2

**Files:**
- Modify: `src/training/trainer.py`
- Modify: `src/utils/reproducibility.py`
- Create: `tests/test_checkpoint_handoff.py`

**Interfaces:**
- Produces: checkpoint payload fields defined by `specs/012-stage-checkpoint-handoff/data-model.md`.
- Produces: `restore_best_stage1_for_handoff() -> HandoffRecord`.
- Consumes: `best_stage1.pt` saved by the same trainer/model configuration.

- [ ] **Step 1: Write a failing non-final-best-epoch test**

Simulate three Stage 1 validations where epoch 2 is best and epoch 3 is worse. Assert the weights visible at the first Stage 2 batch equal epoch 2, not epoch 3.

- [ ] **Step 2: Write failing error-path tests**

When Stage 1 was requested, assert that a missing checkpoint, malformed payload, stage mismatch, or incompatible configuration raises before the Stage 2 optimizer is created.

- [ ] **Step 3: Verify failure**

Run: `python -m pytest tests/test_checkpoint_handoff.py -v`

Expected: FAIL because Stage 2 currently continues from the final in-memory epoch.

- [ ] **Step 4: Save complete checkpoint state**

Save model, optimizer, scheduler, scaler, resolved config, provenance, checkpoint ID, and optional parent ID. Keep Stage 1 and Stage 2 best-loss trackers independent.

- [ ] **Step 5: Restore before constructing Stage 2 optimizer**

Call `restore_best_stage1_for_handoff()` after Stage 1 ends and before `train_stage(stage_name="stage2", ...)` creates its optimizer. Hash the restored model state and save the handoff record into every Stage 2 checkpoint.

- [ ] **Step 6: Run tests and commit**

Run: `python -m pytest tests/test_checkpoint_handoff.py tests/test_model_forward.py -v`

```bash
git add src/training/trainer.py src/utils/reproducibility.py tests/test_checkpoint_handoff.py
git commit -m "fix(training): hand off the best stage one checkpoint"
```

---

### Task 7: Align alpha-word accuracy after insertions and deletions

**Files:**
- Modify: `src/evaluation/metrics.py`
- Modify: `tests/test_methodology_compliance.py`

**Interfaces:**
- Produces: `_alphabetic_words(text: str) -> List[str]`.
- Produces: `_alpha(pred: str, ref: str) -> float` based on word-sequence edit alignment, not positional zip.

- [ ] **Step 1: Add failing shifted-word tests**

Include perfect output, one insertion at the beginning, one deletion in the middle, punctuation, Filipino Unicode letters, empty reference, and case-insensitivity. Assert that a single insertion/deletion does not mark every later matching word wrong.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_methodology_compliance.py -k alpha -v`

Expected: FAIL because current code compares `pred_words[i]` with `ref_words[i]`.

- [ ] **Step 3: Implement one shared Unicode-aware alignment**

Extract alphabetic words with the same Unicode predicate for predictions and references. Compute Levenshtein distance over lower-cased word sequences, then return `max(0.0, 1.0 - distance / len(reference_words))`. Preserve the existing empty-reference convention explicitly in tests.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_methodology_compliance.py -v`

```bash
git add src/evaluation/metrics.py tests/test_methodology_compliance.py
git commit -m "fix(metrics): align alphabetic words before scoring"
```

---

### Task 8: Support label-free forward passes safely

**Files:**
- Modify: `src/models/fixed_compression_byt5.py`
- Modify: `src/models/noise_adaptive_byt5.py`
- Modify: `tests/test_model_forward.py`

**Interfaces:**
- Changes: `forward(..., labels=None, decoder_input_ids=None)` requires one of `labels` or `decoder_input_ids` when logits are requested.
- Preserves: `generate(...)` as the supported free-generation interface.

- [ ] **Step 1: Write failing behavior tests**

Test an explicit `decoder_input_ids` label-free call and a call with neither decoder input nor labels. The first must return logits; the second must raise a clear `ValueError` directing callers to `generate()`.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_model_forward.py -k label_free -v`

Expected: FAIL because the current decoder is called without `input_ids`.

- [ ] **Step 3: Implement the contract identically in both models**

Read `decoder_input_ids` from the explicit parameter, not arbitrary `kwargs`. Shift labels when labels exist; otherwise validate decoder input. Keep output keys consistent between branches.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_model_forward.py -v`

```bash
git add src/models/fixed_compression_byt5.py src/models/noise_adaptive_byt5.py tests/test_model_forward.py
git commit -m "fix(models): define label-free forward behavior"
```

---

### Task 9: Make the runtime batch endpoint genuinely batched and non-blocking

**Files:**
- Modify: `backend/app.py`
- Create: `tests/test_backend.py`

**Interfaces:**
- Produces: `normalize_texts(texts: List[str], model_name: str, max_length: int, num_beams: int) -> Tuple[List[str], float]`.
- Changes: `/normalize/batch` performs one tokenizer call and one `model.generate` call.
- Clarifies: endpoint latency is presentation telemetry, not the controlled Chapter 3 benchmark.

- [ ] **Step 1: Write failing API tests**

Mock tokenizer/model calls. Assert two submitted texts cause exactly one `generate` call, preserve response order, and reject an empty list. Assert the event loop is not blocked by running inference through a worker thread or by using synchronous route handlers.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_backend.py -v`

Expected: FAIL because the current batch endpoint loops over `normalize_text` and async handlers call synchronous model work directly.

- [ ] **Step 3: Implement batched inference**

Tokenize with `padding="longest"`, `truncation=True`, and one tensor batch. Time the single generation call with CUDA synchronization where applicable, decode with `batch_decode`, and return total batch latency plus clearly labeled amortized per-item latency.

- [ ] **Step 4: Move blocking model work off the event loop**

Either change inference routes to ordinary `def` handlers (FastAPI executes them in its threadpool) or call `await run_in_threadpool(normalize_texts, ...)`. Use one approach consistently and retain `async` only for lifespan handling.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_backend.py -v`

```bash
git add backend/app.py tests/test_backend.py
git commit -m "fix(api): batch normalization and isolate blocking inference"
```

---

### Task 10: Remove the experiment tokenizer hardcode

**Files:**
- Modify: `scripts/run_experiment.py`
- Create: `tests/test_experiment_configuration.py`

**Interfaces:**
- Consumes: one authoritative `BaseConfig.model_name` for the shared tokenizer and all model variants.
- Produces: provenance containing the resolved tokenizer identifier.

If the Base migration plan's Task 1 already removed this hardcode, keep this task as a regression-verification checkpoint and do not edit the same lines again.

- [ ] **Step 1: Write a failing configuration-spy test**

Monkeypatch `AutoTokenizer.from_pretrained`, set `base_config.model_name = "example/model"`, and assert the tokenizer receives exactly `"example/model"`.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_experiment_configuration.py -v`

Expected: FAIL because `run_experiment.py` currently contains `google/byt5-small`.

- [ ] **Step 3: Load from the shared config**

Construct `base_config` before the tokenizer and call `AutoTokenizer.from_pretrained(base_config.model_name)`. Assert each per-variant config resolves to the same model name before training begins; fail with a message listing divergent variants.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_experiment_configuration.py -v`

```bash
git add scripts/run_experiment.py tests/test_experiment_configuration.py
git commit -m "fix(experiment): load tokenizer from shared model configuration"
```

---

### Task 11: Verify the complete manuscript-aligned pipeline

**Files:**
- Modify: `README.md`
- Modify: `docs/CHAPTER3-COMPLIANCE.md`
- Modify: `specs/FINDINGS.md`
- Modify: `specs/DECISIONS.md`
- Modify: `specs/PROVENANCE.md`

**Interfaces:**
- Produces: an auditable compliance matrix with status `implemented`, `externally blocked`, or `not applicable` for every finding.

- [ ] **Step 1: Run the full automated suite**

Run: `python -m pytest tests -v`

Expected: PASS. If collection hangs, run `python -X importtime -m pytest --collect-only -vv`, identify the import causing the stall, and fix that import before continuing.

- [ ] **Step 2: Run static repository checks**

Run: `rg -n 'padding="max_length"|from_pretrained\("google/byt5-small"\)|async def normalize|self\.cn\b|\.mean\(\).*attn' src scripts backend configs`

Expected: no forbidden production occurrences. Documentation may mention old code only inside an explicitly dated historical finding.

- [ ] **Step 3: Run a tiny offline three-variant smoke test**

Use mocked/tiny T5 fixtures, two variable-length sentences, Stage 1 with a non-final best epoch, Stage 2 handoff, generation for all three variants, and the statistical report. Assert identical sample order and decoding settings across variants.

- [ ] **Step 4: Update documentation from evidence**

Mark only verified items as fixed. Mark production synthetic-noise calibration as externally blocked until approved category bounds and reviewed lexicons exist. State that API timings are demonstrations and Chapter 3 numbers come only from the benchmark harness.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/CHAPTER3-COMPLIANCE.md specs/FINDINGS.md specs/DECISIONS.md specs/PROVENANCE.md
git commit -m "docs: record manuscript alignment verification"
```

## Completion gate

Do not call the system manuscript-aligned until Tasks 1–11 pass. Even after code completion, the experiment remains ineligible for thesis reporting until approved synthetic probability bounds, reviewed lexicon resources, the required gold/synthetic dataset sizes, trained checkpoints, and GPU benchmark results exist.
