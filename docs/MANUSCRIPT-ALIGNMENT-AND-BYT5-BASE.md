# TAHIMIK Implementation Guide: Manuscript Alignment and ByT5-Base Migration

**Status:** Code and automated-test verification completed on 2026-09-08.  
**Audience:** Researchers, developers, panel members, and future maintainers.  
**Scope:** Explains what changed, why it changed, the responsible code paths, the mathematical logic, how to run the system, and what must still be supplied before reporting thesis results.

---

## 1. Executive summary

TAHIMIK is a controlled three-condition experiment for Filipino/Taglish text normalization:

| Condition | Role in experiment | Compression behavior |
|---|---|---|
| ByT5-Base | Accuracy/reference baseline | No deletion; encoder processes all retained bytes |
| Fixed-compression ByT5 (MrT5-style) | Fixed-rate efficiency baseline | One shared deletion target regardless of sentence noise |
| TAHIMIK | Proposed method | Deletion target adapts to predicted sentence noise |

The key methodological rule is that the three conditions must differ only in their compression mechanism. They now share the same pretrained `google/byt5-base` backbone family, tokenizer identity, byte ceilings, data splits, optimizer family, decoding configuration, and evaluation protocol.

Two groups of work were completed:

1. **Manuscript-alignment remediation.** This repaired issues that could confound accuracy, efficiency, adaptive-gate behavior, synthetic Stage 1 training, checkpoint handoff, metrics, or demo inference.
2. **ByT5-Base migration.** This moved the entire controlled comparison from ByT5-Small to ByT5-Base, while protecting the experiment from accidental Small checkpoint loading and making Base training feasible through memory controls.

The result is code that can start a valid controlled training run once the external research inputs are ready. It is not permission to report thesis results yet: approved category bounds, reviewed local resources, finalized gold data, trained checkpoints, and controlled GPU benchmark results remain required.

---

## 2. Why the ByT5-Base migration was necessary

### 2.1 The original problem

The original implementation used a Small backbone in some places, while the intended experimental design needed one authoritative backbone configuration across every condition. A mixed or partially migrated setup is dangerous because the comparison could become:

```text
compression mechanism
plus a changed tokenizer/model size/checkpoint family
```

instead of measuring compression alone.

MrT5 does not offer a directly comparable released Base checkpoint. The study therefore does **not** load a pretrained Stanford MrT5 checkpoint for only the fixed-compression comparator. All three variants initialize their shared pretrained backbone independently from Google's ByT5-Base weights; TAHIMIK-only modules (noise estimator and adaptive gate) and fixed-gate-specific parameters are trained within this study.

### 2.2 Single source of truth

[`configs/base.py`](../configs/base.py) is now the authoritative identity:

```python
@dataclass
class BaseConfig:
    model_name: str = "google/byt5-base"
    max_input_length: int = 1024
    max_target_length: int = 1024
```

All model wrappers use `config.model_name`, rather than embedding a model string:

```python
self.model = T5ForConditionalGeneration.from_pretrained(config.model_name)
self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
```

This applies to:

- [`src/models/byt5_baseline.py`](../src/models/byt5_baseline.py)
- [`src/models/fixed_compression_byt5.py`](../src/models/fixed_compression_byt5.py)
- [`src/models/noise_adaptive_byt5.py`](../src/models/noise_adaptive_byt5.py)
- [`scripts/train.py`](../scripts/train.py)
- [`scripts/run_experiment.py`](../scripts/run_experiment.py)
- [`scripts/evaluate.py`](../scripts/evaluate.py)
- [`scripts/benchmark.py`](../scripts/benchmark.py)
- [`backend/app.py`](../backend/app.py)

This means changing `BaseConfig.model_name` intentionally changes all conditions together. A user should not set one model variant to Small while leaving the others at Base.

### 2.3 Dynamic dimensions rather than Small-specific numbers

ByT5-Base does not have the same hidden dimension as ByT5-Small. Therefore custom modules derive their shape from the loaded backbone configuration:

```python
hidden_dim = self.model.config.d_model
self.delete_gate = DeleteGate(hidden_dim=hidden_dim, ...)
self.noise_estimator = NoiseEstimator(hidden_dim=hidden_dim, ...)
```

Hard-coding a Small-specific hidden size would cause either a shape error or a subtle architecture mismatch. The dynamic approach keeps the custom TAHIMIK components compatible with the configured backbone.

### 2.4 Memory plan: physical batches and gradient accumulation

Base is larger, so the code separates **physical** and **effective** batch size:

| Training stage | Physical batch | Accumulation steps | Effective batch |
|---|---:|---:|---:|
| Stage 1 synthetic pretraining | 2 | 8 | 16 |
| Stage 2 gold fine-tuning | 2 | 4 | 8 |

The relevant configuration is in [`configs/base.py`](../configs/base.py). The trainer divides each microbatch loss before backward propagation:

```python
scaled_loss = losses["total_loss"] / accum_steps
self.scaler.scale(scaled_loss).backward()
```

It performs one optimizer/scheduler step at each accumulation boundary or at the final microbatch:

```python
if (step % accum_steps == 0) or (step == num_items):
    self.scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
    self.scaler.step(optimizer)
    self.scaler.update()
    scheduler.step()
    optimizer.zero_grad()
```

This preserves the manuscript batch semantics while reducing instantaneous memory demand. The trainer also supports fp16/bf16 validation and gradient checkpointing. Run [`scripts/preflight_base.py`](../scripts/preflight_base.py) before a real Base run to inspect the resolved settings and device feasibility.

### 2.5 Checkpoint architecture protection

Every saved best checkpoint contains architecture metadata, including model name, `d_model`, encoder/decoder layer counts, and vocabulary size. [`validate_checkpoint_architecture()`](../src/training/trainer.py) compares those fields with the target model before loading weights.

The purpose is practical: a Small checkpoint must never be loaded into a Base model merely because the filename looks plausible. A legacy checkpoint missing architecture metadata requires an explicit development-only opt-in and is not thesis-reportable.

---

## 3. Manuscript alignment: encoder position bias and deletion

### 3.1 Why relative position bias mattered

ByT5/T5 encoder layers use learned relative position bias. The earlier compressed-model loops did not reliably carry that bias from one layer to the next, while the uncompressed baseline did. That would have made the experiment unfair: the compressed models could lose positional information for reasons unrelated to deletion.

The shared solution is [`src/models/encoder_layers.py`](../src/models/encoder_layers.py). Its `run_encoder_layers()` function:

1. Builds the extended encoder attention mask once.
2. Adds the soft deletion penalty when training.
3. Passes the current `position_bias` into every T5 block.
4. Reads the updated bias from `layer_output[1]`.

Conceptually:

```python
for index in range(start_layer, end_layer):
    output = encoder.block[index](
        hidden_states,
        attention_mask=extended_mask,
        position_bias=position_bias,
    )
    hidden_states = output[0]
    position_bias = output[1]
```

Both fixed and adaptive compression models call this same helper. That removes duplicated encoder loops and prevents one variant drifting from the other in a future modification.

### 3.2 Soft deletion in training

During training, deletion is differentiable. The gate produces a keep probability \(p_i\) for each input byte. The model converts it into an additive attention-logit bias:

\[
b_i = \log(p_i + \epsilon)
\]

and adds this bias to the attention mask. Low keep probability decreases attention to that position without physically removing its hidden state, so the gate can receive gradients.

### 3.3 Hard deletion in generation/evaluation

During generation and evaluation, actual compression is needed for the efficiency measurement. [`DeleteGate.apply_hard_deletion()`](../src/models/delete_gate.py) selects retained positions, gathers them into shorter sequences, and returns:

```python
HardDeletionResult(
    hidden_states=compressed_hidden_states,
    attention_mask=compressed_attention_mask,
    source_positions=retained_original_positions,
)
```

The source positions are essential. [`compress_position_bias()`](../src/models/encoder_layers.py) gathers the original position-bias query and key axes using precisely the retained indices. The compressed encoder therefore keeps internally consistent positional information after physical deletion.

---

## 4. TAHIMIK's adaptive delete-gate logic

### 4.1 Noise estimator and gradient isolation

The noise estimator predicts a sentence-level score \(n \in [0,1]\) from early encoder hidden states. Its supervision target \(n^*\) is normalized byte-level edit distance between noisy and clean text:

\[
n^* = \frac{\operatorname{Levenshtein}(\text{noisy bytes},\text{clean bytes})}
{\max(|\text{noisy bytes}|, |\text{clean bytes}|)}
\]

This implementation is in [`src/data/noise_label.py`](../src/data/noise_label.py). The predicted score is trained through the noise-estimation loss \(L_{NE}\). At gate/rate-target consumption points, it is detached so deletion losses do not train the estimator through an unintended shortcut.

### 4.2 Adaptive shift

The gate shift follows the manuscript principle:

\[
\text{shift} = c_n(n - \bar n)
\]

where \(\bar n\) is a training-time moving average. A noisy sentence above the average receives a positive retention shift, so it is compressed less aggressively. A cleaner sentence can be compressed more.

[`DeleteGate`](../src/models/delete_gate.py) now parameterizes \(c_n\) as:

\[
c_n = \operatorname{softplus}(\text{raw\_cn}) \ge 0
\]

```python
@property
def adaptive_coefficient(self) -> torch.Tensor:
    return F.softplus(self.raw_cn)
```

This matters because an unconstrained negative coefficient would reverse the thesis behavior: noisier sentences would be deleted more aggressively. Legacy negative coefficients are rejected rather than silently converted.

### 4.3 Adaptive deletion target

For TAHIMIK, the per-sentence rate target is:

\[
d_{target} = d_{max}(1-n)
\]

As estimated noise rises, the desired deletion rate falls. The fixed-compression comparator instead uses its same fixed rate for all sentences. This is the independent variable that addresses the thesis question.

### 4.4 Diagnostics and regularization

The model returns `adaptive_coefficient`, `noise_average`, `noise_scores`, `deletion_rate`, and `gate_attention_mask`. The trainer aggregates mean noise/deletion rate and deletion rates across fixed noise bands:

```text
[0.0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0]
```

The gate regularizer is padding-aware:

\[
L_{attn} =
\frac{\sum_i 4p_i(1-p_i)m_i}{\max(\sum_i m_i, 1)}
\]

where \(m_i\) is the original pre-deletion attention mask. Padding cannot dilute the regularizer.

---

## 5. Dynamic padding and honest efficiency measurement

### 5.1 The original risk

Using `padding="max_length"` at 1,024 bytes for every sentence would make short Filipino social-media inputs mostly padding. Compression could appear fast simply because it deleted padding, rather than because it efficiently handled real input. It would also waste computation during training.

### 5.2 Current data flow

[`NormalizationDataset`](../src/data/dataset.py) tokenizes with:

```python
padding=False
truncation=True
max_length=1024
```

Thus 1,024 is a byte-sequence ceiling, not a per-sample padded length. [`NormalizationCollator`](../src/data/dataset.py) pads each batch only to its longest input. It pads:

- `input_ids` with the tokenizer pad ID;
- `attention_mask` with `0`;
- labels with `-100`, so cross-entropy ignores label padding.

When alignment is requested for mixed precision, inputs and labels round independently to the requested multiple. This prevents an input of length 10 and target of length 3 from incorrectly producing a 9-long target merely because the input required six more positions.

### 5.3 Benchmark protocol

[`EfficiencyBenchmark`](../src/evaluation/efficiency.py) enforces the manuscript's primary latency design:

- batch size 1;
- five warmup passes;
- twenty timed passes;
- exact dynamic input length rather than multiple-of-eight benchmark padding;
- CUDA synchronization before and after generation;
- per-sentence durations and independent per-run peak-memory observations.

API latency is presentation telemetry, not the Chapter 3 controlled benchmark. The batch API does, however, synchronize CUDA around its single batched `model.generate()` call so it does not underreport GPU execution time.

---

## 6. Synthetic Stage 1 data: manifest, lineage, and fail-closed policy

### 6.1 Why a manifest is required

Hardcoded noise probabilities and silent defaults make Stage 1 impossible to audit. They can also create many copy pairs, where noisy input equals target, producing almost no useful normalization signal and making adaptive rates resemble fixed rates.

The solution is [`ProbabilityManifest`](../src/data/noise_policy.py). It records category counts/rates, lower and upper bounds, resolved probability, training-split fingerprint, seed, resource versions, creation state, and canonical manifest ID.

`require_ready()` verifies, among other things:

```text
- readiness_state is explicitly "ready"
- source split is "train"
- identity and training fingerprint are present
- approved resource versions are present
- category names and category groups are recognized
- 0 <= lower_bound <= upper_bound <= 1
- observed and resolved probabilities are within [0, 1]
- canonical manifest digest matches manifest_id
```

The loader is deliberately strict:

```python
manifest = load_probability_manifest(args.noise_manifest)
manifest.require_ready()
```

### 6.2 Stage 1 command contract

Both training entry points now require a manifest whenever synthetic generation is enabled:

```powershell
python scripts/train.py `
  --variant tahimik `
  --gold_data data/gold.csv `
  --clean_corpus data/clean.txt `
  --noise_manifest data/noise-manifest.json
```

The equivalent full-experiment invocation is:

```powershell
python scripts/run_experiment.py `
  --gold_data data/gold.csv `
  --clean_corpus data/clean.txt `
  --noise_manifest data/noise-manifest.json `
  --output_dir outputs/experiment
```

If `--clean_corpus` is present but `--noise_manifest` is absent, the scripts stop before Stage 1 generation.

### 6.3 Pair-generation logic

For every clean base sentence, the intended transition is:

```text
clean base
  -> preserved augmentation applied to target
  -> copy target into source
  -> correctable corruption applied only to source
  -> verify source != target
  -> pair + lineage
```

The implementation is [`TagalogNoiseGenerator.generate_pair()`](../src/data/noise_generator.py). A pair with unchanged source/target is rejected, including short inputs that cannot be safely corrupted. The pipeline does not fall back to the older `apply_noise()` path for reporting Stage 1 generation.

[`DataPipeline.generate_synthetic_pairs()`](../src/data/preprocessing.py) creates exactly the requested number of pairs, records lineage, and computes:

```json
{
  "manifest_id": "...",
  "requested_count": 1000000,
  "generated_count": 1000000,
  "mean_noise": 0.0,
  "median_noise": 0.0,
  "noise_quantiles": {"q25": 0.0, "q75": 0.0},
  "zero_noise_fraction": 0.0,
  "category_counts": {"abbreviation": 0}
}
```

The real values naturally depend on the manifest and corpus. Scripts write `synthetic_lineage.json` and `synthetic_diagnostics.json` in the selected output directory.

### 6.4 Current external block: code-switching and Taglish morphology

Slang and emoji have supported local generation behavior. Code-switching and Taglish-morphology categories are recognized by the policy but cannot be used until reviewed local resources and their category-specific generation functions are supplied. If a manifest asks for an unsupported category, generation fails immediately.

This is intentional. A false claim that those categories were generated would be worse than an explicit block. It means the system is **code-ready but not Chapter 3-report-ready** until those external resources and approved bounds are available.

---

## 7. Two-stage checkpoint handoff and reproducibility

### 7.1 Correct Stage 1 to Stage 2 handoff

Stage 2 must start from the **best validation-loss Stage 1 checkpoint**, not merely the final in-memory Stage 1 epoch. The trainer now restores `best_stage1.pt` before constructing the Stage 2 optimizer and scheduler.

The saved payload includes:

```text
schema_version
checkpoint_id
stage, epoch, validation loss
architecture metadata
model_state_dict
optimizer_state_dict
scheduler_state_dict
scaler_state_dict
resolved configuration
provenance
model fingerprint
parent_checkpoint_id
handoff record (for Stage 2)
```

The Stage 2 checkpoint therefore identifies the exact best Stage 1 parent used to initialize it. This makes the handoff inspectable without reading terminal logs.

### 7.2 Why optimizer state is saved but not carried over

Stage 2 begins from Stage 1's best **model weights** but creates fresh Stage 2 optimization state. This is deliberate because the stages have different training data and schedules. Saving optimizer/scheduler/scaler state remains important for auditability and potential exact resumption of a stage.

---

## 8. Metrics, statistical tests, and runtime API

### 8.1 Alpha-word accuracy

The previous positional comparison would mark every later word wrong after one insertion/deletion. [`NormalizationMetrics`](../src/evaluation/metrics.py) now extracts Unicode alphabetic words, lowercases them, calculates word-sequence Levenshtein distance, then computes:

\[
\alpha = \max\left(0, 1 - \frac{\operatorname{distance}(prediction, reference)}{|reference|}\right)
\]

This scores shifted but otherwise correct content fairly.

### 8.2 Batch demo endpoint

[`backend/app.py`](../backend/app.py) exposes `normalize_texts()`. It receives a list, performs one tokenizer call with dynamic longest padding, then makes one batched `model.generate()` call:

```python
inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=max_length,
    num_beams=num_beams,
)
```

Routes are synchronous FastAPI handlers, so blocking model work is handled through FastAPI's normal threadpool rather than blocking an async event loop. The API returns total batch latency and clearly amortized per-item timing.

---

## 9. Tests and evidence

The merged implementation added or strengthened regression tests for:

- Base model configuration and Base preflight;
- Small-vs-Base checkpoint compatibility;
- dynamic dimensions in custom models;
- gradient accumulation;
- dynamic padding and independent label alignment;
- position-bias propagation and post-deletion bias gathering;
- adaptive coefficient monotonicity/legacy migration/diagnostics;
- manifest validation, identity alternatives, copy-pair prevention, and lineage;
- best Stage 1 checkpoint restoration;
- label-free forward behavior;
- alpha-word sequence alignment;
- batch API behavior;
- primary efficiency benchmark constraints.

The final repository test command was:

```powershell
python -m pytest tests -q -p no:cacheprovider
```

Result: **84 passed**. The remaining warning comes from a test intentionally monkeypatching an optimizer step to count gradient-accumulation updates; it is not a model-training failure.

---

## 10. What “ready to train” means

### Code-ready

The implementation is ready to launch training in the following sense:

- all three conditions resolve to ByT5-Base;
- compression preserves position-bias semantics;
- Base memory controls preserve intended effective batches;
- synthetic data cannot silently use unapproved defaults;
- Stage 2 restores the correct Stage 1 model;
- benchmark and API timing logic are honest about GPU synchronization;
- automated regression coverage passes.

### Not yet thesis-result-ready

Do **not** present accuracy, efficiency, or significance claims until all of the following exist:

1. Finalized and quality-checked gold noisy/clean data.
2. Approved per-category probability bounds based only on the gold training split.
3. Reviewed, versioned local lexicon/template resources.
4. Implemented reviewed resources for code-switching and Taglish morphology if those categories remain part of the manuscript's synthetic policy.
5. A valid Stage 1 manifest derived from the training labels.
6. Three trained checkpoints produced under the same controls.
7. GPU benchmark artifacts using five warmups, twenty timed runs, and batch size one.
8. A clean/recorded repository state and saved provenance artifacts.

The important distinction is simple: the system now prevents an invalid experiment from quietly looking valid. It is ready to train when the research inputs are available, and ready to report only after the experiment itself has been executed under the manuscript protocol.

---

## 11. Related documents

- [Chapter 3 compliance notes](CHAPTER3-COMPLIANCE.md)
- [Manuscript alignment remediation plan](superpowers/plans/2026-09-08-manuscript-alignment-remediation.md)
- [ByT5-Base migration plan](superpowers/plans/2026-09-08-byt5-base-migration.md)
- [Synthetic noise governance design](superpowers/specs/2026-09-08-synthetic-noise-governance-design.md)
- [Checkpoint lineage design](superpowers/specs/2026-09-08-checkpoint-lineage-design.md)
- [Base migration specification](../specs/015-byt5-base-migration/spec.md)
