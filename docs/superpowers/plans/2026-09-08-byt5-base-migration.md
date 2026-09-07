# ByT5 Base Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the controlled ByT5, fixed-compression, and TAHIMIK experiment from `google/byt5-small` to the manuscript-required `google/byt5-base` backbone without introducing tokenizer, checkpoint, memory, or comparison confounds.

**Architecture:** Make `BaseConfig.model_name` the single model/tokenizer authority, keep all three variants initialized independently from the same pretrained ByT5-base checkpoint, and randomly initialize only the study-specific gate/estimator modules. Add effective-batch gradient accumulation and checkpoint-architecture validation so Base fits available hardware and Small checkpoints cannot be loaded accidentally.

**Tech Stack:** Python 3.11, PyTorch, Hugging Face Transformers, ByT5, pytest, CUDA mixed precision.

**Spec:** `specs/003-byt5-baseline/spec.md`; `specs/004-fixed-rate-compression/spec.md`; `specs/005-noise-adaptive-byt5/spec.md`; `specs/PROVENANCE.md`

## Global Constraints

- Use `google/byt5-base` for all three variants and the shared tokenizer.
- Initialize each backbone from Google's same pretrained checkpoint; do not randomly initialize the full Base model.
- Do not load `stanfordnlp/mrt5-small` or transplant its 1,472-wide gate into ByT5-base's 1,536-wide hidden states.
- Describe the fixed comparator as “ByT5-base with an MrT5-style fixed-rate delete gate,” not as an official pretrained “MrT5-base.”
- Keep `delete_gate_layer=3` for both compressed variants unless the manuscript is formally amended and a gate-position ablation is added.
- Preserve the manuscript's effective Stage 1 batch size of 16 and Stage 2 batch size of 8 through gradient accumulation if physical batches are reduced.
- The alignment plan's position-bias and dynamic-padding tasks are prerequisites for any reportable Base experiment.
- Tests must not download the 582M-parameter checkpoint; network-backed smoke validation is an explicit manual/preflight step.
- Preserve unrelated working-tree changes.

## Gemini 3.8 execution contract

- Execute this plan after Tasks 1–10 of the manuscript-alignment plan, or verify those prerequisites before starting.
- Execute one numbered task at a time and stop after its test/commit checkpoint for review.
- Before editing, read every file and spec named by that task and inspect the existing diff for those paths.
- Preserve user-owned changes and avoid formatting or documentation rewrites outside the task.
- Never download ByT5-base as part of unit tests; use mocks/tiny configurations until the explicit GPU preflight task.
- Do not silently fall back to Small after CUDA OOM. Lower physical batch size, increase accumulation proportionally, and record the resolved values.
- After each task, report changed files, exact commands, pass/fail totals, peak memory when applicable, and whether the result is eligible for thesis reporting.

---

### Task 1: Establish one authoritative Base model identifier

**Files:**
- Modify: `configs/base.py`
- Modify: `scripts/run_experiment.py`
- Test: `tests/test_model_configuration.py`

**Interfaces:**
- Produces: `BaseConfig.model_name == "google/byt5-base"`.
- Consumes: `config.model_name` in every model and tokenizer loader.

If Task 10 of the manuscript-alignment plan already removed the tokenizer hardcode, retain that code and change only the shared model identifier plus the Base-specific assertions.

- [ ] **Step 1: Write failing authority tests**

```python
def test_all_variants_use_byt5_base():
    configs = [ByT5Config(), MrT5Config(), TAHIMIKConfig()]
    assert {config.model_name for config in configs} == {"google/byt5-base"}


def test_experiment_has_no_hardcoded_small_identifier():
    source = Path("scripts/run_experiment.py").read_text(encoding="utf-8")
    assert 'from_pretrained("google/byt5-small")' not in source
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_model_configuration.py -v`

Expected: FAIL because `BaseConfig` and `run_experiment.py` currently use Small.

- [ ] **Step 3: Switch the shared configuration and tokenizer**

Set:

```python
model_name: str = "google/byt5-base"
```

Construct the shared config before the tokenizer in `run_experiment.py` and load `AutoTokenizer.from_pretrained(base_config.model_name)`. Do not add per-variant overrides.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_model_configuration.py -v`

```bash
git add configs/base.py scripts/run_experiment.py tests/test_model_configuration.py
git commit -m "feat(config): make byt5 base the shared backbone"
```

---

### Task 2: Prove model-specific modules derive their dimensions dynamically

**Files:**
- Modify: `tests/test_model_forward.py`
- Modify only if tests expose a defect: `src/models/fixed_compression_byt5.py`
- Modify only if tests expose a defect: `src/models/noise_adaptive_byt5.py`
- Modify only if tests expose a defect: `src/models/noise_estimator.py`
- Modify only if tests expose a defect: `src/models/delete_gate.py`

**Interfaces:**
- Consumes: `self.model.config.d_model` as the sole input width for gates and estimator.
- Produces: fixed gate `in_features == d_model` and TAHIMIK estimator first-layer `in_features == d_model`.

- [ ] **Step 1: Add architecture-agnostic tests**

Parameterize the tiny model fixture over two distinct hidden dimensions and encoder depths, including 18 encoder layers with a small test-only width. Assert both compressed models construct and complete forward/generate without hardcoded Small dimensions.

```python
@pytest.mark.parametrize("d_model,num_layers", [(32, 4), (48, 18)])
def test_compression_modules_follow_backbone_shape(...):
    model = build_with_tiny_backbone(d_model=d_model, num_layers=num_layers)
    assert model.delete_gate.gate_linear.in_features == d_model
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_model_forward.py -k backbone_shape -v`

Expected: PASS with the current dynamic construction. If it fails, make only the minimal replacement of a hardcoded width/depth with `model.config.d_model` or `len(model.encoder.block)`.

- [ ] **Step 3: Commit the regression tests**

```bash
git add tests/test_model_forward.py src/models/fixed_compression_byt5.py src/models/noise_adaptive_byt5.py src/models/noise_estimator.py src/models/delete_gate.py
git commit -m "test(models): verify base-compatible dynamic dimensions"
```

---

### Task 3: Add gradient accumulation while preserving effective batch sizes

**Files:**
- Modify: `configs/base.py`
- Modify: `src/training/trainer.py`
- Create: `tests/test_gradient_accumulation.py`

**Interfaces:**
- Produces configuration fields `stage1_gradient_accumulation_steps: int = 8` and `stage2_gradient_accumulation_steps: int = 4`.
- Changes physical batch sizes to Stage 1 `2` and Stage 2 `2`, preserving effective sizes `16` and `8`.
- Changes `train_stage(..., gradient_accumulation_steps: int)`.

- [ ] **Step 1: Write failing accumulation tests**

With five microbatches and accumulation factor two, assert three optimizer/scheduler steps occur, the final partial accumulation is not discarded, and the loss is divided by the accumulation factor before backpropagation.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_gradient_accumulation.py -v`

Expected: FAIL because the trainer currently steps the optimizer after every physical batch.

- [ ] **Step 3: Add explicit physical/effective batch configuration**

Set:

```python
stage1_batch_size: int = 2
stage1_gradient_accumulation_steps: int = 8
stage2_batch_size: int = 2
stage2_gradient_accumulation_steps: int = 4
```

Validate positive integers and log both physical and effective batch sizes. Compute scheduler training steps with ceiling division:

```python
updates_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
num_training_steps = updates_per_epoch * epochs
```

- [ ] **Step 4: Implement accumulation safely**

Divide total loss before scaled backward. Step/unscale/clip/update/schedule only at accumulation boundaries or the final microbatch. Zero gradients before the loop and after each optimizer step.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_gradient_accumulation.py tests/test_model_forward.py -v`

```bash
git add configs/base.py src/training/trainer.py tests/test_gradient_accumulation.py
git commit -m "feat(training): preserve effective batches with accumulation"
```

---

### Task 4: Add Base memory controls without changing the experiment

**Files:**
- Modify: `configs/base.py`
- Modify: `src/training/trainer.py`
- Modify: `scripts/train.py`
- Modify: `scripts/run_experiment.py`
- Create: `tests/test_memory_configuration.py`

**Interfaces:**
- Produces: `gradient_checkpointing: bool = True` and `precision: Literal["fp32", "fp16", "bf16"]`.
- Preserves identical memory settings across all variants.

- [ ] **Step 1: Write failing configuration tests**

Assert that every variant inherits identical precision/checkpointing values and that unsupported precision/device combinations fail before model training.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_memory_configuration.py -v`

- [ ] **Step 3: Implement shared memory settings**

Enable model gradient checkpointing before training when configured. Prefer BF16 on supported A100-class hardware; otherwise use FP16 on CUDA and FP32 on CPU. Record the resolved precision in provenance. Do not enable checkpointing only for a compressed variant.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_memory_configuration.py tests/test_model_configuration.py -v`

```bash
git add configs/base.py src/training/trainer.py scripts/train.py scripts/run_experiment.py tests/test_memory_configuration.py
git commit -m "feat(training): add shared base-model memory controls"
```

---

### Task 5: Reject Small or incompatible checkpoints explicitly

**Files:**
- Modify: `src/training/trainer.py`
- Modify: `src/utils/reproducibility.py`
- Create: `tests/test_checkpoint_compatibility.py`

**Interfaces:**
- Produces checkpoint metadata: `model_name`, `model_type`, `d_model`, encoder/decoder layer counts, vocabulary size, and configuration fingerprint.
- Produces: `validate_checkpoint_architecture(checkpoint, model) -> None`.

- [ ] **Step 1: Write failing compatibility tests**

Create synthetic checkpoint metadata for Small (`d_model=1472`, 12/4 layers) and Base (`d_model=1536`, 18/6 layers). Assert Base accepts Base and rejects Small before `load_state_dict` with a message listing every mismatch.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_checkpoint_compatibility.py -v`

- [ ] **Step 3: Store and validate architecture identity**

Save the resolved identity in every checkpoint and validate it before loading weights. Legacy checkpoints without identity are development-only and require an explicit `allow_legacy_checkpoint=True`; that flag must make provenance ineligible for thesis reporting.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_checkpoint_compatibility.py tests/test_checkpoint_handoff.py -v`

```bash
git add src/training/trainer.py src/utils/reproducibility.py tests/test_checkpoint_compatibility.py
git commit -m "fix(checkpoints): prevent small-to-base weight loading"
```

---

### Task 6: Add an explicit Base preflight command

**Files:**
- Create: `scripts/preflight_base.py`
- Create: `tests/test_base_preflight.py`

**Interfaces:**
- Produces CLI: `python scripts/preflight_base.py --device cuda --max-input-length 1024`.
- Produces JSON containing model/tokenizer identity, parameter count, architecture fields, forward/generate success, peak allocation, and recommended physical batch viability.

- [ ] **Step 1: Write failing preflight unit tests**

Mock `AutoConfig`, tokenizer, model, and CUDA counters. Assert failure if the resolved model is not `google/byt5-base`, if tokenizer/model vocabularies differ, or if any variant resolves a different backbone.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_base_preflight.py -v`

- [ ] **Step 3: Implement the preflight CLI**

The default mode may download/load Base and run one short sentence through each variant without training. Add `--metadata-only` to inspect configuration without loading weights. Catch CUDA OOM, clear only tensors created by the script, and exit nonzero with a recommendation to lower physical batch size while preserving effective batch size through accumulation.

- [ ] **Step 4: Run offline tests**

Run: `python -m pytest tests/test_base_preflight.py -v`

Expected: PASS without network access.

- [ ] **Step 5: Run the network/GPU preflight on experiment hardware**

Run: `python scripts/preflight_base.py --device cuda --max-input-length 1024`

Expected: JSON reports `google/byt5-base` for all variants and successful generation. This is a preflight result, not thesis evidence.

- [ ] **Step 6: Commit**

```bash
git add scripts/preflight_base.py tests/test_base_preflight.py
git commit -m "feat(scripts): add byt5 base hardware preflight"
```

---

### Task 7: Update manuscript-facing names and reproducibility documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/DEFENSE-PREP.md`
- Modify: `docs/CHAPTER3-COMPLIANCE.md`
- Modify: `specs/DECISIONS.md`
- Modify: `specs/FINDINGS.md`
- Modify: `specs/PROVENANCE.md`

**Interfaces:**
- Produces three unambiguous names: `ByT5-base`, `ByT5-base + fixed-rate deletion`, and `TAHIMIK (ByT5-base + noise-adaptive deletion)`.

- [ ] **Step 1: Replace stale development claims**

Remove statements that the actual experiment will switch later or that Small is the current reportable backbone. Retain historical notes only when labeled with their decision date.

- [ ] **Step 2: Document initialization and comparator scope**

State that every backbone starts independently from `google/byt5-base`; fixed/adaptive modules start randomly; no released MrT5 checkpoint is used; and “MrT5-style” refers to the delete mechanism, not an official Base checkpoint.

- [ ] **Step 3: Document the gate-layer decision**

Record that layer 3 is retained as the absolute gate location for fidelity to the chosen manuscript design and is shared by fixed/adaptive variants. Do not claim proportional equivalence to MrT5 Small's 12-layer encoder.

- [ ] **Step 4: Run documentation checks**

Run: `rg -n "byt5-small|MrT5-base|switch to byt5-base|development default" README.md docs specs`

Expected: every remaining occurrence is an explicitly historical statement or explains why the released Small checkpoint is not used.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/DEFENSE-PREP.md docs/CHAPTER3-COMPLIANCE.md specs/DECISIONS.md specs/FINDINGS.md specs/PROVENANCE.md
git commit -m "docs: align experiment records with byt5 base"
```

---

### Task 8: Final controlled-comparison verification

**Files:**
- Modify only if evidence exposes a defect: `tests/test_model_configuration.py`
- Modify only if evidence exposes a defect: `tests/test_checkpoint_compatibility.py`

**Interfaces:**
- Produces a verified configuration/provenance record showing that backbone, tokenizer, gate layer, decoding, data, and effective batch controls are identical where required.

- [ ] **Step 1: Run the complete test suite**

Run: `python -m pytest tests -v`

Expected: PASS.

- [ ] **Step 2: Verify no production Small identifiers remain**

Run: `rg -n 'google/byt5-small|stanfordnlp/mrt5-small' configs src scripts backend`

Expected: no production loader uses either identifier. A validation error message may mention the forbidden Stanford checkpoint.

- [ ] **Step 3: Print resolved control variables**

Run:

```bash
python -c "from configs.byt5_config import ByT5Config; from configs.mrt5_config import MrT5Config; from configs.tahimik_config import TAHIMIKConfig; cs=[ByT5Config(),MrT5Config(),TAHIMIKConfig()]; print([(c.variant_name,c.model_name,c.stage1_batch_size*c.stage1_gradient_accumulation_steps,c.stage2_batch_size*c.stage2_gradient_accumulation_steps) for c in cs])"
```

Expected: all model names are `google/byt5-base`; effective batches are 16 and 8.

- [ ] **Step 4: Run Base preflight on the target GPU**

Run: `python scripts/preflight_base.py --device cuda --max-input-length 1024`

Expected: success for all three variants with recorded peak memory.

- [ ] **Step 5: Commit any test-only corrections**

```bash
git add tests/test_model_configuration.py tests/test_checkpoint_compatibility.py
git commit -m "test(experiment): verify controlled byt5 base migration"
```

## Completion gate

The migration is complete only when all production loaders resolve `google/byt5-base`, all three variants pass the same configuration checks, Small checkpoints are rejected, the target GPU passes preflight, and the alignment plan's position-bias/dynamic-padding work is complete. Model training and final benchmarking occur after this gate and must produce new Base checkpoints; existing Small results cannot be relabeled or reused.
