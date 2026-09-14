# Transformer implementation review

**Scope.** This review traces the active implementation paths in `src/models`, `src/data`, `src/training`, `src/evaluation`, and `scripts`, against the project-local Spec Kit specifications and constitution. It is a read-only review: no source code was changed. Historical entries in `specs/FINDINGS.md` were not repeated unless they remain true in the current files.

## 1. Executive summary

The uncompressed ByT5 wrapper is a thin and generally correct use of the Hugging Face model. The two compressed variants have a material training/inference inconsistency: their custom training forward path omits the backbone's conditional decoder-output scaling, whereas generation uses the backbone's normal forward path. This can make both compression conditions learn from a different logit distribution than the one used for reported predictions, so the study is **not ready** for thesis experiments until corrected and regression-tested.

The data, scheduler, padding, gradient accumulation, hard-deletion, CUDA synchronization, and current statistical-test paths are largely thoughtfully implemented. Main remaining validity risks are independent re-splitting at each entry point, truncation after noise labels are computed, and the fact that the soft surrogate rate trained by the gate is not the hard deletion rate measured at inference. Reproducibility is incomplete because production entry points do not enable the repository's required deterministic mode.

## 2. Model-by-model summary

| Model | Intended / implemented architecture | Status |
|---|---|---|
| ByT5 baseline | `google/byt5-small` loaded as `T5ForConditionalGeneration`; full native encoder/decoder; no gate; deletion rate is zero. | Architecture, teacher forcing, padding, and beam generation are sound. Config and README consistently select *small*, but the older retrofit specs and constitution still say *base*, so the documented study contract must be reconciled. |
| MrT5 fixed | Stanford remote MrT5 backbone with embedded gate disabled; wrapper inserts a copied local gate after encoder block 2, applies soft attention bias in training and hard gather-based pruning in eval/generation. | Custom masking and position-bias propagation are internally consistent. The manual decoder logit path is wrong when `config.scale_decoder_outputs` is enabled; hard-rate behavior is not directly optimized. |
| TAHIMIK adaptive | MrT5 path plus an MLP on pre-gate states, EMA-centred positive score shift, per-example rate target, and detached gate inputs. | The intended estimator lifecycle and gradient isolation are correctly wired. It inherits the compressed-decoder mismatch and hard-rate calibration concern. |

## 3. Prioritised action items

### P0 — Critical

#### C1. Compressed training logits omit the backbone's decoder-output scaling

**Affected models:** MrT5 fixed and TAHIMIK  
**Affected code:** `src/models/fixed_compression.py:176-191`; `src/models/noise_adaptive.py:205-222`

**Problem and evidence.** Both wrappers call `self.model.decoder(...)` and immediately pass `decoder_outputs[0]` to `self.model.lm_head`. In the installed Hugging Face `T5ForConditionalGeneration.forward`, the same operation first performs `sequence_output *= self.model_dim ** -0.5` when `self.config.scale_decoder_outputs` is true, then applies `lm_head`. The wrappers do not check or apply that flag. Their `generate()` methods do call `self.model.generate()`, which uses the backbone generation/forward code and therefore uses the conditional scaling.

**Impact on thesis.** With the normal T5 setting enabled, CE gradients, validation loss, and checkpoint selection for both compressed variants use logits at a different scale from the logits used for generation and scored predictions. This can materially alter optimization and makes their comparison with the native ByT5 forward invalid.

**Recommended fix.** Create one small helper shared by both wrappers (or exactly mirror the conditional from the installed backbone) and apply it immediately before every `lm_head` call in their manual decoder paths. Preserve behavior when the remote MrT5 configuration disables the flag. Add a tiny-model regression test that compares wrapper logits/loss with an equivalent native T5 decoder/LM-head computation for both flag settings, plus an end-to-end training/generation consistency test.

**Fix prompt.**

> In `src/models/fixed_compression.py` and `src/models/noise_adaptive.py`, correct the manual decoder path used by `forward(labels=...)` and `forward(decoder_input_ids=...)`. Current behavior calls `self.model.lm_head(decoder_outputs[0])` directly. Hugging Face T5 conditionally scales `sequence_output` by `self.model.model_dim ** -0.5` when `self.model.config.scale_decoder_outputs` is true before its LM head. Mirror that conditional exactly before every manual LM-head call, without changing generation, gate behavior, losses, or model configuration. Factor shared code only if it remains small and explicit. Add regression tests in `tests/test_model_forward.py` using the existing tiny T5 fixture: test true and false values of `scale_decoder_outputs`, verify logits equal a reference native computation, and verify the two compressed variants still produce finite loss and gradients. Run the relevant tests and the full suite.

### P1 — High

#### H1. Noise labels are calculated before independent input/target truncation

**Affected models:** All models for sequence/target integrity; TAHIMIK directly for `L_NE`  
**Affected code:** `src/data/dataset.py:48-95`; `scripts/train.py:147-175`; `scripts/evaluate.py:84-103`

**Problem and evidence.** Scripts compute `n*` from complete strings, then `NormalizationDataset.__getitem__` independently truncates noisy input and clean target to their separate maximum lengths. Therefore a long pair can be trained/evaluated as truncated byte sequences while TAHIMIK receives an estimator target calculated from content the model never sees; separate truncation can also leave an incomplete normalization pair.

**Impact on thesis.** This is a direct supervision error for TAHIMIK and can alter CE, accuracy, and cross-model fairness for overlength gold examples. It also makes the reported maximum-length policy hard to defend.

**Recommended fix.** Decide and document one policy: reject overlength gold pairs with an auditable count, or deterministically pre-truncate/segment pairs before computing noise labels and before every split. Do not silently tokenize away differing suffixes.

**Fix prompt.**

> Make sequence-length handling consistent in `src/data/dataset.py`, `src/data/preprocessing.py`, and the train/evaluate/experiment entry points. Current `n*` is calculated from full noisy/clean strings, but `NormalizationDataset` later truncates input and target independently at `max_input_length` and `max_target_length`. Implement a single documented policy before noise-label computation and splitting: either reject overlength gold pairs with a logged/auditable reason, or deterministically transform pairs into valid aligned examples. The policy must apply identically to train, validation, test, and all variants; it must not silently retain an `n*` calculated from discarded text. Add tests for a long pair, including the resulting label and dataset length, and verify no train/test leakage is introduced.

#### H2. The rate loss trains a soft probability surrogate but reports a different hard threshold rate

**Affected models:** MrT5 fixed and TAHIMIK  
**Affected code:** `src/models/delete_gate.py:224-250`; `src/training/losses.py:68-107`; `src/models/fixed_compression.py:109-161`; `src/models/noise_adaptive.py:130-189`

**Problem and evidence.** During training, `deletion_rate = 1 - mean(keep_prob)`, but inference reports `1 - mean(gate_outputs > k/2)`. `L_rate` constrains only the former; no objective or calibration verifies that the hard deletion fraction approaches the configured fixed target or the adaptive target. The soft/hard split is differentiable and intentional, but these two quantities are not equivalent.

**Impact on thesis.** Reported compression, latency, and memory can differ from the trained deletion objective. The effect may differ by variant and noise band, confounding the claim that TAHIMIK targets a particular adaptive compression behavior.

**Recommended fix.** Retain the differentiable surrogate, but add post-training hard-rate calibration/validation and report both soft training rate and actual hard rate per noise band. If a controller/straight-through calibration is needed to achieve the stated target, specify it in the thesis and apply it equally to compressed models.

**Fix prompt.**

> In the MrT5 and TAHIMIK gate/training paths, make the distinction between the differentiable soft deletion rate and the hard inference deletion rate explicit and testable. Current `DeleteGate.forward` trains `L_rate` on `1 - mean(keep_prob)` in train mode but reports `1 - mean(kept_mask)` after the `k/2` threshold in eval mode; the code has no acceptance criterion that hard rates meet fixed/adaptive targets. Preserve differentiability and existing soft-mask training. Add validation diagnostics that compute the hard rate in eval mode on the validation set, including TAHIMIK noise bands, and persist both rates. Add tests proving the values can differ and that reported efficiency telemetry always uses the hard rate. Do not add an unvalidated heuristic controller; if calibration is introduced, document and test it for both compressed variants.

#### H3. Thesis-required deterministic execution is not enabled by the experiment entry points

**Affected models:** All  
**Affected code:** `scripts/train.py:134-136`, `scripts/evaluate.py:75-76`, `scripts/benchmark.py:70-73`, `scripts/run_experiment.py`; `src/utils/reproducibility.py:18-28`

**Problem and evidence.** The scripts seed Torch, but do not call the already-provided `configure_determinism()`, which configures Python/NumPy/Torch seeds, cuDNN flags, CUBLAS workspace, and deterministic algorithms. This violates Constitution Principle III and leaves Gumbel noise, dataloader shuffling, and CUDA kernels insufficiently controlled.

**Impact on thesis.** A rerun may produce different checkpoints, generations, and timing/accuracy summaries. Results are not bit-reproducible as required by the project’s own experimental contract.

**Recommended fix.** Call the shared setup at the top of every results-producing entry point, log the deterministic setting, and add a small reproducibility regression test. Document any operation that cannot run deterministically instead of silently disabling the requirement.

**Fix prompt.**

> Enforce Constitution Principle III at all experiment entry points. In `scripts/train.py`, `scripts/evaluate.py`, `scripts/benchmark.py`, and `scripts/run_experiment.py`, replace ad hoc Torch-only seeding with `src.utils.reproducibility.configure_determinism(seed)` before any model, tokenizer, dataset, or generator is constructed. Ensure it seeds Python, NumPy, CPU/CUDA Torch, sets deterministic cuDNN/CUBLAS behavior, and enables deterministic algorithms as the existing helper intends. Log the resolved deterministic mode and fail clearly for unsupported operations. Add tests that mock CUDA as needed and prove each entry point uses the shared helper; add a small CPU synthetic-generation repeatability test.

### P2 — Medium

#### M1. Dataset split identity is re-derived independently, not persisted with checkpoints/results

**Affected models:** All  
**Affected code:** `src/data/preprocessing.py:209-263`; `scripts/train.py:153-159`; `scripts/evaluate.py:88-96`; `scripts/run_experiment.py`

**Problem and evidence.** Each entry point recreates a shuffled 80/10/10 split from input order and seed. This works only if the source dataset, its normalization/deduplication order, and RNG call history remain identical. Checkpoints save no split indices/fingerprint, while later evaluation recomputes them.

**Impact on thesis.** Dataset revisions or reordered files can silently change the test set used to score an existing checkpoint. This is not current leakage, but it weakens provenance and makes comparisons difficult to reproduce.

**Recommended fix.** Persist a versioned split manifest with source digest, normalized pair IDs/indices, seed, and split fingerprint; require evaluation to load it from the checkpoint/run directory.

**Fix prompt.**

> Add durable split provenance for thesis runs. Update `DataPipeline.split_data` and the train/evaluate/run-experiment flow so training writes a versioned manifest containing the source-data digest, normalized pair identifiers or indices, seed, ratios, and exact train/validation/test membership. Store its path and digest in checkpoint and result provenance. Evaluation must load and validate that manifest instead of reshuffling the current source file. Preserve the existing 80/10/10 policy and add tests showing source-row reordering cannot silently change a checkpoint’s test set.

#### M2. Standalone evaluation/benchmark bypass checkpoint architecture and provenance validation

**Affected models:** All  
**Affected code:** `scripts/evaluate.py:114-118`; `scripts/benchmark.py:100-103`; `src/training/trainer.py:97-160`

**Problem and evidence.** `TAHIMIKTrainer` and backend loading call `validate_checkpoint_architecture`, but the CLI evaluator and benchmark call `torch.load(..., weights_only=True)` followed directly by `model.load_state_dict`. They do not validate model source/name, stage, architecture metadata, or eligibility.

**Impact on thesis.** A wrong-but-shape-compatible or Stage 1 checkpoint can be benchmarked and reported under the requested variant without the safety checks available elsewhere.

**Recommended fix.** Route CLI loads through one shared safe loader that calls architecture validation, rejects ineligible/legacy checkpoints by default, checks stage policy, and records provenance.

**Fix prompt.**

> Make `scripts/evaluate.py` and `scripts/benchmark.py` use the same checkpoint validation policy as `TAHIMIKTrainer` and `backend/app.py`. Before `load_state_dict`, call `validate_checkpoint_architecture` with the requested variant’s expected model name; reject missing architecture metadata unless an explicit development-only override is supplied, and reject checkpoints whose stage is not allowed for thesis reporting. Keep `weights_only=True` where supported. Add tests for wrong backbone/source metadata, absent metadata, and a Stage 1 checkpoint passed to final evaluation.

### P3 — Low

#### L1. Active configuration and several retrofit specs describe different backbone decisions

**Affected models:** All  
**Affected code:** `configs/base.py:25`, `README.md`, `specs/003-byt5-baseline/spec.md`, `specs/004-fixed-rate-compression/spec.md`, constitution references

**Problem and evidence.** Active configs and README specify Small checkpoints, while older retrofit specifications/constitution passages still state the manuscript requires Base. `specs/FINDINGS.md` marks this historical remediation as resolved, but the checked-in current config is Small.

**Impact on thesis.** This is documentation/provenance ambiguity rather than an execution defect, but it could make the experimental setup indefensible unless the authoritative decision is explicit.

**Recommended fix.** Reconcile the manuscript, constitution, README, specs, and current configs through an explicit recorded decision; do not silently change model size.

**Fix prompt.**

> Reconcile model-size documentation without changing any model code or checkpoint behavior. The active configs and README use ByT5/MrT5 Small, but retrofit specs and the constitution include stale text saying the thesis requires Base. Create or update the project’s explicit decision record, identify the authoritative study configuration, and update all stale references consistently. Include the exact checkpoint IDs, parameter-count implications, and whether prior runs remain eligible. Add a documentation check or test that prevents config/spec drift.

## 4. Cross-model consistency findings

| Area | ByT5 | MrT5 fixed | TAHIMIK | Consistency issue |
|---|---|---|---|---|
| Architecture | Native full T5 | Manual split encoder + local gate | MrT5 path + estimator | Intentional compression differences; compressed manual decoder needs native scaling parity. |
| Input processing | Dynamic tokenizer/collator | Same | Same | Same policy, but labels predate potential truncation. |
| Attention/padding | Native mask | Additive gate bias; hard compressed mask | Same | Correctly masks padding in gate metrics. |
| Training | Native HF CE | Manual CE, rate + commitment loss | Manual CE, rate + commitment + MSE | Compressed CE scaling differs from baseline/native inference. |
| Inference/decoding | Beam generation | Hard prune then native generation | Same plus estimator | Same beams/max length; hard rate is factual telemetry. |
| Evaluation | Same metrics/test reconstruction | Same | Same | Test membership is regenerated rather than persisted. |
| Runtime/memory | Same 5 warmups/20 runs, CUDA sync | Same | Same | Good reference pattern is `EfficiencyBenchmark`; all variants use it. |
| Reproducibility | Torch seed only | Torch seed + stochastic gate in train | Same | Shared deficiency: deterministic mode not enabled. |

The `NormalizationCollator`, `EfficiencyBenchmark`, shared `TAHIMIKLoss`, and `run_encoder_layers` are the stronger standardization patterns. They avoid the duplicated behavior that is causing the decoder-scale omission.

## 5. Experimental validity checklist

- [x] **PASS** — No direct train/test overlap is created by a single `split_data` invocation, but duplicate-group handling is not guaranteed.
- [~] **NEEDS VERIFICATION** — Same test set across models depends on identical input order and repeated seeded splitting; persist a manifest.
- [x] **PASS** — Same tokenizer limits, collator, beams, and benchmark protocol are configured across variants.
- [~] **NEEDS VERIFICATION** — Output/reference list lengths are assumed by `zip`; add explicit equality checks before metrics.
- [x] **PASS** — Padding is excluded from gate rates and attention regularization.
- [~] **NEEDS VERIFICATION** — Generation uses hard deletion correctly, but it has not been calibrated against the trained soft rate.
- [x] **PASS** — No evaluation-side fallback substitutes a model output.
- [x] **PASS** — CUDA timing is synchronized and warmups are discarded.
- [x] **PASS** — Peak CUDA allocation is reset per timed run; CPU explicitly reports memory unavailable.
- [~] **NEEDS VERIFICATION** — Small versus Base documentation must be reconciled before claiming the configuration matches the manuscript.
- [ ] **FAIL** — Deterministic mode required by the constitution is not invoked by result-producing scripts.
- [~] **NEEDS VERIFICATION** — Checkpoint selection uses validation loss, not test performance, but final CLI evaluation does not enforce checkpoint stage/provenance.
- [~] **NEEDS VERIFICATION** — Fresh generations are produced, but results lack durable split/checkpoint provenance.

## 6. Quick wins

- Replace unused `collate_fn` imports in `src/training/trainer.py`, `src/evaluation/efficiency.py`, and evaluator modules after verifying no public import relies on them.
- Add `ValueError` checks in metric entry points when prediction/reference/noisy list lengths differ, before `zip` can drop observations.
- Replace broad `except Exception` in `load_mrt5_pretrained_gate` with logged expected download/format errors; preserve fail-closed construction behavior.
- Update stale Small/Base language in historical spec sections only after an authoritative configuration decision.

**Consolidated prompt:**

> Make only low-risk cleanup changes: remove confirmed unused `collate_fn` imports, add explicit equal-length validation to public metric methods before any `zip`, and replace the silent broad exception in `load_mrt5_pretrained_gate` with narrow, logged expected failures while preserving its `False` return and the constructor’s fail-closed RuntimeError. Do not alter model math, data policy, checkpoint semantics, or model-size selection. Add focused tests and run the full suite.

## 7. Redundancy removal log

- `collate_fn` imports in `src/training/trainer.py` and `src/evaluation/efficiency.py` → imported but these modules construct and use `NormalizationCollator` directly; no local reference exists.

No files, classes, model modules, configuration fields, or feature flags were verified safe to delete. In particular, the local gate, position-bias utilities, telemetry methods, and compatibility loader are referenced by the active compressed-model paths.

## 8. Final readiness assessment

### Overall status

**NOT READY**

**Must fix before training/evaluation:** C1 (compressed decoder scaling), H1 (pre-label truncation policy), and H3 (deterministic mode). Also verify H2’s hard-rate calibration before reporting compression/efficiency claims.

**Should fix before final thesis experiments:** M1 (persisted split manifest) and M2 (validated CLI checkpoint loading).

**Can defer until cleanup:** L1 and the Quick Wins, provided the authoritative architecture decision is separately recorded before reporting.

**Most important verification:** after C1 is fixed, use the exact same compressed encoder states and decoder inputs to prove that the wrappers’ teacher-forced logits match the underlying backbone’s configured LM-head behavior, then run a small end-to-end train/generate regression for both compressed variants.

## Verification notes

I inspected the active code and the local Spec Kit contracts, then attempted `python -m pytest tests -q`. The run progressed past 67% but the available command channel returned before a final pass/fail summary, so this review does **not** claim the full suite passed. Focused source inspection confirmed the decoder-scaling behavior from the installed Transformers implementation.
