# Feature Specification: Noise-Adaptive ByT5 (TAHIMIK)

**Feature Branch**: `feat/005-noise-adaptive-byt5`

**Created**: 2026-08-27

**Status**: Draft — **Class B provenance**, see [../PROVENANCE.md](../PROVENANCE.md)

**Input**: Retrofit spec for `src/models/noise_adaptive_byt5.py` and the
noise-adaptive mode of `src/models/delete_gate.py`.

> **This is the study's contribution.** Everything else in `specs/` exists to
> make this variant's claim measurable. Read this one closely.

## Plain-language summary

The two baselines are all-or-nothing: ByT5 throws away nothing, MrT5 always
throws away the same fixed share. Both are wrong in the same way — they ignore
the sentence in front of them.

TAHIMIK looks first. It asks the noise estimator "how messy is this?", then
adjusts how much it throws away based on the answer. A clean sentence can lose
a lot safely, so compress hard and go fast. A messy sentence needs its details
to be reconstructed, so keep more.

The claim is that this buys most of MrT5's speed without paying MrT5's accuracy
cost on exactly the noisy sentences that matter for Tagalog/Taglish social media
text.

## What the manuscript specifies

Two distinct mechanisms, both driven by the noise score `n`:

**1. The gate shift** — the score nudges every byte's keep/delete decision:

```
G_shifted = G + cn · (n − navg)
```

where `cn` is learned, `navg` is a running average of noise scores (momentum
0.99). A noisier-than-average sentence shifts scores up, so fewer bytes get
deleted.

**2. The deletion target** — the score sets what the rate loss aims for:

```
d_target(xᵢ) = d_max · (1 − nᵢ)
```

Clean (`n≈0`) targets `d_max`; noisy (`n≈1`) targets near zero.

**3. The gradient boundary**, stated directly:

> "Although n appears in the gate's conditioning term and in the deletion
> target, it is detached at both points, so the estimator receives gradients
> only from L_NE."

> Backpropagation routes each loss term: "(1) from L_CE, it gradients to the
> ByT5 encoder-decoder; (2) from L_CE + w_rate·L_rate + w_attention·L_attention,
> it gradients to the delete gate; (3) from L_NE, it gradients to the noise
> estimator."

**4. Total loss**: `L = L_CE + w_rate·L_rate + w_attn_reg·L_attn_reg + L_NE`

## User Scenarios & Testing

### User Story 1 - Compression adapts to sentence noise (Priority: P1)

**Actor**: the training loop and the inference path.

This is the contribution in one sentence: two sentences of identical length but
different messiness must receive *different* compression.

**Independent Test**: Feed the gate two identical hidden-state batches with
different noise scores; confirm the noisier one is compressed less.

**Acceptance Scenarios**:

1. **Given** two batches identical except for their noise scores, **When** the
   gate runs, **Then** the batch with the higher noise score has a **lower**
   deletion rate. *If this fails, the study has no contribution.*
2. **Given** a noise score `n`, **When** the deletion target is computed,
   **Then** it equals `d_max · (1 − n)` per sentence, not a shared constant.
3. **Given** training mode, **When** a batch completes, **Then** `navg` updates
   as an exponential moving average of that batch's mean noise score.

### User Story 2 - Gradient isolation holds under the full loss (Priority: P1)

**Actor**: the training loop.

Equal priority to Story 1, because if this fails the comparison is unfair and
the contribution is unmeasurable regardless of how well Story 1 works.

**Independent Test**: Backpropagate `L_rate` alone and confirm no gradient
reaches the noise estimator; backpropagate `L_NE` and confirm it does.

**Acceptance Scenarios**:

1. **Given** the full loss, **When** `L_rate` backpropagates, **Then** the
   noise estimator's parameters receive **no** gradient.
2. **Given** the full loss, **When** `L_NE` backpropagates, **Then** the noise
   estimator's parameters **do** receive gradient.
3. **Given** the noise score used in the gate shift and in the deletion target,
   **When** either is computed, **Then** the value used is detached at both
   points.

### User Story 3 - Physical compression at inference (Priority: P2)

Same hard-deletion requirement as `004`, but with the noise-conditioned gate.
This is where the efficiency claim is actually earned.

### Edge Cases

- **`navg` at initialization**: registered as a buffer starting at `0.5`
  (`delete_gate.py:81`). Until enough batches have passed, the shift is
  computed against a prior rather than an observed average.
  `NEEDS CLARIFICATION: does the manuscript specify an initialization value or
  a warmup period for navg? The code chooses 0.5; the manuscript text located
  so far does not state one.`
- **Gate shift pushing scores out of range**: clamped back to `[k, 0]`
  (`delete_gate.py:151`). Defined behavior, but the clamp silently caps how
  far noise adaptation can push the decision.
  `NEEDS CLARIFICATION: is saturation at the clamp boundary expected and
  acceptable, or should cn be constrained so it cannot saturate?`

## Requirements

- **FR-001**: The variant MUST run the noise estimator on pre-gate hidden
  states before the gate, at the same layer, with no second encoder pass.
- **FR-002**: The gate MUST shift scores by `cn · (n − navg)` where `cn` is a
  learned scalar parameter.
- **FR-003**: `navg` MUST be an exponential moving average updated only during
  training, with momentum from config, and MUST NOT be a learnable parameter.
- **FR-004**: The deletion target MUST be per-sentence, `d_max · (1 − n)`.
- **FR-005**: `n` MUST be detached at both consumption points — the gate shift
  and the deletion target.
- **FR-006**: A higher noise score MUST result in less deletion, all else equal.
- **FR-007**: The total loss MUST be `L_CE + w_rate·L_rate + w_attn_reg·L_attn_reg + L_NE`.
- **FR-008**: Soft deletion in training, hard deletion at inference, as in `004`.

## Success Criteria

### Checkable now

- **SC-001**: Higher noise score produces measurably lower deletion rate.
- **SC-002**: `L_rate` produces zero gradient on the noise estimator.
- **SC-003**: `L_NE` produces nonzero gradient on the noise estimator.
- **SC-004**: Per-sentence deletion targets differ within a batch of mixed
  noise scores.
- **SC-005**: `navg` changes across training batches and is frozen in eval.

### BLOCKED on the gold-standard dataset

- **SC-006** *(blocked)*: TAHIMIK's accuracy versus ByT5 and MrT5 on real data.
- **SC-007** *(blocked)*: TAHIMIK's inference time and peak GPU memory versus
  both baselines.
- **SC-008** *(blocked)*: Whether TAHIMIK's advantage concentrates on noisy
  sentences, which is the mechanism's actual claim.

## Assumptions

- The noise estimator itself is specified in `002` and is not re-specified here.
- The delete gate's fixed-rate mode is specified in `004`; this covers only its
  noise-adaptive mode.
- Whether the accuracy/efficiency tradeoff is *favorable* is an empirical
  question for `007`, not a requirement here.

## Convergence — assessed against the implementation

| ID | Gap | Severity | Evidence |
|----|-----|----------|----------|
| F1 | **contradicts** | **HIGH** | `byt5-small` vs manuscript's `byt5-base`. Cross-cutting. |
| F2 | **contradicts** | MEDIUM | `L_attn_reg` deviates from MrT5 Appendix D — inherited from `004`, affects TAHIMIK's loss identically. |
| F3 | partial | MEDIUM | `navg` initialization (`0.5`) and the gate-shift clamp are code decisions the manuscript does not appear to specify. Both marked `NEEDS CLARIFICATION` above. |
| F4 | partial | LOW | `delete_gate.py:136` assigns `self.noise_avg = ...` directly on a registered buffer inside a `training` branch. Works, but the update runs inside the autograd-tracked forward pass rather than under `torch.no_grad()`; `n_detached` makes it safe in practice. Worth confirming intent. |

**FR-001 through FR-008: satisfied.** Verified directly:
- Gate shift at `delete_gate.py:143` — `shift = self.cn * (n_detached - self.noise_avg)`
- `cn` is `nn.Parameter` (`delete_gate.py:77`); `noise_avg` is a buffer, not a parameter (`:81`)
- Per-sentence target at `noise_adaptive_byt5.py:187` — `self.d_max * (1.0 - noise_scores.detach())`
- **Both detach points present**, exactly as the manuscript requires
- Existing test `tests/test_model_forward.py::test_noise_estimator_is_trained_only_by_l_ne` already proves SC-002 and SC-003
- Existing test `tests/test_delete_gate.py:251-252` already compares clean vs noisy deletion rates — appears to cover SC-001

### Convergence tasks

- [ ] T001 Resolve the two `NEEDS CLARIFICATION` items above (`navg` initialization, gate-shift clamp saturation) against the manuscript, or record them as deliberate implementation choices the manuscript leaves open (partial) — **author decision, not a code question**
- [ ] T002 Confirm `tests/test_delete_gate.py:251-252` fully covers SC-001, and if so cite it here rather than duplicating (partial)
- [ ] T003 [P] Write tests for SC-004 and SC-005, which no existing test appears to cover: per-sentence target variation within a mixed batch, and `navg` updating in train / frozen in eval (missing)
- [ ] T004 [P] Confirm whether the `navg` EMA update should be wrapped in `torch.no_grad()` for clarity at `src/models/delete_gate.py:136` (partial)
