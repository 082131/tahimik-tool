# Phase 1 Data Model: Noise-Adaptive ByT5 (TAHIMIK)

The entities that carry the contribution. Two of them — `cn` and `navg` — are
model state rather than data, and the distinction between them is the
mechanism's core.

## Noise score (`n`)

**Shape**: `(batch,)`, values in `[0, 1]`. Produced by the noise estimator
(spec `002`), consumed here.

**Two consumption points, both detached**:

| Consumer | Expression | Where | Effect |
|----------|------------|-------|--------|
| Gate shift | `cn · (n − navg)` | `delete_gate.py:143` | Changes scores directly, every forward pass, training **and** inference |
| Deletion target | `d_max · (1 − n)` | `noise_adaptive_byt5.py:187` | Changes what the rate loss asks for, training only |

**Why detached at both**: see `research.md`. Undetached at the shift, the
estimator learns to emit whatever makes compression convenient. Undetached at
the target, the rate loss can satisfy itself by moving the target instead of
moving the rate.

## `cn` — learned conditioning coefficient

**Type**: `nn.Parameter`, scalar, initialised `1.0` (`delete_gate.py:77`).

**What it controls**: how strongly noise shifts the keep/delete decision. Larger
means more aggressive adaptation.

**Unguarded property**: its **sign is unconstrained**. A negative `cn` inverts
the mechanism — noisy sentences compressed more — while the model continues to
run and produce plausible numbers. Nothing currently detects this. See
`research.md` and the convergence tasks.

## `navg` — running average of noise scores

**Type**: registered **buffer**, not a parameter. Initialised `0.5`
(`delete_gate.py:81`).

**Why a buffer and not a parameter**: it is an observed statistic, not something
to optimize. Making it learnable would let the model move the reference point to
suit the loss, defeating the purpose of having a reference point at all.

**Update rule**: `navg ← momentum·navg + (1−momentum)·batch_mean`, momentum
0.99, **training only**. Frozen at inference so evaluation does not depend on
batch composition.

**Open**: the `0.5` initialisation and any warmup period are unspecified in the
manuscript. With momentum 0.99 the EMA half-life is roughly 69 batches, so early
training conditions the gate against a possibly-wrong centre.
`NEEDS CLARIFICATION`.

## The `cn` / `navg` distinction

Easy to conflate, and the mechanism needs both:

- `navg` answers **"what counts as typical?"** — an observation, updated from
  data, never optimized.
- `cn` answers **"how much should atypical matter?"** — a decision, learned by
  optimization.

One is measurement, the other is policy. Neither substitutes for the other.

## Per-sentence deletion target

**Shape**: `(batch,)`. **This is what makes the variant adaptive.** MrT5's target
is one scalar broadcast across the batch; TAHIMIK's is a vector with a distinct
value per sentence.

If this ever collapses to a single repeated value across a batch of mixed noise
scores, the variant has silently become MrT5. SC-004 exists to catch that, and
no existing test covers it.

## Forward-pass result

Everything MrT5 emits, plus:

| Key | Shape | Notes |
|-----|-------|-------|
| `noise_scores` | `(batch,)` | The estimator's output, **undetached here** — `losses.py` needs the graph to compute `L_NE` |
| `target_deletion_rate` | `(batch,)` | Per-sentence, computed from **detached** `n` |
| `ne_loss` | scalar | `MSE(noise_scores, noise_level)` when `noise_level` is supplied |

**Absent**: `fixed_deletion_target`. This variant has no shared constant target —
that absence is the difference.

## Config fields read

| Field | Value | Role |
|-------|-------|------|
| `d_max` | 0.5 | Ceiling on deletion for a perfectly clean sentence |
| `noise_avg_momentum` | 0.99 | EMA momentum for `navg` |
| `gate_k` | −30.0 | Gate score bound; also sets the clamp the shift saturates against |
| `delete_gate_layer` | 3 | **Must equal MrT5's**, or placement confounds the result |
| `noise_estimator_hidden_dim` | 256 | Estimator width (spec `002`) |
| `w_rate`, `w_attn_reg` | 1.0, 0.01 | Loss weights. `L_NE` carries implicit weight 1.0 |
