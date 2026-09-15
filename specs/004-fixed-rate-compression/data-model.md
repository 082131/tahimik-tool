# Phase 1 Data Model: Fixed-Rate Compression

> **Current parameter/source authority:**
> `specs/018-huggingface-mrt5-baseline/data-model.md`.

No persistent data entity. The entities here are the tensors the gate produces
and the config values that govern it.

## Gate output (per forward pass)

The delete gate returns a 4-tuple. Each element has a distinct job, and
conflating them is the main way this component gets misused.

| Name | Shape | Differentiable | Purpose |
|------|-------|----------------|---------|
| `gate_outputs` | `(batch, seq, 1)` | yes | Per-byte scores in `[k, 0]`. Added to attention logits as a log-space bias during training. |
| `keep_prob` | `(batch, seq)` | yes | Keep probability in `[0, 1]`, derived from the gate score. The differentiable quantity the rate loss uses. |
| `kept_mask` | `(batch, seq)` | **no** | Binary keep/delete decision from thresholding. Used for physical deletion at inference. Non-differentiable by construction. |
| `deletion_rate` | `(batch,)` | in training only | Fraction deleted per sentence. Computed from `keep_prob` in training so the rate loss has a gradient; from `kept_mask` at inference so the reported number is the real one. |

**The `deletion_rate` dual definition is deliberate and worth understanding.**
Using the hard mask during training would produce a constant with no gradient,
and the rate loss would be inert — the gate would never learn its target. Using
the soft probability at inference would report a compression that did not
physically happen. Each mode uses the definition that is correct for it.

## Config values read

| Field | Value | Role |
|-------|-------|------|
| `delete_gate_layer` | 3 | Where the encoder splits. **Control variable** — must equal TAHIMIK's. |
| `gate_k` | −30.0 | Lower bound of the gate score range. Sets the hard threshold at `k/2`. |
| `fixed_deletion_target` | 0.5 | What the rate loss aims for. Applied identically to every sentence — this is what makes the variant "fixed". |
| `w_rate` | 1.0 | Rate loss weight. |
| `w_attn_reg` | 0.01 | Attention regulariser weight. See the open item in `research.md`. |
| `noise_adaptive` | `False` | Selects the gate's fixed-rate mode. The difference this baseline exists to isolate. |

None are written by this feature.

## Forward-pass result

Adds to the baseline's keys: `gate_outputs`, `keep_prob`, `kept_mask`,
`deletion_rate`, and `fixed_deletion_target`.

`fixed_deletion_target` is emitted so the loss uses **this variant's**
configured target rather than falling back to a hardcoded default — a subtle
correctness detail the implementation already handles
(`fixed_compression_byt5.py`, comment at the `result` dict).

**Absent by design**: `noise_scores`, `target_deletion_rate`, `ne_loss`. This
variant has no noise estimator, and its target is a shared constant rather than
a per-sentence value.

## Sequence length

The one entity whose *shape* changes between modes:

- **Training**: unchanged. Soft deletion masks; it does not remove.
- **Inference**: shortened to the longest surviving sequence in the batch, with
  shorter ones re-padded. Minimum length 1 is enforced, so the decoder always
  receives something even if a sentence is entirely deleted.
