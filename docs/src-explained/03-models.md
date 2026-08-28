# `src/models/` — the three neural networks (exhaustive, every line)

**Flow position:** the **second** stage. These consume tensors from
[`src/data`](02-data.md) and are trained by [`src/training`](04-training.md).

**Files (build order — each adds one piece):**
1. `byt5_baseline.py` — plain ByT5, no compression (the accuracy ceiling)
2. `delete_gate.py` — the compression engine (shared building block)
3. `fixed_compression_byt5.py` — ByT5 + gate = **MrT5** (fixed 50% deletion)
4. `noise_estimator.py` — predicts how noisy a sentence is (shared block)
5. `noise_adaptive_byt5.py` — MrT5 + estimator = **TAHIMIK** (the contribution)
6. `__init__.py` — re-exports

**How they connect:**
```
delete_gate.py ──────────┐            noise_estimator.py ──┐
                         ▼                                 ▼
byt5_baseline.py   fixed_compression_byt5.py        noise_adaptive_byt5.py
   (ByT5 only)        (ByT5 + gate)               (ByT5 + gate + estimator)
        └──────── all trained by src/training, served by backend/ ──────────┘
```

> **Format:** every line of the real source is shown and explained (including
> comments and `return`s). Each block opens with **▸ What this block does**.
> Distinct methods get a **Worked walkthrough**; near-duplicates get a short
> "same as X" note.

---

## Terms & abbreviations used across this whole folder
| Term | Full form / plain meaning |
|---|---|
| NN | neural network — a model made of learnable layers |
| `nn.Module` | PyTorch's base class for every network piece; you subclass it |
| layer | a small transformation with learnable numbers ("weights") |
| forward pass | running data through the model to get an output |
| tensor | a grid of numbers (PyTorch's data container) |
| shape | a tensor's dimensions, e.g. `(batch, seq_len, hidden_dim)` |
| batch | how many sentences processed at once |
| seq_len | sequence length — how many bytes per sentence |
| hidden_dim / d_model | how many numbers represent each byte inside the model |
| encoder | the half of ByT5 that *reads* the input into vectors |
| decoder | the half that *writes* the output one byte at a time |
| hidden states | the encoder's internal vectors for each byte |
| logits | raw output scores before turning them into probabilities |
| gradient | the direction to nudge a weight to reduce error (learning signal) |
| differentiable | smooth enough for gradients to flow (needed for learning) |
| `self.training` | a flag: True after `model.train()`, False after `model.eval()` |
| inference | using the trained model (no learning) |
| beam search | decoding that explores several candidate outputs and keeps the best |
| ByT5 / T5 | Byte-level T5 / Text-to-Text Transfer Transformer (the model family) |
| MrT5 | Merge-then-T5 (Kallini et al., 2025) — the compression method extended here |
| `Linear` | a layer that does a matrix multiply (learned) |
| `LayerNorm` | a layer that normalizes numbers to keep them stable |
| sigmoid | a function squashing any number into (0,1) |
| GELU | Gaussian Error Linear Unit — a smooth activation function |
| dropout | randomly zeroing some values during training (prevents overfitting) |
| `nn.Parameter` | a number the model *learns* |
| buffer | a number the model *stores* but does not learn |
| broadcasting | auto-expanding a smaller tensor to match a bigger one in math |
| `.detach()` | cut the gradient link so learning can't flow through that value |
| EMA | exponential moving average — a slowly-updated running average |
| `**kwargs` | "keyword arguments" — a catch-all for extra named parameters |
| `.squeeze(d)` / `.unsqueeze(d)` | remove / insert a size-1 dimension at position `d` |
| `.view(...)` | reshape a tensor without copying its data |

---

## `byt5_baseline.py` — plain ByT5 (the ceiling)

**Role:** The simplest variant — standard ByT5, no compression. Every byte is
processed → most accurate but slowest. The yardstick for the other two.

- **Input (forward):** `input_ids`, `attention_mask`, optional `labels`
- **Output:** dict with `loss` (if labels), `logits`, `encoder_last_hidden_state`,
  `deletion_rate` (always zeros here)
- **Used by:** `scripts/train.py`, `backend/app.py`, `run_experiment.py`
- **Connects to:** `configs/byt5_config.py`, Hugging Face `T5ForConditionalGeneration`

### Where this fits in TAHIMIK
This is the **experimental control — the accuracy ceiling.** A comparison study
needs a "no compression" reference so you can measure what compression *costs*. Any
accuracy that MrT5 or TAHIMIK loses is measured as a drop from this model. It's also
the simplest variant, so it's the right place to first meet the `forward` → dict →
loss shape the other two mimic.

**What breaks without it:** no baseline to say "TAHIMIK keeps X% of the ceiling's
accuracy at Y× the speed." The comparison needs this anchor.

**Terms & abbreviations in this file**
| Term | Full form / plain meaning |
|---|---|
| `T5ForConditionalGeneration` | Hugging Face's ready-made ByT5 (encoder+decoder+output head) |
| `.from_pretrained(name)` | download a model/tokenizer's trained weights by name |
| `.size(0)` | the length of a tensor's first dimension (here, the batch size) |
| device | where a tensor lives — GPU or CPU |
| `early_stopping` | stop beam search once good candidates finish |

### Imports

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from typing import Dict, Optional
```
**▸ What this block does:** pull in the tools this file uses.
- `import torch` — PyTorch itself (tensors, math).
- `import torch.nn as nn` — the neural-network toolbox, under the short alias `nn`.
- `from transformers import AutoTokenizer, T5ForConditionalGeneration` — Hugging
  Face's tokenizer loader and the ready-made ByT5 model class.
- `from typing import Dict, Optional` — type hints; `Optional[X]` = "an X or `None`."

### Class + constructor

```python
class ByT5Baseline(nn.Module):
```
**▸ What this block does:** declare the model class; `(nn.Module)` makes it a
PyTorch network that inherits gradient machinery.

```python
    def __init__(self, config):
        super().__init__()
```
**▸ What this block does:** the constructor. `config` is the settings object.
`super().__init__()` runs `nn.Module`'s own setup — **required** or the model won't
register its layers.

```python
        self.config = config
        self.model = T5ForConditionalGeneration.from_pretrained(
            config.model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
```
**▸ What this block does:** store the config and download the pretrained model +
tokenizer.
- `self.config = config` — keep the settings for later.
- `T5ForConditionalGeneration.from_pretrained(config.model_name)` — download the
  ByT5 weights named by the config (`"google/byt5-small"`); store in `self.model`.
- `AutoTokenizer.from_pretrained(config.model_name)` — download the matching
  tokenizer into `self.tokenizer`.

### Method `forward`

**Variables at a glance**
| Variable | Type / shape | Holds |
|---|---|---|
| `input_ids` (param) | `(batch, seq_len)` | Input byte IDs |
| `attention_mask` (param) | `(batch, seq_len)` | 1=real, 0=pad |
| `labels` (param) | `(batch, target_len)` or None | Target IDs (padding = −100) |
| `outputs` | HF output object | ByT5's raw result |
| `result` | `dict` | The repackaged output |

```python
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
```
**▸ What this block does:** declares the forward pass and its inputs.
- `input_ids`, `attention_mask` — the batched tensors from `dataset.py`.
- `labels: Optional[...] = None` — the target IDs; optional (absent at inference).
- `**kwargs` — a catch-all that absorbs any extra named arguments (like
  `noise_level`) so all three variants can be **called identically**; this one
  ignores what it doesn't need.
- `-> Dict[str, torch.Tensor]` — returns a dict of tensors.

```python
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
```
**▸ What this block does:** run the standard ByT5. When `labels` are provided, ByT5
computes the cross-entropy loss internally. The result object is stored in
`outputs`.

```python
        result = {
            "logits": outputs.logits,
            "encoder_last_hidden_state": outputs.encoder_last_hidden_state,
            # No compression — deletion rate is always 0
            "deletion_rate": torch.zeros(
                input_ids.size(0), device=input_ids.device
            ),
        }
```
**▸ What this block does:** repackage ByT5's output into the **same dict shape** the
compressed variants use, so downstream code doesn't care which model produced it.
- `"logits": outputs.logits` — the raw prediction scores.
- `"encoder_last_hidden_state": outputs.encoder_last_hidden_state` — the encoder's
  final vectors.
- The comment `# No compression — deletion rate is always 0` explains the next line.
- `torch.zeros(input_ids.size(0), device=input_ids.device)` — a tensor of zeros, one
  per sentence. `input_ids.size(0)` is the **batch size** (first dimension);
  `device=input_ids.device` keeps it on the same GPU/CPU (mixing devices errors).

```python
        if labels is not None:
            result["loss"] = outputs.loss

        return result
```
**▸ What this block does:** if labels were given, include ByT5's computed loss, then
return the dict.
- `if labels is not None:` — only during training/validation.
- `result["loss"] = outputs.loss` — add the loss under the `"loss"` key.
- `return result` — hand the dict back.

### Method `generate`

```python
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 1024,
        num_beams: int = 4,
        **kwargs,
    ) -> torch.Tensor:
```
**▸ What this block does:** declares the inference method — turn input IDs into
output text IDs.
- `max_length=1024` — cap on how long the output can be.
- `num_beams=4` — beam-search width (default 4).
- `**kwargs` — absorb extras. `-> torch.Tensor` — returns generated byte IDs.

```python
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )
```
**▸ What this block does:** call ByT5's built-in text generator with beam search and
return its output directly.
- `num_beams=num_beams` — explore that many candidate outputs, keep the best.
- `early_stopping=True` — stop once complete candidates are found.

---

## `delete_gate.py` — the compression engine

**Role:** The shared machine that scores every byte keep-or-delete and can remove
low-value bytes. Used by both compressed variants.

- **Input (forward):** `hidden_states`, `attention_mask`, optional `noise_scores`
- **Output:** `gate_outputs`, `keep_prob`, `kept_mask`, `deletion_rate`
- **Used by:** `fixed_compression_byt5.py`, `noise_adaptive_byt5.py`
- **Connects to:** the MrT5 reference implementation (see `ATTRIBUTIONS.md`)

### Where this fits in TAHIMIK
This is **the compression engine — the single reason MrT5 and TAHIMIK are faster
than plain ByT5.** Both compressed variants embed it (composition, not copy-paste):
MrT5 in fixed mode, TAHIMIK in noise-adaptive mode. In TAHIMIK, Step 2's
`cn·(n−navg)` shift is literally **where the thesis contribution lives**.

**What breaks without it:** no compression → no efficiency story → no thesis.

**Example** (the thesis claim, which a test in `tests/` checks)
```
same batch of hidden states, gate in eval mode:
  noise_scores = 0.05 (clean)  → deletion_rate ≈ 0.50   (compress hard)
  noise_scores = 0.95 (noisy)  → deletion_rate ≈ 0.10   (preserve for correction)
```

**Terms & abbreviations in this file**
| Term | Full form / plain meaning |
|---|---|
| gate | the mechanism scoring each byte keep vs delete |
| soft deletion | mute a byte smoothly (keeps gradients; used in training) |
| hard deletion | physically remove a byte (used at inference for speed) |
| `k` | a large negative constant (−30) bounding the gate scores |
| keep_prob | keep probability in [0,1] (1 = keep, 0 = delete) |
| kept_mask | a 0/1 hard keep/delete decision per byte |
| deletion_rate | fraction of real bytes removed |
| Gumbel noise | small random noise added in training so the gate explores choices |
| `nn.Parameter` | a learned number (here, `cn`) |
| buffer | a stored, non-learned number (here, `noise_avg`) |
| `.clamp(min, max)` | force values to stay within a range |
| `.sum(dim=1)` | add up along the sequence dimension |
| `cumsum` | running cumulative total |
| `scatter_add_` / `gather` | tensor ops that place / pick values at given positions |
| `.detach()` | cut the gradient link |
| `.bool()` / `.long()` / `.float()` | convert a tensor to True/False / integer / decimal |
| `eps` | a tiny number added to avoid invalid math (e.g. `log(0)`) |
| fp16 | 16-bit floating point (half precision) |

### Imports

```python
import torch
import torch.nn as nn
from typing import Tuple, Optional
```
**▸ What this block does:** import PyTorch, the NN toolbox, and type hints.
- `Tuple` — hint for a fixed group of return values; `Optional[X]` = "X or None."

### Helper `gumbel_noise_like`

**▸ What this function does (whole thing):** generate random "Gumbel" noise shaped
like the input, used to jitter the gate scores during training so it explores both
keep and delete rather than locking in early. (Adapted from the MrT5 reference.)

```python
def gumbel_noise_like(x: torch.Tensor) -> torch.Tensor:
    """..."""
    eps = 3e-4 if x.dtype == torch.float16 else 1e-10
    uniform = torch.empty_like(x).uniform_(eps, 1 - eps)
    return -(-uniform.log()).log()
```
**▸ line by line:**
- `def gumbel_noise_like(x): ...` — takes a tensor `x` whose shape/type the noise
  should match.
- `eps = 3e-4 if x.dtype == torch.float16 else 1e-10` — a tiny guard value. `3e-4`
  (= 0.0003) under 16-bit (`float16`), `1e-10` otherwise — it keeps the logs below
  from hitting `log(0)` (invalid). `x.dtype` is the tensor's number type.
- `uniform = torch.empty_like(x).uniform_(eps, 1 - eps)` — make a same-shaped tensor
  (`empty_like`) and fill it in place (`uniform_`) with random values between `eps`
  and `1 - eps`.
- `return -(-uniform.log()).log()` — the standard formula converting uniform
  randomness into the **Gumbel** distribution (`.log()` = natural logarithm).

### Class `DeleteGate` — constructor

**Variables at a glance**
| Variable | Type | Holds |
|---|---|---|
| `hidden_dim` (param) | `int` | Size of each byte's vector (d_model) |
| `k` (param) | `float` (−30) | Bounds gate scores to [k, 0] |
| `noise_adaptive` (param) | `bool` | Turn TAHIMIK's noise shift on/off |
| `noise_avg_momentum` (param) | `float` (0.99) | EMA slowness |
| `use_gumbel_noise` (param) | `bool` | Whether to add Gumbel noise in training |
| `self.layer_norm` | `nn.LayerNorm` | Normalizer before scoring |
| `self.gate_linear` | `nn.Linear` | Collapses a byte vector → 1 score |
| `self.cn` | `nn.Parameter` | Learned strength of the noise shift |
| `self.noise_avg` | buffer | Running average of noise |
| `self.hard_threshold` | `float` (−15) | Keep/delete cutoff |

```python
class DeleteGate(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        k: float = -30.0,
        noise_adaptive: bool = True,
        noise_avg_momentum: float = 0.99,
        use_gumbel_noise: bool = True,
    ):
        super().__init__()
```
**▸ What this block does:** declare the gate and its options; run the parent setup.
- `hidden_dim` — the width of each byte vector. `k=-30.0` — the bound.
  `noise_adaptive=True` — the switch that enables TAHIMIK's shift.
  `noise_avg_momentum=0.99` — the EMA slowness. `use_gumbel_noise=True` — toggle.

```python
        self.k = k
        self.noise_adaptive = noise_adaptive
        self.noise_avg_momentum = noise_avg_momentum
        self.use_gumbel_noise = use_gumbel_noise
```
**▸ What this block does:** store all four settings on the object.

```python
        # ── Gate scoring layers (Equation 1 from MrT5) ─────────────────
        # G = k * sigmoid(LayerNorm(H) @ W + b)
        # This produces per-byte scores in [k, 0].
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.gate_linear = nn.Linear(hidden_dim, 1)
```
**▸ What this block does:** create the two learnable scoring layers (comments quote
the MrT5 formula they implement).
- `nn.LayerNorm(hidden_dim)` — normalizes each byte's vector (mean 0, variance 1)
  for numerical stability.
- `nn.Linear(hidden_dim, 1)` — a learned matrix mapping a byte's whole vector down
  to a single raw score.

```python
        # ── Noise-adaptive conditioning ─────────────────────────────────
        # cn is a learned scalar coefficient that controls how strongly
        # the noise score shifts the keep/delete decision.
        if noise_adaptive:
            self.cn = nn.Parameter(torch.tensor(1.0))

            # Running average of noise scores (not a learnable parameter —
            # updated via EMA during training, frozen during inference)
            self.register_buffer("noise_avg", torch.tensor(0.5))
```
**▸ What this block does:** only for TAHIMIK, create the learned shift-strength dial
and the stored noise average.
- `if noise_adaptive:` — MrT5 skips this whole block.
- `self.cn = nn.Parameter(torch.tensor(1.0))` — `cn`, a **learned** scalar starting
  at 1.0 (a `Parameter` gets updated by training).
- `self.register_buffer("noise_avg", torch.tensor(0.5))` — `noise_avg`, a **buffer**
  starting at 0.5: saved and moved to GPU with the model, but **not** learned;
  updated by hand via EMA.

```python
        # Hard deletion threshold: midpoint of the gate range [k, 0]
        self.hard_threshold = k / 2.0
```
**▸ What this block does:** set the hard keep/delete cutoff to −15 (the midpoint of
[−30, 0]).

### Method `forward`

**Variables at a glance**
| Variable | Type / shape | Holds |
|---|---|---|
| `hidden_states` (param) | `(batch, seq, hidden)` | Byte vectors at the gate |
| `attention_mask` (param) | `(batch, seq)` | 1=real, 0=pad |
| `noise_scores` (param) | `(batch,)` or None | Predicted noise per sentence |
| `normed` | `(batch, seq, hidden)` | Normalized vectors |
| `logits` | `(batch, seq, 1)` | Raw per-byte scores |
| `gate_outputs` | `(batch, seq, 1)` | Scores in [k, 0] |
| `keep_prob` | `(batch, seq)` | Smooth keep probability |
| `kept_mask` | `(batch, seq)` | Hard 0/1 keep decision |
| `deletion_rate` | `(batch,)` | Fraction deleted per sentence |

```python
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        noise_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
```
**▸ What this block does:** declare the forward pass; returns a 4-tuple of tensors.
- `noise_scores` is optional — only TAHIMIK passes it.

```python
        batch_size, seq_len, _ = hidden_states.shape
```
**▸ What this block does:** read the sizes from the input's shape.
- `hidden_states.shape` is `(batch, seq_len, hidden_dim)`.
- `batch_size, seq_len, _ = ...` — **unpacking**: name the first two, discard the
  third (hidden_dim) with `_`.

```python
        # ── Step 1: Compute raw gate scores ─────────────────────────────
        # LayerNorm → Linear → rescaled sigmoid in [k, 0]
        normed = self.layer_norm(hidden_states)
        logits = self.gate_linear(normed)  # (batch, seq, 1)
```
**▸ What this block does:** normalize each byte vector, then collapse it to one raw
score.
- `normed` — normalized vectors, same shape.
- `logits` — shape `(batch, seq, 1)`, one raw score per byte.

```python
        if self.training and self.use_gumbel_noise:
            logits = logits + gumbel_noise_like(logits)
```
**▸ What this block does:** during training only, jitter the scores so the gate
explores.
- `self.training` is True only after `model.train()`; at inference this is skipped,
  so the same input compresses identically every call.

```python
        # Rescaled sigmoid: k * sigmoid(logits), bounded in [k, 0]
        gate_outputs = self.k * torch.sigmoid(logits)
```
**▸ What this block does:** squash each score into [−30, 0].
- `torch.sigmoid(logits)` maps any number into (0,1); `* self.k` (−30) flips/scales
  it into (−30, 0). Score near **0 = keep**, near **−30 = delete**.

```python
        # ── Step 2: Apply noise-adaptive shift ──────────────────────────
        if self.noise_adaptive and noise_scores is not None:
            # Detach noise scores so gate gradients don't flow back
            # to the noise estimator (manuscript: "n is detached here")
            n_detached = noise_scores.detach()
```
**▸ What this block does:** only when noise-adaptive and a noise score was passed,
detach it (cut the gradient) so using it here can't train the estimator.

```python
            # Update running average during training
            if self.training:
                batch_avg = n_detached.mean()
                self.noise_avg = (
                    self.noise_avg_momentum * self.noise_avg
                    + (1 - self.noise_avg_momentum) * batch_avg
                )
```
**▸ What this block does:** during training, nudge the running noise average toward
this batch's average, slowly.
- `batch_avg = n_detached.mean()` — this batch's average noise.
- The assignment is an **EMA**: `0.99*old + 0.01*new` — `noise_avg` drifts gradually
  and stays stable.

```python
            # Shift = cn * (n - navg)
            # Shape: (batch_size,) → (batch_size, 1, 1) for broadcasting
            shift = self.cn * (n_detached - self.noise_avg)
            shift = shift.unsqueeze(1).unsqueeze(2)
```
**▸ What this block does:** compute the per-sentence shift and reshape it to
broadcast over byte positions.
- `self.cn * (n_detached - self.noise_avg)` — positive when a sentence is noisier
  than average, negative when cleaner; `cn` scales the effect. Shape `(batch,)`.
- `.unsqueeze(1).unsqueeze(2)` — insert two size-1 dims → `(batch, 1, 1)` so it adds
  across all positions of `gate_outputs` `(batch, seq, 1)`.

```python
            # Positive shift (noisy) → scores closer to 0 → keep more
            # Negative shift (clean) → scores closer to k → delete more
            gate_outputs = gate_outputs + shift

            # Clamp back to valid range [k, 0]
            gate_outputs = gate_outputs.clamp(min=self.k, max=0.0)
```
**▸ What this block does:** add the shift (noisier → keep more; cleaner → delete
more), then `clamp` any overshoot back into [−30, 0].

```python
        # ── Step 3: Keep probability (differentiable) ───────────────────
        # Map the gate score from [k, 0] onto a keep probability in [0, 1]:
        #   gate = 0  → p = 1  (definitely keep)
        #   gate = k  → p = 0  (definitely delete)
        # This is the differentiable quantity the rate loss is computed from.
        keep_prob = 1.0 + (gate_outputs.squeeze(-1) / abs(self.k))
        keep_prob = keep_prob * attention_mask.float()  # padding never counts
```
**▸ What this block does:** convert the [−30, 0] score into a smooth [0,1] keep
probability, and zero out padding.
- `gate_outputs.squeeze(-1)` — drop the trailing size-1 dim → `(batch, seq)`.
- `1.0 + (score / abs(k))` — `abs(k)` = 30; so 0 → 1.0, −15 → 0.5, −30 → 0.0.
- `* attention_mask.float()` — `.float()` turns 1/0 into 1.0/0.0; multiplying forces
  **padding positions to 0**. This smooth value is what the rate loss trains on.

```python
        # ── Step 4: Hard keep/delete decision ───────────────────────────
        # Used for physical deletion at inference and for reporting the true
        # rate. Non-differentiable by construction (it is a threshold).
        kept_mask = (gate_outputs.squeeze(-1) > self.hard_threshold).float()
        kept_mask = kept_mask * attention_mask.float()
```
**▸ What this block does:** make a 0/1 keep decision per byte, for physical removal
and true-rate reporting.
- `(... > self.hard_threshold)` — True where score > −15 (keep), else False.
- `.float()` — True/False → 1.0/0.0.
- `* attention_mask.float()` — zero out padding.

```python
        # ── Step 5: Deletion rate ───────────────────────────────────────
        # real_tokens excludes padding, so the rate is always a genuine
        # fraction of the sentence in [0, 1].
        real_tokens = attention_mask.float().sum(dim=1).clamp(min=1.0)
```
**▸ What this block does:** count the real (non-padding) bytes per sentence, the
denominator for the rate.
- `.sum(dim=1)` — add across the sequence. `.clamp(min=1.0)` — avoid divide-by-zero.

```python
        if self.training:
            # Soft (differentiable) rate — this is what makes L_rate able to
            # push the gate toward its target. Using the hard mask here would
            # produce a constant with no gradient.
            deletion_rate = 1.0 - (keep_prob.sum(dim=1) / real_tokens)
        else:
            # Hard rate — the compression actually achieved at inference.
            deletion_rate = 1.0 - (kept_mask.sum(dim=1) / real_tokens)
```
**▸ What this block does:** compute the deletion rate — from the *smooth* `keep_prob`
during training (so gradients flow and L_rate can push the gate), from the *hard*
mask at inference (the real compression achieved).

```python
        return gate_outputs, keep_prob, kept_mask, deletion_rate
```
**▸ What this block does:** return the four tensors the models consume.

### Method `apply_hard_deletion` (inference only)

**▸ What this method does (whole function):** physically remove deleted bytes and
left-pack the survivors into a shorter sequence (padding the rest), for the whole
batch at once with fast tensor ops. **This is where inference gets faster.**

```python
    def apply_hard_deletion(
        self,
        hidden_states: torch.Tensor,
        kept_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
```
**▸ What this block does:** declare the method; returns (compressed states, new
mask).

```python
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device
```
**▸ What this block does:** read the sizes and remember which device (GPU/CPU) to
build new tensors on.

```python
        keep = kept_mask.bool()
        kept_counts = keep.sum(dim=1)
```
**▸ What this block does:** turn the mask into True/False and count how many bytes
each sentence keeps.
- `.bool()` — 1/0 → True/False. `keep.sum(dim=1)` — kept-count per sentence.

```python
        # At least one position, so the decoder always receives something even
        # if the gate deleted an entire sentence.
        new_seq_len = max(int(kept_counts.max().item()), 1)
```
**▸ What this block does:** set the new (shorter) length to the largest kept-count,
but never below 1.
- `.max()` — biggest count; `.item()` — as a plain Python int; `max(..., 1)` —
  guarantee at least one position so generation always has something to attend to.

```python
        # cumsum over the keep mask gives each kept position its destination
        # index; deleted positions repeat the previous index and contribute a
        # source index of 0, which scatter_add_ then adds harmlessly.
        target_pos = (torch.cumsum(keep.long(), dim=1) - 1).clamp(min=0)
```
**▸ What this block does:** compute, for each byte, its destination slot in the
shortened sequence.
- `keep.long()` — True/False → 1/0. `torch.cumsum(..., dim=1)` — a **running total**
  along the sequence, so each kept byte gets an increasing index.
- `- 1` — make it 0-based. `.clamp(min=0)` — keep it valid at the start.

```python
        positions = torch.arange(seq_len, device=device).expand(batch_size, -1)
        positions = positions * keep.long()
```
**▸ What this block does:** build the list of original source positions, zeroing out
deleted ones.
- `torch.arange(seq_len, ...)` — `[0,1,...,seq_len-1]`.
- `.expand(batch_size, -1)` — repeat that row for every sentence.
- `* keep.long()` — deleted positions become 0 (they'll be ignored below).

```python
        src_positions = torch.zeros(
            batch_size, new_seq_len, device=device, dtype=torch.long
        )
        src_positions.scatter_add_(1, target_pos, positions)
```
**▸ What this block does:** for each destination slot, record which original
position feeds it.
- `torch.zeros(batch_size, new_seq_len, ...)` — an empty destination-index table.
- `scatter_add_(1, target_pos, positions)` — scatter each source position into its
  destination slot (from `target_pos`); an in-place op (`_`).

```python
        compressed_states = torch.gather(
            hidden_states, 1,
            src_positions.unsqueeze(-1).expand(-1, -1, hidden_dim),
        )
```
**▸ What this block does:** copy the kept byte vectors into the shortened tensor.
- `torch.gather(hidden_states, 1, index)` — pick, along the sequence dim, the vectors
  at the recorded source positions.
- `src_positions.unsqueeze(-1).expand(-1, -1, hidden_dim)` — reshape the index table
  to `(batch, new_seq_len, hidden_dim)` so it selects whole vectors.

```python
        # Positions beyond a row's kept count are padding.
        new_attention_mask = (
            torch.arange(new_seq_len, device=device).expand(batch_size, -1)
            < kept_counts.unsqueeze(1)
        ).to(kept_mask.dtype)
```
**▸ What this block does:** build the new mask — real where the slot index is below
that sentence's kept-count, padding beyond.
- `arange(new_seq_len) < kept_counts` — a True/False comparison per slot.
- `.to(kept_mask.dtype)` — convert True/False back to the mask's number type.

```python
        # Zero the padded tail so it carries no stale hidden state.
        compressed_states = compressed_states * new_attention_mask.unsqueeze(-1).to(
            compressed_states.dtype
        )
```
**▸ What this block does:** multiply by the new mask so the padded tail is zeroed
(no leftover vectors leak into the decoder).

```python
        return compressed_states, new_attention_mask
```
**▸ What this block does:** return the shortened states and their mask.

---

## `fixed_compression_byt5.py` — MrT5

**Role:** ByT5 with a delete gate that compresses **every** sentence at a fixed
rate (50%), regardless of noise. The efficiency baseline.

- **Input (forward):** `input_ids`, `attention_mask`, optional `labels`
- **Output:** dict with `loss`, `logits`, `gate_outputs`, `keep_prob`,
  `kept_mask`, `deletion_rate`, `fixed_deletion_target`
- **Used by:** the scripts + backend. **Contains** a `DeleteGate(noise_adaptive=False)`.
- **Connects to:** `delete_gate.py`, `configs/mrt5_config.py`

### Where this fits in TAHIMIK
The **efficiency baseline — "compression without intelligence."** It proves byte
compression works (faster than ByT5) but applies the **same** 50% deletion to every
sentence. TAHIMIK's value is proven by **beating this specific model** on noisy
sentences at similar speed — showing the *adaptivity*, not just the compression,
helped.

**What breaks without it:** you couldn't separate "compression is faster" from
"being smart about which sentences to compress matters." This is the control that
isolates the contribution.

**Terms & abbreviations in this file**
| Term | Full form / plain meaning |
|---|---|
| encoder split | run the first few encoder layers, insert the gate, then the rest |
| `embed_tokens` | the layer turning byte IDs into vectors |
| extended attention mask | HF's internal additive mask (0 for real, huge negative for pad) |
| gate_bias | the gate score added onto attention logits (soft deletion) |
| `_shift_right` | prepares the decoder's input (standard seq2seq trick) |
| cross-entropy | the loss for predicting the correct next byte |
| `.view(...)` | reshape a tensor without copying data |
| `BaseModelOutput` | HF wrapper so `generate()` accepts a precomputed encoder output |

### Imports + constructor

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from typing import Dict, Optional

from src.models.delete_gate import DeleteGate
```
**▸ What this block does:** import PyTorch, ByT5 + tokenizer, HF's `BaseModelOutput`
wrapper (needed by `generate`), type hints, and — the key connection — the
`DeleteGate` class from `delete_gate.py`.

```python
class FixedCompressionByT5(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model = T5ForConditionalGeneration.from_pretrained(
            config.model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
```
**▸ What this block does:** load ByT5 and its tokenizer (same as the baseline).

```python
        # Gate placement: after this encoder layer
        self.delete_gate_layer = config.delete_gate_layer
```
**▸ What this block does:** record where to split the encoder — after layer 3.

```python
        # The delete gate in non-adaptive mode (fixed rate)
        hidden_dim = self.model.config.d_model
        self.delete_gate = DeleteGate(
            hidden_dim=hidden_dim,
            k=config.gate_k,
            noise_adaptive=False,
            use_gumbel_noise=getattr(config, "use_gumbel_noise", True),
        )
```
**▸ What this block does:** create the gate in **fixed** mode.
- `hidden_dim = self.model.config.d_model` — ByT5's vector width, passed to the gate.
- `DeleteGate(..., noise_adaptive=False, ...)` — the same class as TAHIMIK but with
  the noise behavior **off** (this is how one class serves both variants).
- `getattr(config, "use_gumbel_noise", True)` — read the config's flag, defaulting to
  `True` if it's absent (`getattr(obj, name, default)`).

```python
        # Fixed deletion target for the rate loss
        self.fixed_deletion_target = config.fixed_deletion_target
```
**▸ What this block does:** store the fixed target rate (0.5) used by the rate loss.

### Helper `_run_encoder_layers`

**▸ What this method does (whole function):** run a *range* of the encoder's layers
on the hidden states, optionally adding the gate's score as a soft attention
penalty.

```python
    def _run_encoder_layers(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        start_layer: int,
        end_layer: int,
        gate_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
```
**▸ What this block does:** declare the helper; `start_layer`/`end_layer` pick the
range, `gate_bias` is the optional soft-deletion penalty.

```python
        encoder = self.model.encoder
        # Pass only the two positional arguments. The third parameter is
        # `device` in transformers 4.x but `dtype` in 5.x, so passing a device
        # positionally raises TypeError on 5.x. Two args works on both.
        extended_mask = encoder.get_extended_attention_mask(
            attention_mask, hidden_states.shape[:2]
        )
```
**▸ What this block does:** grab the encoder and convert the 0/1 mask into HF's
internal additive form.
- `encoder = self.model.encoder` — the encoder half of ByT5.
- `get_extended_attention_mask(mask, shape)` — produces a mask that's 0 for real
  positions and a huge negative for padding. The comment explains why only two
  arguments are passed (version compatibility between transformers 4.x and 5.x).
- `hidden_states.shape[:2]` — the `(batch, seq_len)` part of the shape.

```python
        if gate_bias is not None:
            # (batch, seq) -> (batch, 1, 1, seq) to broadcast over heads and
            # query positions, then add as a log-space attention penalty.
            extended_mask = extended_mask + gate_bias[:, None, None, :]
```
**▸ What this block does:** if a gate bias was passed (soft deletion), **add** it
onto the mask.
- `gate_bias[:, None, None, :]` — reshape `(batch, seq)` → `(batch, 1, 1, seq)` so it
  broadcasts across attention heads and query positions.
- **Adding, not multiplying**, is critical: HF turns a binary mask into
  `(1-mask)*−1e34`, so multiplying a soft 0.99997 keep would become a hard delete
  and kill the gradient. Adding keeps it soft.

```python
        for i in range(start_layer, end_layer):
            layer = encoder.block[i]
            layer_output = layer(
                hidden_states,
                attention_mask=extended_mask,
            )
            hidden_states = layer_output[0]

        return hidden_states
```
**▸ What this block does:** run each encoder layer in the chosen range, feeding each
layer's output into the next, then return the result.
- `for i in range(start_layer, end_layer):` — loop the layer indices (e.g. 0,1,2).
- `encoder.block[i]` — the i-th encoder layer.
- `layer(hidden_states, attention_mask=extended_mask)` — run it.
- `layer_output[0]` — the hidden states are the first element of the layer's output
  tuple.
- `return hidden_states` — the transformed vectors after the range.

### Method `forward`

```python
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        encoder = self.model.encoder
        num_layers = len(encoder.block)
```
**▸ What this block does:** declare the forward pass and grab the encoder plus its
total layer count.
- `num_layers = len(encoder.block)` — how many encoder layers there are (used as the
  end of the post-gate range).

```python
        # ── Step 1: Embedding ───────────────────────────────────────────
        inputs_embeds = encoder.embed_tokens(input_ids)
        hidden_states = encoder.dropout(inputs_embeds)
```
**▸ What this block does:** turn byte IDs into vectors, then apply dropout.
- `embed_tokens(input_ids)` — look up a learned vector for each byte ID.
- `encoder.dropout(...)` — randomly zero some values (training regularization).

```python
        # ── Step 2: Pre-gate encoder layers ─────────────────────────────
        hidden_states = self._run_encoder_layers(
            hidden_states, attention_mask,
            start_layer=0,
            end_layer=self.delete_gate_layer,
        )
```
**▸ What this block does:** run the pre-gate layers (0 up to layer 3) to build
context before scoring bytes.

```python
        # ── Step 3: Delete gate ─────────────────────────────────────────
        gate_outputs, keep_prob, kept_mask, deletion_rate = self.delete_gate(
            hidden_states, attention_mask
        )
```
**▸ What this block does:** score the bytes with the gate. No `noise_scores` is
passed, so it stays fixed-rate.

```python
        # ── Step 4/5: Apply deletion, then post-gate encoder layers ─────
        if self.training:
            # SOFT deletion: gate score added to attention logits as a
            # log-space penalty. Differentiable; length unchanged.
            modified_mask = attention_mask
            hidden_states = self._run_encoder_layers(
                hidden_states, attention_mask,
                start_layer=self.delete_gate_layer,
                end_layer=num_layers,
                gate_bias=gate_outputs.squeeze(-1),
            )
```
**▸ What this block does:** the **training** branch — keep full length, pass the gate
score as `gate_bias` so disliked bytes become progressively harder to attend to
(differentiable).
- `modified_mask = attention_mask` — the decoder later uses this; unchanged here.
- `gate_bias=gate_outputs.squeeze(-1)` — drop the size-1 dim → `(batch, seq)`.

```python
        else:
            # HARD deletion: bytes physically removed — the actual speedup.
            hidden_states, modified_mask = self.delete_gate.apply_hard_deletion(
                hidden_states, kept_mask
            )
            hidden_states = self._run_encoder_layers(
                hidden_states, modified_mask,
                start_layer=self.delete_gate_layer,
                end_layer=num_layers,
            )
```
**▸ What this block does:** the **inference** branch — physically remove bytes (a
shorter, faster sequence), then run the remaining layers on it.
- `apply_hard_deletion(...)` — returns the shortened states + their new mask
  (`modified_mask`).

```python
        # Final layer norm
        hidden_states = encoder.final_layer_norm(hidden_states)
        hidden_states = encoder.dropout(hidden_states)
```
**▸ What this block does:** apply the encoder's final normalization and dropout.

```python
        if self.training:
            # Suppress deleted bytes in what the decoder sees. Applied after
            # the final norm, since RMS normalisation would undo it.
            hidden_states = hidden_states * keep_prob.unsqueeze(-1)
```
**▸ What this block does:** in training, multiply by `keep_prob` so the decoder also
sees deleted bytes suppressed — applied **after** the norm on purpose (T5's
normalization would otherwise cancel it).
- `keep_prob.unsqueeze(-1)` — `(batch, seq)` → `(batch, seq, 1)` to broadcast over
  the hidden dimension.

```python
        # ── Step 6: Decoder ─────────────────────────────────────────────
        encoder_outputs = (hidden_states,)
```
**▸ What this block does:** wrap the encoder states in a 1-element tuple.
- *(Note: this local `encoder_outputs` is not actually used below — it's the dead
  assignment recorded in `FINDINGS.md` under minor items.)*

```python
        if labels is not None:
            decoder_input_ids = self.model._shift_right(labels)
            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=modified_mask,
            )
            sequence_output = decoder_outputs[0]
            lm_logits = self.model.lm_head(sequence_output)
```
**▸ What this block does:** the training/validation branch — run the decoder and
produce output scores.
- `_shift_right(labels)` — shift the target right by one to make the decoder's input
  (it predicts each byte from the previous ones).
- `self.model.decoder(...)` — run the decoder, attending to the (possibly shortened)
  encoder states via `encoder_attention_mask=modified_mask`.
- `decoder_outputs[0]` — the decoder's hidden states (`sequence_output`).
- `self.model.lm_head(sequence_output)` — the final layer turning those into
  vocabulary scores (`lm_logits`).

```python
            result = {
                "logits": lm_logits,
                "gate_outputs": gate_outputs,
                "keep_prob": keep_prob,
                # Emitted so the loss uses THIS variant's configured
                # target rather than silently falling back to 0.5.
                "fixed_deletion_target": self.fixed_deletion_target,
                "kept_mask": kept_mask,
                "deletion_rate": deletion_rate,
                "encoder_last_hidden_state": hidden_states,
            }
```
**▸ What this block does:** build the output dict with everything the loss and logs
need. The `fixed_deletion_target` is included so `TAHIMIKLoss` uses this variant's
configured 0.5 rather than a hardcoded default.

```python
            # Cross-entropy loss (computed by the external loss module)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            result["loss"] = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1),
            )

            return result
```
**▸ What this block does:** compute the cross-entropy loss and return the dict.
- `nn.CrossEntropyLoss(ignore_index=-100)` — the loss; `ignore_index=-100` is where
  the `-100` padding labels from `dataset.py` are skipped.
- `lm_logits.view(-1, lm_logits.size(-1))` — reshape to `(all_positions, vocab)`;
  `labels.view(-1)` → `(all_positions,)` so the loss can compare them.
- `return result` — done for the training branch.

```python
        else:
            decoder_outputs = self.model.decoder(
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=modified_mask,
            )
            lm_logits = self.model.lm_head(decoder_outputs[0])

            return {
                "logits": lm_logits,
                "gate_outputs": gate_outputs,
                "keep_prob": keep_prob,
                "fixed_deletion_target": self.fixed_deletion_target,
                "kept_mask": kept_mask,
                "deletion_rate": deletion_rate,
                "encoder_last_hidden_state": hidden_states,
            }
```
**▸ What this block does:** the no-labels branch — run the decoder without shifting
labels (there are none), compute logits, and return the same-shaped dict minus the
loss.

### Method `generate`

**▸ What this method does (whole function):** the inference path the API calls —
embed → pre-gate → gate → hard-delete → post-gate → beam search.

```python
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 1024,
        num_beams: int = 4,
        **kwargs,
    ) -> torch.Tensor:
        self.eval()

        encoder = self.model.encoder
        num_layers = len(encoder.block)
```
**▸ What this block does:** declare `generate`, force eval mode, and grab the encoder
+ layer count.
- `self.eval()` — switch to inference behavior (hard deletion, no dropout, no Gumbel
  noise).

```python
        # Embedding + pre-gate layers
        inputs_embeds = encoder.embed_tokens(input_ids)
        hidden_states = encoder.dropout(inputs_embeds)
        hidden_states = self._run_encoder_layers(
            hidden_states, attention_mask,
            start_layer=0,
            end_layer=self.delete_gate_layer,
        )
```
**▸ What this block does:** embed the bytes and run the pre-gate layers (same as
`forward`).

```python
        # Delete gate (hard mode in eval)
        gate_outputs, keep_prob, kept_mask, deletion_rate = self.delete_gate(
            hidden_states, attention_mask
        )
        hidden_states, compressed_mask = self.delete_gate.apply_hard_deletion(
            hidden_states, kept_mask
        )
```
**▸ What this block does:** score the bytes, then physically remove the deleted ones,
producing shortened states + `compressed_mask`.

```python
        # Post-gate layers + final norm
        hidden_states = self._run_encoder_layers(
            hidden_states, compressed_mask,
            start_layer=self.delete_gate_layer,
            end_layer=num_layers,
        )
        hidden_states = encoder.final_layer_norm(hidden_states)
```
**▸ What this block does:** run the remaining encoder layers on the shortened
sequence, then the final normalization.

```python
        # Beam-search decode.
        # encoder_outputs must be a BaseModelOutput, not a bare tuple —
        # generate() reads .last_hidden_state from it to size the beams. Given
        # a tuple it cannot find the encoder states, falls through to the
        # "no input_ids" branch, and raises:
        #   ValueError: `bos_token_id` has to be defined when no `input_ids`...
        return self.model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden_states),
            attention_mask=compressed_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )
```
**▸ What this block does:** decode with beam search over the compressed encoder
output, returning output byte IDs.
- `BaseModelOutput(last_hidden_state=hidden_states)` — the comment explains why this
  wrapper is required (a bare tuple triggers a `bos_token_id` error).
- `attention_mask=compressed_mask` — tells the decoder which compressed positions are
  real. `num_beams`, `early_stopping=True` — beam search settings.

---

## `noise_estimator.py` — predicting the noise

**Role:** A tiny network that *guesses* a sentence's noise from the encoder vectors
(at inference there's no clean reference to compute n\* from). Its guess conditions
TAHIMIK's gate.

- **Input (forward):** `hidden_states`, `attention_mask`
- **Output:** `noise_scores` — one number in [0,1] per sentence
- **Used by:** `noise_adaptive_byt5.py`. Trained by the **L_NE** loss.
- **Connects to:** `configs/tahimik_config.py` (`noise_estimator_hidden_dim`)

### Where this fits in TAHIMIK
The **"sensor" half of the contribution.** At training you know the true noise (n\*);
at inference there's no clean reference, so the model must *guess* it. This MLP is
that guesser. Without it, TAHIMIK collapses into MrT5.

**Example**
```
estimator(hidden_states, attention_mask)
  hidden_states shape (8, 1024, 512)   → noise_scores shape (8,)
  e.g. tensor([0.08, 0.71, 0.15, ...])  ← sentence 2 looks noisy
```

**Terms & abbreviations in this file**
| Term | Full form / plain meaning |
|---|---|
| MLP | multi-layer perceptron — a couple of linear layers with an activation between |
| mean-pooling | averaging many vectors into one summary vector |
| `nn.Sequential` | a container that runs its layers in order |
| GELU / dropout / sigmoid | smooth activation / random zeroing / squash into (0,1) |
| `intermediate_dim` | the MLP's hidden-layer width (256) |

### Imports + constructor

```python
import torch
import torch.nn as nn
```
**▸ What this block does:** import PyTorch and the NN toolbox.

```python
class NoiseEstimator(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int = 256):
        super().__init__()
```
**▸ What this block does:** declare the estimator; `hidden_dim` = the encoder width,
`intermediate_dim` = the MLP's hidden width (256).

```python
        # Two-layer MLP: hidden_dim → intermediate_dim → 1
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_dim, 1),
        )
```
**▸ What this block does:** build the MLP as an ordered stack.
- `nn.Sequential(...)` — runs the listed layers in order.
- `nn.Linear(hidden_dim, intermediate_dim)` — shrink from d_model to 256.
- `nn.GELU()` — a smooth activation adding non-linearity.
- `nn.Dropout(0.1)` — randomly zero 10% of values during training.
- `nn.Linear(intermediate_dim, 1)` — map 256 → a single number.

```python
        # Final sigmoid maps the output to [0, 1]
        self.sigmoid = nn.Sigmoid()
```
**▸ What this block does:** create the sigmoid that squashes the final number into
[0,1] (matching n\*'s range).

### Method `forward`

```python
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
```
**▸ What this block does:** declare the forward pass — hidden states in, one noise
score per sentence out.

```python
        # Expand mask for broadcasting with hidden_dim
        # Shape: (batch_size, seq_len, 1)
        mask_expanded = attention_mask.unsqueeze(-1).float()
```
**▸ What this block does:** reshape the mask so it can multiply the hidden states.
- `.unsqueeze(-1)` — `(batch, seq)` → `(batch, seq, 1)`. `.float()` — 1/0 → 1.0/0.0.

```python
        # Mean-pool over non-padding positions
        # Numerator: sum of hidden states at real positions
        summed = (hidden_states * mask_expanded).sum(dim=1)
```
**▸ What this block does:** sum the byte vectors, ignoring padding.
- `hidden_states * mask_expanded` — zeros out padding positions.
- `.sum(dim=1)` — add across the sequence → `(batch, hidden_dim)`.

```python
        # Denominator: number of real tokens per sentence
        # Clamp to avoid division by zero for fully-padded sequences
        counts = mask_expanded.sum(dim=1).clamp(min=1.0)
```
**▸ What this block does:** count the real bytes per sentence (the averaging
denominator), clamped to ≥1 to avoid divide-by-zero.

```python
        # Sentence vector: (batch_size, hidden_dim)
        sentence_vector = summed / counts
```
**▸ What this block does:** divide sum by count → the **mean-pooled** sentence
vector, one per sentence.

```python
        # MLP + sigmoid → noise score: (batch_size, 1) → (batch_size,)
        noise_scores = self.sigmoid(self.network(sentence_vector)).squeeze(-1)

        return noise_scores
```
**▸ What this block does:** run the MLP, squash with sigmoid, drop the size-1 dim,
and return one score per sentence.
- `self.network(sentence_vector)` — the MLP → shape `(batch, 1)`.
- `self.sigmoid(...)` — squash into [0,1].
- `.squeeze(-1)` — `(batch, 1)` → `(batch,)`.

---

## `noise_adaptive_byt5.py` — TAHIMIK (the contribution)

**Role:** MrT5 plus the noise estimator. Predicts each sentence's noise and uses it
to compress clean sentences hard, noisy ones gently.

- **Input (forward):** `input_ids`, `attention_mask`, optional `labels`, optional
  `noise_level` (ground-truth n\* for the estimator's loss)
- **Output:** dict with `loss`, `ne_loss`, `logits`, `noise_scores`,
  `gate_outputs`, `keep_prob`, `kept_mask`, `deletion_rate`, `target_deletion_rate`
- **Used by:** the scripts + backend. **Contains** a `NoiseEstimator` **and** a
  `DeleteGate(noise_adaptive=True)`.
- **Connects to:** `noise_estimator.py`, `delete_gate.py`, `tahimik_config.py`

### Where this fits in TAHIMIK
**This is the contribution — the whole thesis in one class.** It assembles the two
halves (sensor + gate) onto ByT5: run early layers → estimate `n` → gate shifts by
`n` → run the rest → decode. It's the file the panel scrutinizes hardest.

**What breaks without it:** there is no proposed model — just two baselines.

**Example** (end-to-end intent)
```
clean input  "salamat po"         → estimator n≈0.05 → gate deletes ~50% → fast
noisy input  "slmt p0 grabeee 😭"  → estimator n≈0.70 → gate deletes ~15% → accurate
```

**Terms & abbreviations in this file**
| Term | Full form / plain meaning |
|---|---|
| n\* | ground-truth noise (from `noise_label.py`), here the param `noise_level` |
| L_NE | noise-estimator loss (MSE between predicted `n` and `n*`) |
| MSE | mean squared error |
| `d_max` | maximum deletion fraction for a perfectly clean sentence (0.5) |
| target_deletion_rate | the adaptive per-sentence goal `d_max*(1−n)` |
| `mse_loss` | PyTorch's mean-squared-error function |
| `BaseModelOutput` | HF wrapper so `generate()` accepts a precomputed encoder output |

### Imports + constructor

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from typing import Dict, Optional

from src.models.noise_estimator import NoiseEstimator
from src.models.delete_gate import DeleteGate
```
**▸ What this block does:** import ByT5 + tokenizer + `BaseModelOutput` + type hints,
and — the two key connections — the `NoiseEstimator` and `DeleteGate` classes.

```python
class NoiseAdaptiveByT5(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model = T5ForConditionalGeneration.from_pretrained(
            config.model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
```
**▸ What this block does:** load ByT5 and its tokenizer (same as the other variants).

```python
        hidden_dim = self.model.config.d_model
        self.delete_gate_layer = config.delete_gate_layer
```
**▸ What this block does:** read ByT5's vector width and the encoder-split layer (3).

```python
        # ── Noise Estimator ─────────────────────────────────────────────
        self.noise_estimator = NoiseEstimator(
            hidden_dim=hidden_dim,
            intermediate_dim=config.noise_estimator_hidden_dim,
        )
```
**▸ What this block does:** build the noise estimator (the extra piece vs MrT5),
sized by the config's `noise_estimator_hidden_dim` (256).

```python
        # ── Delete Gate (noise-adaptive mode) ───────────────────────────
        self.delete_gate = DeleteGate(
            hidden_dim=hidden_dim,
            k=config.gate_k,
            noise_adaptive=True,
            noise_avg_momentum=config.noise_avg_momentum,
            use_gumbel_noise=getattr(config, "use_gumbel_noise", True),
        )
```
**▸ What this block does:** build the gate in **noise-adaptive** mode — the switch
that turns on Step 2's shift inside the gate.

```python
        # Maximum deletion fraction for the noise-adaptive rate target:
        # d_target(x_i) = d_max * (1 - n_i)
        self.d_max = config.d_max
```
**▸ What this block does:** store `d_max` (0.5) — the most a perfectly clean sentence
may be compressed; used to compute the adaptive target.

### Helper `_run_encoder_layers`

Identical to MrT5's version (same code, same purpose): run a range of encoder layers
and, if given, add the gate score to attention as a soft penalty. See the
**MrT5 `_run_encoder_layers`** section above for the full line-by-line — the source
is the same.

### Method `forward` (the full source)

**Variables at a glance**
| Variable | Type / shape | Holds |
|---|---|---|
| `noise_scores` | `(batch,)` | Predicted noise `n` per sentence |
| `target_deletion_rate` | `(batch,)` | Adaptive goal `d_max·(1−n)` |
| `modified_mask` | `(batch, seq')` | Mask for the decoder (full or compressed) |
| `result` | `dict` | The output dict |
| `ne_loss` | scalar tensor | L_NE (estimator's MSE) |

```python
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        noise_level: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
```
**▸ What this block does:** declare the forward pass. Note the extra `noise_level`
parameter — the ground-truth n\* used to train the estimator.

```python
        encoder = self.model.encoder
        num_layers = len(encoder.block)

        # ── Step 1: Embedding ───────────────────────────────────────────
        inputs_embeds = encoder.embed_tokens(input_ids)
        hidden_states = encoder.dropout(inputs_embeds)
```
**▸ What this block does:** grab the encoder + layer count, embed the byte IDs, apply
dropout (same as MrT5).

```python
        # ── Step 2: Pre-gate encoder layers ─────────────────────────────
        hidden_states = self._run_encoder_layers(
            hidden_states, attention_mask,
            start_layer=0,
            end_layer=self.delete_gate_layer,
        )
```
**▸ What this block does:** run the pre-gate layers (0–2) to build context.

```python
        # ── Step 3: Noise estimation ────────────────────────────────────
        noise_scores = self.noise_estimator(hidden_states, attention_mask)
```
**▸ What this block does:** **the first addition over MrT5** — predict the noise `n`
from the current hidden states (calls `noise_estimator.py`).

```python
        # ── Step 4: Delete gate (noise-conditioned) ─────────────────────
        gate_outputs, keep_prob, kept_mask, deletion_rate = self.delete_gate(
            hidden_states, attention_mask, noise_scores=noise_scores
        )
```
**▸ What this block does:** **the second addition** — pass `noise_scores` into the
gate so its Step-2 shift activates.

```python
        # Per-sentence adaptive deletion target:
        # d_target(x_i) = d_max * (1 - n_i)
        # Detach noise_scores here so the target doesn't create a gradient
        # path through the noise estimator.
        target_deletion_rate = self.d_max * (1.0 - noise_scores.detach())
```
**▸ What this block does:** **the third addition** — compute each sentence's adaptive
target rate.
- `1.0 - noise_scores` — clean (n≈0) → ≈1; noisy (n≈1) → ≈0.
- `* self.d_max` (0.5) — clean → target ≈ 0.5 (compress hard); noisy → ≈ 0 (preserve).
- `.detach()` — again prevents a gradient backdoor into the estimator.

```python
        # ── Step 5/6: Apply deletion, then post-gate encoder layers ─────
        if self.training:
            # SOFT deletion. The gate score is added to the attention logits
            # as a log-space penalty, so a byte the gate wants to drop becomes
            # progressively harder to attend to while staying differentiable.
            # The sequence length is unchanged.
            modified_mask = attention_mask
            hidden_states = self._run_encoder_layers(
                hidden_states, attention_mask,
                start_layer=self.delete_gate_layer,
                end_layer=num_layers,
                gate_bias=gate_outputs.squeeze(-1),
            )
```
**▸ What this block does:** the **training** branch — soft deletion (same mechanism as
MrT5): keep full length, add the gate score as an attention penalty.

```python
        else:
            # HARD deletion: bytes are physically removed, which is where the
            # computational saving comes from.
            hidden_states, modified_mask = self.delete_gate.apply_hard_deletion(
                hidden_states, kept_mask
            )
            hidden_states = self._run_encoder_layers(
                hidden_states, modified_mask,
                start_layer=self.delete_gate_layer,
                end_layer=num_layers,
            )
```
**▸ What this block does:** the **inference** branch — physically remove bytes, then
run the remaining layers on the shortened sequence.

```python
        # Final layer norm
        hidden_states = encoder.final_layer_norm(hidden_states)
        hidden_states = encoder.dropout(hidden_states)
```
**▸ What this block does:** apply the encoder's final normalization and dropout.

```python
        if self.training:
            # Suppress deleted bytes in what the decoder cross-attends to.
            # Applied AFTER the final norm — T5 uses RMS normalisation, which
            # would otherwise rescale the suppression straight back out.
            hidden_states = hidden_states * keep_prob.unsqueeze(-1)
```
**▸ What this block does:** in training, scale by `keep_prob` after the norm so the
decoder also sees deleted bytes suppressed.

```python
        # ── Step 7: Decoder ─────────────────────────────────────────────
        result = {
            "noise_scores": noise_scores,
            "gate_outputs": gate_outputs,
            "keep_prob": keep_prob,
            "kept_mask": kept_mask,
            "deletion_rate": deletion_rate,
            "target_deletion_rate": target_deletion_rate,
            "encoder_last_hidden_state": hidden_states,
        }
```
**▸ What this block does:** build the output dict. Note it additionally carries
`noise_scores` and `target_deletion_rate` (vs MrT5) so the loss can build L_NE and
the *adaptive* L_rate.

```python
        if labels is not None:
            decoder_input_ids = self.model._shift_right(labels)
            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=modified_mask,
            )
            sequence_output = decoder_outputs[0]
            lm_logits = self.model.lm_head(sequence_output)

            result["logits"] = lm_logits
```
**▸ What this block does:** the training branch — shift labels, run the decoder,
compute logits, and store them under `"logits"` (same decoder wiring as MrT5).

```python
            # Cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            result["loss"] = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1),
            )
```
**▸ What this block does:** compute the cross-entropy accuracy loss (ignoring `-100`
padding) and store it under `"loss"`.

```python
            # Noise estimator loss: MSE(predicted_n, ground_truth_n*)
            if noise_level is not None:
                ne_loss = nn.functional.mse_loss(noise_scores, noise_level)
                result["ne_loss"] = ne_loss
```
**▸ What this block does:** compute **L_NE** — the estimator's error — when the
ground-truth n\* was supplied.
- `nn.functional.mse_loss(noise_scores, noise_level)` — mean squared error between
  the predicted `n` and the true `n*`. This is the *only* signal that trains the
  estimator. Stored as `"ne_loss"` for the loss module to combine.

```python
        else:
            decoder_outputs = self.model.decoder(
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=modified_mask,
            )
            lm_logits = self.model.lm_head(decoder_outputs[0])
            result["logits"] = lm_logits

        return result
```
**▸ What this block does:** the no-labels branch — run the decoder without labels,
store the logits, and return the dict.

### Method `generate`

TAHIMIK's `generate` is the **same shape as MrT5's** (embed → pre-gate → estimate
noise → gate → hard-delete → post-gate → final norm → beam search over a
`BaseModelOutput`). The only difference from MrT5's `generate` is the extra
`noise_scores = self.noise_estimator(...)` call before the gate, and passing
`noise_scores=noise_scores` into the gate. See the **MrT5 `generate`** section above
for the full line-by-line; the wrapper/beam-search details are identical.

---

## `__init__.py`

```python
from src.models.noise_estimator import NoiseEstimator
from src.models.delete_gate import DeleteGate
from src.models.byt5_baseline import ByT5Baseline
from src.models.fixed_compression_byt5 import FixedCompressionByT5
from src.models.noise_adaptive_byt5 import NoiseAdaptiveByT5
```
**▸ What this block does:** re-export the five public classes so other code can do
`from src.models import NoiseAdaptiveByT5`, etc. Each line pulls one class up to the
package level.

- **Inputs:** none (runs on import)
- **Outputs:** the five names above, importable from `src.models`
