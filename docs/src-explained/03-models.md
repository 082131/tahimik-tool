# `src/models/` — the three neural networks (exhaustive line-by-line)

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

> **Format reminder:** each block starts with **▸ What this block does**, then
> breaks down every line, variable, and technical term.

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
accuracy that MrT5 or TAHIMIK loses is measured as a drop from this model. It
processes every byte, so it should be the most accurate and the slowest — the top
of the accuracy scale and the bottom of the speed scale.

**What breaks without it:** you'd have no baseline to say "TAHIMIK keeps X% of the
ceiling's accuracy while running Y× faster." The whole comparison needs this anchor.
It's also the simplest variant, so it's the right place to first understand the
`forward` → dict → loss shape the other two mimic.

**Terms & abbreviations in this file**
| Term | Full form / plain meaning |
|---|---|
| `T5ForConditionalGeneration` | Hugging Face's ready-made ByT5 (encoder+decoder+output head) |
| `.from_pretrained(name)` | download a model/tokenizer's trained weights by name |
| `**kwargs` | "keyword arguments" — a catch-all for extra named parameters |
| `.size(0)` | the length of a tensor's first dimension (here, the batch size) |
| device | where a tensor lives — GPU or CPU |
| `early_stopping` | stop beam search once good candidates finish |

### Module-level imports

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from typing import Dict, Optional
```
**▸ What this block does:** imports PyTorch, its NN toolbox (aliased `nn`), the
ByT5 model + tokenizer loaders, and type hints.
- `import torch.nn as nn` — `as nn` gives it a short **alias**, so `nn.Linear` etc.
- `Optional[X]` — hint meaning "an X **or** `None`."

### Class `ByT5Baseline`

```python
class ByT5Baseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = T5ForConditionalGeneration.from_pretrained(config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
```
**▸ What this block does:** builds the model — downloads pretrained ByT5 and its
tokenizer, named by the config.
- `class ByT5Baseline(nn.Module):` — a network; inherits PyTorch's `nn.Module`.
- `super().__init__()` — run the parent class's setup first (**required** for
  every `nn.Module`).
- `self.config = config` — store the settings object.
- `T5ForConditionalGeneration.from_pretrained(config.model_name)` — download the
  ByT5 model weights (name e.g. `"google/byt5-small"`). Stored in `self.model`.
- `AutoTokenizer.from_pretrained(...)` — download the matching tokenizer.

### Method `forward`

**▸ What this method does (whole function):** run the standard ByT5 and repackage
its output into the same dict shape the other two variants use, so training code
doesn't care which variant it holds.

**Variables at a glance**
| Variable | Type / shape | Holds |
|---|---|---|
| `input_ids` (param) | tensor `(batch, seq_len)` | Input byte IDs |
| `attention_mask` (param) | tensor `(batch, seq_len)` | 1=real, 0=pad |
| `labels` (param) | tensor `(batch, target_len)` or None | Target IDs (padding = −100) |
| `outputs` | HF output object | ByT5's raw result |
| `result` | `dict` | The repackaged output |

```python
    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask, labels=labels)
```
**▸ What this block does:** delegate the whole computation to built-in ByT5.
- `**kwargs` — **keyword arguments** catch-all; absorbs extras like `noise_level`
  that the other variants accept, so all three can be **called the same way** and
  this one just ignores what it doesn't need.
- `self.model(...)` — run ByT5. When `labels` are given, it computes the loss
  internally. Result stored in `outputs`.

```python
        result = {
            "logits": outputs.logits,
            "encoder_last_hidden_state": outputs.encoder_last_hidden_state,
            "deletion_rate": torch.zeros(input_ids.size(0), device=input_ids.device),
        }
```
**▸ What this block does:** build the standardized output dict; this model deletes
nothing, so its deletion rate is all zeros.
- `outputs.logits` — the raw prediction scores.
- `outputs.encoder_last_hidden_state` — the encoder's final vectors.
- `torch.zeros(input_ids.size(0), ...)` — a tensor of zeros, one per sentence.
  `input_ids.size(0)` = the **batch size** (first dimension). `device=input_ids
  .device` puts it on the same GPU/CPU as the inputs (mixing devices errors).

```python
        if labels is not None:
            result["loss"] = outputs.loss
        return result
```
**▸ What this block does:** if labels were provided, include ByT5's computed loss,
then return the dict.

### Method `generate`

```python
    def generate(self, input_ids, attention_mask, max_length=1024, num_beams=4, **kwargs):
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                   max_length=max_length, num_beams=num_beams,
                                   early_stopping=True)
```
**▸ What this block does:** produce output text at inference using beam search.
- `self.model.generate(...)` — ByT5's built-in text generator.
- `max_length=1024` — cap on output length.
- `num_beams=4` — **beam search** width: explore 4 candidate outputs, keep the best.
- `early_stopping=True` — stop once complete candidates are found.

---

## `delete_gate.py` — the compression engine

**Role:** The shared machine that scores every byte keep-or-delete and can remove
low-value bytes. Used by both compressed variants. In MrT5 it deletes a fixed
amount; in TAHIMIK it also shifts by noise.

- **Input (forward):** `hidden_states`, `attention_mask`, optional `noise_scores`
- **Output:** `gate_outputs`, `keep_prob`, `kept_mask`, `deletion_rate`
- **Used by:** `fixed_compression_byt5.py`, `noise_adaptive_byt5.py`
- **Connects to:** the MrT5 reference implementation (see `ATTRIBUTIONS.md`)

### Where this fits in TAHIMIK
This is **the compression engine — the single reason MrT5 and TAHIMIK are faster
than plain ByT5 at all.** It's a shared `nn.Module` that both compressed variants
*reuse* (composition, not copy-paste): MrT5 uses it in fixed mode, TAHIMIK in
noise-adaptive mode. And in TAHIMIK, this is literally **where the thesis
contribution lives** — Step 2's `cn·(n−navg)` shift is the one place noise changes
the compression.

**What breaks without it:** no compression → no efficiency story → no thesis. Both
`fixed_compression_byt5.py` and `noise_adaptive_byt5.py` import and embed it, so
it's the most-depended-on model file.

**Example** (the thesis claim, which a test in `tests/` actually checks)
```
same batch of hidden states, gate in eval mode:
  noise_scores = 0.05 (clean)  → deletion_rate ≈ 0.50   (compress hard)
  noise_scores = 0.95 (noisy)  → deletion_rate ≈ 0.10   (preserve for correction)
# noisier input is compressed LESS — the adaptive shift working
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
| `.unsqueeze(d)` | insert a size-1 dimension at position `d` (for broadcasting) |
| `.squeeze(-1)` | remove the size-1 last dimension |
| `.sum(dim=1)` | add up along the sequence dimension |
| `cumsum` | running cumulative total |
| `gather` | pick specific positions out of a tensor |
| `.detach()` | cut the gradient link |
| EMA | exponential moving average |

### Helper `gumbel_noise_like`

**▸ What this function does (whole thing):** generate random "Gumbel" noise shaped
like the input, used to jitter the gate scores during training so it explores both
keep and delete rather than locking in early. (Adapted from the MrT5 reference.)

```python
def gumbel_noise_like(x: torch.Tensor) -> torch.Tensor:
    eps = 3e-4 if x.dtype == torch.float16 else 1e-10
    uniform = torch.empty_like(x).uniform_(eps, 1 - eps)
    return -(-uniform.log()).log()
```
**▸ line notes:**
- `x` — a tensor whose shape/dtype the noise should match.
- `eps` — a tiny number to keep the logs below from hitting zero (`log(0)` is
  invalid). `3e-4` = 0.0003 (scientific notation). It's larger under `float16`
  (16-bit) because tiny values vanish there. `x.dtype` = the tensor's number type.
- `torch.empty_like(x)` — a same-shaped uninitialized tensor.
- `.uniform_(eps, 1 - eps)` — fill it with uniform random values in that range.
  The trailing `_` means it modifies the tensor **in place**.
- `-(-uniform.log()).log()` — the standard formula turning uniform randomness into
  the **Gumbel** distribution (`.log()` = natural logarithm).

### Class `DeleteGate` — constructor

**Variables at a glance**
| Variable | Type | Holds |
|---|---|---|
| `hidden_dim` (param) | `int` | Size of each byte's vector (d_model) |
| `k` (param) | `float` (−30) | Bounds gate scores to [k, 0] |
| `noise_adaptive` (param) | `bool` | Turn TAHIMIK's noise shift on/off |
| `noise_avg_momentum` (param) | `float` (0.99) | EMA slowness |
| `self.layer_norm` | `nn.LayerNorm` | Normalizer before scoring |
| `self.gate_linear` | `nn.Linear` | Collapses a byte vector → 1 score |
| `self.cn` | `nn.Parameter` | Learned strength of the noise shift |
| `self.noise_avg` | buffer | Running average of noise |
| `self.hard_threshold` | `float` (−15) | Keep/delete cutoff |

```python
    def __init__(self, hidden_dim, k=-30.0, noise_adaptive=True,
                 noise_avg_momentum=0.99, use_gumbel_noise=True):
        super().__init__()
        self.k = k
        self.noise_adaptive = noise_adaptive
        self.noise_avg_momentum = noise_avg_momentum
        self.use_gumbel_noise = use_gumbel_noise
```
**▸ What this block does:** store the gate's settings. `noise_adaptive` is the
switch that makes one class serve both MrT5 (`False`) and TAHIMIK (`True`).

```python
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.gate_linear = nn.Linear(hidden_dim, 1)
```
**▸ What this block does:** create the two learnable scoring layers.
- `nn.LayerNorm(hidden_dim)` — normalizes each byte's vector (mean 0, variance 1)
  so scoring is numerically stable.
- `nn.Linear(hidden_dim, 1)` — a learned matrix that maps a byte's whole
  `hidden_dim`-length vector down to a single number (its raw score).

```python
        if noise_adaptive:
            self.cn = nn.Parameter(torch.tensor(1.0))
            self.register_buffer("noise_avg", torch.tensor(0.5))
```
**▸ What this block does:** only for TAHIMIK — create the learned strength dial and
the stored noise average.
- `nn.Parameter(torch.tensor(1.0))` — `cn`, a **learned** scalar starting at 1.0.
  Being a `Parameter` means training updates it.
- `self.register_buffer("noise_avg", torch.tensor(0.5))` — `noise_avg`, a
  **buffer**: saved with the model and moved to GPU with it, but **not** learned;
  it's updated by hand via EMA. Starts at 0.5.

```python
        self.hard_threshold = k / 2.0
```
**▸ What this block does:** set the hard keep/delete cutoff to the midpoint of
[k, 0], i.e. −15.

### Method `forward` — the scoring

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
    def forward(self, hidden_states, attention_mask, noise_scores=None):
        batch_size, seq_len, _ = hidden_states.shape
```
**▸ What this block does:** read the sizes from the input's shape.
- `hidden_states.shape` — the tuple `(batch, seq_len, hidden_dim)`.
- `batch_size, seq_len, _ = ...` — **unpacking**: assign the three parts to
  variables; `_` throws away the third (hidden_dim, not needed here).

**Step 1 — raw scores:**
```python
        normed = self.layer_norm(hidden_states)
        logits = self.gate_linear(normed)
```
**▸ What this block does:** normalize each byte vector, then collapse it to one raw
score.
- `normed` — normalized vectors, same shape.
- `logits` — shape `(batch, seq, 1)`; one raw score per byte.

```python
        if self.training and self.use_gumbel_noise:
            logits = logits + gumbel_noise_like(logits)
```
**▸ What this block does:** during training only, jitter the scores so the gate
explores. `self.training` is True only after `model.train()`; inference stays
deterministic (same input → same result).

```python
        gate_outputs = self.k * torch.sigmoid(logits)
```
**▸ What this block does:** squash each score into the range [k, 0] = [−30, 0].
- `torch.sigmoid(logits)` — maps any number into (0,1).
- `* self.k` (−30) — flips and scales it into (−30, 0). Score near **0 = keep**,
  near **−30 = delete**. Negative because Step 5/6 adds this onto attention as a
  penalty (a big negative hides a byte).

**Step 2 — noise-adaptive shift (TAHIMIK only):**
```python
        if self.noise_adaptive and noise_scores is not None:
            n_detached = noise_scores.detach()
```
**▸ What this block does:** only when noise-adaptive and a noise score was passed —
detach it so using it here can't train the noise estimator.
- `.detach()` — returns the same numbers but **cut off** from the gradient graph.
  Crucial: the estimator must be trained *only* by its own loss (L_NE).

```python
            if self.training:
                batch_avg = n_detached.mean()
                self.noise_avg = (self.noise_avg_momentum * self.noise_avg
                                  + (1 - self.noise_avg_momentum) * batch_avg)
```
**▸ What this block does:** during training, nudge the running noise average toward
this batch's average, slowly.
- `n_detached.mean()` — average noise over the batch.
- The formula is an **EMA** (exponential moving average): `0.99*old + 0.01*new`.
  Momentum 0.99 = keep 99% of the old value, so `noise_avg` drifts gradually and
  stays stable.

```python
            shift = self.cn * (n_detached - self.noise_avg)
            shift = shift.unsqueeze(1).unsqueeze(2)
            gate_outputs = gate_outputs + shift
            gate_outputs = gate_outputs.clamp(min=self.k, max=0.0)
```
**▸ What this block does:** compute the per-sentence shift and add it to every
byte's score, then clamp back into range.
- `self.cn * (n_detached - self.noise_avg)` — positive when a sentence is
  **noisier than average**, negative when cleaner. `cn` scales the effect. Shape
  `(batch,)`.
- `.unsqueeze(1).unsqueeze(2)` — insert two size-1 dimensions → shape
  `(batch, 1, 1)`, so the one-per-sentence shift can be **broadcast** (auto-added)
  across all byte positions of `gate_outputs` `(batch, seq, 1)`.
- `+ shift` — noisier → scores rise toward 0 → keep more; cleaner → scores fall
  toward −30 → delete more.
- `.clamp(min=self.k, max=0.0)` — force any values that overshot back into [−30, 0].

**Step 3 — keep probability (smooth, for training):**
```python
        keep_prob = 1.0 + (gate_outputs.squeeze(-1) / abs(self.k))
        keep_prob = keep_prob * attention_mask.float()
```
**▸ What this block does:** convert the [−30, 0] score into a [0,1] keep
probability, and zero out padding.
- `gate_outputs.squeeze(-1)` — drop the trailing size-1 dim → `(batch, seq)`.
- `/ abs(self.k)` — divide by 30 (`abs` = absolute value). So score 0 → 1.0,
  −15 → 0.5, −30 → 0.0.
- `* attention_mask.float()` — multiply by the mask (`.float()` turns 1/0 ints into
  1.0/0.0) so **padding positions become 0** and never count. This smooth quantity
  is what the rate loss trains on.

**Step 4 — hard decision (yes/no, for inference):**
```python
        kept_mask = (gate_outputs.squeeze(-1) > self.hard_threshold).float()
        kept_mask = kept_mask * attention_mask.float()
```
**▸ What this block does:** make a 0/1 keep decision per byte, used to physically
remove bytes and to report the true rate.
- `(... > self.hard_threshold)` — a True/False tensor: keep if score > −15.
- `.float()` — turn True/False into 1.0/0.0.
- `* attention_mask.float()` — zero out padding.

**Step 5 — deletion rate:**
```python
        real_tokens = attention_mask.float().sum(dim=1).clamp(min=1.0)
        if self.training:
            deletion_rate = 1.0 - (keep_prob.sum(dim=1) / real_tokens)
        else:
            deletion_rate = 1.0 - (kept_mask.sum(dim=1) / real_tokens)
        return gate_outputs, keep_prob, kept_mask, deletion_rate
```
**▸ What this block does:** compute the fraction of bytes deleted — from the smooth
`keep_prob` during training (so gradients flow), from the hard mask at inference
(the real number) — and return all four tensors.
- `attention_mask.float().sum(dim=1)` — number of real (non-pad) bytes per
  sentence. `.sum(dim=1)` adds along the sequence dimension. `.clamp(min=1.0)`
  avoids divide-by-zero.
- `1.0 - kept/real` — fraction deleted.

### Method `apply_hard_deletion` — the physical shortening (inference only)

**▸ What this method does (whole function):** removes deleted bytes and left-packs
the survivors into a shorter sequence (padding the rest), for the whole batch at
once using fast tensor ops instead of a Python loop. **This is where inference gets
faster.**

```python
        keep = kept_mask.bool()
        kept_counts = keep.sum(dim=1)
        new_seq_len = max(int(kept_counts.max().item()), 1)
```
**▸ What this block does:** figure out the new (shorter) length = the most bytes any
sentence keeps (at least 1).
- `.bool()` — turn the 0/1 mask into True/False.
- `keep.sum(dim=1)` — how many bytes each sentence keeps.
- `.max().item()` — the biggest such count as a plain Python int; `max(..., 1)`
  guarantees at least one position so the decoder always gets something.

```python
        target_pos = (torch.cumsum(keep.long(), dim=1) - 1).clamp(min=0)
        positions = torch.arange(seq_len, device=device).expand(batch_size, -1)
        positions = positions * keep.long()
        src_positions = torch.zeros(batch_size, new_seq_len, device=device, dtype=torch.long)
        src_positions.scatter_add_(1, target_pos, positions)
        compressed_states = torch.gather(hidden_states, 1,
            src_positions.unsqueeze(-1).expand(-1, -1, hidden_dim))
```
**▸ What this block does:** compute, for each kept byte, its new slot, then copy the
kept vectors into the shortened tensor.
- `torch.cumsum(keep.long(), dim=1)` — a **running total** of kept bytes; each kept
  position learns its destination index. `- 1` makes it 0-based; `.clamp(min=0)`
  keeps it valid.
- `torch.arange(seq_len, ...)` — `[0,1,...,seq_len-1]`; `.expand(batch_size, -1)`
  repeats it for every sentence (source position numbers).
- `scatter_add_` + `torch.gather` — the vectorized machinery that gathers exactly
  the kept positions, in order, into `compressed_states`. (`gather` = "pick these
  positions.") Done for the whole batch without a slow Python loop.

```python
        new_attention_mask = (torch.arange(new_seq_len, device=device)
            .expand(batch_size, -1) < kept_counts.unsqueeze(1)).to(kept_mask.dtype)
        compressed_states = compressed_states * new_attention_mask.unsqueeze(-1).to(...)
        return compressed_states, new_attention_mask
```
**▸ What this block does:** build the new mask (positions beyond a sentence's kept
count are padding), zero the padded tail, and return the shortened states + mask.
- `arange(new_seq_len) < kept_counts` — True for real positions, False for padding.
- `* new_attention_mask.unsqueeze(-1)` — zero out the padded tail so no stale
  vectors leak.

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
This is **the efficiency baseline — the "compression without intelligence"
comparison point.** It proves that byte compression *works* (faster than ByT5), but
applies the **same** 50% deletion to every sentence regardless of noise. TAHIMIK's
entire value is proven by **beating this specific model**: if the noise-adaptive
version scores higher on noisy sentences at similar speed, the adaptivity — not
just the compression — is what helped.

**What breaks without it:** you couldn't separate two effects. Beating plain ByT5
on speed only shows "compression is faster"; beating MrT5 on accuracy shows "*being
smart about which sentences to compress* matters." This is the control that
isolates your contribution.

**Terms & abbreviations in this file**
| Term | Full form / plain meaning |
|---|---|
| encoder split | running the first few encoder layers, inserting the gate, then the rest |
| `embed_tokens` | the layer turning byte IDs into vectors |
| extended attention mask | HF's internal additive mask form (0 for real, huge negative for pad) |
| gate_bias | the gate score added onto attention logits (soft deletion) |
| `_shift_right` | prepares the decoder's input (standard seq2seq trick) |
| cross-entropy | the loss for predicting the correct next byte |
| `.view(...)` | reshape a tensor without copying data |
| `BaseModelOutput` | HF wrapper so `generate()` accepts a precomputed encoder output |

### The key idea — splitting the encoder
```
Encoder layers 0,1,2  →  DeleteGate  →  Encoder layers 3..N  →  Decoder
```

### Constructor

```python
    def __init__(self, config):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.delete_gate_layer = config.delete_gate_layer   # 3
        self.delete_gate = DeleteGate(hidden_dim=self.model.config.d_model,
                                      k=config.gate_k, noise_adaptive=False, ...)
        self.fixed_deletion_target = config.fixed_deletion_target   # 0.5
```
**▸ What this block does:** load ByT5, then build a `DeleteGate` with
`noise_adaptive=False` (same class as TAHIMIK, noise behavior off), and record
where to split (layer 3) and the fixed target rate (0.5).
- `self.model.config.d_model` — ByT5's hidden size, passed to the gate.

### Helper `_run_encoder_layers`

**▸ What this method does (whole function):** run a *range* of the encoder's layers
on the hidden states, optionally adding the gate's score as a soft attention
penalty.

```python
        extended_mask = encoder.get_extended_attention_mask(attention_mask, hidden_states.shape[:2])
        if gate_bias is not None:
            extended_mask = extended_mask + gate_bias[:, None, None, :]
        for i in range(start_layer, end_layer):
            hidden_states = encoder.block[i](hidden_states, attention_mask=extended_mask)[0]
        return hidden_states
```
**▸ line notes:**
- `get_extended_attention_mask(...)` — converts the 0/1 mask into HF's internal
  additive form (0 for real positions, a huge negative for padding).
- `+ gate_bias[:, None, None, :]` — **add** the gate scores onto that mask.
  `[:, None, None, :]` reshapes `(batch, seq)` → `(batch, 1, 1, seq)` so it
  broadcasts across attention heads and query positions. **Adding, not
  multiplying, is critical** — HF turns a binary mask into `(1-mask)*−1e34`, so
  multiplying a soft 0.99997 keep would become a hard delete and kill the gradient.
- `encoder.block[i](...)` — run encoder layer number `i`. `[0]` takes the hidden
  states out of its output tuple.
- The loop runs layers `start_layer` up to (not including) `end_layer`.

### Method `forward`

```python
        inputs_embeds = encoder.embed_tokens(input_ids)
        hidden_states = encoder.dropout(inputs_embeds)
        hidden_states = self._run_encoder_layers(hidden_states, attention_mask,
                                                 start_layer=0, end_layer=self.delete_gate_layer)
```
**▸ What this block does:** turn byte IDs into vectors, apply dropout, then run the
**pre-gate** layers (0–2).
- `embed_tokens(input_ids)` — look up a learned vector for each byte ID.
- `encoder.dropout(...)` — randomly zero some values (training regularization).

```python
        gate_outputs, keep_prob, kept_mask, deletion_rate = self.delete_gate(
            hidden_states, attention_mask)
```
**▸ What this block does:** score the bytes with the gate (no `noise_scores` passed
→ stays fixed-rate).

```python
        if self.training:
            modified_mask = attention_mask
            hidden_states = self._run_encoder_layers(..., gate_bias=gate_outputs.squeeze(-1))
        else:
            hidden_states, modified_mask = self.delete_gate.apply_hard_deletion(hidden_states, kept_mask)
            hidden_states = self._run_encoder_layers(hidden_states, modified_mask, ...)
```
**▸ What this block does:** the soft/hard fork.
- **Training:** keep full length; pass the gate score as `gate_bias` so disliked
  bytes get progressively harder to attend to (differentiable).
- **Inference:** `apply_hard_deletion` physically removes bytes (shorter, faster),
  then run the remaining layers on the shortened sequence.

```python
        hidden_states = encoder.final_layer_norm(hidden_states)
        hidden_states = encoder.dropout(hidden_states)
        if self.training:
            hidden_states = hidden_states * keep_prob.unsqueeze(-1)
```
**▸ What this block does:** apply the encoder's final normalization, then in
training multiply by `keep_prob` so the decoder also sees deleted bytes suppressed.
- Applied **after** the norm on purpose — the comment notes T5's normalization
  would otherwise undo the suppression.
- `keep_prob.unsqueeze(-1)` — `(batch, seq)` → `(batch, seq, 1)` to broadcast over
  the hidden dimension.

```python
        if labels is not None:
            decoder_input_ids = self.model._shift_right(labels)
            decoder_outputs = self.model.decoder(input_ids=decoder_input_ids,
                encoder_hidden_states=hidden_states, encoder_attention_mask=modified_mask)
            lm_logits = self.model.lm_head(decoder_outputs[0])
            ...
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            result["loss"] = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
```
**▸ What this block does:** run the decoder and compute the accuracy loss.
- `_shift_right(labels)` — shift the target right by one to make the decoder's
  input (it predicts each byte from the previous ones).
- `self.model.decoder(...)` — run the decoder, attending to the (possibly
  shortened) encoder states via `encoder_attention_mask`.
- `lm_head(...)` — the final layer turning decoder vectors into vocabulary scores
  (`lm_logits`).
- `nn.CrossEntropyLoss(ignore_index=-100)` — the loss; `ignore_index=-100` is where
  the `-100` padding labels from `dataset.py` get skipped.
- `.view(-1, lm_logits.size(-1))` — reshape logits to `(all_positions, vocab)` and
  `labels.view(-1)` to `(all_positions,)` so the loss can compare them.

`generate()` mirrors the inference path (embed → pre-gate → gate → hard-delete →
post-gate → beam search). It wraps the encoder output in a `BaseModelOutput` so
HF's `generate` can read `.last_hidden_state` — passing a bare tuple triggers a
`bos_token_id` error.

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
This is **the "sensor" half of the contribution.** At *training* time you know the
true noise (n\*, from `noise_label.py`). But at *inference* — the real tool, a user
pasting text — there is **no clean reference**, so n\* cannot be computed. The model
has to *guess* the noise itself. This tiny MLP is that guesser: it reads the
encoder's hidden states and outputs a predicted noise `n`. The delete gate then
uses `n` to decide how hard to compress.

**What breaks without it:** TAHIMIK would have nothing to condition on at inference
and would collapse into MrT5 (fixed compression). The estimator + the gate's shift
together *are* "noise-adaptive."

**Example**
```
estimator(hidden_states, attention_mask)
  hidden_states shape (8, 1024, 512)   ← 8 sentences
  → noise_scores shape (8,)            ← one predicted n per sentence
  e.g. tensor([0.08, 0.71, 0.15, ...]) ← sentence 2 looks noisy
```

**Terms & abbreviations in this file**
| Term | Full form / plain meaning |
|---|---|
| MLP | multi-layer perceptron — a couple of linear layers with an activation between |
| mean-pooling | averaging many vectors into one summary vector |
| `nn.Sequential` | a container that runs its layers in order |
| GELU | Gaussian Error Linear Unit — a smooth activation |
| dropout | randomly zeroing values in training |
| sigmoid | squash into (0,1) |
| `intermediate_dim` | the MLP's hidden-layer width (256) |

### Constructor

```python
    def __init__(self, hidden_dim, intermediate_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_dim, 1),
        )
        self.sigmoid = nn.Sigmoid()
```
**▸ What this block does:** build a 2-layer MLP that maps a sentence vector to one
number, plus a sigmoid to squash it to [0,1].
- `nn.Sequential(...)` — runs the listed layers in order.
- `nn.Linear(hidden_dim, intermediate_dim)` — shrink from d_model to 256.
- `nn.GELU()` — a smooth activation adding non-linearity (so it can learn more than
  a straight line).
- `nn.Dropout(0.1)` — randomly zero 10% of values during training.
- `nn.Linear(intermediate_dim, 1)` — map 256 → a single number.

### Method `forward`

```python
        mask_expanded = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask_expanded).sum(dim=1)
        counts = mask_expanded.sum(dim=1).clamp(min=1.0)
        sentence_vector = summed / counts
```
**▸ What this block does:** **mean-pool** the byte vectors into one sentence vector,
ignoring padding.
- `attention_mask.unsqueeze(-1).float()` — `(batch, seq)` → `(batch, seq, 1)` so it
  multiplies cleanly against `(batch, seq, hidden)`.
- `(hidden_states * mask_expanded).sum(dim=1)` — zero out padding, then sum across
  the sequence → `(batch, hidden)`.
- `counts` — number of real bytes per sentence (`.clamp(min=1.0)` avoids /0).
- `summed / counts` — the average = one vector per sentence.

```python
        noise_scores = self.sigmoid(self.network(sentence_vector)).squeeze(-1)
        return noise_scores
```
**▸ What this block does:** run the MLP, squash to [0,1], drop the size-1 dim → one
noise score per sentence, shape `(batch,)`.

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
halves (the sensor + the gate) onto the ByT5 backbone: run early encoder layers →
estimate noise `n` → let the gate shift its keep/delete scores by `n` → run the
rest → decode. It is the file the panel will scrutinize hardest, because everything
else exists to build, train, or measure *this*.

**What breaks without it:** there is no proposed model — just two baselines. Every
result the study reports for "TAHIMIK" comes from this class.

**Example** (end-to-end intent)
```
clean input  "salamat po"         → estimator n≈0.05 → gate deletes ~50% → fast
noisy input  "slmt p0 grabeee 😭"  → estimator n≈0.70 → gate deletes ~15% → accurate
# same model, different compression, decided by predicted noise
```

**Terms & abbreviations in this file**
| Term | Full form / plain meaning |
|---|---|
| n\* | ground-truth noise level (from `noise_label.py`), here the param `noise_level` |
| L_NE | noise-estimator loss (MSE between predicted `n` and `n*`) |
| MSE | mean squared error |
| `d_max` | maximum deletion fraction for a perfectly clean sentence (0.5) |
| target_deletion_rate | the adaptive per-sentence goal `d_max*(1−n)` |
| `mse_loss` | PyTorch's mean-squared-error function |

### The architecture
```
Encoder 0,1,2  →  NoiseEstimator (predict n)  →  DeleteGate (shift by n)  →  Encoder 3..N  →  Decoder
```
Identical to MrT5 **except** the estimator inserted at the split and the gate
receiving the noise score. The helper `_run_encoder_layers`, the soft/hard fork,
the final norm, the decoder, and `generate` are the **same as MrT5** (see above).

### Constructor — the differences from MrT5

```python
        self.noise_estimator = NoiseEstimator(hidden_dim=hidden_dim,
            intermediate_dim=config.noise_estimator_hidden_dim)
        self.delete_gate = DeleteGate(hidden_dim=hidden_dim, k=config.gate_k,
            noise_adaptive=True, noise_avg_momentum=config.noise_avg_momentum, ...)
        self.d_max = config.d_max   # 0.5
```
**▸ What this block does:** build the two extra pieces — the estimator and a gate
with `noise_adaptive=True` (the switch that turns on the gate's Step 2 shift) — and
store `d_max`.

### Method `forward` — the three additions over MrT5

```python
        noise_scores = self.noise_estimator(hidden_states, attention_mask)
```
**▸ What this block does:** after the pre-gate layers, **predict the noise** from
the current hidden states (calls `noise_estimator.py`).

```python
        gate_outputs, keep_prob, kept_mask, deletion_rate = self.delete_gate(
            hidden_states, attention_mask, noise_scores=noise_scores)
```
**▸ What this block does:** pass that prediction into the gate, so its
noise-adaptive shift activates.

```python
        target_deletion_rate = self.d_max * (1.0 - noise_scores.detach())
```
**▸ What this block does:** compute each sentence's **adaptive target** deletion
rate.
- `1.0 - noise_scores` — clean (n≈0) → ≈1, noisy (n≈1) → ≈0.
- `* d_max` (0.5) — clean → target ≈ 0.5 (compress hard); noisy → target ≈ 0
  (barely compress).
- `.detach()` — again prevents a gradient backdoor into the estimator.

```python
        if noise_level is not None:
            ne_loss = nn.functional.mse_loss(noise_scores, noise_level)
            result["ne_loss"] = ne_loss
```
**▸ What this block does:** compute **L_NE** — the only signal that trains the
estimator.
- `nn.functional.mse_loss(noise_scores, noise_level)` — **mean squared error**:
  average of `(predicted − truth)²`. `noise_level` is the ground-truth n\* from
  `dataset.py`. Stored as `ne_loss` for the loss module to combine.

Everything else in `forward` (embedding, the soft/hard fork, final norm + the
`keep_prob` scaling, the decoder + cross-entropy) is identical to MrT5. The result
dict additionally carries `noise_scores` and `target_deletion_rate` so the loss
module can build L_NE and the *adaptive* L_rate.

---

## `__init__.py`

```python
from src.models.noise_estimator import NoiseEstimator
from src.models.delete_gate import DeleteGate
from src.models.byt5_baseline import ByT5Baseline
from src.models.fixed_compression_byt5 import FixedCompressionByT5
from src.models.noise_adaptive_byt5 import NoiseAdaptiveByT5
```
**▸ What this block does:** re-exports the five public names so other code can do
`from src.models import NoiseAdaptiveByT5`, etc.

- **Inputs:** none (runs on import)
- **Outputs:** the five names above, importable from `src.models`
