# `src/training/` — the loss and the training loop (exhaustive line-by-line)

**Flow position:** the **third** stage. Takes the models from
[`src/models`](03-models.md) and the tensors from [`src/data`](02-data.md), teaches
the model, and saves a checkpoint.

**Files (dependency order):**
1. `losses.py` — combine the four loss terms into one number
2. `trainer.py` — the loop that runs training and saves checkpoints
3. `__init__.py` — re-exports

**How they connect:**
```
dataset.py ─(batches)─► trainer.py ─(model outputs)─► losses.py ─(total loss)─► trainer.py
                             │                                                      │
                             └──── nudges the model's weights, saves best_stageN.pt ┘
```

---

## Terms & abbreviations used across this folder
| Term | Full form / plain meaning |
|---|---|
| loss | one number measuring how wrong the model is (training minimizes it) |
| optimizer | the algorithm that adjusts weights using gradients |
| AdamW | Adam with decoupled weight decay — the optimizer used here |
| weight decay | a mild pull of weights toward zero (regularization) |
| LR | learning rate — how big each weight update is |
| scheduler | changes the LR over time (warm up, then decay) |
| warmup | ramping the LR up from 0 at the start of training |
| epoch | one full pass over the training data |
| batch | a group of examples processed together |
| DataLoader | feeds the model batches from a Dataset (shuffling, grouping) |
| gradient (grad) | the direction to nudge a weight to reduce loss |
| backpropagation / `.backward()` | computing gradients from the loss |
| gradient clipping | capping the size of an update so it can't blow up |
| fp16 / mixed precision | doing math in 16-bit for speed/memory |
| GradScaler | keeps tiny fp16 numbers from vanishing during backprop |
| autocast | runs a block in mixed precision automatically |
| checkpoint | a saved snapshot of the model's learned weights |
| validation | measuring loss on held-out data (not trained on) |
| overfitting | learning the training data too specifically, hurting new data |
| CE / MSE | cross-entropy / mean squared error (loss types) |

> **Format reminder:** each block starts with **▸ What this block does**, then
> breaks down every line, variable, and technical term.

---

## `losses.py` — the four-part loss

**Role:** Combine up to four loss terms into one `total_loss`, activating only the
ones a given variant needs.

- **Input:** the model's output dict, plus the ground-truth `noise_level`
- **Output:** a dict of individual losses + `total_loss`
- **Used by:** `trainer.py` (and the scripts). Reads outputs from the models in
  [`src/models`](03-models.md).

### Where this fits in TAHIMIK
This file **defines what "good" means during training** — it's the single number
the optimizer pushes down. It's also what lets **one trainer train all three
variants**: the two switches (`use_compression`, `noise_adaptive`) turn the extra
terms on or off, so ByT5 sees only L_CE, MrT5 adds the compression terms, and
TAHIMIK adds the estimator term — without any special-casing in the training loop.
Each term is a *different job* the model is being taught: be accurate (L_CE), hit
the target compression (L_rate), commit to a decision (L_attn_reg), and sense noise
correctly (L_NE).

**What breaks without it:** the trainer wouldn't know how to score a model's
output. Specifically, TAHIMIK's noise estimator (trained *only* by L_NE) and its
gate (trained by L_rate) would get **no learning signal** — the contribution
wouldn't train.

**Example** (what it returns for TAHIMIK on one batch)
```
loss_fn(model_outputs, noise_level)
# {
#   "l_ce":       tensor(0.81),   ← accuracy
#   "l_rate":     tensor(0.04),   ← compression on-target?
#   "l_attn_reg": tensor(0.12),   ← gate decisiveness
#   "l_ne":       tensor(0.02),   ← noise prediction error
#   "total_loss": tensor(0.97),   ← the number that gets .backward()
# }
```

**Terms & abbreviations in this file**
| Term | Full form / plain meaning |
|---|---|
| L_CE | cross-entropy loss — are the output bytes correct? (accuracy) |
| L_rate | deletion-rate loss — did the gate hit its target rate? |
| L_attn_reg | attention-regularizer loss — push the gate to *decide* |
| L_NE | noise-estimator loss — did the estimator predict the noise? |
| w_rate / w_attn_reg | weights (multipliers) setting how much a term matters |
| `torch.full_like(t, v)` | a tensor shaped like `t`, filled with value `v` |
| `.mean()` | average of all elements |
| `**` | exponent (`x ** 2` = x squared) |
| `.get(key, default)` | dict lookup returning `default` if the key is missing |

### The formula
```
L = L_CE  +  w_rate · L_rate  +  w_attn_reg · L_attn_reg  +  L_NE
```
| Term | Active for | Meaning |
|---|---|---|
| **L_CE** | all 3 | Cross-entropy: are the output bytes correct? |
| **L_rate** | MrT5, TAHIMIK | Did the gate hit its target deletion rate? |
| **L_attn_reg** | MrT5, TAHIMIK | Push the gate to *decide* (not sit at 0.5) |
| **L_NE** | TAHIMIK only | Did the estimator predict the noise correctly? |

### Constructor

```python
    def __init__(self, w_rate=1.0, w_attn_reg=0.01,
                 use_compression=True, noise_adaptive=False):
        super().__init__()
        self.w_rate = w_rate
        self.w_attn_reg = w_attn_reg
        self.use_compression = use_compression
        self.noise_adaptive = noise_adaptive
```
**▸ What this block does:** store the two weights and two switches.
- `w_rate`, `w_attn_reg` — multipliers for those loss terms.
- `use_compression` — if True, add L_rate + L_attn_reg (for MrT5 + TAHIMIK).
- `noise_adaptive` — if True, add L_NE (for TAHIMIK only). These switches let one
  loss class serve all three variants.

### Method `forward`

**▸ What this method does (whole function):** start from the model's cross-entropy
loss and add whichever extra terms the variant needs, returning every term plus the
combined total.

**Variables at a glance**
| Variable | Type / shape | Holds |
|---|---|---|
| `model_outputs` (param) | `dict` | The model's forward output |
| `noise_level` (param) | tensor `(batch,)` or None | Ground-truth n\* |
| `losses` | `dict` | Each term, for logging |
| `l_ce` | scalar tensor | Cross-entropy loss |
| `total` | scalar tensor | The running combined loss |
| `deletion_rate`, `target_rate` | `(batch,)` | Actual vs target rate |
| `l_rate`, `l_attn_reg`, `l_ne` | scalar tensors | The extra terms |

```python
    def forward(self, model_outputs, noise_level=None):
        losses = {}
        l_ce = model_outputs["loss"]
        losses["l_ce"] = l_ce
        total = l_ce
```
**▸ What this block does:** begin the running total with L_CE (the model already
computed it), and start a dict to log each term.
- `model_outputs["loss"]` — the cross-entropy loss the model returned.
- `total` — the accumulator; other terms get added to it below.

```python
        if self.use_compression:
            deletion_rate = model_outputs["deletion_rate"]
            if self.noise_adaptive:
                target_rate = model_outputs["target_deletion_rate"]
            else:
                target_rate = torch.full_like(deletion_rate,
                    model_outputs.get("fixed_deletion_target", 0.5))
            l_rate = ((deletion_rate - target_rate) ** 2).mean()
            losses["l_rate"] = l_rate
            total = total + self.w_rate * l_rate
```
**▸ What this block does:** for compressed models, add **L_rate** — a penalty for
missing the target deletion rate.
- `deletion_rate` — what the gate actually did (from the model).
- `target_rate` — the goal: TAHIMIK's per-sentence adaptive target, or MrT5's fixed
  0.5. `torch.full_like(deletion_rate, 0.5)` makes a tensor of 0.5s the same shape.
  `.get("fixed_deletion_target", 0.5)` reads the model's configured target, falling
  back to 0.5.
- `((deletion_rate - target_rate) ** 2).mean()` — the squared gap, averaged (this
  is MSE between actual and target rate). `** 2` = squared.
- `total = total + self.w_rate * l_rate` — add it, scaled by its weight.

```python
            keep_prob = model_outputs["keep_prob"]
            l_attn_reg = (4.0 * keep_prob * (1.0 - keep_prob)).mean()
            losses["l_attn_reg"] = l_attn_reg
            total = total + self.w_attn_reg * l_attn_reg
```
**▸ What this block does:** add **L_attn_reg** — a term that pushes each keep
probability toward a clear decision.
- `4.0 * keep_prob * (1.0 - keep_prob)` — this expression is **largest (1.0) at
  p=0.5** (maximum indecision) and **zero at p=0 or p=1** (a clear delete/keep).
  Minimizing it discourages the gate from hedging. It's **symmetric** (no
  preference for keep vs delete — that's decided by L_rate).
- `.mean()` — average over all bytes.
- *(Note: this is a deliberate, documented substitute for MrT5's paper version —
  see `specs/FINDINGS.md` #8.)*

```python
        if self.noise_adaptive:
            if "ne_loss" in model_outputs:
                l_ne = model_outputs["ne_loss"]
            elif noise_level is not None and "noise_scores" in model_outputs:
                l_ne = nn.functional.mse_loss(model_outputs["noise_scores"], noise_level)
            else:
                l_ne = torch.tensor(0.0, device=l_ce.device)
            losses["l_ne"] = l_ne
            total = total + l_ne
```
**▸ What this block does:** for TAHIMIK, add **L_NE** — the estimator's error.
- Prefer `ne_loss` if the model already computed it; else compute MSE between the
  predicted `noise_scores` and the ground-truth `noise_level` here; else fall back
  to 0.0 (a scalar tensor on the same device). Added with implicit weight 1.0.

```python
        losses["total_loss"] = total
        return losses
```
**▸ What this block does:** record the combined total and return every term (the
individual ones for logging, `total_loss` for training to minimize).

---

## `trainer.py` — the training loop

**Role:** Run the full two-stage training — Stage 1 (synthetic pretraining) then
Stage 2 (gold fine-tuning) — saving the best model of each stage.

- **Input:** a model, its config, a `TAHIMIKLoss`, and the datasets
- **Output:** checkpoint files (`best_stage1.pt`, `best_stage2.pt`) + a history dict
- **Used by:** `scripts/train.py`, `scripts/run_experiment.py`
- **Connects to:** `losses.py`, `dataset.py` (+ its `collate_fn`), `logging_utils.py`

### Where this fits in TAHIMIK
This is **the engine that turns a model + data into a trained checkpoint** — the
`.pt` file the backend later serves. It runs the two-stage schedule (synthetic
pretraining → gold fine-tuning) and is **shared, unchanged, by all three variants**,
which is exactly what makes the comparison fair: same optimizer, same schedule,
same early-stopping — only the model differs. It's the piece the scripts call to
produce `best_stage2.pt`.

**What breaks without it:** nothing gets trained, so no checkpoint exists — which is
**literally today's state**: the tool returns HTTP 503 precisely because this file
hasn't been run on the (not-yet-collected) gold data. Running this is the step that
lifts the project from "code complete" to "results exist."

**Example** (the observability it produces during a run)
```
[..] tahimik.trainer | INFO | Starting stage2: 10 epochs, batch_size=8
[..] tahimik.trainer | INFO |   [stage2] Epoch 1/10 (498.2s) — train_loss=1.20 val_loss=1.05
[..] tahimik.trainer | INFO |   Saved checkpoint: checkpoints/tahimik_noise_adaptive/best_stage2.pt (val_loss=1.05)
...
[..] tahimik.trainer | INFO | Training complete.
```

**Terms & abbreviations in this file**
| Term | Full form / plain meaning |
|---|---|
| `torch.device` | an object naming where computation happens (`cuda`/`cpu`) |
| `.to(device)` | move a model/tensor onto a device |
| `zero_grad()` | clear old gradients before computing new ones |
| `state_dict()` | the model's learned weights as a saveable dictionary |
| `torch.save` / `torch.load` | write / read a checkpoint file |
| `@torch.no_grad()` | a decorator disabling gradient tracking (faster, no learning) |
| `float("inf")` | positive infinity (a "worse than anything" starting value) |
| `.item()` | pull a plain Python number out of a size-1 tensor |
| `num_workers` | how many background processes load data (0 = main process) |
| `pin_memory` | a speed optimization for moving data to the GPU |

### Constructor

```python
    def __init__(self, model, config, loss_fn):
        self.model = model
        self.config = config
        self.loss_fn = loss_fn
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.fp16 = config.fp16 and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.fp16)
        self.best_val_loss = float("inf")
```
**▸ What this block does:** wire up the trainer — store the model/config/loss, pick
a device, move the model onto it, set up mixed precision, and initialize the
best-loss tracker.
- `torch.device(config.device if torch.cuda.is_available() else "cpu")` — use the
  configured device (usually `"cuda"`, the GPU) if a GPU exists, else fall back to
  `"cpu"`.
- `self.model.to(self.device)` — move the model's weights onto that device.
- `self.fp16 = config.fp16 and self.device.type == "cuda"` — use 16-bit math only
  if requested **and** on a GPU.
- `GradScaler(enabled=self.fp16)` — the helper that keeps tiny fp16 numbers from
  underflowing during backprop.
- `self.best_val_loss = float("inf")` — start at infinity so the first validation
  always counts as an improvement.

### Method `_create_optimizer`

```python
    def _create_optimizer(self):
        return torch.optim.AdamW(self.model.parameters(),
            lr=self.config.learning_rate, betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon, weight_decay=self.config.weight_decay)
```
**▸ What this block does:** build the **AdamW** optimizer over all learnable
weights.
- `self.model.parameters()` — every learnable weight in the model (what gets
  updated).
- `lr` — learning rate. `betas`, `eps` — Adam's internal smoothing settings.
- `weight_decay` — the mild pull toward zero (regularization).

### Method `_create_scheduler`

```python
    def _create_scheduler(self, optimizer, num_training_steps):
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        return get_scheduler(name=self.config.lr_scheduler_type, optimizer=optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
```
**▸ What this block does:** build the LR scheduler — warm up for the first 6% of
steps, then decay along a cosine curve.
- `num_training_steps` — total number of weight updates over the whole stage.
- `int(... * warmup_ratio)` — 6% of them are warmup.
- `get_scheduler(...)` — Hugging Face's helper building the named schedule
  (`"cosine"`).

### Method `_train_epoch`

**▸ What this method does (whole function):** run one full pass over the training
data — for each batch: forward → loss → backward → update.

```python
    def _train_epoch(self, dataloader, optimizer, scheduler):
        self.model.train()
        ...
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            optimizer.zero_grad()
```
**▸ What this block does:** switch to training mode and, for each batch, move it to
the device and clear old gradients.
- `self.model.train()` — sets `self.training=True` in every layer (soft deletion,
  dropout on).
- `{k: v.to(self.device) for k, v in batch.items()}` — a **dict comprehension**
  moving every tensor in the batch onto the device.
- `optimizer.zero_grad()` — reset gradients from the previous step (they accumulate
  otherwise).

```python
            with autocast(enabled=self.fp16):
                model_outputs = self.model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"], labels=batch["labels"],
                    noise_level=batch.get("noise_level"))
                losses = self.loss_fn(model_outputs, noise_level=batch.get("noise_level"))
```
**▸ What this block does:** run the forward pass and compute the loss, in mixed
precision. **This is the models → losses connection.**
- `with autocast(enabled=self.fp16):` — run this block in 16-bit where safe.
- `self.model(...)` — the forward pass. `batch.get("noise_level")` passes n\* for
  all variants; baseline/MrT5 ignore it.
- `self.loss_fn(...)` — build the combined loss from the outputs.

```python
            self.scaler.scale(losses["total_loss"]).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(optimizer)
            self.scaler.update()
            scheduler.step()
```
**▸ What this block does:** compute gradients, clip them, and apply the weight
update, then advance the LR.
- `self.scaler.scale(...).backward()` — **backpropagation**: compute gradients.
  `scale(...)` multiplies the loss up first so tiny fp16 gradients don't vanish.
- `self.scaler.unscale_(optimizer)` — undo that scaling before clipping.
- `nn.utils.clip_grad_norm_(..., max_grad_norm)` — cap the total gradient size so
  one bad batch can't cause a huge, destabilizing update.
- `self.scaler.step(optimizer)` — apply the update (AdamW nudges the weights).
- `self.scaler.update()` — adjust the scaler for next time.
- `scheduler.step()` — move the LR one step along its schedule.

The rest of the method accumulates each loss term into `total_losses` and returns
the per-batch averages (for logging).

### Method `_validate`

```python
    @torch.no_grad()
    def _validate(self, dataloader):
        self.model.eval()
        ...
```
**▸ What this block does:** measure loss on held-out data without learning.
- `@torch.no_grad()` — a decorator turning off gradient tracking for the whole
  method (faster, less memory; no updates happen).
- `self.model.eval()` — inference mode (hard deletion, dropout off). It runs the
  same forward + loss but only *measures*.

### Method `_save_checkpoint`

```python
    def _save_checkpoint(self, epoch, stage, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            ...
            torch.save({"epoch": epoch, "stage": stage,
                        "model_state_dict": self.model.state_dict(),
                        "val_loss": val_loss}, path)
```
**▸ What this block does:** save the model **only when validation loss improved**,
so you keep the best version, not the last.
- `if val_loss < self.best_val_loss:` — only save on improvement; update the record.
- `self.model.state_dict()` — the model's learned weights, as a saveable dict.
- `torch.save({...}, path)` — write a checkpoint file (`best_stageN.pt`) holding the
  weights plus metadata (epoch, stage, loss).

### Method `train_stage`

**▸ What this method does (whole function):** run one whole stage — build the data
loaders, optimizer, and scheduler, then loop epochs of train → validate →
maybe-save.

```python
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
            shuffle=True, collate_fn=collate_fn, num_workers=0,
            pin_memory=self.device.type == "cuda")
        val_loader = DataLoader(val_dataset, batch_size=self.config.eval_batch_size,
            shuffle=False, collate_fn=collate_fn, ...)
```
**▸ What this block does:** create the batch feeders.
- `DataLoader(...)` — wraps a Dataset and serves batches.
- `batch_size` — examples per batch.
- `shuffle=True` for training (order shouldn't matter, reduces bias), `False` for
  validation (consistent measurement).
- `collate_fn=collate_fn` — the stacking function from `dataset.py` (the
  **dataset → trainer** connection).
- `num_workers=0` — load data in the main process. `pin_memory=True` on GPU speeds
  host→GPU transfers.

```python
        optimizer = self._create_optimizer()
        num_training_steps = len(train_loader) * epochs
        scheduler = self._create_scheduler(optimizer, num_training_steps)
        for epoch in range(1, epochs + 1):
            train_losses = self._train_epoch(train_loader, optimizer, scheduler)
            val_losses = self._validate(val_loader)
            ... log ...
            self._save_checkpoint(epoch, stage_name, val_losses["total_loss"])
        return history
```
**▸ What this block does:** set up training math, then loop epochs.
- `len(train_loader)` — number of batches per epoch; `× epochs` = total steps.
- The loop: `_train_epoch` (learn), `_validate` (measure), `_save_checkpoint`
  (keep the best). `history` collects per-epoch losses.

### Method `train` — the two-stage driver

```python
    def train(self, stage1_train=None, stage1_val=None, stage2_train=None, stage2_val=None):
        if stage1_train is not None and stage1_val is not None:
            self.best_val_loss = float("inf")
            full_history["stage1"] = self.train_stage("stage1", ...)
        if stage2_train is not None and stage2_val is not None:
            self.best_val_loss = float("inf")
            full_history["stage2"] = self.train_stage("stage2", ...)
        return full_history
```
**▸ What this block does:** run Stage 1 (only if synthetic data was given), then
Stage 2 (only if gold data was given).
- `self.best_val_loss = float("inf")` is **reset between stages** so each stage
  selects its own best checkpoint independently.

### Method `load_checkpoint`

```python
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
```
**▸ What this block does:** reload a saved model — read the file and copy its
weights back into the model (used by evaluation/inference).
- `torch.load(..., map_location=self.device, weights_only=True)` — read the file
  onto the current device; `weights_only=True` is a safety setting (only load
  tensors, not arbitrary code).
- `self.model.load_state_dict(...)` — copy the saved weights into the live model.

---

## `__init__.py`

```python
from src.training.losses import TAHIMIKLoss
from src.training.trainer import TAHIMIKTrainer
```
**▸ What this block does:** re-exports so other code can do
`from src.training import TAHIMIKTrainer`.

- **Inputs:** none (runs on import)
- **Outputs:** the two names above, importable from `src.training`
