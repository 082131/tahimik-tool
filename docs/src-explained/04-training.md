# `src/training/` — the loss and the training loop (exhaustive, every line)

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

> **Format:** every line of the real source is shown and explained (including
> comments, `logger.info`, and `return`s). Each block opens with **▸ What this
> block does**.

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
| backprop / `.backward()` | computing gradients from the loss |
| gradient clipping | capping the size of an update so it can't blow up |
| fp16 / mixed precision | doing math in 16-bit for speed/memory |
| GradScaler | keeps tiny fp16 numbers from vanishing during backprop |
| autocast | runs a block in mixed precision automatically |
| checkpoint | a saved snapshot of the model's learned weights |
| validation | measuring loss on held-out data (not trained on) |
| overfitting | learning the training data too specifically, hurting new data |
| CE / MSE | cross-entropy / mean squared error (loss types) |
| `@torch.no_grad()` | a decorator disabling gradient tracking (faster, no learning) |
| `float("inf")` | positive infinity (a "worse than anything" starting value) |
| `.item()` | pull a plain Python number out of a size-1 tensor |
| `state_dict()` | the model's learned weights as a saveable dictionary |

---

## `losses.py` — the four-part loss

**Role:** Combine up to four loss terms into one `total_loss`, activating only the
ones a given variant needs.

- **Input:** the model's output dict, plus the ground-truth `noise_level`
- **Output:** a dict of individual losses + `total_loss`
- **Used by:** `trainer.py` (and the scripts). Reads outputs from the models in
  [`src/models`](03-models.md).

### Where this fits in TAHIMIK
This file **defines what "good" means during training** — the single number the
optimizer pushes down. It also lets **one trainer train all three variants**: two
switches (`use_compression`, `noise_adaptive`) turn the extra terms on/off, so ByT5
sees only L_CE, MrT5 adds the compression terms, TAHIMIK adds the estimator term.

**What breaks without it:** the trainer couldn't score a model's output; TAHIMIK's
estimator (L_NE) and gate (L_rate) would get no learning signal.

**Example** (what it returns for TAHIMIK on one batch)
```
{
  "l_ce": tensor(0.81), "l_rate": tensor(0.04),
  "l_attn_reg": tensor(0.12), "l_ne": tensor(0.02),
  "total_loss": tensor(0.97),   ← the number that gets .backward()
}
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

### Imports + constructor

```python
import torch
import torch.nn as nn
from typing import Dict, Optional
```
**▸ What this block does:** import PyTorch, the NN toolbox, and type hints.

```python
class TAHIMIKLoss(nn.Module):
    def __init__(
        self,
        w_rate: float = 1.0,
        w_attn_reg: float = 0.01,
        use_compression: bool = True,
        noise_adaptive: bool = False,
    ):
        super().__init__()
```
**▸ What this block does:** declare the loss module and its options; run the parent
setup. `(nn.Module)` — even the loss is a PyTorch module.

```python
        self.w_rate = w_rate
        self.w_attn_reg = w_attn_reg
        self.use_compression = use_compression
        self.noise_adaptive = noise_adaptive
```
**▸ What this block does:** store the two weights and two switches.
- `w_rate`, `w_attn_reg` — multipliers for the L_rate and L_attn_reg terms.
- `use_compression` — if True, add L_rate + L_attn_reg (MrT5 + TAHIMIK).
- `noise_adaptive` — if True, add L_NE (TAHIMIK only).

### Method `forward`

**Variables at a glance**
| Variable | Type / shape | Holds |
|---|---|---|
| `model_outputs` (param) | `dict` | The model's forward output |
| `noise_level` (param) | `(batch,)` or None | Ground-truth n\* |
| `losses` | `dict` | Each term, for logging |
| `l_ce` | scalar tensor | Cross-entropy loss |
| `total` | scalar tensor | The running combined loss |
| `deletion_rate`, `target_rate` | `(batch,)` | Actual vs target rate |
| `l_rate`, `l_attn_reg`, `l_ne` | scalar tensors | The extra terms |

```python
    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        noise_level: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
```
**▸ What this block does:** declare the loss's forward pass — takes the model's
output dict plus the ground-truth n\*, returns a dict of losses.

```python
        losses = {}

        # ── L_CE: Cross-entropy (always present) ───────────────────────
        l_ce = model_outputs["loss"]
        losses["l_ce"] = l_ce
        total = l_ce
```
**▸ What this block does:** start the running total with L_CE (the model already
computed it) and start a dict to log each term.
- `losses = {}` — the per-term log dict.
- `l_ce = model_outputs["loss"]` — the cross-entropy loss from the model.
- `losses["l_ce"] = l_ce` — record it.
- `total = l_ce` — the accumulator other terms add to.

```python
        if self.use_compression:
            # ── L_rate: Deletion rate loss ──────────────────────────────
            deletion_rate = model_outputs["deletion_rate"]
```
**▸ What this block does:** for compressed models only, begin the rate loss by
reading the actual deletion rate the gate achieved.

```python
            if self.noise_adaptive:
                # TAHIMIK: target is noise-adaptive
                target_rate = model_outputs["target_deletion_rate"]
            else:
                # MrT5: fixed target for all sentences
                target_rate = torch.full_like(
                    deletion_rate,
                    model_outputs.get("fixed_deletion_target", 0.5),
                )
```
**▸ What this block does:** pick the *target* rate to compare against.
- TAHIMIK: the per-sentence adaptive `target_deletion_rate` from the model.
- MrT5: a fixed value for every sentence. `torch.full_like(deletion_rate, v)` makes a
  tensor the same shape as `deletion_rate` filled with `v`;
  `model_outputs.get("fixed_deletion_target", 0.5)` reads the model's configured
  target (0.5), falling back to 0.5 if absent.

```python
            l_rate = ((deletion_rate - target_rate) ** 2).mean()
            losses["l_rate"] = l_rate
            total = total + self.w_rate * l_rate
```
**▸ What this block does:** compute L_rate as the average squared gap between actual
and target rate, record it, and add it (scaled by `w_rate`) to the total.
- `(deletion_rate - target_rate) ** 2` — squared difference per sentence (`**` =
  power). `.mean()` — average (this is MSE between actual and target rate).

```python
            # ── L_attn_reg: Gate commitment regularizer ─────────────────
            # ... (long comment explaining the term and the deviation) ...
            keep_prob = model_outputs["keep_prob"]

            l_attn_reg = (4.0 * keep_prob * (1.0 - keep_prob)).mean()
            losses["l_attn_reg"] = l_attn_reg
            total = total + self.w_attn_reg * l_attn_reg
```
**▸ What this block does:** compute L_attn_reg — a term that pushes each keep
probability toward a clear decision — record it, and add it (scaled by `w_attn_reg`).
- `keep_prob = model_outputs["keep_prob"]` — the smooth keep probabilities.
- `4.0 * keep_prob * (1.0 - keep_prob)` — largest (1.0) at p=0.5 (indecision), zero
  at p=0 or p=1 (a clear delete/keep). Minimizing it discourages the gate from
  hedging. It is **symmetric** — no preference for keep vs delete (that's decided by
  L_rate).
- `.mean()` — average over all bytes.
- *(The source has a long comment here noting this is a deliberate, documented
  substitute for MrT5's paper version — see `specs/FINDINGS.md` #8.)*

```python
        if self.noise_adaptive:
            # ── L_NE: Noise estimator loss ──────────────────────────────
            if "ne_loss" in model_outputs:
                l_ne = model_outputs["ne_loss"]
            elif noise_level is not None and "noise_scores" in model_outputs:
                l_ne = nn.functional.mse_loss(
                    model_outputs["noise_scores"], noise_level
                )
            else:
                l_ne = torch.tensor(0.0, device=l_ce.device)
```
**▸ What this block does:** for TAHIMIK only, obtain L_NE (the estimator's error).
- `if "ne_loss" in model_outputs:` — prefer the value the model already computed.
- `elif noise_level is not None and "noise_scores" in model_outputs:` — otherwise
  compute MSE between the predicted `noise_scores` and the ground-truth `noise_level`
  right here.
- `else: l_ne = torch.tensor(0.0, device=l_ce.device)` — if neither is available,
  fall back to a zero scalar on the same device.

```python
            losses["l_ne"] = l_ne
            total = total + l_ne
```
**▸ What this block does:** record L_NE and add it to the total (implicit weight 1.0).

```python
        losses["total_loss"] = total
        return losses
```
**▸ What this block does:** record the combined total under `"total_loss"` and return
the full dict — the individual terms for logging, `total_loss` for training to
minimize.

---

## `trainer.py` — the training loop

**Role:** Run the full two-stage training — Stage 1 (synthetic pretraining) then
Stage 2 (gold fine-tuning) — saving the best model of each stage.

- **Input:** a model, its config, a `TAHIMIKLoss`, and the datasets
- **Output:** checkpoint files (`best_stage1.pt`, `best_stage2.pt`) + a history dict
- **Used by:** `scripts/train.py`, `scripts/run_experiment.py`
- **Connects to:** `losses.py`, `dataset.py` (+ `collate_fn`), `logging_utils.py`

### Where this fits in TAHIMIK
This is **the engine that turns a model + data into a trained checkpoint** — the
`.pt` file the backend later serves. It runs the two-stage schedule and is
**shared, unchanged, by all three variants**, which is what makes the comparison
fair (same optimizer, schedule, early-stopping — only the model differs).

**What breaks without it:** nothing gets trained, so no checkpoint exists — **today's
state** (the tool returns HTTP 503). Running this lifts the project from
"code complete" to "results exist."

**Example** (the log it produces during a run)
```
[..] tahimik.trainer | INFO | Starting stage2: 10 epochs, batch_size=8
[..] tahimik.trainer | INFO |   [stage2] Epoch 1/10 (498.2s) — train_loss=1.20 val_loss=1.05
[..] tahimik.trainer | INFO |   Saved checkpoint: checkpoints/tahimik_noise_adaptive/best_stage2.pt (val_loss=1.05)
```

**Terms & abbreviations in this file**
| Term | Full form / plain meaning |
|---|---|
| `torch.device` | an object naming where computation happens (`cuda`/`cpu`) |
| `.to(device)` | move a model/tensor onto a device |
| `zero_grad()` | clear old gradients before computing new ones |
| `num_workers` | how many background processes load data (0 = main process) |
| `pin_memory` | a speed optimization for moving data to the GPU |
| `os.makedirs` | create a directory (and parents) |
| `get_scheduler` | Hugging Face helper that builds an LR schedule |

### Imports + module setup

```python
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import get_scheduler
from typing import Dict, Optional

from src.training.losses import TAHIMIKLoss
from src.data.dataset import NormalizationDataset, collate_fn
from src.utils.logging_utils import setup_logger
```
**▸ What this block does:** import everything the trainer needs.
- `os` — file/directory operations (making checkpoint folders).
- `time` — measuring epoch duration.
- `torch`, `nn` — tensors + NN utilities (gradient clipping).
- `DataLoader` — serves batches from a Dataset.
- `GradScaler, autocast` — mixed-precision (fp16) helpers.
- `get_scheduler` — HF's LR-scheduler builder.
- The last three are the **connections**: the loss class, the Dataset + its
  `collate_fn`, and the logger factory.

```python
logger = setup_logger("tahimik.trainer")
```
**▸ What this block does:** create this module's logger, named `tahimik.trainer`, so
its log lines are labeled with that name.

### Constructor

```python
class TAHIMIKTrainer:
    def __init__(self, model, config, loss_fn: TAHIMIKLoss):
        self.model = model
        self.config = config
        self.loss_fn = loss_fn
```
**▸ What this block does:** store the model, config, and loss function on the trainer.

```python
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
```
**▸ What this block does:** pick a device and move the model onto it.
- `torch.device(config.device if torch.cuda.is_available() else "cpu")` — use the
  configured device (usually `"cuda"`) if a GPU exists, else `"cpu"`.
- `self.model.to(self.device)` — move the model's weights there.

```python
        self.fp16 = config.fp16 and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.fp16)
```
**▸ What this block does:** enable mixed precision only on a GPU when the config asks,
and build the gradient scaler.
- `config.fp16 and self.device.type == "cuda"` — both conditions must hold.
- `GradScaler(enabled=self.fp16)` — keeps tiny fp16 gradients from underflowing.

```python
        # Best validation loss for checkpoint selection
        self.best_val_loss = float("inf")
```
**▸ What this block does:** initialize the best-loss tracker to infinity so the first
real validation always counts as an improvement.

### Method `_create_optimizer`

```python
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with the config's hyperparameters."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay,
        )
```
**▸ What this block does:** build and return the **AdamW** optimizer over all
learnable weights.
- `self.model.parameters()` — every learnable weight (what gets updated).
- `lr` — the learning rate. `betas`, `eps` — Adam's internal smoothing settings.
- `weight_decay` — the mild pull toward zero (regularization).

### Method `_create_scheduler`

```python
    def _create_scheduler(self, optimizer, num_training_steps):
        """Create cosine LR scheduler with warmup."""
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        return get_scheduler(
            name=self.config.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
```
**▸ What this block does:** build the LR scheduler — warm up for the first 6% of
steps, then decay along a cosine curve.
- `int(num_training_steps * self.config.warmup_ratio)` — 6% of the total steps are
  warmup.
- `get_scheduler(name="cosine", ...)` — HF's helper building the named schedule.

### Method `_train_epoch`

**▸ What this method does (whole function):** run one full pass over the training
data — for each batch: forward → loss → backward → update.

```python
    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
    ) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_losses = {}
        num_batches = 0
```
**▸ What this block does:** declare the method, switch to training mode, and start
accumulators.
- `self.model.train()` — sets `self.training=True` everywhere (soft deletion, dropout
  on, Gumbel noise on).
- `total_losses = {}` — running sums of each loss term. `num_batches = 0` — a counter.

```python
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            optimizer.zero_grad()
```
**▸ What this block does:** loop over batches; move each batch to the device; clear
old gradients.
- `{k: v.to(self.device) for k, v in batch.items()}` — a **dict comprehension** that
  moves every tensor in the batch onto the device.
- `optimizer.zero_grad()` — reset gradients from the previous step (they accumulate
  otherwise).

```python
            with autocast(enabled=self.fp16):
                model_outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    noise_level=batch.get("noise_level"),
                )

                losses = self.loss_fn(
                    model_outputs,
                    noise_level=batch.get("noise_level"),
                )
```
**▸ What this block does:** run the forward pass and compute the loss, in mixed
precision. **This is the models → losses connection.**
- `with autocast(enabled=self.fp16):` — run this block in 16-bit where safe.
- `self.model(...)` — the forward pass. `batch.get("noise_level")` passes n\* for all
  variants; baseline/MrT5 ignore it (they don't use it).
- `self.loss_fn(...)` — build the combined loss dict from the model outputs.

```python
            # Backward pass with gradient scaling
            self.scaler.scale(losses["total_loss"]).backward()
```
**▸ What this block does:** compute gradients from the total loss.
- `self.scaler.scale(...)` — multiply the loss up first so tiny fp16 gradients don't
  vanish. `.backward()` — **backpropagation**: compute every weight's gradient.

```python
            # Gradient clipping
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
```
**▸ What this block does:** undo the loss scaling, then cap the gradient size.
- `self.scaler.unscale_(optimizer)` — remove the earlier scaling before clipping.
- `nn.utils.clip_grad_norm_(params, max_grad_norm)` — cap the total gradient norm so
  one bad batch can't cause a huge, destabilizing update.

```python
            self.scaler.step(optimizer)
            self.scaler.update()
            scheduler.step()
```
**▸ What this block does:** apply the weight update and advance the LR.
- `self.scaler.step(optimizer)` — AdamW nudges the weights (scaler-aware).
- `self.scaler.update()` — adjust the scaler for next time.
- `scheduler.step()` — move the LR one step along its schedule.

```python
            # Accumulate losses for logging
            for key, val in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += val.item()
            num_batches += 1
```
**▸ What this block does:** add each loss term into the running sums (for later
averaging) and count the batch.
- `for key, val in losses.items():` — loop the loss dict.
- `if key not in total_losses: total_losses[key] = 0.0` — initialize on first sight.
- `total_losses[key] += val.item()` — `.item()` turns a size-1 tensor into a plain
  float and adds it.
- `num_batches += 1` — increment the counter.

```python
        # Average over batches
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses
```
**▸ What this block does:** divide each running sum by the number of batches to get
per-epoch averages, and return them.

### Method `_validate`

```python
    @torch.no_grad()
    def _validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run validation and return average losses."""
        self.model.eval()
        total_losses = {}
        num_batches = 0
```
**▸ What this block does:** run validation without learning.
- `@torch.no_grad()` — a decorator turning off gradient tracking for the whole method
  (faster, less memory, no updates).
- `self.model.eval()` — inference mode (hard deletion, dropout off).

```python
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with autocast(enabled=self.fp16):
                model_outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    noise_level=batch.get("noise_level"),
                )

                losses = self.loss_fn(
                    model_outputs,
                    noise_level=batch.get("noise_level"),
                )

            for key, val in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += val.item()
            num_batches += 1
```
**▸ What this block does:** the same forward + loss + accumulate as `_train_epoch`,
but with **no backward/optimizer/scheduler** — it only *measures*.

```python
        avg_losses = {k: v / max(num_batches, 1) for k, v in total_losses.items()}
        return avg_losses
```
**▸ What this block does:** average the losses (with `max(num_batches, 1)` guarding
divide-by-zero) and return them.

### Method `_save_checkpoint`

```python
    def _save_checkpoint(self, epoch: int, stage: str, val_loss: float):
        """Save model checkpoint if validation loss improved."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            checkpoint_dir = os.path.join(
                self.config.checkpoint_dir,
                self.config.variant_name,
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
```
**▸ What this block does:** only when validation improved, update the record and make
sure the checkpoint folder exists.
- `if val_loss < self.best_val_loss:` — save only on improvement (keep the *best*,
  not the last).
- `os.path.join(...)` — build the folder path `<checkpoint_dir>/<variant_name>`.
- `os.makedirs(..., exist_ok=True)` — create it (don't error if present).

```python
            path = os.path.join(
                checkpoint_dir, f"best_{stage}.pt"
            )
            torch.save({
                "epoch": epoch,
                "stage": stage,
                "model_state_dict": self.model.state_dict(),
                "val_loss": val_loss,
            }, path)
            logger.info(f"  Saved checkpoint: {path} (val_loss={val_loss:.4f})")
```
**▸ What this block does:** write the checkpoint file and log it.
- `path = os.path.join(checkpoint_dir, f"best_{stage}.pt")` — e.g.
  `.../best_stage2.pt`.
- `torch.save({...}, path)` — save a dict holding the epoch, stage, the model's
  learned weights (`self.model.state_dict()`), and the validation loss.
- `logger.info(...)` — log the save with the path and loss (`:.4f` = 4 decimals).

### Method `train_stage`

**▸ What this method does (whole function):** run one whole stage — build the data
loaders, optimizer, and scheduler, then loop epochs of train → validate →
maybe-save.

```python
    def train_stage(
        self,
        stage_name: str,
        train_dataset: NormalizationDataset,
        val_dataset: NormalizationDataset,
        epochs: int,
        batch_size: int,
    ) -> Dict[str, list]:
```
**▸ What this block does:** declare the method with the stage name, its two datasets,
and the epoch/batch settings.

```python
        logger.info(f"{'='*60}")
        logger.info(f"Starting {stage_name}: {epochs} epochs, batch_size={batch_size}")
        logger.info(f"  Train samples: {len(train_dataset)}")
        logger.info(f"  Val samples:   {len(val_dataset)}")
        logger.info(f"{'='*60}")
```
**▸ What this block does:** log a banner announcing the stage and its data sizes.
- `f"{'='*60}"` — a line of 60 `=` characters (a visual separator).
- The other lines report the stage name, epochs, batch size, and dataset sizes.

```python
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
```
**▸ What this block does:** build the training batch feeder.
- `batch_size` — examples per batch. `shuffle=True` — reorder each epoch (order
  shouldn't matter; reduces bias).
- `collate_fn=collate_fn` — the stacking function from `dataset.py` (the
  **dataset → trainer** connection).
- `num_workers=0` — load data in the main process.
- `pin_memory=self.device.type == "cuda"` — a speed optimization when on GPU.

```python
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
```
**▸ What this block does:** build the validation feeder — `shuffle=False` (consistent
measurement) and the config's eval batch size.

```python
        optimizer = self._create_optimizer()
        num_training_steps = len(train_loader) * epochs
        scheduler = self._create_scheduler(optimizer, num_training_steps)
```
**▸ What this block does:** create the optimizer and scheduler for this stage.
- `len(train_loader)` — number of batches per epoch; `× epochs` = total steps, which
  the scheduler needs to plan warmup + decay.

```python
        history = {"train": [], "val": []}
```
**▸ What this block does:** start a history dict that will collect per-epoch train and
val loss dicts.

```python
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Train
            train_losses = self._train_epoch(train_loader, optimizer, scheduler)
            # Validate
            val_losses = self._validate(val_loader)

            epoch_time = time.time() - epoch_start
```
**▸ What this block does:** the epoch loop — time it, train one epoch, then validate.
- `range(1, epochs + 1)` — epochs numbered 1..epochs.
- `time.time()` before/after → `epoch_time` in seconds.

```python
            history["train"].append(train_losses)
            history["val"].append(val_losses)
```
**▸ What this block does:** record this epoch's train and val loss dicts.

```python
            # Log
            logger.info(
                f"  [{stage_name}] Epoch {epoch}/{epochs} "
                f"({epoch_time:.1f}s) — "
                f"train_loss={train_losses['total_loss']:.4f} "
                f"val_loss={val_losses['total_loss']:.4f}"
            )
```
**▸ What this block does:** log the epoch summary — stage, epoch number, duration, and
the train/val total losses.

```python
            # Log component losses if present
            for key in ["l_ce", "l_rate", "l_attn_reg", "l_ne"]:
                if key in train_losses:
                    logger.info(
                        f"    {key}: train={train_losses[key]:.4f} "
                        f"val={val_losses[key]:.4f}"
                    )
```
**▸ What this block does:** log each individual loss term that exists for this variant
(so you can see, e.g., `l_ne` falling for TAHIMIK).
- `for key in [...]:` — check each possible term.
- `if key in train_losses:` — only log terms this variant actually produced.

```python
            # Checkpoint
            self._save_checkpoint(epoch, stage_name, val_losses["total_loss"])

        return history
```
**▸ What this block does:** try to save a checkpoint (saves only if val improved),
then, after all epochs, return the history.

### Method `train` — the two-stage driver

**▸ What this method does (whole function):** run Stage 1 (if synthetic data given)
then Stage 2 (if gold data given), resetting the best-loss tracker between them.

```python
    def train(
        self,
        stage1_train: Optional[NormalizationDataset] = None,
        stage1_val: Optional[NormalizationDataset] = None,
        stage2_train: Optional[NormalizationDataset] = None,
        stage2_val: Optional[NormalizationDataset] = None,
    ) -> Dict[str, Dict]:
        full_history = {}
```
**▸ What this block does:** declare the driver with four optional datasets and start
an empty overall-history dict.

```python
        # ── Stage 1: Synthetic Pretraining ──────────────────────────────
        if stage1_train is not None and stage1_val is not None:
            self.best_val_loss = float("inf")
            stage1_history = self.train_stage(
                stage_name="stage1",
                train_dataset=stage1_train,
                val_dataset=stage1_val,
                epochs=self.config.stage1_epochs,
                batch_size=self.config.stage1_batch_size,
            )
            full_history["stage1"] = stage1_history
        else:
            logger.info("Skipping Stage 1 (no synthetic data provided)")
```
**▸ What this block does:** run Stage 1 only if both synthetic datasets were given.
- `self.best_val_loss = float("inf")` — **reset** so Stage 1 picks its own best
  checkpoint independently.
- `train_stage(...)` — run the stage with the config's Stage-1 epochs/batch size.
- `full_history["stage1"] = ...` — record it.
- `else: logger.info(...)` — otherwise log that Stage 1 is skipped.

```python
        # ── Stage 2: Gold Standard Fine-tuning ──────────────────────────
        if stage2_train is not None and stage2_val is not None:
            self.best_val_loss = float("inf")
            stage2_history = self.train_stage(
                stage_name="stage2",
                train_dataset=stage2_train,
                val_dataset=stage2_val,
                epochs=self.config.stage2_epochs,
                batch_size=self.config.stage2_batch_size,
            )
            full_history["stage2"] = stage2_history
        else:
            logger.info("Skipping Stage 2 (no gold standard data provided)")
```
**▸ What this block does:** the same for Stage 2 (gold fine-tuning), again resetting
`best_val_loss` first so each stage's checkpoint is chosen independently.

```python
        logger.info("Training complete.")
        return full_history
```
**▸ What this block does:** log completion and return the combined history (with
`"stage1"` and/or `"stage2"` keys).

### Method `load_checkpoint`

```python
    def load_checkpoint(self, checkpoint_path: str):
        """Load a saved checkpoint into the model."""
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            f"Loaded checkpoint from {checkpoint_path} "
            f"(epoch={checkpoint['epoch']}, "
            f"stage={checkpoint['stage']}, "
            f"val_loss={checkpoint['val_loss']:.4f})"
        )
```
**▸ What this block does:** reload a saved model (used by evaluation/inference).
- `torch.load(path, map_location=self.device, weights_only=True)` — read the file
  onto the current device; `weights_only=True` is a safety setting (load only
  tensors, not arbitrary code).
- `self.model.load_state_dict(checkpoint["model_state_dict"])` — copy the saved
  weights back into the live model.
- `logger.info(...)` — log which checkpoint was loaded, with its metadata.

---

## `__init__.py`

```python
from src.training.losses import TAHIMIKLoss
from src.training.trainer import TAHIMIKTrainer
```
**▸ What this block does:** re-export both public classes so other code can do
`from src.training import TAHIMIKTrainer`. Each line lifts one class to the package
level.

- **Inputs:** none (runs on import)
- **Outputs:** the two names above, importable from `src.training`
