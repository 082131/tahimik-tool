# Quickstart: Validating the Training Pipeline

How to prove the schedule works without a GPU and without the gold standard.

## Prerequisites

- `pip install -r requirements.txt`
- No GPU required — fp16 auto-disables on CPU.
- No gold dataset required. Every run today is Stage-1-only by necessity.

## Run the tests

```bash
python -m pytest tests/test_trainer.py -v
```

**This file does not exist yet.** The trainer is currently the largest untested
component in the repository, which is why T006 exists.

## Confirm the schedule's control flow

The behaviours worth checking, in order of how much they matter today:

### 1. Stage 1 skipping works (SC-004)

Load-bearing right now, not a nicety. With no gold data, *every* run is
Stage-1-only. Save as `check_skip.py`:

```python
from configs.byt5_config import ByT5Config
from src.training.losses import TAHIMIKLoss

config = ByT5Config()
loss_fn = TAHIMIKLoss(use_compression=False, noise_adaptive=False)
print("constructed; trainer.train() with no datasets should log two skips")
```

Then confirm `train()` called with all four datasets as `None` logs
"Skipping Stage 1" and "Skipping Stage 2" and returns an empty history rather
than raising.

### 2. Each stage uses its own hyperparameters (SC-002)

```bash
python -c "from configs.base import BaseConfig as C; c=C(); print(f'stage1: {c.stage1_epochs} epochs, batch {c.stage1_batch_size}'); print(f'stage2: {c.stage2_epochs} epochs, batch {c.stage2_batch_size}')"
```

Expect Stage 1 to have fewer epochs and larger batches than Stage 2 — more data,
fewer passes.

### 3. All three variants share the schedule (FR-002)

The control-variable requirement, checkable directly:

```bash
python -c "from configs.byt5_config import ByT5Config as A; from configs.mrt5_config import MrT5Config as B; from configs.tahimik_config import TAHIMIKConfig as C; ks=['learning_rate','stage1_epochs','stage2_epochs','seed','gold_train_ratio']; rows=[{k:getattr(c(),k) for k in ks} for c in (A,B,C)]; print(rows[0]); assert rows[0]==rows[1]==rows[2], 'variants disagree on a control variable'; print('all three variants agree on the shared schedule')"
```

If this ever fails, the comparison has been silently confounded.

### 4. `n*` labels are produced and in range (SC-005)

```bash
python -c "from src.data.noise_label import compute_noise_level as f; print(f('grabeeee ang init','grabe ang init')); print(f('same text','same text')); assert f('same text','same text')==0.0"
```

## Known gaps, not covered by any check above

Both are Constitution Principle III violations recorded in `research.md`:

- **Determinism**: seeding happens correctly at every entry point, but
  `torch.use_deterministic_algorithms(True)`, the cuDNN flags, and
  `CUBLAS_WORKSPACE_CONFIG` are absent. GPU runs still vary between executions
  on the same seed.
- **Checkpoint traceability**: no git SHA, dirty flag, or resolved config is
  stored, so a checkpoint cannot be tied to the commit that produced it.

Neither can be validated until T001 and T002 are done — there is nothing to
check yet.

## What this does not cover

- Whether Stage 2 improves on Stage 1 — blocked on the gold standard (SC-006).
- Reproduction of the Samuel and Straka ERR gain — blocked (SC-007).
- Inter-annotator agreement on the gold set — blocked (SC-008); `krippendorff`
  is already a dependency for it.
- Whether `noise_generator.py` implements all nine manuscript noise categories —
  **not audited**, recorded as T005 rather than assumed either way.
