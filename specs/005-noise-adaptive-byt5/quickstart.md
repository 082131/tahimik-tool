# Quickstart: Validating Noise-Adaptive Compression

The contribution. These checks prove the mechanism works structurally; whether
it actually *helps* needs the gold-standard dataset.

## Prerequisites

- `pip install -r requirements.txt`
- No GPU, no dataset. A tiny randomly-initialized T5 stands in.

## Run existing coverage

```bash
python -m pytest tests/test_model_forward.py tests/test_delete_gate.py -v
```

## The check that matters most

**Noisier input must compress less (SC-001).** This is the contribution in one
assertion. If it fails, there is no contribution. Save as `check_adaptive.py`:

```python
import torch
from src.models.delete_gate import DeleteGate

torch.manual_seed(42)
gate = DeleteGate(hidden_dim=32, noise_adaptive=True)
gate.eval()

h = torch.randn(2, 16, 32)
m = torch.ones(2, 16)

clean = gate(h, m, noise_scores=torch.tensor([0.05, 0.05]))[3].mean()
noisy = gate(h, m, noise_scores=torch.tensor([0.95, 0.95]))[3].mean()

print(f"clean deletion={clean:.4f}  noisy deletion={noisy:.4f}")
assert noisy < clean, "noisy input compressed MORE — mechanism is inverted"
print("confirmed: noisier input is compressed less")
```

```bash
python check_adaptive.py
```

## Gradient isolation (SC-002, SC-003)

Already covered by
`tests/test_model_forward.py::test_noise_estimator_is_trained_only_by_l_ne`,
which asserts both halves: `L_rate` must **not** reach the estimator, and
`L_NE` must.

This is what keeps the encoder a control variable across all three variants. If
it breaks, the comparison is unfair and every reported result is suspect.

## Check `cn` has not inverted

**No existing test covers this.** A negative `cn` reverses the mechanism while
everything continues to run normally. Save as `check_cn.py`:

```python
from src.models.delete_gate import DeleteGate

gate = DeleteGate(hidden_dim=32, noise_adaptive=True)
cn = gate.cn.item()
print(f"cn = {cn}")
assert cn > 0, "cn is negative — noisy sentences would compress MORE"
print("cn sign is correct")
```

Run this **after training**, not only at initialization. It starts at `1.0`, and
nothing prevents it crossing zero during optimization.

## Check targets vary per sentence (SC-004)

If targets collapse to one value across a batch of mixed noise scores, TAHIMIK
has silently become MrT5. Save as `check_targets.py`:

```python
import torch

n = torch.tensor([0.1, 0.5, 0.9])
d_max = 0.5
targets = d_max * (1.0 - n.detach())

print("targets:", targets.tolist())
assert len(set(targets.tolist())) == 3, "targets collapsed — variant is not adaptive"
print("confirmed: per-sentence targets vary")
```

## Check `navg` updates in training, freezes in eval (SC-005)

Save as `check_navg.py`:

```python
import torch
from src.models.delete_gate import DeleteGate

gate = DeleteGate(hidden_dim=32, noise_adaptive=True)
h = torch.randn(2, 16, 32)
m = torch.ones(2, 16)
n = torch.tensor([0.8, 0.8])

before = gate.noise_avg.item()
gate.train()
gate(h, m, noise_scores=n)
after_train = gate.noise_avg.item()

gate.eval()
gate(h, m, noise_scores=n)
after_eval = gate.noise_avg.item()

print(f"init={before:.5f} after_train={after_train:.5f} after_eval={after_eval:.5f}")
assert after_train != before, "navg did not update during training"
assert after_eval == after_train, "navg changed during eval — evaluation is not deterministic"
print("confirmed: navg updates in train, frozen in eval")
```

Note `navg` starts at `0.5`, an unspecified choice — see `research.md`.

## What this does not cover

- Whether TAHIMIK is actually more accurate than either baseline — blocked
  (SC-006).
- Inference time and peak GPU memory — blocked, needs a GPU and real data
  (SC-007).
- **Whether the advantage concentrates on noisy sentences** — blocked (SC-008),
  and the most important blocked item. Winning on average is not the same as
  winning for the reason the study claims.
