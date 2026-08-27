# Quickstart: Validating Fixed-Rate Compression

## Prerequisites

- `pip install -r requirements.txt`
- No GPU, no dataset. Tests use a tiny randomly-initialized T5.

## Run the existing tests

This variant is already well covered:

```bash
python -m pytest tests/test_delete_gate.py tests/test_model_forward.py -v
```

`test_delete_gate.py` exercises the gate and losses in isolation;
`test_model_forward.py` runs real forward and backward passes for both
compression variants.

## What passing proves

1. **Gate scores are bounded** (SC-001) — deletion rate stays in `[0, 1]` for
   any batch, including padded ones.
2. **Training-mode compression is differentiable** (SC-002) — `deletion_rate`
   carries a `grad_fn`. Without this the rate loss is inert and the gate cannot
   learn its target.
3. **Inference physically compresses** (SC-003) — encoder output is no longer
   than the input. This is where the efficiency claim is earned.
4. **The gate actually moves toward its target** (SC-004) —
   `test_short_training_run_moves_deletion_rate_toward_target` runs 40 optimizer
   steps and asserts the measured rate gets closer to the configured target.

That last one is the most valuable test in the file. It would fail if the
log-space-bias subtlety were broken, which is the failure mode that otherwise
produces working-looking code that silently never learns.

## Confirm this variant is genuinely noise-blind

The defining property — it must ignore content entirely:

```bash
python -c "
import torch
from src.models.delete_gate import DeleteGate
torch.manual_seed(42)
gate = DeleteGate(hidden_dim=32, noise_adaptive=False)
h = torch.randn(2, 16, 32); m = torch.ones(2, 16)
r1 = gate(h, m)[3]
r2 = gate(h, m, noise_scores=torch.tensor([0.05, 0.95]))[3]
assert torch.allclose(r1, r2), 'fixed-rate gate reacted to noise scores'
print('confirmed: fixed-rate mode ignores noise scores entirely')
"
```

If this ever fails, the baseline has become adaptive and the study has no
contrast to measure TAHIMIK against.

## Confirm the gate layer matches TAHIMIK

```bash
python -c "
from configs.mrt5_config import MrT5Config
from configs.tahimik_config import TAHIMIKConfig
a, b = MrT5Config().delete_gate_layer, TAHIMIKConfig().delete_gate_layer
assert a == b, f'gate layers differ: MrT5={a}, TAHIMIK={b}'
print(f'both variants gate at layer {a}')
"
```

Different layers would confound gate *placement* with noise *adaptation*.

## What this does not cover

- Accuracy cost of fixed compression on real data — blocked (SC-005).
- Whether fixed compression degrades more on noisy sentences than clean ones —
  blocked (SC-006). This is the premise TAHIMIK is built on, and it is currently
  unproven.
- Actual speed and memory savings — spec `007`, needs a GPU.
