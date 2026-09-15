# Quickstart: Validating the ByT5 Baseline

How to prove this variant works, with no GPU and no dataset.

## Prerequisites

- `pip install -r requirements.txt`
- No GPU required.
- No real dataset required — a tiny randomly-initialized T5 stands in for a
  real checkpoint, following `tests/test_model_forward.py`.

## Run the tests

```bash
python -m pytest tests/test_byt5_baseline.py -v
```

If that file does not exist yet, T003 has not been done. The commands below
describe the target state.

## What passing proves

1. **It trains** (SC-001) — a forward pass with labels returns a finite loss.
   Not NaN, not missing.
2. **It deletes nothing** (SC-002) — `deletion_rate` is exactly `0.0` for every
   sentence in every batch. This is the defining property of the variant; if it
   is ever non-zero, the accuracy ceiling is not a ceiling.
3. **It generates** (SC-003) — `generate()` returns byte IDs with the batch
   dimension preserved.

## Confirm it is genuinely uncompressed

Worth checking by hand once, because it is the whole point of the variant:

```bash
python -c "
from configs.byt5_config import ByT5Config
import src.models.byt5_baseline as m
import inspect
src = inspect.getsource(m)
assert 'DeleteGate' not in src, 'baseline imported a delete gate'
assert 'NoiseEstimator' not in src, 'baseline imported a noise estimator'
print('baseline is clean: no gate, no estimator')
"
```

A gate configured to delete nothing would still add parameters and still
perturb training. The requirement is absence, not neutralisation.

## Confirm the encoder is a shared control variable

```bash
python -c "
from configs.byt5_config import ByT5Config
from configs.mrt5_config import MrT5Config
from configs.tahimik_config import TAHIMIKConfig
names = {c().variant_name: c().model_name for c in (ByT5Config, MrT5Config, TAHIMIKConfig)}
print(names)
assert names['byt5_baseline'] == 'google/byt5-small'
assert names['mrt5_fixed'] == 'stanfordnlp/mrt5-small'
assert names['tahimik_noise_adaptive'] == 'stanfordnlp/mrt5-small'
print('all variants use their designated Small source')
"
```

Expect the native baseline to print `google/byt5-small` and the two compressed
variants to print `stanfordnlp/mrt5-small`, as specified by Specs 017–019.

## What this does not cover

- Actual accuracy scores — blocked on the gold-standard dataset (SC-004).
- Whether this variant is in fact the most accurate of the three — expected,
  unproven (SC-005).
- Inference time and GPU memory — measured in spec `007`, needs a GPU.
