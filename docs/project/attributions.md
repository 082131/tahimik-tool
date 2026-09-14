# Attributions

Third-party code and assets used in this repository, with their licences.

---

## MrT5 — delete gate techniques

**Source**: [jkallini/mrt5](https://github.com/jkallini/mrt5) — the reference
implementation for Kallini et al. (2025), *MrT5: Dynamic Token Merging for
Efficient Byte-level Language Models* (ICLR 2025).

**Licence**: Apache License 2.0

**What is used**: Stanford's `SigmoidDeleteGate` initialization and
`ScaledSigmoid(-logit)` convention for the MrT5 baseline, plus two techniques
from `models/modeling_mrt5.py`, adapted into `src/models/delete_gate.py`:

1. **Gumbel noise on the gate logits during training**, from
   `SigmoidDeleteGate.forward` and the `gumbel_noise_like` helper.
2. **Vectorised hard deletion** using `cumsum` / `scatter_add_` / `gather`,
   from `MrT5Block.__get_new_positions_and_mask` and
   `__hard_delete_hidden_states`.

Both are marked at their call sites in `src/models/delete_gate.py`.

**Model and gate source**: the compressed variants load
[`stanfordnlp/mrt5-small`](https://huggingface.co/stanfordnlp/mrt5-small) via
`AutoModelForSeq2SeqLM(..., trust_remote_code=True)`. TAHIMIK begins with the
same Stanford gate then applies its noise-adaptive shift. The wrapper disables
the embedded gate before applying its imported gate so deletion occurs once.

### Apache 2.0 notice

```
Copyright 2024-2025 Julie Kallini and contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## ByT5

**Source**: `google/byt5-small` via HuggingFace `transformers`.
**Licence**: Apache License 2.0.
Loaded as a pretrained model; no code is vendored.

---

## Frontend assets

See [`frontend/ATTRIBUTIONS.md`](../../frontend/ATTRIBUTIONS.md).
