# Quickstart: ByT5 Base Migration

Commands to verify and inspect the ByT5-Base configuration and run hardware preflight.

---

## 1. Run Offline Preflight (Metadata Mode)

Verifies that all three variants resolve `google/byt5-base` with effective batch sizes of 16 and 8 without downloading weights:

```bash
python scripts/preflight_base.py --metadata-only
```

---

## 2. Run Online Preflight (GPU Smoke Test)

Runs a single forward pass and generation per variant on target GPU hardware, checking memory allocations:

```bash
python scripts/preflight_base.py --device cuda --max-input-length 1024
```

---

## 3. Verify Controlled Hyperparameters

Print the resolved backbone and effective batch sizes across all variants:

```bash
python -c "from configs.byt5_config import ByT5Config; from configs.mrt5_config import MrT5Config; from configs.tahimik_config import TAHIMIKConfig; cs=[ByT5Config(),MrT5Config(),TAHIMIKConfig()]; print([(c.variant_name,c.model_name,c.stage1_batch_size*c.stage1_gradient_accumulation_steps,c.stage2_batch_size*c.stage2_gradient_accumulation_steps) for c in cs])"
```

---

## 4. Run Migration Regression Tests

```bash
python -m pytest tests/test_model_configuration.py tests/test_memory_configuration.py tests/test_gradient_accumulation.py tests/test_checkpoint_compatibility.py tests/test_base_preflight.py -v
```
