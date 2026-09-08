# Data Model: ByT5 Base Migration

Defines the schemas and entity relationships for configuration, checkpoint architecture fingerprints, and preflight outputs.

---

## 1. Architecture Fingerprint Schema

Stored inside every checkpoint dictionary under the `"architecture"` key:

```python
{
    "model_name": "google/byt5-base",        # str: Backbone model identifier
    "model_type": "t5",                      # str: Backbone architecture family
    "d_model": 1536,                         # int: Hidden state dimension
    "num_encoder_layers": 18,                # int: Total encoder layers
    "num_decoder_layers": 6,                 # int: Total decoder layers
    "vocab_size": 384,                       # int: Byte-level vocabulary size
    "config_fingerprint": "abc123..."        # str: SHA-256 of resolved model hyperparameters
}
```

### Comparison Matrix: Base vs Small

| Attribute | `google/byt5-small` | `google/byt5-base` (Authoritative) | Match Enforcement |
|---|---|---|---|
| Parameters | ~300M | ~582M | Exact architecture |
| `d_model` | 1472 | 1536 | Strict match |
| Encoder Layers | 12 | 18 | Strict match |
| Decoder Layers | 4 | 6 | Strict match |
| Vocabulary | 384 | 384 | Strict match |

---

## 2. Configuration Entity (`configs/base.py`)

```python
@dataclass
class BaseConfig:
    model_name: str = "google/byt5-base"
    stage1_batch_size: int = 2
    stage1_gradient_accumulation_steps: int = 8      # Effective batch = 16
    stage2_batch_size: int = 2
    stage2_gradient_accumulation_steps: int = 4      # Effective batch = 8
    gradient_checkpointing: bool = True
    precision: Literal["fp32", "fp16", "bf16"] = "fp16"
```

---

## 3. Preflight Output Schema (`scripts/preflight_base.py`)

```json
{
  "timestamp": "ISO-8601 UTC string",
  "requested_device": "cuda",
  "resolved_device": "cuda",
  "cuda_available": true,
  "metadata_only": false,
  "all_models_base": true,
  "variants": {
    "byt5": {
      "model_name": "google/byt5-base",
      "is_byt5_base": true,
      "stage1_physical_batch": 2,
      "stage1_accumulation_steps": 8,
      "stage1_effective_batch": 16,
      "stage2_physical_batch": 2,
      "stage2_accumulation_steps": 4,
      "stage2_effective_batch": 8,
      "precision": "fp16",
      "gradient_checkpointing": true,
      "parameter_count": 582455808,
      "generate_success": true,
      "latency_ms": 142.5,
      "peak_memory_mb": 2450.0
    }
  },
  "success": true
}
```
