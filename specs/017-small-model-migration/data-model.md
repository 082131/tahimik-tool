# Data Model: Small Model Migration

## Variant Source Map

| Variant | Model source | Role |
|---|---|---|
| `byt5` | `google/byt5-small` | Uncompressed accuracy baseline |
| `mrt5` | `stanfordnlp/mrt5-small` | Fixed-rate compression baseline |
| `tahimik` | `stanfordnlp/mrt5-small` | Noise-adaptive proposed variant |

## Checkpoint Architecture Record

- `model_name`: exact pretrained source identifier
- `d_model`: 1472
- `num_encoder_layers`: 12
- `num_decoder_layers`: 4
- `vocab_size`: 384

Compatibility requires both matching dimensions and matching `model_name`.
