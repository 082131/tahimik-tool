# Data Model: Hugging Face MrT5 Baseline

## Model Source

- Identifier: `stanfordnlp/mrt5-small`
- Loader: `AutoModelForSeq2SeqLM.from_pretrained`
- Loader option: `trust_remote_code=True`

## Gate Transfer

```text
remote model
  -> encoder.block[config.delete_gate_layer].delete_gate
  -> capture state
  -> disable embedded has_delete_gate
  -> map feed_forward/layer_norm weights
  -> local DeleteGate
```

The resulting wrapper exposes loss, deletion-rate, and telemetry fields expected by the shared trainer and backend.
