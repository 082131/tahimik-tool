# Quickstart

```powershell
python -m pytest tests/test_provenance.py -q
```

Run any command (`train.py`, `run_experiment.py`, `evaluate.py`, or `benchmark.py`) and inspect its JSON/checkpoint metadata. Confirm the same schema groups exist, `configuration` contains resolved primitive values, GPU/CUDA fields are explicit, lineage points to the exact inputs, and any development condition has a machine-readable ineligibility reason.
