# Quickstart

Run the focused tests:

```powershell
python -m pytest tests/test_checkpoint_handoff.py -q
```

For a two-stage run, inspect the resulting Stage 2 checkpoint and confirm its `parent_checkpoint_id`, handoff source epoch, source validation loss, and restored model fingerprint match `best_stage1.pt`. If Stage 1 is explicitly skipped in a development command, confirm the handoff is recorded as not applicable rather than successful.
