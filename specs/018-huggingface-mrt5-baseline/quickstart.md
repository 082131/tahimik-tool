# Quickstart: Verify the MrT5 Baseline

```powershell
python -m pytest tests/test_mrt5_model_loading.py tests/test_delete_gate.py tests/test_model_forward.py -v
```

Check that MrT5 resolves to `stanfordnlp/mrt5-small`, uses `trust_remote_code=True`, disables its embedded gate, and refuses a random-gate fallback.
