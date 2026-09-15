# Quickstart: Verify Small Model Selection

```powershell
python -m pytest tests/test_experiment_configuration.py tests/test_model_configuration.py tests/test_checkpoint_compatibility.py tests/test_gradient_accumulation.py -v
```

Expected configuration:

- ByT5: `google/byt5-small`
- MrT5: `stanfordnlp/mrt5-small`
- TAHIMIK: `stanfordnlp/mrt5-small`

Then run `python -m pytest tests/ -v`.
