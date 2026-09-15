# Quickstart: Verify the TAHIMIK Baseline

```powershell
python -m pytest tests/test_mrt5_model_loading.py tests/test_model_forward.py tests/test_training_diagnostics.py -v
```

Confirm that both compressed variants load `stanfordnlp/mrt5-small`, only one gate executes, and the estimator is trained only by `L_NE`.
