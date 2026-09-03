# Quickstart

Run the exact-method tests:

```powershell
python -m pytest tests/test_metrics_exact.py tests/test_statistics_exact.py tests/test_annotation_reliability.py tests/test_efficiency_protocol.py tests/test_methodology_integration.py -q
```

Then run the full suite:

```powershell
python -m pytest -q
```

Before accepting a Chapter 3 result, confirm: corpus and sentence metrics are both present; bootstrap reports 1,000 paired resamples; each efficiency family names exactly latency and memory; memory has 20 paired CUDA observations; per-category alpha is at least 0.80; and every result includes spec 013 provenance.
