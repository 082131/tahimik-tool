# Quickstart: Paired Length Policy

1. Load the active tokenizer used by the selected study variant.
2. Run the shared preparation function on all gold or synthetic pairs.
3. Log the returned audit summary.
4. Split only the prepared noisy strings, clean strings, labels, and source
   indices.
5. Construct `NormalizationDataset` with `truncation=False` validation.

Run verification:

```powershell
python -m pytest tests/test_paired_length_policy.py -v
python -m pytest tests/ -v
```
