# Quickstart

1. Export finalized gold pairs and reliability labels using the contracts in `contracts/`.
2. Keep the CSV files outside Git (for example under a locally ignored data directory).
3. Run the contract tests:

```powershell
python -m pytest tests/test_import_contracts.py -q
```

4. Run `scripts/run_experiment.py` with the two export paths and seed. The command must print/write an eligibility report before loading a tokenizer or model.
5. Confirm `eligible=true`, exact split counts, zero rejection counts, and saved fingerprints before treating the run as thesis evidence.

Development scripts may accept small fixtures, but their output is not Chapter 3 eligible.
