# Quickstart

1. Provide the accepted gold training IDs and long-form reliability labels from spec 010.
2. Provide an approved bounds configuration for every declared category and any reviewed lexicon CSV resources.
3. Resolve and save the probability manifest before generation.
4. Run focused tests:

```powershell
python -m pytest tests/test_noise_policy.py tests/test_noise_generator_contract.py -q
```

5. Generate with the saved manifest and seed. Re-running with identical inputs must reproduce pairs and lineage.

Until final category bounds are supplied, development fixtures may be used, but `scripts/run_experiment.py` must reject the run as Chapter 3 ineligible.
