# Research: Small Model Migration

## Decision

Use `google/byt5-small` for the uncompressed accuracy baseline and `stanfordnlp/mrt5-small` for both compressed variants. Model source is part of architecture identity, not merely a download location.

## Evidence in the system

- `configs/byt5_config.py` owns the Google Small override.
- `configs/base.py` supplies the Stanford MrT5 Small default inherited by MrT5 and TAHIMIK.
- `tests/test_experiment_configuration.py` and `tests/test_model_configuration.py` lock the three resolved identifiers.
- `tests/test_checkpoint_compatibility.py` records the Small fingerprint and rejects Base or different-source checkpoints.

## Preserved controls

The physical batches are 4 with accumulation 4 in Stage 1 and 4 with accumulation 2 in Stage 2. These preserve effective batches of 16 and 8. Data, seed 42, truncation ceilings, optimizer settings, two-stage schedule, and evaluation contracts remain shared.

## Rejected alternatives

- Loading Base weights into Small models: incompatible hidden width and layer counts.
- Treating Google Small and Stanford MrT5 Small as interchangeable: their source/training identity differs even when dimensions match.
