# Data Model: Paired Length Policy

## PreparedPair

| Field | Meaning |
|---|---|
| `source_index` | Original zero-based collection index |
| `noisy` | Complete-word retained noisy prefix |
| `clean` | Complete-word retained clean prefix |
| `noise_level` | `compute_noise_level(noisy, clean)` after preparation |
| `was_transformed` | Whether either retained text differs from source |

## PairPreparationAudit

| Field | Meaning |
|---|---|
| `total_pairs` | Supplied pair count |
| `unchanged_pairs` | Pairs retained as supplied |
| `transformed_pairs` | Pairs shortened at an aligned boundary |
| `excluded_pairs` | Pairs with no fitting first aligned group |
| `exclusions` | `(source_index, reason)` entries |
