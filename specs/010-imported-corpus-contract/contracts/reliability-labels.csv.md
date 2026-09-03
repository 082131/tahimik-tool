# Reliability-label CSV Contract

Required header, in any column order:

```csv
sentence_id,annotator_id,noise_category,present
```

- `sentence_id` must reference the accepted gold export.
- `annotator_id` is a stable pseudonymous identifier.
- `noise_category` must match the configured category vocabulary exactly.
- `present` is binary: `0` (absent) or `1` (present).
- Duplicate sentence/annotator/category keys, unknown categories, missing cells, or non-binary values are invalid.
- A full run requires 3,000 unique sentences and three complete annotators per sentence.
