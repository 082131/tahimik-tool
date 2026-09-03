# Gold-pairs CSV Contract

Required header, in any column order:

```csv
sentence_id,noisy,clean,review_status,anonymization_status
```

- UTF-8 with a single header row.
- Every value is required; IDs are opaque strings.
- `review_status` must be `approved`; `anonymization_status` must be `complete`.
- `noisy` and `clean` must each contain at least four words and at most 1,024 UTF-8 bytes.
- Sentence IDs and pair records must be unique. Conflicting clean targets for the same noisy input are invalid.
- Additional columns may be retained by the external system but are ignored unless explicitly versioned into this contract.
