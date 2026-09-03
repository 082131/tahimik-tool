# Data Model

## RunProvenance

Top-level groups: `schema_version`, `provenance_id`, `code`, `configuration`, `operation`, `randomness`, `software`, `hardware`, `determinism`, `data`, `lineage`, and `eligibility`.

## Eligibility

Fields: boolean `eligible`, sorted unique `reason_codes`, and `scope` (`chapter3_full`, `accuracy_only`, or `development`). Eligibility is computed; callers cannot set it directly.

## ArtifactLineage

Fields may include parent checkpoint ID, evaluated checkpoint ID, gold/split fingerprints, probability manifest ID, and lexicon resource versions.

## Invariants

- `provenance_id` hashes canonical content excluding itself.
- Equivalent resolved configurations have identical fingerprints.
- Every required key is present even when its value is unavailable.
- No raw text or secret environment values are stored.
