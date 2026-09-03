# Data Model

## CategoryProbability

Fields: `category`, `group`, `positive_count`, `total_count`, `observed_rate`, `lower_bound`, `upper_bound`, and `resolved_probability`. Bounds satisfy `0 <= lower <= upper <= 1`; resolved probability is the clamped observed rate.

## ProbabilityManifest

Fields: schema version, manifest ID, source gold/split fingerprints, seed, category records, lexicon resource versions, creation timestamp, and readiness state. Its ID is a SHA-256 digest of canonical content excluding the ID itself.

## LexiconEntry

Fields: `resource_id`, `resource_version`, `category`, `source_form`, `target_form`, `review_status`, `source_reference`, and `license_note`. Only `approved` entries may be used.

## SyntheticPairLineage

Fields: pair ID, base sentence ID, manifest ID, resource versions, applied preserved categories, applied correctable categories, and seed derivation. Lineage metadata may be stored separately from text.

## Generation transition

`clean base -> preserved augmented target -> copy to input -> input-only corruption -> pair + lineage`
