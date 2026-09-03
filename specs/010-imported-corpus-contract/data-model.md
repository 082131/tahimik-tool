# Data Model

## GoldPair

| Field | Type | Rule |
|---|---|---|
| `sentence_id` | string | non-empty and unique |
| `noisy` | string | at least 4 whitespace-delimited words; at most 1,024 UTF-8 bytes |
| `clean` | string | same length rules as `noisy` |
| `review_status` | enum | exactly `approved` |
| `anonymization_status` | enum | exactly `complete` |

## ReliabilityLabel

| Field | Type | Rule |
|---|---|---|
| `sentence_id` | string | references an accepted gold pair |
| `annotator_id` | string | non-empty; pseudonymous |
| `noise_category` | string | member of configured category set |
| `present` | integer | `0` or `1` |

The full run requires exactly 3,000 unique reliability sentence IDs, three distinct annotators per sentence, and one judgment per required category per annotator.

## EligibilityReport

Contains `eligible`, input fingerprints, accepted/rejected counts, reason-code counts, rejected IDs, split sizes/fingerprints, and reliability coverage. It must not contain raw text.

## State transitions

`loaded -> structurally_validated -> corpus_validated -> split_assigned -> eligible`

Any failure transitions to `ineligible`; strict orchestration raises before tokenizer/model creation.
