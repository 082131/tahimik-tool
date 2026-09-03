# Research and Decisions

## Scope boundary

**Decision:** Treat the separate gathering and annotation systems as authoritative for consent, anonymization, language, spam, advertising, link-only, and sensitive-content review.

**Rationale:** Reclassifying those semantic decisions with heuristics would silently change the approved corpus. This repository verifies explicit approval fields and structural rules only.

## CSV shape

**Decision:** Use one row per gold pair and one row per sentence/annotator/category reliability judgment.

**Rationale:** Long-form labels make missing annotator-category cells detectable and allow nominal alpha to be computed per category.

## Duplicate policy

**Decision:** Exact repeated records are reported and rejected; one noisy input mapped to different clean targets is a conflict and always fails.

**Rationale:** Silent deduplication can make declared counts differ from the externally approved export.

## Deterministic splitting

**Decision:** Sort by stable `sentence_id`, apply a seed-controlled permutation, then slice 12,000/1,500/1,500.

**Rationale:** It is independent of CSV row order and reproducible across runs.

## Privacy-safe reporting

**Decision:** Reports contain IDs, counts, fingerprints, and reason codes, but never source or target text.

**Alternatives rejected:** logging invalid rows or automatic text repair, because both leak or mutate reviewed data.
