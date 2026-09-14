# Research: Paired Length Policy

## Decision: paired aligned-word truncation

The existing dataset computes byte edit-distance labels before
`NormalizationDataset` independently applies tokenizer truncation.  The
selected policy instead prepares a pair before labels or splits exist.

Use deterministic word-level Levenshtein alignment. Equal words and single
word substitutions remain individual groups; adjacent insertion/deletion
operations attach to a neighbouring substitution so split/merge corrections
remain atomic without merging independent corrections. Candidate text prefixes
are reconstructed from whole original words, then measured by the active tokenizer with
`truncation=False` and special tokens enabled.

## Rejected alternatives

- Raw byte truncation: can split a UTF-8 character or word and violates the
  confirmed complete-word requirement.
- Independent last-space truncation: can retain a clean group after its noisy
  partner has been omitted.
- Same word-index truncation: fails common split/merge normalizations.
- Dropping every overlength pair: loses usable future data contrary to the
  confirmed policy.
