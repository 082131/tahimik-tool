# Chapter 3 Gap Remediation Design

The repository consumes finalized CSV exports from separate gathering and annotation systems. It does not collect data, reproduce the annotation website, scrape KWF, or modify the manuscript.

## Feature boundaries

1. `010-imported-corpus-contract`: validate imports and fixed experiment counts.
2. `011-synthetic-noise-policy`: derive auditable probabilities and separate preserved augmentation from correctable noise.
3. `012-stage-checkpoint-handoff`: start Stage 2 from the best Stage 1 checkpoint.
4. `013-experiment-provenance`: stamp complete provenance and eligibility on every artifact.
5. `014-methodology-verification`: correct and test Chapter 3 metrics and statistics.

## Settled decisions

- `run_experiment.py` always enforces 15,000 gold pairs, a 12,000/1,500/1,500 split, and 1,000,000 synthetic pairs. There is no manuscript-mode flag.
- Development commands may use smaller inputs, but their outputs are ineligible for thesis reporting.
- The annotation website exports separate gold-pair and long-format binary-label CSV files.
- Nominal Krippendorff alpha is recomputed per category; any value below 0.80 blocks the full experiment.
- Synthetic probabilities use training labels only; category bounds remain required-but-unset until annotation finishes.
- Meaningful slang, emoji, code-switching, and Taglish morphology are injected into both input and target. Correctable corruption affects only the input.
- A reviewed lexicon CSV is supported; KWF is never scraped. A documented API provider may be added after permission is granted.
- Work uses one integration branch, independently reviewable commits, and one final push.
