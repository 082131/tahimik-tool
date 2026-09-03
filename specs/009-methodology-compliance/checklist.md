# Chapter 3 Compliance Checklist

- [ ] Source-aware GLEU+ uses noisy, clean, and predicted text.
- [ ] Bootstrap is 1,000 paired, two-tailed, add-one smoothed, percentile CI, and CI-gated.
- [ ] Wilcoxon is two-sided on independent per-run memory observations with median/IQR/effect size.
- [ ] Holm is step-down and reports raw/adjusted p-values for the two prescribed comparisons.
- [ ] Gold data validation and duplicate/conflict handling are logged.
- [ ] Protected Taglish/slang/emoji behavior is tested.
- [ ] JSON/checkpoints contain reproducibility metadata and dirty-tree status.
- [ ] Annotator website/annotation is explicitly external, not falsely claimed as implemented.
