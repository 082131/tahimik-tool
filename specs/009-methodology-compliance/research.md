# Research Decisions

- Accuracy and latency are paired by sentence; GPU memory is paired by timed run.
- Bootstrap uses the manuscript's two-tailed add-one formula and requires both `p < alpha` and a CI excluding zero.
- Holm is step-down, with adjusted p-values reported within the declared two-comparison families.
- Duplicate gold pairs are removed only when identical; conflicting normalizations are rejected for human review.
- Meaningful code-switching, Taglish morphology, slang, and emoji are protected from synthetic corruption. This avoids creating artificial language errors that the manuscript does not claim to model.
- CPU-only environments do not invent GPU memory values; the corresponding Wilcoxon result is marked unavailable.
