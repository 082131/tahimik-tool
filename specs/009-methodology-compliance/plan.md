# Implementation Plan: Chapter 3 Methodology Compliance

## Technical context

Python 3.10+, NumPy/SciPy, sacrebleu, NLTK, PyTorch, and the existing `src/evaluation`, `src/data`, `src/training`, and `scripts` modules. No new runtime dependency is required.

## Approach

1. Add shared metric/per-sentence helpers and a source-aware GLEU+ implementation.
2. Correct bootstrap, Wilcoxon effect reporting, pair selection, and step-down Holm correction.
3. Benchmark per-sentence latency and per-run GPU memory with CPU-safe output.
4. Add gold-pair validation and configuration-driven protected synthetic noise.
5. Add reproducibility/determinism utilities and attach metadata to JSON/checkpoints.
6. Update experiment orchestration, tests, and Chapter 3 documentation.

## Constraints

The annotator website and human annotation workflow are external deliverables and are not fabricated by this code change. GPU tests are conditional on CUDA availability.
