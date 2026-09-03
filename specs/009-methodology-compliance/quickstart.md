# Verification Quickstart

Run `python -m pytest tests/ -v`.

For a manuscript-eligible run, use a clean git tree, a fixed seed, CUDA, the resolved configuration emitted in the JSON, and inspect `statistical_tests` for exactly two model comparisons. CPU runs are useful for accuracy/latency development checks but are not GPU-memory evidence.
