# Quickstart: Validating Evaluation and Statistics

The components that decide what your thesis reports. Worth checking carefully.

## Prerequisites

- `pip install -r requirements.txt`
- Accuracy metrics run on CPU.
- **Peak GPU memory cannot be validated without CUDA.** On CPU it returns `0.0`,
  which is a sentinel, not a measurement.

## Run the tests

```bash
python -m pytest tests/test_metrics.py tests/test_statistical_tests.py -v
```

**Neither file exists yet.** These are the least-tested components in the
repository and the ones with the most direct influence on reported results.

## The check that would have caught the biggest bug

The bootstrap p-value must be two-tailed with add-one smoothing. Save as
`check_pvalue.py`:

```python
import numpy as np

# Two identical distributions: the true p-value should be near 1.0,
# because there is no real difference to detect.
rng = np.random.RandomState(42)
a = rng.normal(0.5, 0.1, 200)
b = rng.normal(0.5, 0.1, 200)

from src.evaluation.statistical_tests import StatisticalAnalysis
stats = StatisticalAnalysis(alpha=0.05, n_bootstrap=1000)
result = stats.paired_bootstrap(list(a), list(b), "identical_distributions")

print(f"p = {result['p_value']:.4f}")
print(f"CI = [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
print(f"significant = {result['significant']}")

# With no real difference, p should be far from 0 and the CI should span zero.
assert result["ci_lower"] < 0 < result["ci_upper"], "CI should contain zero"
assert result["p_value"] > 0.05, f"p={result['p_value']} — suspiciously low for identical inputs"
print("OK")
```

```bash
python check_pvalue.py
```

Run this against the current code and watch what happens: with a one-tailed
p-value, roughly half of no-difference cases produce a small `p` simply because
the resampled difference landed on one side. The CI correctly spans zero — which
is exactly why the manuscript requires **both** conditions, and why checking the
p-value alone is unsafe.

## Verify the two-tailed formula directly

The manuscript's formula, computed by hand on a known distribution:

```python
import numpy as np

deltas = np.array([0.1] * 700 + [-0.1] * 300)   # 70/30 split
B = len(deltas)

le = np.sum(deltas <= 0)
ge = np.sum(deltas >= 0)
p = 2 * min((1 + le) / (B + 1), (1 + ge) / (B + 1))

print(f"le={le} ge={ge}  two-tailed p = {p:.4f}")   # ~0.6014
print(f"one-tailed (current code) would give: {le / B:.4f}")  # 0.3000
```

The gap between `0.6014` and `0.3000` is the difference between "not
significant" and "borderline" on the same data.

## Check ERR's edge cases

```python
from src.evaluation.metrics import NormalizationMetrics
m = NormalizationMetrics()

# Perfect correction: ERR should be 1.0
print(m.compute_err(["grabe ang init"], ["grabe ang init"], ["grabeeee ang init"]))

# Made it worse: ERR should be NEGATIVE
print(m.compute_err(["completely wrong"], ["grabe ang init"], ["grabeeee ang init"]))

# Nothing to fix: degenerate case
print(m.compute_err(["same text"], ["same text"], ["same text"]))   # expect 1.0
```

A metric that cannot report a negative number would hide the model making text
worse. ERR can, and should.

## Check Holm-Bonferroni against hand computation

```python
from src.evaluation.statistical_tests import StatisticalAnalysis

results = StatisticalAnalysis.holm_bonferroni(
    [("time", 0.01), ("memory", 0.04)], alpha=0.05
)
for r in results:
    print(r)

# m=2. Smallest p compared against 0.05/2 = 0.025; next against 0.05/1 = 0.05.
# So time (0.01 < 0.025) is significant; memory (0.04 < 0.05) is too.
```

## Confirm the benchmark protocol

```bash
python -c "from src.evaluation.efficiency import EfficiencyBenchmark as E; import inspect; s=inspect.signature(E.__init__); print('warmup default:', s.parameters['warmup_passes'].default); print('runs default:', s.parameters['inference_runs'].default)"
```

Expect `5` and `20`, matching the manuscript.

## What this does not cover

- Actual metric values for the three variants — blocked on the gold standard
  (SC-006).
- Whether H₀₁ and H₀₂ are rejected — blocked (SC-007).
- **Peak GPU memory comparisons — blocked twice over** (SC-008): needs a GPU and
  real data. CI cannot produce these numbers at all, so per Constitution
  Principle III they must come from a recorded benchmark run traceable to a
  commit.
