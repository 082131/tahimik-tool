# Quickstart: Validating the Noise Estimator

How a developer proves this feature works, end to end, without needing the
gold-standard dataset. Covers Stories 1 and 3 from `spec.md`; Story 2 stays
blocked until real data exists.

## Prerequisites

- The project's normal environment: `pip install -r requirements.txt`.
- No GPU required — these checks run on CPU, following the existing pattern
  in `tests/test_delete_gate.py`.
- No real dataset required — every check below uses hand-written or
  randomly-initialized stand-ins.

## Run the test suite

```bash
python -m pytest tests/test_noise_estimator.py -v
```

Expected: every test passes. If `tests/test_noise_estimator.py` doesn't
exist yet, that means Phase 2 (`/speckit-tasks`) and the test-first
implementation haven't happened — see Constitution Principle IV, this file
describes the target state, not a currently-runnable command.

## What "passing" actually proves (Story 1)

1. **Shape and range** (SC-001): feed a tiny randomly-initialized T5's
   layer-3 hidden states through the estimator for a batch of a few
   sentences. Confirm the output is one float per sentence, each in
   `[0, 1]`.
2. **Label correctness** (SC-002): compute `n*` for at least three
   hand-written (noisy, clean) pairs where you can count the edits yourself.
   Confirm the code's answer matches your hand count exactly — this is an
   exact deterministic calculation, not an approximate one.
3. **Gradient isolation** (SC-003): run one training step that computes
   `L_NE`, call `.backward()`, then inspect `.grad` on the noise estimator's
   parameters (should be non-`None`) versus the shared encoder's parameters
   (should be `None`, or unchanged from before the step). This is the test
   that directly protects the manuscript's stated guarantee.

## Manually spot-check precomputed labels (Story 3)

Before trusting a full training run, look at a handful of labels by hand:

1. Pick two or three (noisy, clean) pairs where the right answer is obvious
   by inspection — an unchanged sentence, and a heavily garbled one.
2. Run whatever `src/data/noise_label.py` exposes for computing `n*` on
   those pairs.
3. Confirm: the unchanged pair reads `0.0`, and the heavily garbled pair
   reads noticeably higher (close to, but not necessarily exactly, `1.0`,
   depending on how much of the sentence actually changed).

If a label looks wrong here, stop before running a full training job on it —
that's the entire point of this check existing as its own story.

## What this quickstart does not cover

- Whether the estimator's judgment is *good* on real Tagalog/Taglish text —
  blocked on the gold-standard dataset (spec SC-004, SC-005).
- How the delete-gate uses the noise score once produced — out of scope for
  this feature (see spec's "deferred to the delete-gate's own spec" note).
