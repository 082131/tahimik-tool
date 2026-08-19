# Contributing

Working agreement for the TAHIMIK team. The goal is a repository where anyone
can see what changed, why, and whether it still works — including a panelist
reading the history during defense.

---

## Branching

`master` is always working. Nothing is committed directly to it.

Every change happens on a branch named `<type>/<short-description>`:

```
fix/delete-gate-training-signal
feat/normalization-tool
docs/methodology-chapter
exp/gate-layer-ablation
```

| Prefix | Use for |
|--------|---------|
| `feat/` | New capability |
| `fix/` | Correcting broken behaviour |
| `docs/` | Documentation only |
| `test/` | Tests only |
| `refactor/` | Restructuring with no behaviour change |
| `exp/` | Experiments that may never merge |

Keep branches short-lived and focused. One branch that changes the loss
function *and* restyles the frontend is difficult to review and impossible to
revert cleanly.

```bash
git checkout master
git pull
git checkout -b fix/some-thing
```

---

## Commits

[Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <summary in the imperative, under ~70 chars>

Why the change was needed. What was wrong before. What is different now.
Wrap the body at 72 characters.
```

Rules we hold to:

- **One logical change per commit.** Not "fixed stuff and also the README".
- **Explain the why.** The diff already shows the what. Six months from now,
  during revisions, the reasoning is what you will need.
- **Never commit secrets.** `.env` is gitignored. Only `.env.example` is
  tracked, and it holds placeholders.
- **Never commit large binaries.** No checkpoints (`*.pt`), no datasets, no
  `node_modules`. All are gitignored — check `git status` before staging.

Research code specifically: if a commit changes model behaviour, say so
explicitly and state which variants are affected. A one-line message on a
change to the loss function is how a result becomes unreproducible.

---

## Pull requests

Push the branch and open a PR against `master`:

```bash
git push -u origin fix/some-thing
```

The PR body should answer:

1. What problem does this solve?
2. How was it verified? (tests, a training run, a screenshot)
3. Anything reviewers should look at closely?

**At least one teammate reviews before merge.** For changes to `src/models/`
or `src/training/`, the reviewer should be someone who can check the maths.
That is the code the panel will question hardest.

---

## Tests

```bash
python -m pytest tests/ -v
```

Tests must pass before a PR is merged.

When fixing a bug, **write the test first** and confirm it fails against the
broken code. A regression test that never failed proves nothing.
`tests/test_delete_gate.py` is the model to follow: each test names the defect
it guards against and explains why the behaviour matters.

---

## Repository layout

```
configs/              Hyperparameters — the study's control variables
src/
  data/               Noise generation, datasets, preprocessing
  models/             Noise estimator, delete gate, the three variants
  training/           Losses and the two-stage trainer
  evaluation/         Metrics, efficiency benchmarks, statistics
  utils/
scripts/              CLI entry points (train, evaluate, benchmark)
tests/                Regression tests
backend/              FastAPI inference server
frontend/             The TAHIMIK demo tool (React + Vite)
annotation-platform/  Annotator app used to build the gold standard
```

---

## Reproducibility

The study compares three variants under identical control variables. Protect
that:

- Change hyperparameters in `configs/`, never inline in a script.
- A change to `configs/base.py` affects **all three variants** — call it out in
  the commit message.
- Keep `seed = 42` unless you are deliberately testing seed sensitivity.
- Record which commit produced which results. A results file that cannot be
  traced to a commit cannot be defended.
