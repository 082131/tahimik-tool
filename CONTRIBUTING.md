# Contributing

Working agreement for the TAHIMIK team. The goal is a repository where anyone
can see what changed, why, and whether it still works — including a panelist
reading the history during defense.

---

## Branching

`main` is always working. Nothing is committed directly to it.

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
| `ci/` | Build, CI, or workflow configuration |
| `chore/` | Housekeeping with no source change (ignores, tooling, deps) |

Branches are deleted by hand after their PR merges, by whoever merged it.
GitHub's "automatically delete head branches" setting is **off deliberately**
and should stay off — deleting a branch is a decision, not a default.

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

They also run automatically on GitHub. Every push and every pull request
triggers `.github/workflows/tests.yml`, which installs the dependencies and runs
the suite on Python 3.10 and 3.11.

- **Green check on a PR** — the suite passed on both versions.
- **Red X** — something broke. Click **Details** on the failed check to see which
  test failed and why. Do not merge until it is green.

If a run fails only on one Python version, the cause is usually a dependency
that resolved differently — read the install step's log before the test output.

When fixing a bug, **write the test first** and confirm it fails against the
broken code. A regression test that never failed proves nothing.
`tests/test_delete_gate.py` is the model to follow: each test names the defect
it guards against and explains why the behaviour matters.

---

## Repository layout

```
.specify/             Spec-kit templates, scripts, and the constitution
docs/                 Working agreements and process notes
specs/                One directory per feature: spec, plan, tasks
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

## Specs

Before a feature is built — or, for code that already exists, before it is
defended — it gets a spec under `specs/NNN-name/`. The process, the division
of labour between the two installed skill sets, and the order we are working
through the existing code are described in
[docs/SPEC-WORKFLOW.md](docs/SPEC-WORKFLOW.md).

Short version: you are interrogated about the design, your answers become the
spec, and the spec is what the panel reads.

---

## Working with AI assistants

Parts of this repository are written with AI assistance. That is allowed and
declared, but it only stays defensible if every assistant follows the same
rules — including across a change of tool, model, or account.

**If you are an AI assistant working in this repository, read this section,
then read [docs/SPEC-WORKFLOW.md](docs/SPEC-WORKFLOW.md), before proposing
changes.**

### Non-negotiable

- **Never commit directly to `main`.** Branch first, using the prefix table
  above. `main` is not protected on GitHub — the plan is free-tier and
  protected branches require Pro — so this rule is enforced by discipline
  alone. That makes it more important, not less.
- **Never invent an answer to a design question.** If the spec is silent on
  why the delete gate sits at layer 3, ask. A plausible-sounding guess written
  into a spec is worse than a blank, because it will be defended at a panel by
  someone who believes it was a decision.
- **Never commit secrets, datasets, checkpoints, or `node_modules`.** All are
  gitignored. Check `git status` before staging.
- **Never change hyperparameters inline.** They live in `configs/`. A change
  to `configs/base.py` affects all three variants and must be called out in
  the commit message.
- **Tests must pass** before anything is proposed for merge:
  `python -m pytest tests/ -v`

### How work is expected to flow

Features are not written from a prompt. They go through the spec loop in
[docs/SPEC-WORKFLOW.md](docs/SPEC-WORKFLOW.md) — the author is interrogated,
the answers become the spec, and the spec is what gets built against. An
assistant's job in that loop is to ask good questions and record the answers
faithfully, not to fill in the blanks.

Two skill sets are installed and they overlap. The division is set in
SPEC-WORKFLOW.md and is not a matter of preference: Pocock's `grill-me` for
interrogation, spec-kit for artifacts. `to-spec`, `to-tickets`, `triage`, and
`wayfinder` are deliberately unused, because a second artifact tree
describing the same features is worse than one.

If those tools are unavailable — a different agent, a different account — the
process still holds. Interrogate, write `specs/NNN-name/spec.md`, plan, break
into tasks, check the tasks against code that already exists, then build only
the gap. The tooling is a convenience; the sequence is the requirement.

### Commit attribution

Commits made with AI assistance carry a `Co-Authored-By:` trailer naming the
model. Do not remove it. It is the evidence behind the AI-use declaration,
and a history that hides the assistance is harder to defend than one that
states it plainly.

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
