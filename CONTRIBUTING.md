# Contributing

Working agreement for the TAHIMIK team. The goal is a repository where anyone
can see what changed, why, and whether it still works — including a panelist
reading the history during defense.

New here, or wondering why a rule exists? [docs/WHY.md](docs/WHY.md) explains
the reason behind each one in plain language.

---

## Branching

`main` is always working. Nothing is committed directly to it.

"Working" is two checkable things, not a feeling: CI is green, and a
synthetic-noise run started from `main` at `seed = 42` reproduces its previous
output.

Every change happens on a branch named `<type>/NNN-<short-description>`:

```
fix/003-delete-gate-training-signal
feat/007-normalization-tool
docs/012-methodology-chapter
exp/015-gate-layer-ablation
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

**Every branch carries a number, including branches with no spec behind
them.** The next number is one above the highest already used in either
`specs/` or any existing branch name — check both before you cut a branch.

Spec Kit numbers its own branches by scanning `specs/` alone, so it will
happily reuse a number an unspec'd branch already took. When that happens the
number you assigned wins, and the `specs/` directory gets renamed to match.
Spec Kit also omits the `<type>/` prefix, so a branch it creates needs
renaming before you push it:

```bash
git branch -m feat/002-noise-estimator
```

Branches are deleted by hand after their PR merges, by whoever merged it.
GitHub's "automatically delete head branches" setting is **off deliberately**
and should stay off — deleting a branch is a decision, not a default.

Keep branches short-lived and focused. One branch that changes the loss
function *and* restyles the frontend is difficult to review and impossible to
revert cleanly.

```bash
git checkout main
git pull
git checkout -b fix/004-some-thing
```

`exp/` is the one prefix that skips the spec loop — see
[docs/SPEC-WORKFLOW.md](docs/SPEC-WORKFLOW.md). It skips nothing else. An
experiment still obeys `configs/`, the seed, and the tests, because the whole
point of an experiment is that it might produce a number you want to keep.

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

Push the branch and open a PR against `main`:

```bash
git push -u origin fix/some-thing
```

The PR body should answer:

1. What problem does this solve?
2. How was it verified? (tests, a training run, a screenshot)
3. Anything reviewers should look at closely?

**At least one teammate reviews before merge.** For changes to `src/models/`
or `src/training/`, the reviewer must be someone who can check the maths.
That is the code the panel will question hardest.

If no review arrives within **48 hours**, the author may merge and must say so
in the PR, with the reason. That escape hatch exists to stop work deadlocking,
not to be the normal path — if you are reaching for it every time, the review
rule has failed and belongs in a conversation, not in a quiet workaround.

Merge with a **merge commit, never a squash.** Squashing collapses the
per-commit reasoning into one blob, and that reasoning is the part of this
history worth reading.

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

A spec is required for every `feat/` branch and for any change to
`src/models/` or `src/training/`, whatever the branch is called. Fixes, docs,
CI, chores, and refactors elsewhere go straight to a branch and a PR — the
loop protects the contribution, and applying it to a typo would turn it into
ritual.

The principles all of this answers to live in
[`.specify/memory/constitution.md`](.specify/memory/constitution.md). Where
this document and the constitution disagree, the constitution wins and this
file is the one that needs fixing.

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
