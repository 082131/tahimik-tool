# Spec-Driven Workflow

How a change gets from an idea in someone's head into `main`, and why the
paper trail it leaves is part of the thesis rather than overhead.

This sits alongside [CONTRIBUTING.md](../CONTRIBUTING.md). That document
covers branches, commits, and PRs — the mechanics of moving code. This one
covers what happens *before* the first line is written, and what record it
leaves behind.

---

## Why bother

Three reasons, in descending order of how much they matter at defense:

1. **AI-use declaration.** Most venues now require you to state where AI
   assisted the work. "I used Claude" is a weak answer. A directory of specs
   showing the questions asked, the answers given, and the decisions made is
   a strong one — it demonstrates the reasoning was yours and the typing was
   assisted, which is the distinction the declaration is actually asking about.
2. **Reviewability.** A panelist can read `specs/003-noise-estimator/spec.md`
   and understand the design without reading `src/models/noise_estimator.py`.
3. **Reproducibility.** Six months from now, "why is the delete gate at layer
   3?" has a written answer instead of a guess.

---

## The two toolkits, and who does what

Two skill sets are installed. They overlap, so the division is deliberate —
**do not run both halves of the same job.**

| Phase | Tool | Why this one |
|-------|------|--------------|
| Interrogation | `/grill-me` (Pocock) | Open-ended and adversarial. Keeps digging until the design stops wobbling. |
| Specification | `/speckit-specify` | Writes to `specs/NNN-name/spec.md` — numbered, versioned, reviewable. |
| Gap-check | `/speckit-clarify` | Fixed structured questions. Catches what grilling missed. |
| Planning | `/speckit-plan` | Produces `plan.md` against the spec. |
| Task breakdown | `/speckit-tasks` | Produces `tasks.md`. |
| Reality check | `/speckit-converge` | Compares tasks to code that already exists. |
| Consistency audit | `/speckit-analyze` | Cross-checks spec vs plan vs tasks before any code moves. |
| Building | `/tdd` (Pocock) | Red-green-refactor. Better fit for research code than bulk generation. |

**Deliberately unused:** `to-spec`, `to-tickets`, `triage`, `wayfinder`.
They are good skills, but they maintain a *second* artifact tree. One source
of truth or none.

---

## When a spec is required

Not every change earns one. A spec is required for:

- every `feat/` branch, anywhere in the repository
- any change to `src/models/` or `src/training/`, whatever the branch type

Everything else — fixes outside the model code, docs, CI, chores, refactors —
goes straight to a branch and a PR. The loop exists to protect the thesis
contribution. Pointing it at a typo fix turns it into ritual, and rituals get
skipped precisely when they would have mattered.

`exp/` branches skip the loop by design. An experiment is allowed to be
unspecified; it is not allowed to be untraceable, so `configs/`, the seed, and
the tests still apply in full.

---

## The loop, per feature

Run once for the project:

```
/speckit-constitution
```

Then per feature, in order:

```
/grill-me            → figure out what this thing actually is
/speckit-specify     → write it down as specs/NNN-name/spec.md
/speckit-clarify     → close the gaps grilling left open
/speckit-plan        → how it gets built
/speckit-tasks       → broken into work items
/speckit-converge    → which of those already exist in the code
/speckit-analyze     → do spec, plan, and tasks still agree?
/tdd                 → build only the delta
```

Steps 1–3 are a conversation. You answer; nothing is invented on your behalf.
If a question cannot be answered yet, the correct response is "unknown" —
that gets recorded as an open question rather than papered over.

### Open questions

A question you cannot answer is marked `NEEDS CLARIFICATION` in the spec.
Two rules govern them, and they are deliberately different:

- **A spec may merge with open questions.** Some are blocked on the gold
  standard, some on your adviser. Holding the whole spec hostage to a question
  nobody can answer this week helps no one, and an unmerged spec is a spec
  nobody reads.
- **A task that depends on an open question may not be built.** `/tdd` does
  not start on it. This is the rule that makes the first one safe.

Documented and ready-to-build are different states. Blurring them is how a
guess ends up in the codebase wearing a spec's authority.

---

## Retrofitting specs onto code that already exists

Most of TAHIMIK is already written. Spec-kit's default direction is
spec → code, so the retrofit runs slightly differently:

- The spec describes **what the component should do**, written as though the
  code did not exist. Do not read the implementation and transcribe it — that
  produces a spec that agrees with every bug.
- `/speckit-converge` then compares the spec to reality. Three outcomes per
  task: **already done**, **missing**, or **done differently than specified**.
- The third outcome is the valuable one. It is either a bug or an
  undocumented design decision, and both are worth knowing before defense.

### Suggested order

Smallest and most self-contained first, so the process is proven on something
cheap before it is applied to the core contribution.

| # | Feature | Notes |
|---|---------|-------|
| 001 | Noise estimator | Small, isolated MLP. Good pilot. |
| 002 | Delete gate | Already has regression tests to check the spec against. |
| 003 | ByT5 baseline | The control variant. |
| 004 | Fixed-rate compression (MrT5) | Baseline two. |
| 005 | Noise-adaptive ByT5 | The actual contribution. Spec this once the pattern is established. |
| 006 | Two-stage training | Synthetic pretrain → gold fine-tune. |
| 007 | Evaluation + statistics | Metrics, benchmarks, significance tests. |
| 008 | Annotation platform | **Deferred** — still under development, see below. |
| 009 | Demo tool (backend + frontend) | Not part of the contribution; spec last. |

---

## Not in scope yet

The gold-standard dataset and the annotation platform that produces it are
still under development. They do not get spec'd for now — specifying something
whose shape is still changing weekly produces a document that is wrong by the
time it is committed, and rewriting it every week is worse than not having it.

They come into scope once the annotation platform settles and the dataset
exists. Until then, feature 008 in the table above is a placeholder.

---

## What lands in git

```
specs/
  001-noise-estimator/
    spec.md         what it should do, and why
    plan.md         how it gets built
    tasks.md        the work items, with converge results
```

Committed, not gitignored. The record is the point — a spec that only exists
on one laptop proves nothing to anyone.
