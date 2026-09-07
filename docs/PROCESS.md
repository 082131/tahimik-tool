# How we build TAHIMIK

A walkthrough of the development process, written to be presented from or read
cold. If you only have two minutes, read "The one-sentence version" and "What it
caught".

---

## The one-sentence version

Before any component is built or defended, it gets a written specification; the
specification is then checked against the code that actually exists, and every
disagreement between the two is recorded rather than quietly resolved.

---

## Why bother

Three reasons, in the order an examiner would care about them.

**Trust in the numbers.** A thesis is graded on whether its results can be
believed. That requires showing that the code does what the manuscript says it
does. Nobody can verify that by reading 4,000 lines of PyTorch. They can verify
it by reading a specification and a list of where the code diverges from it.

**The AI-use declaration.** Saying "AI assisted this work" is weak. Showing a
directory of specifications, each recording which decisions came from the
author, which came from the manuscript, and which are still open, is strong. It
demonstrates the reasoning was directed by a person.

**Recoverable reasoning.** Six months from now, "why is the delete gate at
layer 3?" has a written answer instead of a guess.

---

## The loop

Each component goes through the same sequence. Roughly one sitting for a small
component.

| Step | What happens |
|------|--------------|
| **Interrogate** | The author is asked pointed questions about the design and answers them. This is where the thinking happens. |
| **Specify** | The answers become `specs/NNN-name/spec.md` — what the component *should* do. |
| **Clarify** | A second structured pass catches gaps the interrogation missed. |
| **Plan** | How it gets built here, in this codebase. |
| **Tasks** | Broken into work items. |
| **Converge** | The code is read and compared against the spec. Every gap is recorded. |
| **Analyze** | Spec, plan, and tasks are checked for internal agreement. |
| **Build** | Only the gap gets built, test-first. |

Two rules make this work rather than becoming paperwork:

**Write the spec without reading the implementation.** If you read
`noise_estimator.py` first and describe what it does, you produce a spec that
agrees with every bug in it. The spec has to say what the component *should* do,
independently, so that convergence has something real to compare against.

**Never invent an answer to an open design question.** If the manuscript is
silent and the author is unsure, it is marked `NEEDS CLARIFICATION` and left
open. A blank gets noticed. A plausible guess gets defended at a panel by
someone who believes it was a decision.

---

## What convergence produces

Four categories, and the third is the one worth reading:

- **missing** — the code doesn't do it yet
- **partial** — it does some of it
- **contradicts** — the code does something that conflicts with the manuscript
  or the project constitution
- **unrequested** — the code does something no spec asked for

Convergence never edits code. It only reports. Fixing is a separate, deliberate
step, so nothing gets "helpfully" changed without a decision.

---

## What it caught

Six components specified, roughly 40 requirements checked. The substantial
majority were already correctly implemented. Full list in
[`specs/FINDINGS.md`](../specs/FINDINGS.md); the headline items:

**The significance test doesn't match the manuscript.**
`src/evaluation/statistical_tests.py:94` computes a one-tailed bootstrap
p-value. The manuscript specifies a two-tailed one with add-one smoothing. A
one-tailed test is roughly twice as easy to pass, so results currently called
significant may not survive the stated method. The Wilcoxon test in the same
file *is* correctly two-sided — the two disagree with each other, which is what
makes it clearly an oversight rather than a choice.

This is the kind of thing that is very hard to catch by reading code and
comparatively easy to catch by comparing code against a written specification.
It is also exactly the kind of thing an examiner might find first.

**Significance ignores the confidence interval.** The manuscript requires both
`p < 0.05` and a CI excluding zero. The code checks only the p-value, having
computed the CI two lines earlier.

**The code ran a smaller model than the manuscript specified.** `byt5-small`
versus `byt5-base`. Deliberate as an early development default (resolved in AD-002;
the codebase now authoritatively runs `google/byt5-base` across all variants),
a problem only if a number from small was reported as a result.

**Runs are seeded but not bit-reproducible.** Seeding is correct everywhere.
PyTorch's deterministic mode is not enabled, so GPU runs can still vary
slightly.

And what was found *correct* matters as much: the gradient isolation the entire
three-variant comparison depends on — the guarantee that the noise estimator's
training signal cannot leak into the shared encoder — is genuinely implemented,
at both points the manuscript requires, not merely asserted in a comment.

---

## The rules underneath

Specifications answer to a ratified project constitution
([`.specify/memory/constitution.md`](../.specify/memory/constitution.md), v1.0.0),
which fixes five principles: branch discipline, spec-driven development,
reproducibility, test-first for model code, and transparent AI assistance.
Day-to-day mechanics live in [`CONTRIBUTING.md`](../CONTRIBUTING.md), and the
reasoning behind each rule is in [`WHY.md`](WHY.md).

The constitution is deliberately harder to change than the code it governs:
amendments go through the same branch-and-review process, with a version bump
and stated reasoning.

---

## Honest limitations

Worth saying out loud rather than being asked:

- **Only `002` was author-interrogated.** Specs `003` through `007` were
  derived from the manuscript and the code under time pressure. This is
  recorded per-spec in [`specs/PROVENANCE.md`](../specs/PROVENANCE.md). The
  manuscript is a legitimate authority — the author wrote it — but it is not
  the same as answering questions live, and the record does not pretend
  otherwise.
- **Nothing found has been fixed yet.** The findings are recorded, not
  resolved.
- **Anything requiring the gold-standard dataset is blocked, not done.** Every
  spec separates what is checkable now from what needs real Tagalog/Taglish
  data, and nothing blocked is claimed as complete.
- **`008` (annotation platform) and `009` (demo tool) are not specified.** The
  first is still changing weekly; specifying it now would produce a document
  that is wrong on arrival. The second is not part of the contribution.
