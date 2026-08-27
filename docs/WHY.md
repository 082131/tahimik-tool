# Why we work this way

The rules in [CONTRIBUTING.md](../CONTRIBUTING.md) and the
[constitution](../.specify/memory/constitution.md) can look strict. This page
explains the reason behind each one in plain language, so nobody has to take
them on faith.

One idea sits under all of it: **a thesis is graded on trust.** An examiner has
to believe your numbers are real and your reasoning is yours. Every rule here
exists to make that trust easy to give.

---

## Why nothing goes straight to `main`

`main` is the version we treat as correct. If someone commits a half-finished
change to it, the "correct" version is now broken, and anyone who pulls it
inherits the mess.

So every change waits on its own branch until it is reviewed and passing. `main`
stays clean by construction, not by luck.

We can't turn on GitHub's automatic protection for this (that needs a paid
plan), so the rule holds only because we follow it. That makes it more
important, not less. There is no safety net underneath.

---

## Why features get a spec before any code

A spec is a short document describing what a feature should do, written before
you build it.

Writing it first forces the thinking to happen while it is still cheap to change
your mind. It also leaves a record. Six months from now, when an examiner asks
"why does the delete gate sit at layer 3," the answer is written down instead of
half-remembered.

The order matters. If you write code first and describe it afterward, the
description just agrees with whatever the code does, bugs included. Spec first
means the spec can catch the bug.

We only require this for real features and for the model code
(`src/models/`, `src/training/`), which is the part an examiner questions
hardest. A typo fix doesn't need a spec. Forcing one would turn the whole
practice into a box-ticking ritual, and rituals get skipped exactly when they
matter.

---

## Why we interrogate before we spec

Before writing a spec, we run a grilling session: pointed questions about the
design, which you answer.

This is the step that makes the work genuinely yours. The questions surface
decisions you hadn't noticed you were making, and your answers become the spec.
An AI assistant asking good questions is helping you think. An AI assistant
inventing answers is putting words in your mouth that you'll have to defend at a
panel. We do the first and forbid the second.

If you can't answer a question yet, the right answer is "I don't know." That gets
written down as an open question, not filled in with a guess.

---

## Why an unanswered question can ship, but its code can't

Some questions can't be answered yet, because they need the dataset or your
adviser. We don't let that freeze everything.

So a spec can be merged with open questions still marked in it. What can't happen
is building the part that depends on an open question. You can write down "we'll
pick the threshold later." You can't have someone code the threshold before
"later" arrives, because then the code becomes the answer, chosen by accident.

Writing something down and being ready to build it are two different stages. This
rule keeps them apart.

---

## Why the seed is fixed, and why that isn't enough

The seed (`42`) controls the random choices in a run. Fix it and a run on a
normal computer repeats exactly.

A GPU is different. It does thousands of additions at once, and the order they
finish in shifts a little each run. Adding the same numbers in a different order
gives a slightly different total, because decimals round at each step. That tiny
difference can reach the third decimal of an accuracy score.

For a thesis built on "model A beats model B," that wobble is a problem. If an
examiner reruns your code and gets different numbers than your table, you have to
explain why on the spot.

So we turn on PyTorch's **deterministic mode**, a setting that tells the GPU to
always do the math the same way. Runs then repeat exactly, from the same commit
and seed. It costs some speed, and a few operations have no repeatable version
and will error. When that happens, we swap in one that works, or, if none exists,
we write down why and report that number as a small range instead of a single
value. Exact is the rule. A documented range is the rare exception.

---

## Why settings live in `configs/`, never in the code

A hyperparameter is a knob that changes how the model behaves, like the learning
rate or the beam width.

If those knobs are scattered through the scripts, no one can see at a glance what
a run actually used, and two runs can differ without anyone noticing. Keeping
them all in `configs/` makes the setup one thing you can read and compare. A knob
that changes results never gets hardcoded. Operational choices that don't change
results, like which device to use, can stay as command-line flags.

---

## Why results carry the commit that made them

A number is only evidence if you can reproduce it, and you can only reproduce it
if you know exactly which version of the code produced it.

So every results file records the commit it came from, whether the code had
uncommitted edits at the time, and the full settings used. This is stamped
automatically, because a rule that depends on remembering fails at the one moment
it matters most, right after you get an exciting result.

---

## Why `exp/` branches skip the spec but keep the basics

An `exp/` branch is for trying something out that might lead nowhere. "What if
the gate moved to layer 4?"

Making you write a full spec before every experiment would kill the point, which
is to explore quickly. So experiments skip the spec loop.

They keep the boring safety parts though, the seed and the config discipline. The
reason is simple, you can't predict which experiment will surprise you. When one
works better than expected and you want it in the paper, you need to reproduce it,
and you can only do that if you kept the ingredients. Skip the recipe, keep the
ingredient list.

The naming rule still applies. An experiment branch is still
`exp/NNN-description`, like every other branch. Skipping the spec doesn't mean
skipping the numbering.

---

## Why AI assistance is labelled, not hidden

Parts of this repository are written with AI help. We say so openly.

Every assisted commit carries a line naming the model that helped, and a short
document ([docs/AI-USE.md](AI-USE.md), once it exists) explains what the tools
did. Hiding the assistance would be both wrong and fragile, since the history
shows it anyway. Stating it plainly is what an examiner can respect, and it's
what the AI-use declaration is actually asking for, evidence that the reasoning
was yours and the typing was assisted.

---

## Why every PR gets checked, and who checks it

A pull request is a proposed change, waiting to join `main`. Checking it before
it merges is how bugs get caught before they become the "correct" version.

Who checks depends on who opened it. On your own PR, your own read plus an AI
check is enough to merge. You're mostly working solo, and waiting on a groupmate
who isn't there would just stall the work. On a groupmate's PR, a human approves,
and an AI review runs only if someone asks for one.

An AI review is a help, not a verdict. It reads the change and flags what looks
wrong, but a person always makes the merge decision, and the formal approval on
GitHub can only come from a human account. For the model code the panel
questions hardest, it's worth getting a second human to look when one is around.
Two sets of eyes catch what one misses.

---

## Why we keep every commit instead of squashing them

When you merge a branch, GitHub can either keep its commits or "squash" them into
one.

We keep them. Each commit explains one change and why it was made, and that trail
of reasoning is the part of the history worth reading. Squashing flattens it into
a single lump and throws the reasoning away. For code nobody will grade, squashing
is fine. For a thesis, the reasoning is the point.
