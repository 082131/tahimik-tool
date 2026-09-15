# How these specs were produced

**Read this before presenting or defending any spec in this directory.**

Not every spec here was produced the same way, and the difference matters for
the AI-use declaration. Constitution Principle II forbids inventing answers to
open design questions. Sourcing an answer from the ratified manuscript is not
inventing — it is reading an authority the author already wrote and submitted.
But it is also not the same as interrogating the author, and the record must
not blur the two.

## Two provenance classes

### Class A — Author-interrogated

The author was grilled question by question, answered in their own words, and
those answers became the spec. Where the author was unsure, the question was
escalated to the manuscript or left open.

| Spec | Rounds | Notes |
|------|--------|-------|
| `002-noise-estimator` | 3 | One question (gradient isolation) escalated to the manuscript and resolved there |

### Class B — Manuscript-derived

The author was not interrogated. Design decisions were taken from the ratified
manuscript (`tmp/thesis/manuscripts/Group9_Final-Revised-Manuscript-v1.docx`) and from
direct assessment of the existing code. Produced under time pressure ahead of a
presentation, at the author's explicit instruction.

| Spec | Primary source |
|------|----------------|
| `003-byt5-baseline` | Manuscript: Scope and Limitation, Conceptual Framework |
| `004-fixed-rate-compression` | Manuscript: Conceptual Framework; MrT5 (Kallini et al., 2025) |
| `005-noise-adaptive-byt5` | Manuscript: Noise-Adaptive Deletion, Loss Computation, Backpropagation |
| `006-two-stage-training` | Manuscript: Research Design, synthetic pretraining discussion |
| `007-evaluation-and-statistics` | Manuscript: Statement of the Problem, Significance Testing |

**What this means in practice.** A Class B spec states what the manuscript
specifies and what the code does. Where those disagree, the disagreement is
recorded as a finding rather than silently resolved. Where the manuscript is
silent on something the code decides, that is marked `NEEDS CLARIFICATION` and
left for the author — it is *not* filled in by inference.

**If asked at a defense**: the honest answer is that the manuscript settled
these decisions and the specs document them; the author's design authority is
in the manuscript, and the specs are a retrofit record built from it. Do not
claim these were interrogated. `002` was.

## What "converge" means in these documents

Every spec has a Convergence section listing gaps between what the spec calls
for and what the code actually does. Four classes:

- **missing** — required work absent from the code.
- **partial** — present but incomplete.
- **contradicts** — the code does something that conflicts with the manuscript
  or the constitution. **These are the ones to read first.**
- **unrequested** — code doing something no spec asked for.

Convergence never edits code. It only records what it found.

## The cross-cutting model-source decision [CURRENT: 2026-09-15]

One issue affected `003`, `004`, and `005` simultaneously, so it is stated once
here rather than three times:

**The manuscript specifies `byt5-base`. The code previously defaulted to `byt5-small` during initial prototyping.**

Manuscript, Scope and Limitation: *"The base variant of ByT5 will be used, and
both MrT5 and the proposed noise-adaptive model will adopt the base variant."*

**Current resolution (2026-09-15, AD-003)**:
`ByT5Config` selects `google/byt5-small`; `MrT5Config` and `TAHIMIKConfig`
select `stanfordnlp/mrt5-small`. Gradient accumulation preserves effective
batches of 16 and 8 using physical microbatches of 4. Checkpoint validation
uses both the exact pretrained source and architecture fingerprint, so Base
checkpoints and cross-source Small checkpoints are rejected. Specs 017–019 are
the current model-source authorities.
