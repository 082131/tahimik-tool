# Feature Specification: Noise Estimator

**Feature Branch**: `feat/002-noise-estimator`

**Created**: 2026-08-27

**Status**: Draft

**Input**: Retrofit specification for `src/models/noise_estimator.py`, produced by
interrogating the author (`/grill-me`) rather than reading the existing
implementation, per Constitution Principle II. Grounded in `README.md`
("Architecture Deep Dive: The Noise Estimator") and the ratified manuscript
(`thesis_code/Group9_Final-Revised-Manuscript-v1.docx`), not in the code.

## Plain-language summary

TAHIMIK cleans up messy Filipino/Taglish text. Before it decides how much of a
sentence it can safely throw away to work faster, it needs to know how messy
that sentence is. The noise estimator is the small component whose only job is
to look at a sentence and answer that one question: a single number from 0
(clean) to 1 (very messy). That number then tells the delete-gate, a separate
component covered by its own future spec, how cautious to be.

This spec covers producing that number. It does not cover how the number gets
used once produced.

## User Scenarios & Testing *(mandatory)*

There is no end user clicking through this component; its consumers are other
parts of the system and the people building and grading it. Framed as such:

### User Story 1 - Training loop obtains a noise score during Stage 1 (Priority: P1)

**Actor**: the training loop (synthetic-pretraining stage, per
`docs/SPEC-WORKFLOW.md` and Constitution Principle III).

During Stage 1, every training example is a computer-generated (noisy, clean)
pair, produced by synthetically corrupting clean text. For each pair, the
training loop needs both a noise estimator prediction (to compute `L_NE`) and
a precomputed ground-truth noise label `n*` (the target `L_NE` compares
against).

**Why this priority**: `L_NE` is active from the first step of Stage 1 (Q5).
Without this working, no training in either stage can start.

**Independent Test**: Feed the noise estimator a batch of hand-written
noisy/clean pairs with hand-computed expected `n*` values. Confirm the
estimator runs, produces one score per sentence, and that `n*` for each pair
matches a value computed by hand from the edit-distance formula.

**Acceptance Scenarios**:

1. **Given** a batch of encoder hidden states for `B` sentences, **When** the
   noise estimator runs a forward pass, **Then** it returns exactly `B` scores,
   each a single float in the closed range `[0, 1]`.
2. **Given** a noisy/clean sentence pair, **When** `n*` is computed from it,
   **Then** the result equals `edit_distance(noisy, clean) / len(longer)`,
   matching a hand-computed reference value for at least three known example
   pairs.
3. **Given** a training step that computes `L_NE`, **When** backpropagation
   runs, **Then** gradients reach only the noise estimator's own parameters —
   never the shared encoder, and never the delete-gate or rate-loss
   parameters. (See Manuscript Confirmation below.)

---

### User Story 2 - Training loop obtains a noise score during Stage 2 (Priority: P2)

**Actor**: the training loop (gold-standard fine-tuning stage).

Same mechanism as User Story 1, but consuming real annotated (noisy, clean)
pairs instead of synthetic ones once the gold standard exists.

**Why this priority**: Lower priority than Stage 1 only because it is
currently **blocked** — the gold-standard dataset does not exist yet
(Constitution: Technology & Data Constraints). The mechanism is identical; the
data source is what's missing.

**Independent Test**: Cannot be independently tested with real data today.
Testable today only by substituting hand-written stand-in pairs for the
missing gold pairs, which exercises the same code path as User Story 1.

**Acceptance Scenarios**:

1. **Given** the noise estimator already satisfies User Story 1, **When** it is
   pointed at gold-standard pairs instead of synthetic pairs, **Then** no code
   change is required — the interface is identical for both stages.
2. `NEEDS CLARIFICATION: none currently` — this story has no open design
   question, only an open data dependency. It is marked BLOCKED, not
   unspecified: see Assumptions.

---

### User Story 3 - Developer inspects precomputed noise labels before training (Priority: P3)

**Actor**: a developer preparing a dataset for a training run.

Because `n*` is precomputed once when the dataset is built rather than
recomputed at every training step (Q4), a developer must be able to open the
precomputed labels and manually verify a sample of them look reasonable before
committing to a full training run.

**Why this priority**: A safety check, not a blocker. Lower priority than
Stories 1–2 because training can technically proceed without a human looking
at the labels, but skipping this check risks discovering a labeling bug only
after a full run completes.

**Independent Test**: Build the precomputed label file for a small hand-picked
set of (noisy, clean) pairs where the "correct" messiness is intuitively
obvious (e.g., an unchanged sentence should score 0; a heavily garbled one
should score close to 1). Open the file and confirm the stored numbers match
that intuition.

**Acceptance Scenarios**:

1. **Given** a dataset-building step, **When** it processes a set of
   (noisy, clean) pairs, **Then** it writes one `n*` value per pair to a
   format a developer can open and read (not recomputed silently inside the
   training loop).
2. **Given** a (noisy, clean) pair where noisy equals clean exactly,
   **When** `n*` is computed, **Then** the result is exactly `0.0`.

### Edge Cases

- **A sentence with zero non-padding tokens** (an entirely-padding row in a
  batch): mean-pooling divides by the count of non-padding positions.
  Dividing by zero is undefined. `NEEDS CLARIFICATION: what should happen
  here — is an all-padding row possible in practice given how batches are
  built elsewhere in the pipeline, and if so, what noise score should it
  produce?` This does not block Stories 1–2 (a real batch from real data is
  not expected to contain an all-padding row), but must be resolved before
  `/speckit-tasks` can write a task claiming full input robustness.
- **Noisy and clean sentences of very different lengths** (e.g., the noisy
  version dropped whole words): the edit-distance-ratio formula still
  produces a value in `[0, 1]` by construction (dividing by the longer
  length), so no special case is needed here — included for completeness,
  not as an open question.
- **A batch mixing Stage-1-style synthetic pairs and Stage-2-style gold
  pairs in the same forward pass**: out of scope. The two stages run
  sequentially per Constitution Principle III (fixed training schedule); the
  training loop does not interleave them within a batch.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The noise estimator MUST accept encoder hidden states from the
  same layer the delete-gate reads from (layer 3), so no additional forward
  pass through the encoder is required to obtain its input. (Q2)
- **FR-002**: The noise estimator MUST reduce per-token hidden states to a
  single per-sentence vector by mean-pooling over non-padding positions only.
- **FR-003**: The noise estimator MUST transform that sentence vector through
  `Linear(d_model → 256) → GELU → Dropout(0.1) → Linear(256 → 1) → Sigmoid`,
  producing one scalar per sentence in `[0, 1]`.
- **FR-004**: The system MUST compute each `n*` label as
  `byte-level edit_distance(noisy, clean) / len(longer sentence)`, once per
  (noisy, clean) pair, at dataset-build time — not recomputed during
  training. (Q4)
- **FR-005**: Training MUST apply `L_NE` (mean squared error between the
  estimator's prediction and `n*`) starting from the first training step of
  Stage 1 (synthetic pretraining), not deferred to Stage 2. (Q5)
- **FR-006**: Gradients from `L_NE` MUST update only the noise estimator's
  own parameters. They MUST NOT reach the shared encoder, the delete-gate, or
  any rate/attention-regularizer parameters. This MUST hold regardless of
  where else the estimator's output value is consumed downstream. (Q3,
  confirmed against the manuscript — see below)
- **FR-007**: The noise estimator's own training (`L_NE`) MUST NOT depend on
  gradients arriving from the delete-gate's conditioning term or from the
  rate loss. Those two consumers use the estimator's output as a read-only
  value.

*Requirement intentionally deferred to the delete-gate's own spec, not this
one, per the Round 3 scope decision:*

- How the delete-gate consumes the noise score (the conditioning term
  `cn · (n − navg)` and the deletion target `d_target = d_max · (1 − n)`) is
  out of scope here. This spec's boundary ends at "produces a valid score";
  it does not extend to "and here is what happens to that score next."

### Manuscript Confirmation (not an open question — recorded for traceability)

The ratified manuscript states directly: *"Although n appears in the gate's
conditioning term and in the deletion target, it is detached at both points,
so the estimator receives gradients only from L_NE."* It further states the
backpropagation routing explicitly: gradients from `L_CE` go to the
ByT5 encoder-decoder; gradients from `L_CE + w_rate·L_rate + w_attn_reg·L_attn_reg`
go to the delete gate; gradients from `L_NE` go to the noise estimator. No
routing path from `L_NE` to the encoder exists in the manuscript's own
description. FR-006 restates this as a testable requirement rather than
leaving it as prose.

### Key Entities

- **Noise score (`n`)**: a single float in `[0, 1]` per sentence, the noise
  estimator's output. Consumed downstream by the delete-gate (out of scope
  here).
- **Noise label (`n*`)**: the precomputed ground-truth target used only
  during training, one per (noisy, clean) sentence pair, computed once at
  dataset-build time.
- **(Noisy, clean) pair**: the unit of training data for this component.
  Comes from synthetic corruption in Stage 1, and from human annotation in
  Stage 2 (blocked — see Assumptions).

## Success Criteria *(mandatory)*

Per Constitution Principle III and `docs/SPEC-WORKFLOW.md`, success criteria
here are split into what is checkable today and what is blocked on data that
does not yet exist. Claiming the second category as "done" without real data
would misrepresent the component's actual state.

### Measurable Outcomes — checkable now

- **SC-001**: Given any batch of encoder hidden states with a valid
  attention/padding mask, the noise estimator produces output of shape
  `(batch_size,)` with every value in `[0, 1]`, on 100% of runs (a shape and
  range invariant, not a statistical claim).
- **SC-002**: Given three or more hand-computed (noisy, clean, expected `n*`)
  triples, the precomputed label matches the hand-computed value exactly
  (edit distance is an exact deterministic calculation, not an approximation).
- **SC-003**: A test demonstrates that `L_NE`'s gradient reaches the noise
  estimator's parameters and does not reach the encoder's parameters,
  verified by checking gradient values directly after a single backward pass
  on a synthetic batch.

### Measurable Outcomes — BLOCKED on the gold-standard dataset

- **SC-004** *(blocked)*: Correlation between the estimator's predicted noise
  score and actual human-perceived messiness on real Tagalog/Taglish text.
  Cannot be measured without gold-standard annotations.
- **SC-005** *(blocked)*: Whether the noise estimator's scores meaningfully
  differ between clean and noisy real-world sentences (as opposed to
  synthetic ones). Cannot be measured without real data.

## Assumptions

- Synthetic (noisy, clean) pairs for Stage 1 are produced by a separate,
  already-existing noise-generation component; this spec assumes that
  component supplies valid pairs and does not re-specify it.
- The gold-standard dataset and the annotation platform that will produce
  Stage 2's (noisy, clean) pairs are still under active development
  (Constitution: Technology & Data Constraints). User Story 2 and Success
  Criteria SC-004–SC-005 are therefore BLOCKED, not unspecified — the
  mechanism is fully defined, only the real data input is missing.
- `d_model`, the encoder's hidden dimension, is inherited from whichever ByT5
  size is configured (`configs/base.py: model_name`); this spec does not
  fix a specific value, since the estimator's `Linear(d_model → 256)` first
  layer must adapt to whatever the shared encoder produces.
- The all-padding-batch edge case (see Edge Cases) is marked
  `NEEDS CLARIFICATION` and does not block this spec from merging, per
  Constitution Principle II ("Open questions"). It blocks only the specific
  task that would claim full robustness to that input shape.
