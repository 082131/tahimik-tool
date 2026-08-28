# Phase 1 Data Model: Noise Estimator

Three entities, all already named in `spec.md`'s Key Entities section,
formalized here with their fields and where each one lives.

## Noise score (`n`)

**What it represents**: The noise estimator's live output — how messy a
sentence looks, according to the model, at inference or training time.

**Fields**:
- A single float per sentence, in the closed range `[0, 1]`.
- Batched: shape `(batch_size,)` for a batch of `batch_size` sentences.

**Produced by**: The noise estimator's forward pass (`src/models/noise_estimator.py`).

**Consumed by**: The delete-gate's conditioning term and deletion target —
*out of scope for this feature*, covered by the delete-gate's own future
spec. This feature's responsibility ends at producing a valid `n`.

**Lifecycle note**: `n` is read-only wherever it's consumed downstream (spec
FR-006, FR-007) — nothing downstream may treat it as something to
backpropagate through back to the encoder.

## Noise label (`n*`)

**What it represents**: The ground-truth target used only during training,
against which the estimator's live prediction `n` is compared to compute
`L_NE`.

**Fields**:
- A single float per (noisy, clean) sentence pair, in `[0, 1]`.
- Formula: `edit_distance(noisy, clean) / len(longer sentence)`.

**Produced by**: `src/data/noise_label.py`, once per pair, at dataset-build
time — not recomputed inside the training loop (spec FR-004).

**Consumed by**: The `L_NE` loss computation in `src/training/losses.py`
(existing wiring; not modified by this feature).

**Validation rule**: For a pair where noisy equals clean exactly, `n*` MUST
be exactly `0.0` (spec Acceptance Scenario, User Story 3).

## (Noisy, clean) sentence pair

**What it represents**: The unit of training data this component consumes.
Not owned by this feature — produced upstream by either the synthetic noise
generator (Stage 1) or, once it exists, the gold-standard annotation
pipeline (Stage 2, currently blocked).

**Fields**: A noisy string and its corresponding clean string. No new fields
introduced by this feature; `n*` (above) is derived from this pair, not
stored as part of it conceptually, though it may live alongside it in
whatever schema `src/data/dataset.py` already uses.

**Relationship**: One (noisy, clean) pair produces exactly one `n*` label.
Many (noisy, clean) pairs, batched together, produce a batch of `n`
predictions during training.

**State note**: Stage 1 pairs exist today (synthetic). Stage 2 pairs do not
exist yet — this is a data-availability gap, not a modeling gap; the entity
shape is identical for both stages (spec User Story 2, Acceptance
Scenario 1).
