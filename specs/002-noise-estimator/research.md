# Phase 0 Research: Noise Estimator

Most of what a normal Phase 0 would need to research was already settled
during `/grill-me`, sourced from the README, the ratified manuscript, and the
project's own dependency list — not invented here. This file covers the one
item genuinely left open, plus two confirmations worth recording so a later
reader doesn't have to re-derive them.

## Open item: all-padding batch row

**Status**: Deliberately unresolved, not blocking.

**The question**: mean-pooling divides by the count of non-padding tokens in
a sentence. If a batch row is entirely padding, that count is zero, and
division by zero is undefined.

**Decision**: Left as `NEEDS CLARIFICATION` in `spec.md`, carried forward
unresolved into this plan.

**Rationale**: Constitution Principle II explicitly permits a spec to merge
with open questions, on the condition that no task depending on the open
question gets built until it's resolved. This item doesn't block User
Story 1 or User Story 2 (a real batch from real data is not expected to
produce an all-padding row), so blocking the whole plan on it would violate
the "don't invent an answer, don't over-block either" balance the
constitution asks for.

**Alternatives considered**:
- *Guess a default (e.g., return 0.0 for an all-padding row) and move on.*
  Rejected — this is exactly the kind of invented answer Principle II
  prohibits. A guess here could hide a real upstream bug (why would a batch
  ever contain an all-padding row in the first place?) behind a
  silently-chosen default.
- *Block the entire plan until this is answered.* Rejected — it would stall
  the two testable, unblocked user stories over an edge case neither of them
  exercises.

**Consequence for tasks.md**: `/speckit-tasks` MUST NOT generate a task
claiming "the noise estimator is robust to all input shapes" until this is
resolved. Tasks scoped to Stories 1–3 as written may proceed.

## Confirmation: edit-distance library

**Decision**: Use `editdistance` (already in `requirements.txt`, version
`>=0.6.0`).

**Rationale**: Already a project dependency, and its existing comment in
`requirements.txt` already names its purpose as "Edit distance for ERR and
noise level (n*)" — this is not a new choice, it's recognizing a decision
the project already made. Introducing a second edit-distance library would
be redundant and would violate the "don't add what already exists" instinct
this repository's `docs/agents/contributing.md` asks for generally.

**Alternatives considered**: None — there is no reason to introduce a second
library when one is already installed for this exact purpose.

## Confirmation: test construction pattern

**Decision**: Follow `tests/test_delete_gate.py` and
`tests/test_model_forward.py`'s existing pattern — build a tiny, randomly
initialized `T5Config` model rather than loading a real ByT5 checkpoint.

**Rationale**: A real checkpoint is ~1.2GB and downloads over the network;
CI cannot and should not depend on that. The existing tests already prove
this pattern is sufficient to validate shapes, wiring, and gradient flow,
which is exactly what this feature's acceptance criteria (SC-001–SC-003)
need.

**Alternatives considered**:
- *Download a real byt5-small checkpoint in CI.* Rejected — slow, and adds a
  network dependency to a test suite that should run offline.
- *Mock the encoder's output entirely (skip building a T5 at all).*
  Rejected — the existing pattern in this repo already builds a real (tiny)
  T5 forward pass, which more faithfully exercises the actual interface
  (padding masks, hidden-state shapes) than a hand-built mock tensor would.
