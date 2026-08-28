# Phase 0 Research: Noise-Adaptive ByT5 (TAHIMIK)

The study's contribution, so grilled harder than the baselines. Two items are
left genuinely open; one new risk was found that no existing test covers.

## Decision: the falsifiable claim

**Decision**: The claim is that *compression conditioned on estimated noise
retains more accuracy than fixed-rate compression at comparable efficiency, on
noisy Tagalog/Taglish text.*

**Rationale**: RQ3 and RQ4 ask whether differences against **both** baselines
are significant on **both** accuracy and efficiency. H₀₁ and H₀₂ state no
significant difference. The contribution requires rejecting both.

**What would disprove it**:
1. Matching MrT5's accuracy while being slower.
2. Matching MrT5's speed while being no more accurate.
3. Being better on average but **not specifically on noisy sentences**.

The third is the dangerous one. It would mean the model improved for some reason
other than the mechanism claimed — the result would be real but the explanation
wrong. This is why SC-008 exists separately from SC-006: winning is not the same
as winning for the stated reason.

**Alternatives considered**: Framing the claim as "TAHIMIK is better" without
the efficiency qualifier. Rejected — an accuracy gain bought by being as slow as
uncompressed ByT5 is not a contribution, it is a regression to the baseline.

## Decision: both the gate shift and the deletion target are required

**Decision**: Keep both mechanisms. They are not redundant.

**Rationale**: They act at different times on different objects.

- **The deletion target** `d_target = d_max·(1−n)` changes what the *loss asks
  for*. It shapes learning over training but is advisory — the rate loss trades
  against cross-entropy.
- **The gate shift** `cn·(n − navg)` changes the *scores themselves*, every
  forward pass, at training **and inference**.

Target alone: at inference there is no loss, so the gate applies an average
learned behaviour with nothing conditioning it per sentence. The adaptation
would exist during training and vanish when it matters.

Shift alone: the gate is nudged per sentence, but the rate loss still pulls
every sentence toward one shared target, actively fighting the shift.

**Consequence**: removing either breaks a different half of the mechanism.
Neither is a simplification opportunity.

## Decision: `navg` is a running average, not a fixed constant

**Decision**: Exponential moving average of observed noise scores, momentum 0.99.

**Rationale**: The shift means "compress this sentence less *than typical*". A
fixed centre makes "typical" wrong. If `navg` were pinned at 0.5 on a corpus
averaging 0.2, nearly every sentence would read as cleaner-than-centre and be
shifted toward more deletion — the adaptation would degenerate into a constant
offset, which is just MrT5 with a different target.

A running average re-centres on the real distribution, so the comparison stays
meaningful.

**Alternatives considered**: A fixed 0.5 midpoint (rejected above); a per-batch
mean with no smoothing (rejected — batch-to-batch variance would make the shift
jitter, and a small batch of unusually clean sentences would swing every
decision in it).

## OPEN: `navg` initialisation and warmup

**Status**: `NEEDS CLARIFICATION` — not resolved, not guessed.

`delete_gate.py:81` registers `noise_avg` initialised to `0.5`. With momentum
0.99 the EMA half-life is roughly 69 batches, so early training conditions the
gate against a prior that may be far from the corpus mean.

The manuscript text located so far specifies the momentum but does not appear to
specify an initialisation value or a warmup period.

**Why it is not guessed**: the choice affects early training dynamics for every
TAHIMIK run, and a wrong-but-plausible value written into a spec becomes a
decision nobody revisits. Options include initialising from the first batch's
mean, warming up before applying the shift, or keeping 0.5 as a deliberate
neutral prior. Each is defensible; none is obviously correct.

## Decision: `n` is detached at both consumption points

**Decision**: Detach in the gate shift **and** in the deletion target.

**Rationale**: The manuscript states it directly — *"it is detached at both
points, so the estimator receives gradients only from L_NE."* The failure modes
differ, and both are silent:

- **Shift not detached**: the rate loss and cross-entropy backpropagate into the
  noise estimator, which then learns to emit whatever makes compression
  convenient rather than what reflects messiness. The noise score stops being a
  noise score while continuing to look like one.
- **Target not detached**: worse. Since `d_target = d_max·(1−n)`, the rate loss
  could reduce its error by moving the **target** toward the achieved rate
  rather than the rate toward the target. The model learns to want what it
  already does, and the rate loss becomes self-satisfying.

**Verified present**: `delete_gate.py:131` and `noise_adaptive_byt5.py:187`.

## Decision: `cn` is learned — with an unguarded failure mode

**Decision**: `cn` is an `nn.Parameter` initialised at 1.0.

**Rationale**: The manuscript describes it as a learned coefficient. The correct
conditioning strength is not knowable in advance and should be fitted.

**Risk identified during grilling, not previously recorded**: nothing constrains
`cn`'s sign. A negative `cn` inverts the mechanism entirely — noisy sentences
would be compressed **more**, the exact opposite of the design. Training would
not necessarily prevent this if a negative value happened to reduce total loss
through some other pathway.

A sign-inverted TAHIMIK would still run, still produce plausible numbers, and
still appear to be a valid comparison. Nothing would flag it.

No existing test appears to assert `cn > 0` after training. Recorded as a task.

**Alternatives considered**: Constraining `cn` to be positive by construction
(e.g. parameterising as `softplus`). Not adopted unilaterally — it deviates from
the manuscript's plain description. Recorded as an option for the author.

## OPEN: gate-shift clamp saturation

**Status**: `NEEDS CLARIFICATION` — not resolved.

`delete_gate.py:151` clamps shifted scores back into `[k, 0]`. For a byte already
scored near 0 (strongly keep), a positive shift is clamped away entirely — that
byte cannot be kept *more* than it already is.

So adaptation saturates precisely on the sentences where it matters most: the
noisiest, where the shift is largest. How often this happens is measurable but
unmeasured.

The manuscript does not appear to address the clamp. Options: accept saturation
as an intended ceiling, constrain `cn` so it cannot saturate, or measure
saturation frequency and report it. Author decision.

## Decision: EMA update location

**Decision**: Keep the update inside the forward pass, but make its safety
explicit.

**Rationale**: `delete_gate.py:136` mutates a registered buffer inside an
autograd-tracked forward pass. It is safe **because** `n_detached` is used, so no
graph is built through the update. But the safety is incidental — it depends on
that detach remaining in place. Removing the detach would introduce buffer
mutation inside the graph, a subtle bug class.

Wrapping in `torch.no_grad()` would make the intent explicit and the safety
structural rather than a side effect. Minor; recorded rather than assumed.
