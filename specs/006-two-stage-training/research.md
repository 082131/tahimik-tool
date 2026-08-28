# Phase 0 Research: Two-Stage Training Pipeline

Grilling answers with sources. No design question is left open — the manuscript
settles the design. The gaps here are constitution-compliance items with known
fixes rather than unresolved decisions.

## Decision: two stages, synthetic then gold

**Decision**: Pretrain on synthetic noisy/clean pairs, then fine-tune on the
gold standard.

**Rationale**: The gold standard targets ~15,000 sentences. A byte-level model
needs considerably more data than that to learn general noise structure. The
manuscript cites direct evidence rather than intuition: *"Samuel and Straka
(2021), whose intermediate synthetic pretraining step produced a substantial
gain in error reduction rate from 59.2% to 64.8%."*

Stage 1 buys volume; Stage 2 buys realism. Gold alone would overfit a small
corpus. Synthetic alone would teach the model to reverse a Python script rather
than to normalise how people actually write.

**Alternatives considered**: Gold-only training (rejected — insufficient data
for the architecture); synthetic-only (rejected — the model would learn the
generator's quirks, not real noise); mixing both in one stage (rejected — the
manuscript specifies a staged schedule, and mixing would let abundant synthetic
data swamp the gold signal).

## Decision: synthetic pretraining is especially well-motivated for ByT5

**Decision**: Treat the synthetic-pretraining evidence as directly transferable,
not merely analogous.

**Rationale**: Manuscript — *"ByT5 processes raw UTF-8 bytes directly, meaning
that character-level corruptions map exactly onto ByT5's input units. This
alignment between the synthetic noise injection strategy and the model's
processing paradigm means that the robustness gains identified by Karpukhin et
al. (2019) are more directly applicable to ByT5 than to subword models."*

The mechanism matters. For a subword model, injecting a typo changes how the
string tokenises, so the model sees a different token sequence rather than a
corrupted one — the corruption is mediated. ByT5 has no such indirection: every
synthetic deletion, insertion, or substitution is a byte operation the model
meets in exactly that form at inference.

**Why this is worth recording**: it is a defensible argument for the study
design, not just a convenience. It belongs in a defense answer to "why should
synthetic data help here?"

## Decision: all three variants share the entire pipeline

**Decision**: One trainer, one schedule, one optimizer configuration, one set of
splits. Only the loss terms differ.

**Rationale**: Manuscript control variables are *"the training data, the
train/validation/test partitioning, the model hyperparameters, and the hardware
used for training and evaluation."* That covers nearly everything except the
compression mechanism itself.

**Consequence**: `TAHIMIKTrainer` accepts any variant, and `TAHIMIKLoss`
activates only the terms that apply — `L_CE` alone for ByT5, plus rate and
attention for MrT5, plus `L_NE` for TAHIMIK. No per-variant branching in the
schedule.

**Alternatives considered**: Per-variant trainers. Rejected — three trainers
would drift apart over time, and the drift would be invisible until it
contaminated a result.

## Decision: synthetic data gets no test split

**Decision**: Gold 80/10/10 train/val/test; synthetic 90/10 train/val only.
Final evaluation uses the gold test set exclusively.

**Rationale**: A test score on synthetic data measures how well the model
reverses the noise script that generated it. A high number there is actively
misleading — it indicates the model learned the generator's quirks, which is the
opposite of the generalisation being claimed.

The 10% synthetic validation split exists solely to detect overfitting during
Stage 1. It is never a reported number.

**Alternatives considered**: A synthetic test split for extra signal. Rejected
for the reason above — the number would be uninterpretable and easy to
misreport.

## Decision: best-validation tracking resets between stages

**Decision**: Reset `best_val_loss` at the start of each stage.

**Rationale**: The two stages train on different data distributions, so their
validation losses are not comparable quantities. Carrying Stage 1's best forward
would mean Stage 2 might never beat it and therefore never checkpoint — you
would fine-tune on the gold standard and silently save nothing.

**Verified present**: `trainer.py:305` and `:319`.

## Decision: a missing stage is skipped, not fatal

**Decision**: Skip with a log message when a stage's dataset is absent.

**Rationale**: More load-bearing than it appears. The gold standard does not yet
exist, so **every run today is Stage-1-only**. If skipping crashed, nothing
could train at all.

**Verified present**: `trainer.py:314` and `:329`.

## Gap: determinism is incomplete

**Status**: Known constitution gap with a known fix. Not an open question.

Seeding **is** correct: `scripts/train.py:117`, `evaluate.py:74`, and
`benchmark.py:72` each call `torch.manual_seed(config.seed)` and
`cuda.manual_seed_all`. Seeding at program entry is the right pattern.

What is absent is the rest of deterministic mode —
`torch.use_deterministic_algorithms(True)`, cuDNN deterministic flags, and
`CUBLAS_WORKSPACE_CONFIG`. GPU runs therefore still vary slightly between
executions on identical seeds.

This is precisely the gap the constitution opened when full determinism was
chosen over a documented tolerance. The fix is mechanical.

**Correction recorded for transparency**: an earlier draft of this assessment
claimed the trainer never seeds anything. That was **wrong** and was corrected
after verification. The accurate finding is materially less severe.

## Gap: checkpoints are not traceable to a commit

**Status**: Known constitution gap with a known fix.

Constitution Principle III requires every results file to carry the git SHA, a
dirty-tree flag, and the fully resolved configuration.
`_save_checkpoint` (`trainer.py:176-195`) records epoch, stage, and validation
loss only.

A checkpoint that produced a reported number cannot currently be tied back to
the code that produced it — the exact failure Principle III exists to prevent.

## Unverified: the nine noise categories

**Status**: Not audited in this pass. Recorded as a task rather than claimed
either way.

The manuscript specifies nine categories of Filipino social media noise:
abbreviation and shortenings, orthographic variation, character elongation,
punctuation variation, capitalization variation, slang and netspeak, Taglish
morphology patterns, emoji-based sentiment markers, and code-switching.

`src/data/noise_generator.py` exists and is wired into the pipeline, but this
pass did not check its implemented categories against that list one by one.
Claiming coverage without checking would be the kind of unverified assertion the
spec process exists to prevent.
