# Research: Hugging Face MrT5 Baseline

## Decision

The fixed-rate comparator uses the released `stanfordnlp/mrt5-small` checkpoint, including Stanford's custom model implementation and pretrained gate. Loading uses the Hugging Face model repository with remote custom code enabled.

## Integration boundary

The project does not execute both Stanford's embedded gate and its wrapper gate. It captures the embedded gate, disables the embedded execution flag, creates the local compatible gate, and imports the pretrained parameters. This keeps the existing loss, telemetry, soft-training, and hard-inference contracts while retaining Stanford initialization.

## Fidelity details

- Gate location: configuration index 2 / third encoder block.
- Scale: `k=-10.0`.
- Target deletion rate: 0.5.
- Scoring convention: T5 RMS norm followed by `k * sigmoid(-logit)`.
- Gumbel perturbation: training only.
- Physical deletion: vectorized gather path in evaluation/generation.

## Failure policy

A randomly initialized gate would no longer represent the selected baseline. Gate-loading failure therefore aborts construction.
