# Synthetic Noise Governance Design

## Goal

Ensure every Stage 1 pair is generated from an approved, verifiable manifest and has reproducible lineage.

## Design

`ProbabilityManifest` becomes a strict, self-verifying contract: it accepts only known categories, valid probability bounds, an approved resource-version map, a training-label fingerprint, and a matching canonical ID. The manifest loader rejects missing or malformed fields.

Stage 1 commands require `--noise-manifest` whenever `--clean-corpus` is supplied. `DataPipeline` receives the manifest, invokes `TagalogNoiseGenerator.generate_pair`, rejects unchanged source/target pairs, persists lineage and distribution diagnostics, and never falls back to embedded probabilities. Preserved categories are enabled only when their approved local resource is present; absent resources make a reporting run ineligible.

## Acceptance criteria

- A Stage 1 command without a manifest fails before generating data.
- Invalid, tampered, incomplete, or unapproved manifests fail validation.
- Every generated pair has distinct source and target plus manifest/resource lineage.
- Diagnostics report count, mean/median/quantiles of n*, zero-noise fraction, categories, and manifest ID.
- Identical manifest, seed, clean corpus, and approved resources reproduce pairs and lineage.
