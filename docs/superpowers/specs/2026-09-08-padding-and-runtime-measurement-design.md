# Padding and Runtime Measurement Design

## Goal

Preserve dynamic-padding efficiency and report truthful interactive batch-demo latency.

## Design

The collator pads encoder inputs and decoder labels independently, each to its own nearest requested multiple. Trainers request multiple-of-eight padding only for CUDA mixed-precision training; benchmark loaders keep exact per-item lengths. The API synchronizes CUDA before and after the one batched `generate` call before presenting latency.

## Acceptance criteria

- Inputs and labels independently round to a requested multiple.
- Mixed-precision training uses multiple-of-eight padding; primary efficiency benchmarking does not.
- A GPU API batch measurement brackets generation with synchronization.
