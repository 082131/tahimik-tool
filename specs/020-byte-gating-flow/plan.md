# Byte-Gating Flow Implementation Plan

**Branch**: `feat/020-byte-gating-flow` | **Date**: 2026-09-15 | **Spec**: `specs/020-byte-gating-flow/spec.md`

## Summary

Restructure the Engine inspector into an ordered three-stage explanation and render only supplied byte decisions.

## Technical Context

**Language**: TypeScript/React
**Testing**: TypeScript check and Vite production build

## Constitution Check

- No research model behavior changes: pass.
- Displayed efficiency/deletion details are labeled data, not measured claims: pass.

## Project Structure

```text
frontend/src/app/pages/Engine.tsx
```

## Implementation Sequence

1. Extend `SentenceData` with optional byte positions.
2. Combine mechanics and decision-map presentation.
3. Separate adaptive and fixed explanations.
4. Replace generated positions with supplied telemetry/unavailable state.
5. Verify type-check and build.
