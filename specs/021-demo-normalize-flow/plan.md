# Demo Normalize Flow Implementation Plan

**Branch**: `feat/021-demo-normalize-flow` | **Date**: 2026-09-15 | **Spec**: `specs/021-demo-normalize-flow/spec.md`

## Summary

Add an explicit demo/live mode boundary to Normalize and restore the full configured presentation scenario in demo mode.

## Technical Context

**Language**: TypeScript/React
**Configuration**: Vite environment variables
**Testing**: TypeScript check and Vite build

## Constitution Check

- Demo output remains distinguishable from measured research output: pass.
- Live inference behavior remains available: pass.

## Project Structure

```text
frontend/src/app/pages/Engine.tsx
```

## Implementation Sequence

1. Resolve data mode once at module load.
2. Branch `handleNormalize` after empty-input validation.
3. Restore all coupled demo state atomically.
4. Leave the live API path unchanged.
5. Verify both build modes.
