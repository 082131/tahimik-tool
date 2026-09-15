# Focused Byte-Gating Example Implementation Plan

**Branch**: `feat/022-focused-byte-gating-example` | **Date**: 2026-09-15 | **Spec**: `specs/022-focused-byte-gating-example/spec.md`

## Summary

Replace the long presentation sentence with a compact ASCII example whose exact byte positions and displayed percentage can be checked by inspection.

## Technical Context

**Language**: TypeScript/React
**Testing**: arithmetic inspection, TypeScript check, Vite build

## Constitution Check

- Values are marked as demonstration data: pass.
- No model, training, or reporting behavior changes: pass.

## Project Structure

```text
frontend/src/app/pages/Engine.tsx
```

## Implementation Sequence

1. Replace the sentence and illustrative outputs.
2. Set the noise/deletion presentation values.
3. Replace reading groups and exact removed positions.
4. Verify 6/19 rounds to 32%.
5. Verify frontend type-check and build.
