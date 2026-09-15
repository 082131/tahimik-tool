# Research: Demo Normalize Flow

## Decision

Use a build-time environment boundary: `demo` is default; `live` must be selected explicitly. Normalize owns restoration of the entire scenario so edited text cannot be displayed beside stale demonstration outputs.

## State restored in demo mode

- input text and the single demo row
- single-sentence mode
- selected row index zero
- hidden comparison summary
- cleared API status message

The live branch retains the backend call and its loading/status behavior.
