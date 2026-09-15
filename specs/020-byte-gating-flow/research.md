# Research: Byte-Gating Flow

## Decision

Use one ordered inspector rather than separate mechanics and decision cards. The explanatory sequence follows the model: sentence-level noise, byte-level scores, then hard retain/remove decisions.

## Truthfulness boundary

The UI may summarize a deletion percentage without exact positions, but it may not fabricate which bytes were removed. `bytePrunedPositions` is therefore optional, and the decision map is gated on its availability and alignment with the input.

## Terminology

Use “retained” and “removed” for bytes. Word boxes only improve readability and must be labeled as such.
