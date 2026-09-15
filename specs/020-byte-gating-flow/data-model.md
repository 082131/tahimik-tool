# Data Model: Byte-Gating Flow

## SentenceData additions

- `input: string`: displayed source text
- `noise: number`: sentence noise estimate
- `pruning: number`: deletion percentage
- `bytePrunedPositions?: number[]`: zero-based input byte positions removed by the gate

## Derived view state

- `byteMapAvailable`: TAHIMIK selected, positions present, and total byte count aligned
- `mappedPrunedBytes`: unique removed positions
- `mappedKeptBytes`: total minus removed
