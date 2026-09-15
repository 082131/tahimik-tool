# Research: Focused Byte-Gating Example

## Decision

Use `aang gandaaaa mooo!` because repeated characters make removals visible at individual byte positions. Its ASCII-only text avoids ambiguity between character and UTF-8 byte offsets in the presentation map.

## Consistency calculation

The input contains 19 bytes. Positions `0, 10, 11, 12, 16, 17` remove six bytes. `6 / 19 = 31.58%`, displayed as 32% after rounding.

All three illustrative outputs are `Ang ganda mo!`; the example teaches the gating flow rather than claiming a measured difference in accuracy.
