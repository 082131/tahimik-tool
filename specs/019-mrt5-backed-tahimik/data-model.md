# Data Model: MrT5-Backed TAHIMIK

## Shared Initialization

```text
stanfordnlp/mrt5-small
  -> custom MrT5 backbone
  -> pretrained delete gate
  -> fixed MrT5 wrapper OR TAHIMIK wrapper
```

## Adaptive Values

| Value | Shape | Meaning |
|---|---|---|
| pre-gate hidden state | batch x bytes x 1472 | estimator input |
| `n` | batch | predicted noise in `[0,1]` |
| `navg` | scalar | EMA center, momentum 0.99 |
| `cn` | scalar | learned non-negative strength |
| target rate | batch | `0.5 * (1 - n)` |

`n` is detached on the adaptive gate/rate path and remains attached for `L_NE`.
