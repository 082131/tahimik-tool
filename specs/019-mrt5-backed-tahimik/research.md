# Research: MrT5-Backed TAHIMIK

## Decision

Use the same `stanfordnlp/mrt5-small` initialization for the fixed comparator and TAHIMIK. This isolates the experimental difference to TAHIMIK's adaptive conditioning rather than model source.

## Adaptive extension

The noise estimator observes the hidden states produced before the gate, masked-mean pools them, and predicts `n`. TAHIMIK sets a per-example target `0.5 * (1 - n)` and shifts gate values by `cn * (n - navg)`. Higher noise preserves more bytes; lower noise permits more deletion.

The score entering the gate is detached. This prevents cross-entropy, rate, and gate-commitment objectives from changing the estimator; supervised `L_NE` remains its training signal.

## Shared baseline controls

Both compressed variants use model source, pretrained gate initialization, gate position index 2, `k=-10`, Gumbel training noise, and hard vectorized inference deletion. They differ in fixed versus per-example targets and the adaptive shift.
