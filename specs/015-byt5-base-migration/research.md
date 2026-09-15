# Research: ByT5 Base Migration

> **Status:** Superseded by `specs/017-small-model-migration/`. The active study
> configuration uses designated Small sources.

Background investigation into ByT5-Base hardware feasibility, layer 3 deletion gate retention, and comparative integrity.

---

## 1. Why `google/byt5-base` Over `google/byt5-small`

- **Manuscript Authority**: The thesis manuscript explicitly specifies `google/byt5-base` across all three conditions. Any result reported with Small creates a direct methodological contradiction (Finding 4 in `specs/FINDINGS.md`).
- **Capacity**: ByT5 operates on raw UTF-8 bytes without subword vocabularies. Sequence lengths are ~4× longer than WordPiece/BPE tokens. The Base model (18 encoder layers, $d_{\text{model}}=1536$) provides the parameter capacity necessary for complex character transformations in code-switched Taglish.

---

## 2. Gate Layer Location Decision

- In `specs/004-fixed-rate-compression/spec.md`, the delete gate is located after layer 3.
- In `google/byt5-small`, layer 3 represents 25% of the encoder depth (3 of 12 layers).
- In `google/byt5-base`, layer 3 represents 16.7% of the encoder depth (3 of 18 layers).
- **Decision**: Retain absolute layer 3 for both compressed variants (`ByT5-base + fixed-rate deletion` and `TAHIMIK (ByT5-base + noise-adaptive deletion)`). Altering the gate position would constitute an unscheduled architectural ablation. Both compressed models share layer 3 under identical conditions.

---

## 3. GPU Memory Feasibility & Gradient Accumulation

- ByT5-Base with 1,024 byte sequences requires ~12GB VRAM under full batch size 16 in FP32.
- **Solution**:
  - Reduce physical microbatch size to 2.
  - Employ PyTorch AMP mixed precision (`fp16` or `bf16`).
  - Enable activation gradient checkpointing (`model.gradient_checkpointing_enable()`).
  - Accumulate gradients over 8 steps in Stage 1 ($2 \times 8 = 16$) and 4 steps in Stage 2 ($2 \times 4 = 8$).
  - Effective batch sizes, learning rates, and warm-up schedules remain mathematically identical to the manuscript specification.
