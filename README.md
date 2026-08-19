# TAHIMIK

**Text Augmentation and Harmonization of Informal and Multilingual Input for Knowledge Extraction**

A noise-adaptive byte-level compression system for Tagalog and Taglish social media text normalization. Built on ByT5 and MrT5, with a learned delete gate conditioned on per-sentence noise estimation.

---

## What TAHIMIK Does

Filipino social media text is full of abbreviations (*slmt* for *salamat*), character elongation (*grabeeee*), code-switching, emoji insertions, and other noise. TAHIMIK normalizes these noisy inputs back into clean Tagalog/Taglish text.

**Core insight:** not all sentences are equally noisy. A clean sentence can be aggressively compressed (deleting redundant bytes) for efficiency, while a noisy sentence should be preserved in full so the decoder has enough information to correct errors. TAHIMIK learns to estimate noise and adjust compression accordingly.

### Three Models Compared Under Identical Conditions

| Variant | Compression | Purpose |
|---------|------------|---------|
| **ByT5 Baseline** | None (every byte processed) | Accuracy ceiling |
| **MrT5 Fixed** | Fixed 50% deletion rate | Efficiency baseline |
| **TAHIMIK (Proposed)** | Noise-adaptive deletion | Best of both worlds |

Same data, same optimizer, same schedule — only the compression mechanism changes.

---

## Project Structure

```
tahimik/
├── configs/                     # Hyperparameters (control variables)
│   ├── base.py                  # Shared: AdamW, LR, splits, seed
│   ├── byt5_config.py           # ByT5 baseline (no compression)
│   ├── mrt5_config.py           # MrT5 (fixed 50% deletion)
│   └── tahimik_config.py        # TAHIMIK (noise-adaptive deletion)
│
├── src/
│   ├── data/                    # Data pipeline
│   │   ├── noise_generator.py   # 9 Filipino noise categories
│   │   ├── noise_label.py       # n* = edit_distance / max_length
│   │   ├── dataset.py           # PyTorch Dataset + collate_fn
│   │   └── preprocessing.py     # Load, clean, split, generate
│   │
│   ├── models/                  # Neural architecture
│   │   ├── noise_estimator.py   # MLP: hidden → noise score n
│   │   ├── delete_gate.py       # Byte deletion + noise conditioning
│   │   ├── byt5_baseline.py     # Variant 1: no compression
│   │   ├── fixed_compression_byt5.py  # Variant 2: fixed-rate
│   │   └── noise_adaptive_byt5.py     # Variant 3: TAHIMIK
│   │
│   ├── training/                # Training loop
│   │   ├── losses.py            # L_CE + L_rate + L_attn_reg + L_NE
│   │   └── trainer.py           # Two-stage: synthetic → gold
│   │
│   ├── evaluation/              # Metrics and statistics
│   │   ├── metrics.py           # GLEU+, chrF, ERR, Alpha-word Acc
│   │   ├── efficiency.py        # Inference time + GPU memory
│   │   ├── statistical_tests.py # Bootstrap, Wilcoxon, Holm-Bonf.
│   │   └── annotation.py        # Krippendorff's alpha (IAA)
│   │
│   └── utils/
│       ├── byte_encoding.py     # UTF-8 byte ↔ ByT5 token IDs
│       └── logging_utils.py     # Console + file logging
│
├── backend/                     # FastAPI inference server
│   └── app.py                   # /normalize endpoint, multi-model registry
│
├── frontend/                    # The TAHIMIK tool (React + Vite)
│   └── src/app/                 # Input/Output workspace + info panels
│
├── annotation-platform/         # Annotator app (React + Supabase)
│   └── ...                      # Used to collect the gold standard
│
├── scripts/                     # Entry points
│   ├── train.py                 # Train one variant
│   ├── evaluate.py              # Evaluate a checkpoint
│   ├── benchmark.py             # Efficiency benchmarking
│   └── run_experiment.py        # Full pipeline: all 3 variants
│
├── requirements.txt
└── .gitignore
```

---

## Architecture Deep Dive

### The Noise Estimator (`src/models/noise_estimator.py`)

A small MLP that reads the encoder's hidden states after the first few layers and predicts how noisy the input sentence is:

```
hidden_states (batch, seq, d_model)
    │
    ▼ mean-pool over non-padding positions
sentence_vector (batch, d_model)
    │
    ▼ Linear(d_model → 256) → GELU → Dropout(0.1) → Linear(256 → 1) → Sigmoid
noise_score n (batch,)     ← value in [0, 1]
```

Trained **only** by L_NE (MSE against n\*). Its output is detached before entering the delete gate so the gate's gradients cannot leak into the estimator.

### The Delete Gate (`src/models/delete_gate.py`)

**Step 1 — Raw gate scores:**
```
G = k · sigmoid(LayerNorm(H) · W + b)   → scores in [k, 0] where k = -30
```

**Step 2 — Noise-adaptive shift (TAHIMIK only):**
```
G_shifted = G + cn · (n - navg)
```
- `cn` is a learned scalar
- `n` is the noise score from the estimator
- `navg` is an exponential moving average of noise scores (momentum 0.99)
- Noisier sentence → positive shift → fewer deletions
- Cleaner sentence → negative shift → more deletions

**Step 3 — Deletion mode:**
- *Training (soft):* gate outputs become attention masks (differentiable)
- *Inference (hard):* bytes below threshold k/2 are physically removed

### Loss Function

```
L = L_CE + w_rate · L_rate + w_attn_reg · L_attn_reg + L_NE
```

| Component | Active For | Description |
|-----------|-----------|-------------|
| L_CE | All | Cross-entropy between predicted and target byte sequences |
| L_rate | MrT5 + TAHIMIK | MSE between actual and target deletion rate |
| L_attn_reg | MrT5 + TAHIMIK | Prevents attention from circumventing the gate |
| L_NE | TAHIMIK only | MSE between predicted noise score and ground truth n\* |

### Two-Stage Training

- **Stage 1 — Synthetic pretraining:** 3 epochs, batch 16, ~1M pairs. Teaches general Filipino noise patterns.
- **Stage 2 — Gold fine-tuning:** 10 epochs, batch 8, ~15K pairs. Calibrates on real social media noise.

---

## Evaluation Metrics

### Normalization Accuracy
- **GLEU+** — Sentence-level BLEU with add-one smoothing (0–100)
- **chrF** — Character-level F-score using char 6-grams (0–100)
- **ERR** — Error Reduction Rate using edit distance
- **Alpha-word Accuracy** — Exact match on alphabetic words

### Computational Efficiency
- **Inference time** — 5 warmup + 20 timed runs, per-sentence average
- **Peak GPU memory** — `torch.cuda.max_memory_allocated()`

### Statistical Significance
- **Paired bootstrap resampling** — 1000 samples, for accuracy metrics
- **Wilcoxon signed-rank** — For GPU memory (non-parametric)
- **Holm-Bonferroni correction** — Controls family-wise error rate across all pairwise tests

---

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train a single variant

```bash
python scripts/train.py --variant tahimik --gold_data data/gold.csv
```

Add `--clean_corpus data/clean.txt` to enable Stage 1 synthetic pretraining.

### Evaluate a checkpoint

```bash
python scripts/evaluate.py \
    --variant tahimik \
    --checkpoint outputs/checkpoints/tahimik_noise_adaptive/best_stage2.pt \
    --gold_data data/gold.csv \
    --output_file outputs/results_tahimik.json
```

### Run the full experiment (all 3 variants + stats)

```bash
python scripts/run_experiment.py \
    --gold_data data/gold.csv \
    --clean_corpus data/clean_corpus.txt \
    --output_dir outputs/experiment
```

### Run the tool (frontend + backend)

Terminal 1 — the inference API:

```bash
pip install -r backend/requirements.txt && python backend/app.py
```

Terminal 2 — the frontend:

```bash
cd frontend && npm install && npm run dev
```

Then open http://localhost:5173, pick a model from the dropdown, paste noisy
text, and press **Normalize** (or Ctrl+Enter).

The API listens on port **8100** by default (8000 is often taken by other
local services). Change it with `PORT=9000 python backend/app.py`, and point
the frontend at it with `VITE_API_URL=http://localhost:9000 npm run dev`.

#### Model registry

The dropdown offers the three variants the study compares. Each is loaded
lazily from the checkpoint `scripts/train.py` writes, so the server starts even
when nothing has been trained yet:

| Option | Checkpoint | Override with |
|--------|-----------|---------------|
| **ByT5** | `checkpoints/byt5_baseline/best_stage2.pt` | `BYT5_CHECKPOINT` |
| **MrT5** | `checkpoints/mrt5_fixed/best_stage2.pt` | `MRT5_CHECKPOINT` |
| **TAHIMIK** | `checkpoints/tahimik_noise_adaptive/best_stage2.pt` | `TAHIMIK_CHECKPOINT` |

`best_stage1.pt` is accepted as a fallback so a partially trained model can
still be demonstrated. Selecting a variant with no checkpoint returns a 503
naming the exact path and the command that produces it.

`GET /health` lists every variant with its resolved checkpoint path and whether
it is available and loaded.

> **Status:** no checkpoints exist yet — the gold-standard dataset is still
> being annotated. The tool runs and reports precisely what is missing, but it
> cannot normalize text until at least one variant is trained.

---

## The Nine Filipino Noise Categories

1. **Abbreviations/Shortenings** — *salamat → slmt*
2. **Orthographic Variation** — *dito → d2*, phonetic spelling
3. **Character Elongation** — *grabe → grabeeee*
4. **Punctuation Variation** — *!!! → !!!!!!!!*
5. **Capitalization Variation** — random CAPS
6. **Slang/Netspeak** — *idol → lodi*
7. **Taglish Morphology** — *nag-download*
8. **Emoji Sentiment Markers** — emoji insertion
9. **Code-Switching** — Tagalog + English mixing

---

## Team

**Group 9 — Polytechnic University of the Philippines**

BS Computer Science Thesis — TAHIMIK
