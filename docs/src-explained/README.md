# `src/` explained — a beginner's walkthrough

This folder documents **every file in `src/`**, written for someone who has been
away from the code for months (or is new to deep learning). Each library and
concept is explained the first time it appears.

## How to read this

Read the files in **flow order** — the order data actually moves through the
system during an experiment:

| # | Doc | Folder | What it does |
|---|-----|--------|--------------|
| 0 | *(concepts)* | — | The vocabulary primer below |
| 1 | [01-utils.md](01-utils.md) | `src/utils/` | Shared toolbox (byte encoding, logging) |
| 2 | [02-data.md](02-data.md) | `src/data/` | Turn raw text into training tensors |
| 3 | [03-models.md](03-models.md) | `src/models/` | The three neural network variants |
| 4 | [04-training.md](04-training.md) | `src/training/` | The loss + the training loop |
| 5 | [05-evaluation.md](05-evaluation.md) | `src/evaluation/` | Scoring accuracy, speed, significance |

## The one-paragraph mental model

> `src/data` makes noisy→clean sentence pairs and scores how noisy each is →
> `src/models` builds three models that compress the input differently →
> `src/training` teaches them with a four-part loss →
> `src/evaluation` measures which model is best and whether the difference is real.
> `src/utils` is the shared toolbox everything borrows from.

---

## Concepts primer (read once)

### Python structure
- **Module** — one `.py` file.
- **Package** — a folder of modules. `src/` is a package.
- **`__init__.py`** — a special file that marks a folder as a package and runs
  when it is imported. Often used to *re-export* names so imports are shorter.
- **`import`** — pull in code from another file or library.
- **Function** — a single action: input → output.
- **Class** — a reusable template bundling data + related functions (**methods**).
  You create one ("instantiate") with `obj = ClassName(...)`. Methods take `self`
  (the object itself) as their first argument. `__init__` is the setup method
  that runs at creation.
- **Decorator** — a tag above a function/method that modifies it, e.g.
  `@staticmethod` (a method that needs no `self`) or `@torch.no_grad()` (turn off
  gradient tracking).

### The libraries
| Library | What it is | Where |
|---|---|---|
| **PyTorch** (`torch`) | Deep-learning engine: tensors, GPU, automatic gradients | everywhere |
| **torch.nn** (`nn`) | Neural-network building blocks (layers, losses) | `models`, `training` |
| **Transformers** (`transformers`) | Pre-built famous models + tokenizers (ByT5) | `models`, `data` |
| **editdistance** | Counts single-character edits between two strings | `data`, `evaluation` |
| **sacrebleu** | Translation-quality metrics (BLEU, chrF) | `evaluation` |
| **scipy / numpy** | Scientific computing + statistics | `evaluation` |
| **krippendorff** | Inter-annotator agreement statistic | `evaluation` |

### Tensors and shapes (the key idea)
A **tensor** is a grid of numbers — the container PyTorch moves through models.
A tensor's **shape** lists its dimensions. Learn to read this one:

```
(batch_size, seq_len, hidden_dim)
```
- **batch_size** — how many sentences at once (e.g. 8)
- **seq_len** — how many bytes per sentence (e.g. 1024)
- **hidden_dim** / **d_model** — how many numbers represent each byte (e.g. 512)

So `(8, 1024, 512)` = 8 sentences × 1024 bytes × 512 numbers per byte.
Most model code is transforming a tensor of one shape into another.

### Gradients (how learning happens)
PyTorch watches every math operation and can compute a **gradient** — the
direction to nudge each number to make the model's error smaller. Training =
repeatedly: run the model → measure error (**loss**) → compute gradients →
nudge. An operation must be **differentiable** (smooth) for gradients to flow;
a hard yes/no cutoff blocks them. This is why the models use "soft" (smooth)
operations during training and "hard" ones only at inference.

### Two phases every model has
- **Training** (`model.train()`) — learning; uses soft/differentiable paths.
- **Inference / evaluation** (`model.eval()`) — using the trained model; uses
  hard, deterministic paths.

---

## Abbreviations & technical terms (glossary)

Every short form used across these docs, spelled out. Each doc also repeats the
ones it uses at the top of its file.

### Programming / Python
| Term | Full form | Meaning |
|---|---|---|
| **param** | parameter | A named input a function/method accepts |
| **arg** | argument | The actual value passed in for a parameter |
| **`self`** | — | "This particular object" — first param of a method |
| **`str`** | string | Text |
| **`int`** | integer | A whole number |
| **`float`** | floating-point number | A decimal number |
| **`bool`** | boolean | True/False |
| **`dict`** | dictionary | A key→value lookup table |
| **RNG** | random number generator | The thing that produces "random" choices |
| **I/O** | input/output | Reading/writing files or streams |
| **CSV** | comma-separated values | A simple table file format |
| **JSON** | JavaScript Object Notation | A structured data file format |
| **TSV** | tab-separated values | Like CSV but tab-delimited |
| **API** | application programming interface | A defined way for programs to talk |
| **CLI** | command-line interface | Running a program by typing a command |

### Text / ByT5
| Term | Full form | Meaning |
|---|---|---|
| **byte** | — | A number 0–255; how computers store text |
| **UTF-8** | Unicode Transformation Format, 8-bit | The standard rule mapping characters → bytes |
| **ID** | identifier | A number standing for a token/byte |
| **EOS** | end of sequence | The token marking where text stops |
| **BOS** | beginning of sequence | The token marking where text starts |
| **pad / padding** | — | Filler added so all sequences share a length |
| **token** | — | One unit the model reads (here, one byte) |
| **tokenizer** | — | The tool that turns text into token IDs |
| **ByT5** | Byte-level T5 | A T5 model that reads raw bytes |
| **T5** | Text-to-Text Transfer Transformer | The encoder-decoder model family ByT5 builds on |
| **MrT5** | Merge-then-T5 (Kallini et al., 2025) | The compression method TAHIMIK extends |

### Deep learning / training
| Term | Full form | Meaning |
|---|---|---|
| **ML** | machine learning | Teaching a model from data |
| **NN** | neural network | The model type used here |
| **MLP** | multi-layer perceptron | The simplest neural network (stacked linear layers) |
| **d_model / hidden_dim** | model dimension | How many numbers represent each token |
| **logits** | — | Raw output scores before probabilities |
| **loss** | — | A number measuring how wrong the model is |
| **CE** | cross-entropy | The loss for "predict the correct category" |
| **MSE** | mean squared error | Average of squared differences (a "how far off" loss) |
| **LR** | learning rate | How big each weight update is |
| **EMA** | exponential moving average | A slowly-updated running average |
| **fp16** | 16-bit floating point | Half-precision numbers (faster/smaller) |
| **AdamW** | Adam with decoupled weight decay | The optimizer that updates weights |
| **GPU** | graphics processing unit | The chip that runs the math fast |
| **CPU** | central processing unit | The regular processor (fallback) |
| **CUDA** | Compute Unified Device Architecture | NVIDIA's GPU programming system |
| **epoch** | — | One full pass over the training data |
| **batch** | — | A group of examples processed together |
| **grad** | gradient | The direction to nudge a weight to reduce loss |

### TAHIMIK-specific
| Term | Full form | Meaning |
|---|---|---|
| **n\*** | "n-star" | The *ground-truth* noise level of a pair (edit-distance ratio) |
| **n** | — | The model's *predicted* noise score |
| **navg** | noise average | Running average of noise scores (`noise_avg`) |
| **cn** | — | Learned coefficient scaling the noise shift |
| **L_CE / L_rate / L_attn_reg / L_NE** | loss terms | Cross-entropy / deletion-rate / attention-regularizer / noise-estimator losses |
| **d_max** | maximum deletion | The most a clean sentence may be compressed |

### Evaluation / statistics
| Term | Full form | Meaning |
|---|---|---|
| **BLEU** | Bilingual Evaluation Understudy | An n-gram overlap accuracy metric |
| **GLEU+** | Google-BLEU (plus) | A sentence-level BLEU variant used here |
| **chrF** | character n-gram F-score | A character-level accuracy metric |
| **ERR** | error reduction rate | Fraction of the input's errors the model fixed |
| **n-gram** | — | A run of *n* consecutive items (words/chars) |
| **CI** | confidence interval | A plausible range for the true value |
| **p-value** | probability value | Chance the result happened by luck |
| **IAA** | inter-annotator agreement | How consistently human labelers agree |
| **α (alpha)** | Krippendorff's alpha | The IAA agreement statistic |
| **std** | standard deviation | How spread out a set of numbers is |
