# `src/data/` — from raw text to training tensors (exhaustive line-by-line)

**Flow position:** the **first** real stage. It produces `(noisy, clean, n*)`
examples and packs them into tensors the models train on.

**Files (flow order):**
1. `noise_generator.py` — corrupt clean text into realistic noisy text
2. `noise_label.py` — score how noisy each pair is (**n\***)
3. `dataset.py` — pack pairs into PyTorch tensors
4. `preprocessing.py` — the conductor: load, clean, generate, split
5. `__init__.py` — re-exports

**How the pieces connect:**
```
preprocessing.py  (the conductor)
   ├── calls noise_generator.py   → makes noisy text
   ├── calls noise_label.py       → computes n*
   └── produces split lists ──────→ dataset.py wraps them into tensors
                                          └── consumed by src/training + src/evaluation
```

> **Format reminder:** each code block starts with **▸ What this block does**
> (group summary), then breaks down every line, variable, and technical term.

---

## `noise_generator.py` — faking Filipino social-media noise

**Role:** Take a *clean* sentence and dirty it, producing a `(noisy, clean)`
training pair for free. Implements the 9 noise categories from the manuscript.

- **Input:** a clean string (e.g. `"Magandang umaga sa lahat!"`)
- **Output:** a noisy string (e.g. `"mgndng umga sa lhat!"`)
- **Used by:** `preprocessing.generate_synthetic_pairs()` (Stage-1 data)

### Where this fits in TAHIMIK
There is no large ready-made corpus of Filipino "noisy → clean" pairs. This file
**manufactures the training data**: give it clean sentences and it produces
realistic noisy versions, so you get `(noisy, clean)` pairs for free — about **1
million** of them for Stage-1 pretraining. Without it, the model would have to
learn every Filipino noise pattern (abbreviations, elongation, code-switching…)
from only the ~15K human-annotated gold pairs, which isn't enough. It's how the
project **bootstraps around not having millions of human annotations.**

**Honest scope note:** this is used **only** for Stage-1 *synthetic* pretraining.
Final evaluation uses the human gold set, never this generator's output — so its
job is to teach *general* noise, not to be the ground truth.

**Example**
```
gen = TagalogNoiseGenerator(seed=42)
gen.apply_noise("Magandang umaga sa lahat salamat")
# one possible output (randomness is seeded):
#   "mgndng umaga sa lahat slmt 😭"
# → paired with the clean original to make one training example
```

**Terms & abbreviations in this file**
| Term | Full form / plain meaning |
|---|---|
| RNG | random number generator — the object that makes "random" choices |
| seed | a fixed starting value that makes an RNG repeat the same sequence every run |
| reproducible | able to get the exact same result again by re-running |
| probability | a number in [0,1]; `p=0.30` = "happens ~30% of the time" |
| `dict` (dictionary) | a key→value lookup table, e.g. `"salamat" → ["slmt"]` |
| regex / `re` | regular expression — a mini-language for find/replace on text patterns |
| stochastic | involving randomness (outcomes vary run to run unless seeded) |
| in place | modifying an existing list rather than making a new one |
| index | the position number of an item in a list (starts at 0) |

### Module-level code

```python
import random
import re
from typing import List, Tuple
```
**▸ What this block does:** imports the tools this file needs — randomness, regex,
and type hints.
- `random` — Python's built-in module for random choices (used via a seeded RNG).
- `re` — the **regular expression** module, for pattern-based find/replace on text.
- `from typing import List, Tuple` — type hints. `List` = "a list";
  `Tuple` = "a fixed group of values" (used to hint a `(noisy, clean)` pair).

```python
ABBREVIATION_MAP = {
    "salamat": ["slmt", "slmat", "tnx", "ty"],
    "magandang": ["mgandang", "mgndng"],
    ...
}
```
**▸ What this block does:** hand-built lookup table of standard Filipino words →
their common texting abbreviations. This is the human knowledge that makes fake
noise realistic.
- `ABBREVIATION_MAP` — a **dictionary** (`dict`). Each **key** (e.g. `"salamat"`)
  maps to a **value** that is a list of possible abbreviated forms.
- Written in `UPPER_CASE` because it's a module-level constant (a fixed table).
- The blocks below it (`VOWELS`, `SLANG_MAP`, `ORTHO_SUBSTITUTIONS`,
  `POSITIVE_EMOJIS`, etc.) are the same idea — reference data for each noise type.

```python
VOWELS = set("aeiouAEIOU")
```
**▸ What this block does:** the set of vowel characters, used by the vowel-dropping
noise.
- `set("aeiou...")` — a **set** is an unordered collection with fast "is this in
  it?" checks. `"x in VOWELS"` is quick.

### Class `TagalogNoiseGenerator`

```python
class TagalogNoiseGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
```
**▸ What this block does:** creates the generator and gives it its own seeded
random source, so the same clean corpus always produces the same noise.
- `def __init__(self, seed=42)` — the **constructor**; runs once when you write
  `TagalogNoiseGenerator()`. `seed` defaults to 42 (the project-wide seed).
- `random.Random(seed)` — builds a **private RNG** seeded with `seed`. Private (its
  own object, not the global `random`) so this generator's randomness is isolated
  and **reproducible**.
- `self.rng` — stores that RNG on the object so every method can use it.

```python
        self.p_abbreviation = 0.30
        self.p_orthographic = 0.20
        self.p_elongation = 0.15
        self.p_punctuation = 0.15
        self.p_capitalization = 0.15
        self.p_slang = 0.10
        self.p_vowel_omission = 0.20
        self.p_emoji_insert = 0.10
        self.p_char_swap = 0.10
```
**▸ What this block does:** sets the **probability** of each noise category firing.
Higher = noisier synthetic data.
- Each `self.p_*` is a `float` in [0,1] stored on the object. `p_abbreviation =
  0.30` means "apply abbreviation to ~30% of sentences." One knob per noise type.

### Method `apply_noise` — the main entry point

**▸ What this method does (whole function):** starts from the clean text and, for
each of the 9 categories independently, rolls the dice and maybe applies that
corruption. Because they're independent, several can stack on one sentence — like
real posts.

**Variables at a glance**
| Variable | Type | Holds |
|---|---|---|
| `clean_sentence` (param) | `str` | The clean input |
| `noisy` | `str` | The progressively-corrupted text |

```python
    def apply_noise(self, clean_sentence: str) -> str:
        noisy = clean_sentence
```
**▸ What this block does:** starts the noisy version as a copy of the clean text;
each step below may modify it.
- `noisy` — the working string; begins equal to `clean_sentence`.

```python
        if self.rng.random() < self.p_abbreviation:
            noisy = self._apply_abbreviation(noisy)
```
**▸ What this block does:** with ~30% probability, abbreviate some words.
- `self.rng.random()` — returns a random `float` in `[0, 1)`.
- `< self.p_abbreviation` — comparing to 0.30 gives a **30% chance** the block runs.
- `self._apply_abbreviation(noisy)` — call the helper (the `_` marks it internal),
  passing the current text and storing the result back in `noisy`.

```python
        if self.rng.random() < self.p_vowel_omission:
            noisy = self._apply_vowel_omission(noisy)
        if self.rng.random() < self.p_orthographic:
            noisy = self._apply_orthographic_variation(noisy)
        ... (elongation, punctuation, capitalization, slang, emoji, char_swap) ...
        return noisy
```
**▸ What this block does:** the same dice-roll pattern for the other 8 categories,
each calling its own helper, then returns the final noisy text. Stacking them is
what produces layered, realistic noise.

### Helper `_apply_abbreviation`

**▸ What this method does (whole function):** looks at each word; if it's in the
abbreviation dictionary, replaces it (about half the time) with a random
abbreviation, keeping any trailing punctuation.

**Variables at a glance**
| Variable | Type | Holds |
|---|---|---|
| `text` (param) | `str` | Sentence to modify |
| `words` | `list[str]` | The sentence split into words |
| `i`, `word` | `int`, `str` | Index + current word (loop) |
| `lower` | `str` | Lowercased, punctuation-stripped word (for lookup) |
| `abbrev` | `str` | The chosen abbreviation |
| `trailing` | `str` | Any punctuation to re-attach |

```python
    def _apply_abbreviation(self, text: str) -> str:
        words = text.split()
```
**▸ What this block does:** breaks the sentence into a list of words so we can
replace individual ones.
- `text.split()` — splits on whitespace into a `list` of words. `words` holds them.

```python
        for i, word in enumerate(words):
```
**▸ What this block does:** loops over each word with its position number.
- `enumerate(words)` — yields `(index, item)` pairs. `i` = the **index**
  (position, starting at 0), `word` = the word there. We need `i` to replace the
  word **in place** later.

```python
            lower = word.lower().strip(".,!?;:")
```
**▸ What this block does:** normalizes the word for dictionary lookup.
- `.lower()` — lowercases it (`"Salamat"` → `"salamat"`).
- `.strip(".,!?;:")` — removes any of those punctuation characters from both ends
  (`"salamat!"` → `"salamat"`), so it matches the dictionary key.
- `lower` — the cleaned lookup key.

```python
            if lower in ABBREVIATION_MAP:
                if self.rng.random() < 0.5:
                    abbrev = self.rng.choice(ABBREVIATION_MAP[lower])
                    trailing = word[len(lower):]
                    words[i] = abbrev + trailing
        return " ".join(words)
```
**▸ What this block does:** if the word is abbreviatable, ~50% of the time swap it
for a random abbreviation while preserving trailing punctuation, then rebuild the
sentence.
- `if lower in ABBREVIATION_MAP:` — dictionary membership test ("is this a key?").
- `if self.rng.random() < 0.5:` — a 50% chance, so not *every* eligible word changes.
- `self.rng.choice(list)` — pick one random element from the list of abbreviations.
- `word[len(lower):]` — slice from the length of the stripped word to the end,
  capturing any **trailing** punctuation that `.strip` removed (e.g. `"!"`).
- `words[i] = abbrev + trailing` — overwrite the word at position `i` (in place)
  with the abbreviation plus its punctuation.
- `" ".join(words)` — glue the word list back into one string with spaces between.

> The remaining helpers follow the **same shape** (split → loop → maybe transform →
> re-join), each doing one kind of corruption:
> - `_apply_vowel_omission` — drops interior vowels (keeps the first letter; skips
>   short words). Uses `if ch in VOWELS`.
> - `_apply_orthographic_variation` — dictionary swap to phonetic/number spellings.
> - `_apply_elongation` — repeats a vowel near the end (`grabe`→`grabeeee`);
>   `self.rng.randint(2, 5)` picks how many repeats (`randint(a,b)` = random whole
>   number between a and b inclusive).
> - `_apply_punctuation_noise` — uses `re.sub(pattern, repl, text)` to multiply or
>   remove punctuation.
> - `_apply_capitalization_noise` — random CAPS / ALLCAPS / lowercase.
> - `_apply_slang` — dictionary swap to slang forms.
> - `_apply_emoji_insert` — inserts an emoji at start/end/middle;
>   `self.rng.choice(pool)` picks which.
> - `_apply_char_swap` — swaps two adjacent letters (a typo);
>   `chars[pos], chars[pos+1] = chars[pos+1], chars[pos]` is Python's one-line swap.

### Method `generate_batch`

**▸ What this method does (whole function):** produces many `(noisy, clean)` pairs
from a list of clean sentences, retrying if the dice happened to change nothing
(so you never train on a useless identical pair).

```python
    def generate_batch(self, clean_sentences, noise_per_sentence=1):
        pairs = []
        for clean in clean_sentences:
            for _ in range(noise_per_sentence):
                noisy = self.apply_noise(clean)
                if noisy != clean:
                    pairs.append((noisy, clean))
                else:
                    noisy_retry = self.apply_noise(clean)
                    pairs.append((noisy_retry, clean))
        return pairs
```
**▸ line notes:**
- `pairs = []` — the output list of `(noisy, clean)` tuples.
- `for clean in clean_sentences:` — loop over each clean sentence.
- `for _ in range(noise_per_sentence):` — repeat `noise_per_sentence` times. `_` is
  a throwaway loop variable (we don't use the count).
- `noisy = self.apply_noise(clean)` — make a noisy version.
- `if noisy != clean:` — if noise actually changed something, keep the pair.
- `else: ... apply_noise(clean)` again — otherwise **retry once** so the pair is
  non-trivial. `pairs.append((noisy, clean))` adds the tuple to the list.
- `return pairs` — the finished list.

---

## `noise_label.py` — the noise score n\*

**Role:** Compute **n\*** — the *ground-truth* noise level of a pair, in [0,1].
This one number is the teaching target for TAHIMIK's noise estimator.

- **Input:** two strings (`noisy_text`, `clean_text`)
- **Output:** one `float` in [0,1] (0 = identical, ~1 = very corrupted)
- **Used by:** `dataset.py`, `preprocessing.py`, training scripts. Feeds the
  **L_NE** loss in [training](04-training.md).

### Where this fits in TAHIMIK
This tiny function is **load-bearing for the whole thesis contribution.** TAHIMIK's
idea is "sense the noise, then adapt compression." For the model to *learn* to
sense noise, it needs a target to aim at — and that target is **n\***, produced
here. It's the bridge between "we have (noisy, clean) pairs" and "the noise
estimator has something to be trained against (L_NE)."

**What breaks without it:** the noise estimator would have no supervision signal,
so `n` would be meaningless, so the adaptive shift `cn·(n−navg)` would be
conditioning on noise — the core mechanism — collapses. Small file, but pull it and
TAHIMIK stops being noise-adaptive.

**Example**
```
compute_noise_level("grabeeee ang init", "grabe ang init")
# 3 extra "e" bytes; longer string is 17 bytes
# → 3 / 17 ≈ 0.18   (mildly noisy)

compute_noise_level("same text", "same text")
# → 0.0             (identical = no noise)
```

**Terms & abbreviations in this file**
| Term | Full form / plain meaning |
|---|---|
| n\* | "n-star" — the ground-truth noise level of a pair |
| edit distance (Levenshtein) | the minimum single-character inserts/deletes/substitutions to turn one string into another |
| byte-level | measured on bytes (not letters), matching how ByT5 reads text |
| normalize (to [0,1]) | scale a raw number into the 0–1 range by dividing by a maximum |

### Full source, line by line

```python
import editdistance
```
**▸ What this block does:** imports the third-party library that computes edit
distance quickly.

```python
def compute_noise_level(noisy_text: str, clean_text: str) -> float:
    noisy_bytes = list(noisy_text.encode("utf-8"))
    clean_bytes = list(clean_text.encode("utf-8"))
```
**▸ What this block does:** converts both strings to lists of byte values, so the
distance is measured at the byte grain ByT5 sees.
- `.encode("utf-8")` — turn each string into raw bytes.
- `list(...)` — turn the `bytes` object into a plain list of integers 0–255.
- `noisy_bytes`, `clean_bytes` — those two lists.

```python
    max_len = max(len(noisy_bytes), len(clean_bytes))
    if max_len == 0:
        return 0.0
```
**▸ What this block does:** finds the longer length (the normalizer) and guards
against dividing by zero when both strings are empty.
- `max(a, b)` — the larger of the two. `len(...)` — how many bytes.
- `max_len` — length of the longer string.
- `if max_len == 0: return 0.0` — both empty → no noise, return `0.0` (a `float`).

```python
    distance = editdistance.eval(noisy_bytes, clean_bytes)
    return distance / max_len
```
**▸ What this block does:** measures how many byte edits separate the two, then
scales that into [0,1].
- `editdistance.eval(a, b)` — the number of single-byte edits to turn `a` into `b`.
- `distance` — that raw count (an `int`).
- `distance / max_len` — divide by the longer length → a `float` in [0,1]. Because
  the clean text *is* the reference, this ratio literally **is** the fraction of
  the sentence that was noise. That's why no human has to label noise.

---

## `dataset.py` — packing pairs into tensors

**Role:** Wrap the text pairs in a PyTorch **Dataset** so the training loop can
fetch one ready-to-use example at a time, already tokenized into tensors.

- **Input:** parallel lists `noisy_texts`, `clean_texts`, a tokenizer, optional
  precomputed `n*`
- **Output:** per example, a dict of tensors: `input_ids`, `attention_mask`,
  `labels`, `noise_level`
- **Used by:** `src/training/trainer.py` and eval scripts, via a `DataLoader`

### Where this fits in TAHIMIK
This is the **adapter between "Python lists of strings" and "batched tensors the
GPU can train on."** The models can't read text — they read number tensors. Every
single training and evaluation example passes through here to be tokenized into
`input_ids` + `attention_mask` + `labels` + `noise_level`. It sits exactly on the
seam **data → models**: `preprocessing.py` produces the split lists, this wraps
them, and the `DataLoader` in `trainer.py` pulls batches out.

**What breaks without it:** there would be no way to feed anything to the models —
training and evaluation both depend on it. It's the mandatory last step of the data
layer.

**Example** (one example the model receives)
```
ds = NormalizationDataset(["slmt po"], ["salamat po"], tokenizer)
ds[0]
# {
#   "input_ids":      tensor([...])  shape (1024,)   ← "slmt po" as byte IDs, padded
#   "attention_mask": tensor([1,1,1,1,1,1,1,0,0,...]) shape (1024,)  ← 1=real, 0=pad
#   "labels":         tensor([...])  shape (1024,)   ← "salamat po", padding = -100
#   "noise_level":    tensor(0.30)                   ← n* for this pair
# }
```

**Terms & abbreviations in this file**
| Term | Full form / plain meaning |
|---|---|
| tensor | a grid of numbers — PyTorch's data container |
| PyTorch `Dataset` | a class with `__len__` + `__getitem__` that serves one example at a time |
| tokenizer | the tool that turns text into model-ready number IDs |
| `input_ids` | the token IDs of the input |
| attention mask | 1 = real token, 0 = padding (tells the model what to ignore) |
| labels | the target token IDs the model should produce |
| `-100` | the special label value PyTorch's cross-entropy loss ignores |
| `collate_fn` | a function that stacks many single examples into one batch |
| `.squeeze()` | remove size-1 dimensions from a tensor's shape |
| `zip(a, b)` | walk two lists together, yielding pairs |
| `assert` | a sanity check that crashes with a message if a condition is false |

### Module-level code

```python
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from transformers import AutoTokenizer
from src.data.noise_label import compute_noise_level
```
**▸ What this block does:** imports PyTorch, the base `Dataset` class, type hints,
the ByT5 tokenizer loader, and — importantly — `compute_noise_level` from the
previous file (a **direct connection** between these two modules).
- `Optional[X]` — a hint meaning "an X **or** `None`."
- `AutoTokenizer` — Hugging Face's loader that fetches the right tokenizer by name.

### Class `NormalizationDataset`

```python
class NormalizationDataset(Dataset):
    def __init__(self, noisy_texts, clean_texts, tokenizer,
                 max_input_length=1024, max_target_length=1024,
                 precomputed_noise_levels=None):
```
**▸ What this block does:** the constructor — receives the text pairs, the
tokenizer, and length limits, and stores them.
- `Dataset` in the parentheses means this class **inherits** from PyTorch's base
  `Dataset` (gets its machinery).
- The parameters are the parallel lists + tokenizer + limits;
  `precomputed_noise_levels=None` lets you supply n\* values or have them computed.

```python
        assert len(noisy_texts) == len(clean_texts), (
            f"Mismatched pair count: {len(noisy_texts)} noisy vs {len(clean_texts)} clean"
        )
```
**▸ What this block does:** a safety check — the two lists must be the same length
(each noisy sentence needs its clean partner), else crash with a clear message.
- `assert CONDITION, MESSAGE` — if the condition is false, stop the program and
  print the message.
- `f"...{x}..."` — an **f-string**: text with `{...}` placeholders filled by the
  values inside (here, the two lengths).

```python
        self.noisy_texts = noisy_texts
        self.clean_texts = clean_texts
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
```
**▸ What this block does:** saves all inputs onto the object so other methods can
use them.

```python
        if precomputed_noise_levels is not None:
            self.noise_levels = precomputed_noise_levels
        else:
            self.noise_levels = [
                compute_noise_level(noisy, clean)
                for noisy, clean in zip(noisy_texts, clean_texts)
            ]
```
**▸ What this block does:** get the n\* value for every pair — reuse supplied ones,
or compute them now, once (not every epoch, which would waste time).
- `if precomputed_noise_levels is not None:` — if values were passed in, reuse them.
- Otherwise a **list comprehension** with `zip`: `zip(noisy_texts, clean_texts)`
  walks both lists in lockstep, giving `(noisy, clean)` pairs; for each,
  `compute_noise_level(...)` (from `noise_label.py`) produces one n\*.
- `self.noise_levels` — the resulting list of floats.

```python
    def __len__(self) -> int:
        return len(self.noisy_texts)
```
**▸ What this block does:** tells PyTorch how many examples exist (required by
`Dataset`). `__len__` is the special method `len(dataset)` calls.

### Method `__getitem__` — fetch one example

**▸ What this method does (whole function):** required by `Dataset`; given a
position `idx`, tokenizes that pair into tensors and returns the four-field dict
one training example needs.

**Variables at a glance**
| Variable | Type / shape | Holds |
|---|---|---|
| `idx` (param) | `int` | Which example to fetch |
| `noisy`, `clean` | `str` | The pair's texts |
| `input_encoding` | dict of tensors | Tokenized noisy input |
| `target_encoding` | dict of tensors | Tokenized clean target |
| `labels` | tensor `(seq_len,)` | Target IDs with padding→`-100` |

```python
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        noisy = self.noisy_texts[idx]
        clean = self.clean_texts[idx]
```
**▸ What this block does:** looks up the noisy and clean text at position `idx`.
- `self.noisy_texts[idx]` — index into the list. `idx` is the example number
  PyTorch asks for.

```python
        input_encoding = self.tokenizer(
            noisy, max_length=self.max_input_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
```
**▸ What this block does:** converts the noisy string into model-ready tensors of a
fixed length.
- `self.tokenizer(noisy, ...)` — run the ByT5 tokenizer on the text.
- `max_length=1024` — the fixed length.
- `padding="max_length"` — pad short inputs up to that length.
- `truncation=True` — cut inputs longer than that length.
- `return_tensors="pt"` — return **P**y**T**orch tensors (`"pt"`).
- Result `input_encoding` is a dict with `input_ids` (the byte IDs) and
  `attention_mask` (1=real, 0=pad).

```python
        target_encoding = self.tokenizer(clean, ...)
```
**▸ What this block does:** the same tokenization for the clean target sentence.

```python
        labels = target_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
```
**▸ What this block does:** prepare the target IDs as `labels`, then hide padding
positions from the loss.
- `target_encoding["input_ids"]` — the target's token IDs (shape `(1, seq_len)`).
- `.squeeze()` — remove the size-1 first dimension → shape `(seq_len,)`.
- `labels[labels == self.tokenizer.pad_token_id] = -100` — **boolean masking**:
  `labels == pad_token_id` is a True/False tensor marking padding positions;
  assigning `-100` there overwrites those with `-100`, the value PyTorch's
  cross-entropy loss ignores. So the model isn't graded on predicting filler.

```python
        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": labels,
            "noise_level": torch.tensor(self.noise_levels[idx], dtype=torch.float32),
        }
```
**▸ What this block does:** returns the four tensors that define one training
example.
- `.squeeze()` on the two input tensors drops their size-1 dimension → shape
  `(seq_len,)`.
- `torch.tensor(self.noise_levels[idx], dtype=torch.float32)` — wrap the plain
  float n\* into a tensor. `dtype=torch.float32` = 32-bit decimal type.

### Function `collate_fn`

**▸ What this function does (whole thing):** the `DataLoader` gathers several
single-example dicts into a list and hands them here; this stacks matching tensors
into one **batch** tensor.

```python
def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "noise_level": torch.stack([b["noise_level"] for b in batch]),
    }
```
**▸ line notes:**
- `batch` — a list of the per-example dicts from `__getitem__`.
- `[b["input_ids"] for b in batch]` — a list comprehension pulling one field out of
  every example.
- `torch.stack([...])` — pile those tensors along a **new** first dimension. Eight
  tensors of shape `(1024,)` become one of shape `(8, 1024)` — i.e. add the batch
  dimension. The `DataLoader` calls this automatically for each batch.

---

## `preprocessing.py` — the conductor

**Role:** Orchestrate the whole data preparation: load files, clean text, generate
synthetic pairs, compute n\*, and split into train/val/test.

- **Input:** file paths (clean corpus and/or gold-standard CSV/JSON)
- **Output:** Python lists of `(noisy, clean, noise_level)`, split into
  train/val/(test)
- **Used by:** the entry scripts. Calls `noise_generator` and `noise_label`.

### Where this fits in TAHIMIK
This is the **conductor of the data layer** — the first thing every script
(`train.py`, `evaluate.py`, `run_experiment.py`) calls. It turns raw files on disk
into the clean train/val/test lists everything downstream needs, orchestrating the
other three data files in order: load → clean → (optionally) synthesize noise →
compute n\* → split by seed 42. Think of `noise_generator`, `noise_label`, and
`dataset` as the instruments; this is the score that tells them when to play.

**What breaks without it:** the scripts would have no data to hand the trainer.
It's the entry point of the entire offline flow.

**Example** (what a script does with it)
```
pipeline = DataPipeline(config, seed=42)
noisy, clean = pipeline.load_gold_standard("data/gold.csv")   # 15,000 pairs
n_stars = [compute_noise_level(x, y) for x, y in zip(noisy, clean)]
splits = pipeline.split_data(noisy, clean, n_stars, 0.8, 0.1, 0.1)
# splits["train"] → (noisy_list, clean_list, n_star_list)   ~12,000 pairs
# splits["val"]   → ~1,500 pairs
# splits["test"]  → ~1,500 pairs   (held out for final scoring)
```

**Terms & abbreviations in this file**
| Term | Full form / plain meaning |
|---|---|
| I/O | input/output — reading/writing files |
| CSV | comma-separated values — a table file |
| JSON | JavaScript Object Notation — a structured data file |
| suffix | a file's extension (`.txt`, `.csv`, `.json`) |
| context manager (`with`) | `with open(...) as f:` safely opens and auto-closes a file |
| `DictReader` | reads a CSV into dictionaries keyed by column name |
| train / val / test | data splits: learn / tune / final unbiased score |
| shuffle | randomly reorder items |
| `//` | integer division (divide and drop the remainder) |
| `%` | modulo (the remainder after division) |
| `.get(key, default)` | dict lookup that returns `default` if the key is missing |
| helper function | a small function used by others (here, at module level) |

### Class `DataPipeline` — constructor

```python
class DataPipeline:
    def __init__(self, config, seed: int = 42):
        self.config = config
        self.rng = random.Random(seed)
        self.noise_gen = TagalogNoiseGenerator(seed=seed)
```
**▸ What this block does:** sets up the pipeline with its config, its own seeded
RNG (for splitting), and an instance of the noise generator (a **direct
connection** to `noise_generator.py`).
- `self.config` — the run's settings object (holds split ratios, lengths, etc.).
- `self.rng` — a seeded RNG used later for shuffling the split.
- `self.noise_gen` — a `TagalogNoiseGenerator`, seeded identically for
  reproducibility.

### Method `load_clean_corpus`

**▸ What this method does (whole function):** reads clean sentences from a `.txt`,
`.csv`, or `.json` file, then filters out ones that are too short or too long.

```python
    def load_clean_corpus(self, filepath: str) -> List[str]:
        path = Path(filepath)
        sentences = []
```
**▸ What this block does:** wraps the path and starts an empty result list.
- `Path(filepath)` — a path object; `.suffix` below reads its extension.

```python
        if path.suffix == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                sentences = [line.strip() for line in f if line.strip()]
```
**▸ What this block does:** for a text file, read it line by line, keeping
non-blank lines.
- `path.suffix` — the file extension (`.txt`).
- `with open(...) as f:` — a **context manager**: opens the file as `f` and
  automatically closes it afterward. `"r"` = read mode, `encoding="utf-8"` for
  Filipino characters.
- `[line.strip() for line in f if line.strip()]` — loop over lines; `.strip()`
  removes surrounding whitespace/newline; the `if line.strip()` filter keeps only
  non-empty lines.

```python
        elif path.suffix == ".csv":
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get("text", row.get("sentence", "")).strip()
                    if text:
                        sentences.append(text)
        elif path.suffix == ".json":
            ...
```
**▸ What this block does:** for CSV, read each row as a dictionary and pull the
`text` (or `sentence`) column; for JSON, handle a list of strings or objects.
- `csv.DictReader(f)` — reads rows as dicts keyed by the header names.
- `row.get("text", row.get("sentence", ""))` — try the `"text"` column; if missing,
  try `"sentence"`; if that's missing too, use `""`. `.get(key, default)` avoids a
  crash on a missing key.

```python
        sentences = [s for s in sentences if len(s.split()) >= 4]
        sentences = [s for s in sentences if len(s.encode("utf-8")) <= 1024]
        return sentences
```
**▸ What this block does:** apply the manuscript's filters — keep sentences with
at least 4 words and at most 1024 bytes.
- `len(s.split()) >= 4` — word count ≥ 4.
- `len(s.encode("utf-8")) <= 1024` — byte length ≤ 1024.

### Method `load_gold_standard`

**▸ What this method does (whole function):** loads the human-annotated
`(noisy, clean)` pairs from CSV/JSON, tolerating different column names.

```python
        noisy = row.get("noisy", row.get("input", "")).strip()
        clean = row.get("clean", row.get("target", row.get("normalized", ""))).strip()
        if noisy and clean:
            noisy_texts.append(noisy)
            clean_texts.append(clean)
```
**▸ line notes:**
- Chained `.get(...)` calls try several possible column names (`noisy`/`input`,
  `clean`/`target`/`normalized`) so different file headers still work.
- `if noisy and clean:` — only keep the pair if **both** are non-empty (a
  non-empty string is "truthy").

### Method `generate_synthetic_pairs`

**▸ What this method does (whole function):** builds the ~1M Stage-1 examples by
running the noise generator over the clean corpus enough times to hit the target
size, computing n\* for each.

```python
    def generate_synthetic_pairs(self, clean_sentences, target_size=1_000_000):
        noisy_texts = []; clean_texts = []; noise_levels = []
        passes = max(1, target_size // len(clean_sentences))
        remainder = target_size % len(clean_sentences)
```
**▸ What this block does:** figure out how many full passes over the corpus are
needed, plus the leftover.
- `1_000_000` — Python lets you put `_` in numbers for readability (= 1,000,000).
- `target_size // len(clean_sentences)` — `//` is **integer division**; how many
  whole passes reach the target. `max(1, ...)` ensures at least one pass.
- `target_size % len(clean_sentences)` — `%` is **modulo**, the leftover count.

```python
        for pass_num in range(passes):
            for clean in clean_sentences:
                noisy = self.noise_gen.apply_noise(clean)
                n_star = compute_noise_level(noisy, clean)
                noisy_texts.append(noisy)
                clean_texts.append(clean)
                noise_levels.append(n_star)
```
**▸ What this block does:** the core generation loop — for each pass, for each
clean sentence, make a noisy version and score it. This is where the three data
files come together.
- `self.noise_gen.apply_noise(clean)` — call **noise_generator**.
- `compute_noise_level(noisy, clean)` — call **noise_label**.
- The three lists grow in lockstep (parallel lists — same index = same example).

```python
        extra = self.rng.sample(clean_sentences, min(remainder, len(clean_sentences)))
        for clean in extra:
            ... (same three appends) ...
        return noisy_texts, clean_texts, noise_levels
```
**▸ What this block does:** top up with `remainder` more randomly-chosen sentences
to hit the exact target, then return the three parallel lists.
- `self.rng.sample(list, k)` — pick `k` **distinct** random items.

### Method `split_data`

**▸ What this method does (whole function):** shuffles the examples (reproducibly)
and slices them into train / val / (test) by the given ratios.

```python
    def split_data(self, noisy_texts, clean_texts, noise_levels,
                   train_ratio, val_ratio, test_ratio=0.0):
        n = len(noisy_texts)
        indices = list(range(n))
        self.rng.shuffle(indices)
```
**▸ What this block does:** build a list of position numbers and shuffle them, so
the split is random but repeatable.
- `n` — total number of examples.
- `list(range(n))` — `[0, 1, 2, ..., n-1]`, the positions.
- `self.rng.shuffle(indices)` — reorder them in place using the seeded RNG.

```python
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
```
**▸ What this block does:** compute the cut points for the splits.
- `int(n * train_ratio)` — e.g. 80% of n, as a whole number (`int(...)` drops
  decimals).
- `val_end` — where validation ends (train end + 10% of n).

```python
        splits = {}
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        splits["train"] = ([noisy_texts[i] for i in train_idx],
                           [clean_texts[i] for i in train_idx],
                           [noise_levels[i] for i in train_idx])
        splits["val"] = (... val_idx ...)
        if test_ratio > 0:
            test_idx = indices[val_end:]
            splits["test"] = (... test_idx ...)
        return splits
```
**▸ What this block does:** slice the shuffled positions into three groups and
gather the matching items from all three parallel lists into `(noisy, clean,
noise_levels)` tuples — exactly the shape `NormalizationDataset` expects.
- `indices[:train_end]` / `[train_end:val_end]` / `[val_end:]` — slices of the
  shuffled positions.
- `[noisy_texts[i] for i in train_idx]` — gather the items at those positions.
- `splits` — a dict keyed `"train"`, `"val"`, optionally `"test"`.

### Module-level helper functions

```python
def _replace_mentions(text): return re.sub(r"@\w+", "@ANON", text)
def _replace_urls(text):     return re.sub(r"https?://\S+|www\.\S+", "<URL>", text)
```
**▸ What this block does:** two small privacy helpers (defined outside the class),
used by `clean_text`.
- `re.sub(pattern, replacement, text)` — regex find-and-replace.
- `r"@\w+"` — a **raw string** (`r"..."`, so backslashes are literal) pattern:
  `@` then one-or-more word characters (`\w+`) → replaced with `@ANON`.
- `r"https?://\S+|www\.\S+"` — matches `http`/`https` links **or** (`|`) `www.`
  links (`\S+` = one-or-more non-space characters) → replaced with `<URL>`.

---

## `__init__.py`

```python
from src.data.noise_generator import TagalogNoiseGenerator
from src.data.dataset import NormalizationDataset
from src.data.preprocessing import DataPipeline
from src.data.noise_label import compute_noise_level
```
**▸ What this block does:** re-exports the four public names so other code can do
`from src.data import DataPipeline`, etc.

- **Inputs:** none (runs on import)
- **Outputs:** the four names above, importable from `src.data`
