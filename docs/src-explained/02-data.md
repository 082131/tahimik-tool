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

> **Format reminder:** every line of code is covered individually. Each block opens
> with **▸ What this block does**, then explains every line, variable, and technical
> term in full (never compressed). Distinct functions also get a **Worked
> walkthrough** tracing concrete values through the code; near-identical repeats get
> a short "same as X, but…" example.

---

## Current methodology update (2026-09-03)

### Configurable correctable-noise probabilities

```python
def __init__(self, seed: int = 42, probabilities=None):
    self.rng = random.Random(seed)
    probabilities = probabilities or {}
    self.p_abbreviation = probabilities.get("abbreviation", 0.30)
    self.p_orthographic = probabilities.get("orthographic", 0.20)
    self.p_elongation = probabilities.get("elongation", 0.15)
    self.p_punctuation = probabilities.get("punctuation", 0.15)
    self.p_capitalization = probabilities.get("capitalization", 0.15)
    self.p_slang = 0.0
    self.p_vowel_omission = probabilities.get("vowel_omission", 0.20)
    self.p_emoji_insert = 0.0
    self.p_char_swap = probabilities.get("char_swap", 0.10)
```

**▸ Every changed line:**

- `probabilities=None` lets a caller supply a category → probability dictionary.
- `random.Random(seed)` creates this generator's isolated, repeatable RNG.
- `probabilities or {}` converts a missing dictionary into an empty one.
- Each `.get("name", default)` uses the supplied value when present and the shown
  fallback otherwise.
- `p_slang = 0.0` and `p_emoji_insert = 0.0` disable those transformations in the
  current input-only corruption path. They were disabled because slang and emoji
  should not be treated as mistakes that disappear from the target.
- The `apply_noise` method still draws `self.rng.random()` for each category and
  applies a helper only when that draw is below the corresponding probability.

**Example:** with `{"abbreviation": 1.0, "char_swap": 0.0}`, abbreviation is
always attempted, character swapping is never attempted, and unspecified
categories use their fallback values.

**Important current limitation:** `DataPipeline.__init__` still constructs
`TagalogNoiseGenerator(seed=seed)` without passing resolved probabilities. Thus
this constructor is configurable in isolation, but the main pipeline does not yet
satisfy spec 011's manifest-driven configuration. Slang, emoji, code-switching,
and Taglish morphology also do not yet use the planned two-pass “same feature in
input and target” augmentation path.

### Gold-pair conflict detection

```python
normalized = {}
for noisy, clean in zip(noisy_texts, clean_texts):
    noisy, clean = self.clean_text(noisy), self.clean_text(clean)
    if noisy in normalized and normalized[noisy] != clean:
        raise ValueError(f"conflicting clean targets for noisy sentence: {noisy!r}")
    normalized[noisy] = clean
noisy_texts = list(normalized.keys())
clean_texts = list(normalized.values())
```

**▸ Every line and syntax:**

- `normalized = {}` stores one clean target per normalized noisy sentence.
- `zip(...)` walks noisy and clean lists as aligned pairs.
- The tuple assignment cleans both sides on one line.
- The `if` detects the same noisy input paired with a different clean target.
- `raise ValueError(...)` stops instead of silently choosing one annotation;
  `{noisy!r}` uses Python's representation form so whitespace is visible.
- Assignment keeps the pair. An exact duplicate overwrites the same dictionary
  entry, effectively deduplicating it.
- `keys()` and `values()` rebuild aligned lists in insertion order.

**Worked example:** `("ang sarappp ng food ngayon", "ang sarap ng food ngayon")`
followed by the same noisy sentence with target `"masarap ang food ngayon"` raises
an error because one source has two incompatible gold answers.

**Privacy warning:** the current exception includes raw sentence text. Spec 010
requires future eligibility reports to use stable IDs and reason codes without
copying private text into logs.

### What the loaders enforce today

- The clean synthetic corpus enforces at least four whitespace-separated words
  and at most 1,024 UTF-8 bytes.
- The gold loader currently checks non-empty fields, cleans text, removes exact
  duplicate sources, and rejects conflicting targets.
- It does **not yet** enforce the four-word/byte rules on gold rows, approval and
  anonymization columns, exact 15,000 size, reliability CSV coverage, or the
  deterministic 12,000/1,500/1,500 membership contract. Those are spec-010 tasks.
- `generate_synthetic_pairs` accepts `target_size=1_000_000`, but an individual
  caller can choose another size and the function itself does not certify Chapter
  3 eligibility.

**Defense-ready summary:** “The present code can generate repeatable synthetic
errors and reject conflicting targets, while the new specs deliberately prevent
us from claiming full corpus compliance until the external CSV contracts,
training-derived probability manifest, and exact-size gates are implemented.”
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
    ...   # ~35 entries in total — shown abbreviated here, not an elision of logic
}
```
**▸ What this block does:** hand-built lookup table of standard Filipino words →
their common texting abbreviations. This is the human knowledge that makes fake
noise realistic. (The `...` above just means the dictionary continues with more
word→abbreviation entries; it's *data*, not skipped code.)
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
```
**▸ What this block does:** with ~20% probability, drop interior vowels from some
words (`punta` → `pnta`).
- `self.rng.random() < self.p_vowel_omission` — a fresh dice roll, 20% chance.
- `self._apply_vowel_omission(noisy)` — run that helper on the *current* `noisy`
  (which may already have abbreviations applied from the block above), storing the
  result back into `noisy`.

```python
        if self.rng.random() < self.p_orthographic:
            noisy = self._apply_orthographic_variation(noisy)
```
**▸ What this block does:** with ~20% probability, apply phonetic/number spellings
(`dito` → `d2`). Same dice-roll-then-helper pattern.

```python
        if self.rng.random() < self.p_elongation:
            noisy = self._apply_elongation(noisy)
```
**▸ What this block does:** with ~15% probability, stretch letters for emphasis
(`grabe` → `grabeeee`).

```python
        if self.rng.random() < self.p_punctuation:
            noisy = self._apply_punctuation_noise(noisy)
```
**▸ What this block does:** with ~15% probability, mangle punctuation (`!` → `!!!!`,
or remove it).

```python
        if self.rng.random() < self.p_capitalization:
            noisy = self._apply_capitalization_noise(noisy)
```
**▸ What this block does:** with ~15% probability, mess up capitalization (random
CAPS, ALL CAPS, or all lowercase).

```python
        if self.rng.random() < self.p_slang:
            noisy = self._apply_slang(noisy)
```
**▸ What this block does:** with ~10% probability, swap words for slang (`idol` →
`lodi`).

```python
        if self.rng.random() < self.p_emoji_insert:
            noisy = self._apply_emoji_insert(noisy)
```
**▸ What this block does:** with ~10% probability, insert an emoji somewhere.

```python
        if self.rng.random() < self.p_char_swap:
            noisy = self._apply_char_swap(noisy)
```
**▸ What this block does:** with ~10% probability, swap two adjacent letters to
simulate a typo (`ang` → `nag`).

```python
        return noisy
```
**▸ What this block does:** returns the final text after all the dice rolls. Because
each `if` was independent, `noisy` may have accumulated several kinds of corruption
— which is exactly the layered noise real posts have.

**Worked walkthrough** (one call, seeded)
```
apply_noise("Salamat idol grabe")

start:  noisy = "Salamat idol grabe"

roll 1 (abbreviation, <0.30?)  → hits → "slmt idol grabe"
roll 2 (vowel omission, <0.20?)→ misses → unchanged
roll 3 (orthographic, <0.20?)  → misses → unchanged
roll 4 (elongation, <0.15?)    → hits → "slmt idol grabeeee"
roll 5 (punctuation, <0.15?)   → misses
roll 6 (capitalization, <0.15?)→ misses
roll 7 (slang, <0.10?)         → hits → "slmt lodi grabeeee"
roll 8 (emoji, <0.10?)         → misses
roll 9 (char swap, <0.10?)     → misses

returns "slmt lodi grabeeee"
# paired with the clean "Salamat idol grabe" → one training example
```
(The exact rolls depend on the seed; this shows how corruptions *stack* across the
independent checks.)

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

### The other 8 helper methods

All follow the **same shape as `_apply_abbreviation`** above — *split the sentence
into words (or characters), loop, maybe transform, re-join* — so rather than repeat
that structure eight times, here is each one with the one thing that makes it
different and a short worked example. (This is the "brief for repeats" treatment;
`_apply_abbreviation` above is the full pattern.)

**`_apply_vowel_omission`** — drops interior vowels from a word to mimic texting.
Keeps the first letter, skips short words (≤3 chars), and only drops each vowel with
some probability (`if ch in VOWELS and self.rng.random() < 0.6`).
```
"punta"  → keep "p", drop some of "u/a" → "pnta"
"gabi"   → "gbi"
```

**`_apply_orthographic_variation`** — dictionary swap (like abbreviation, but using
`ORTHO_SUBSTITUTIONS`) for phonetic/number spellings.
```
"dito"  → "d2"   ;   "hindi" → "hnd"
```

**`_apply_elongation`** — repeats a vowel near the end of a word for emphasis. The
count comes from `self.rng.randint(2, 5)` (`randint(a, b)` = a random whole number
from a to b **inclusive**).
```
"grabe" → repeat the last vowel 4× → "grabeeee"
"sarap" → "saraaap"
```

**`_apply_punctuation_noise`** — uses `re.sub(pattern, repl, text)` (regex
find-replace) to either multiply terminal punctuation, remove some, or turn `.`
into `...`.
```
"grabe!"  → "grabe!!!!!"
"tama."    → "tama..."
```

**`_apply_capitalization_noise`** — randomly chooses one of three styles: sprinkle
random CAPS, ALL-CAPS a word or two, or lowercase everything.
```
"grabe naman" → "GRABE naman"   (allcaps on one word)
"Grabe Naman" → "grabe naman"   (nocaps)
```

**`_apply_slang`** — dictionary swap using `SLANG_MAP`.
```
"idol"    → "lodi"   ;   "grabe" → "grabiii"
```

**`_apply_emoji_insert`** — picks one emoji with `self.rng.choice(pool)` and inserts
it at the start, end, or middle.
```
"late na ako"  → "late na ako 😭"
```

**`_apply_char_swap`** — swaps two adjacent alphabetic characters to simulate a
typo. The swap itself is Python's one-line tuple swap:
`chars[pos], chars[pos+1] = chars[pos+1], chars[pos]`.
```
"ang"  → swap positions 0,1 → "nag"
```

### Method `generate_batch`

**▸ What this method does (whole function):** produces many `(noisy, clean)` pairs
from a list of clean sentences, retrying if the dice happened to change nothing
(so you never train on a useless identical pair).

**Variables at a glance**
| Variable | Type | Holds |
|---|---|---|
| `clean_sentences` (param) | `list[str]` | Clean sentences to corrupt |
| `noise_per_sentence` (param) | `int` (default 1) | How many noisy variants per clean sentence |
| `pairs` | `list[tuple]` | The `(noisy, clean)` results |
| `clean` | `str` | Current clean sentence (outer loop) |
| `noisy` / `noisy_retry` | `str` | A generated noisy version |

```python
    def generate_batch(self, clean_sentences, noise_per_sentence=1):
        pairs = []
```
**▸ What this block does:** starts an empty list that will collect the finished
`(noisy, clean)` pairs.
- `pairs` — a Python list; each element will be a **tuple** `(noisy, clean)`.

```python
        for clean in clean_sentences:
```
**▸ What this block does:** loops over every clean sentence you passed in.
- `clean` — the current clean sentence being corrupted.

```python
            for _ in range(noise_per_sentence):
```
**▸ What this block does:** repeats the corruption `noise_per_sentence` times, so
you can make several different noisy versions of the *same* clean sentence.
- `range(noise_per_sentence)` — counts from 0 up to that number.
- `_` — a throwaway loop variable; the underscore signals "I don't use the count,
  I just want to repeat this many times."

```python
                noisy = self.apply_noise(clean)
```
**▸ What this block does:** produce one noisy version of the current clean sentence
by calling `apply_noise` (the method above).

```python
                if noisy != clean:
                    pairs.append((noisy, clean))
```
**▸ What this block does:** if the noise actually changed the text, keep the pair.
- `noisy != clean` — "did anything change?" (all the dice rolls could have missed).
- `pairs.append((noisy, clean))` — add the tuple to the results list.

```python
                else:
                    noisy_retry = self.apply_noise(clean)
                    pairs.append((noisy_retry, clean))
```
**▸ What this block does:** if nothing changed, **retry once** and keep whatever
that produces — so you never store a useless `(clean, clean)` pair that teaches the
model nothing.
- `noisy_retry` — a second attempt (a fresh set of dice rolls).
- `pairs.append((noisy_retry, clean))` — store it regardless (even if the retry also
  changed nothing, one such pair is harmless).

```python
        return pairs
```
**▸ What this block does:** returns the full list of pairs.

**Worked walkthrough**
```
generate_batch(["Salamat idol", "Kumain ka na"], noise_per_sentence=2)

"Salamat idol":
  attempt 1 → "slmt idol"   (changed → keep)          pairs += ("slmt idol", "Salamat idol")
  attempt 2 → "Salamat lodi"(changed → keep)          pairs += ("Salamat lodi", "Salamat idol")
"Kumain ka na":
  attempt 1 → "Kumain ka na"(no change → retry once)
             retry → "kumain ka na" (keep the retry)  pairs += ("kumain ka na", "Kumain ka na")
  attempt 2 → "kmain ka na" (changed → keep)           pairs += ("kmain ka na", "Kumain ka na")

returns 4 pairs (2 per clean sentence)
```

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

**Worked walkthrough**
```
compute_noise_level("grabeee", "grabe")

noisy_bytes = list("grabeee".encode("utf-8"))
            = [103, 114, 97, 98, 101, 101, 101]   # g r a b e e e  → length 7
clean_bytes = list("grabe".encode("utf-8"))
            = [103, 114, 97, 98, 101]             # g r a b e      → length 5

max_len = max(7, 5) = 7
max_len == 0?  no → continue

distance = editdistance.eval(noisy_bytes, clean_bytes)
         = 2        # delete the 2 extra "e" bytes to turn "grabeee" into "grabe"

return distance / max_len
     = 2 / 7
     ≈ 0.286        # ~29% of the sentence was noise
```
Contrast: `compute_noise_level("grabe", "grabe")` → distance `0` → `0 / 5 = 0.0`
(no noise). So a small n\* = clean, a large n\* = heavily corrupted — the exact
signal the noise estimator learns to reproduce.

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
        target_encoding = self.tokenizer(
            clean, max_length=self.max_target_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
```
**▸ What this block does:** the exact same tokenization, but for the **clean target**
sentence (the answer the model should produce), using `max_target_length`. Result
`target_encoding` is a dict with the target's `input_ids` and `attention_mask`; only
its `input_ids` are used below (as the `labels`).

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

**Worked walkthrough** (fetching one example, shortened to length 5 for readability)
```
ds = NormalizationDataset(["slmt", "d2"], ["salamat", "dito"], tokenizer)
ds[0]                       # idx = 0

noisy = self.noisy_texts[0] = "slmt"
clean = self.clean_texts[0] = "salamat"

input_encoding = tokenizer("slmt", ...)
  input_encoding["input_ids"]      = tensor([[118, 111, 112, 119, 1]])   shape (1, 5)
  input_encoding["attention_mask"] = tensor([[1, 1, 1, 1, 1]])

target_encoding = tokenizer("salamat", ...)
  target_encoding["input_ids"]     = tensor([[118, 100, 111, 100, 1]])   shape (1, 5)

labels = target_encoding["input_ids"].squeeze()   = tensor([118, 100, 111, 100, 1])
# (no padding here, so no positions become -100; if it were padded,
#  those pad IDs would be replaced with -100)

returns {
  "input_ids":      tensor([118, 111, 112, 119, 1]),   # .squeeze() → shape (5,)
  "attention_mask": tensor([1, 1, 1, 1, 1]),
  "labels":         tensor([118, 100, 111, 100, 1]),
  "noise_level":    tensor(0.57),                       # n* for ("slmt","salamat")
}
```
(The byte-ID numbers are illustrative. In the real project everything is padded to
1024, so each tensor above would be shape `(1024,)` with the tail filled by pad-IDs
in `input_ids`/`attention_mask` and by `-100` in `labels`.)

### Function `collate_fn`

**▸ What this function does (whole thing):** `collate_fn` combines individual
examples into one **batch** that the model can process at once. A neural network is
far faster when it processes many sentences together (as one big tensor) than one
sentence at a time. `__getitem__` (above) produces **one** example at a time; this
function is the step that glues a handful of them together. The `DataLoader` calls
it automatically every time it forms a batch — you never call it yourself.

**Variables at a glance**
| Variable | Type / shape | Holds |
|---|---|---|
| `batch` (param) | `list[dict]` | Several single-example dicts from `__getitem__` |
| (return) | `dict` of tensors | The same fields, but stacked across the batch |

#### Line by line

```python
def collate_fn(batch):
```
**▸ What this block does:** defines the function. `batch` is the list of individual
examples the `DataLoader` collected.
- `batch` — a Python **list**, where each element is one dict exactly like what
  `__getitem__` returns (`input_ids`, `attention_mask`, `labels`, `noise_level`).
  If the batch size is 8, this list has 8 dicts.

```python
    return {
```
**▸ What this block does:** starts building — and returning — one **new** dictionary.
It has the same four field names as a single example, but this time each field will
hold data for the *entire batch* rather than one sentence.

```python
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
```
**▸ What this block does:** builds the batched `input_ids` tensor. Read it in two
parts:
- `[b["input_ids"] for b in batch]` — a **list comprehension** that loops through
  every example `b` in the list and pulls out just its `input_ids` tensor,
  producing a plain list of same-sized 1-D tensors (one per sentence).
- `torch.stack([...])` — takes that list of same-sized tensors and **stacks** them
  along a brand-new first dimension, producing a single 2-D tensor. If each
  `input_ids` was shape `(1024,)` and there are 8 of them, the result is
  `(8, 1024)`. That new leading `8` is the **batch dimension**.

```python
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
```
**▸ What this block does:** exactly the same operation for the attention masks —
collect each example's `attention_mask` and stack them into one `(8, 1024)` tensor,
so the model knows which positions are real vs padding for every sentence at once.

```python
        "labels": torch.stack([b["labels"] for b in batch]),
```
**▸ What this block does:** the same again for the target `labels` (the clean-text
byte IDs, with padding marked as `-100`). Result shape `(8, 1024)`.

```python
        "noise_level": torch.stack([b["noise_level"] for b in batch]),
```
**▸ What this block does:** the same for the noise scores. Each example's
`noise_level` is a single number (a **scalar** tensor, shape `()`), so stacking 8 of
them gives a 1-D tensor of shape `(8,)` — one n\* per sentence in the batch.

```python
    }
```
**▸ What this block does:** closes and returns the finished batched dictionary. The
`DataLoader` hands this dict straight to the training loop, which passes its fields
into the model.

#### Worked walkthrough (batch size 2)

Imagine the `DataLoader` requests a batch size of 2. It first calls `__getitem__()`
twice, producing this Python list (shortened to length 5 for readability):

```
batch = [
    {
        "input_ids":      tensor([107, 108, 1, 0, 0]),
        "attention_mask": tensor([1, 1, 1, 0, 0]),
        "labels":         tensor([107, 108, 1, -100, -100]),
        "noise_level":    tensor(0.2),
    },
    {
        "input_ids":      tensor([100, 101, 1, 0, 0]),
        "attention_mask": tensor([1, 1, 1, 0, 0]),
        "labels":         tensor([100, 101, 1, -100, -100]),
        "noise_level":    tensor(0.6),
    },
]
```

Each dictionary is one sentence pair. Now trace the code:

`def collate_fn(batch):` — `batch` is that list of two examples.

`return {` — returns one new dictionary where every field holds data for the *whole*
batch.

`[b["input_ids"] for b in batch]` — loops through each example `b` and collects its
`input_ids`:

```
[
    tensor([107, 108, 1, 0, 0]),
    tensor([100, 101, 1, 0, 0]),
]
```

`torch.stack([...])` — stacks those same-sized tensors into one tensor by adding a
new first dimension:

```
tensor([
    [107, 108, 1, 0, 0],
    [100, 101, 1, 0, 0],
])
```

The shape changes from:

```
one example:  (5,)
two examples: (2, 5)
```

So the line `"input_ids": torch.stack([b["input_ids"] for b in batch]),` creates the
batch of input token IDs. The remaining lines do exactly the same for the matching
fields.

`"attention_mask": torch.stack([b["attention_mask"] for b in batch]),` becomes:

```
tensor([
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
])
```

`"labels": torch.stack([b["labels"] for b in batch]),` becomes:

```
tensor([
    [107, 108, 1, -100, -100],
    [100, 101, 1, -100, -100],
])
```

`"noise_level": torch.stack([b["noise_level"] for b in batch]),` combines the scalar
noise scores:

```
tensor([0.2, 0.6])
```

The final output is:

```
{
    "input_ids":      tensor of shape (2, 5),
    "attention_mask": tensor of shape (2, 5),
    "labels":         tensor of shape (2, 5),
    "noise_level":    tensor of shape (2,),
}
```

With the project's actual configuration — a batch size of 8 and
`max_input_length = 1024` — the same code gives:

```
input_ids:      (8, 1024)
attention_mask: (8, 1024)
labels:         (8, 1024)   # assuming target length is also 1024
noise_level:    (8,)
```

The model then processes all eight examples together, which is much faster than
training one sentence at a time.

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
```
**▸ What this block does:** for a CSV file, read each row as a dictionary and pull
out the sentence text.
- `csv.DictReader(f)` — reads the file's rows as dicts keyed by the header names
  (so `row["text"]` works).
- `for row in reader:` — loop over every row.
- `row.get("text", row.get("sentence", ""))` — try the `"text"` column; if it's
  missing, try `"sentence"`; if that's missing too, use `""`. `.get(key, default)`
  avoids a crash when a column doesn't exist.
- `.strip()` — trim surrounding whitespace.
- `if text: sentences.append(text)` — keep only non-empty text.

```python
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        sentences.append(item.strip())
                    elif isinstance(item, dict):
                        text = item.get("text", item.get("sentence", ""))
                        if text:
                            sentences.append(text.strip())
```
**▸ What this block does:** for a JSON file, load it and handle two possible shapes —
a plain list of strings, or a list of objects with a `text`/`sentence` field.
- `json.load(f)` — parse the file into Python objects (lists/dicts).
- `isinstance(data, list)` — check the top level is a list (`isinstance(x, T)` = "is
  `x` of type `T`?").
- `for item in data:` — loop over each entry.
- `if isinstance(item, str):` — if the entry is already a plain string, keep it
  (stripped).
- `elif isinstance(item, dict):` — otherwise if it's an object, pull its
  `text`/`sentence` field the same way as the CSV branch, and keep it if non-empty.

```python
        sentences = [s for s in sentences if len(s.split()) >= 4]
        sentences = [s for s in sentences if len(s.encode("utf-8")) <= 1024]

        logger.info(f"Loaded {len(sentences)} clean sentences from {filepath}")
        return sentences
```
**▸ What this block does:** apply the manuscript's filters, log the result, and
return the surviving sentences.
- `len(s.split()) >= 4` — keep only sentences with word count ≥ 4 (first filter).
- `len(s.encode("utf-8")) <= 1024` — keep only sentences ≤ 1024 bytes (second
  filter).
- `logger.info(f"Loaded {len(sentences)} clean sentences from {filepath}")` — a
  progress log with the final count and source file (e.g.
  `Loaded 41230 clean sentences from data/clean.txt`). Observability only.
- `return sentences` — hand back the filtered list of clean sentences.

### Method `load_gold_standard`

**▸ What this method does (whole function):** loads the human-annotated
`(noisy, clean)` pairs from a CSV/JSON file, tolerating different column names, and
returns them as two parallel lists.

```python
    def load_gold_standard(self, filepath: str) -> Tuple[List[str], List[str]]:
        path = Path(filepath)
        noisy_texts = []
        clean_texts = []
```
**▸ What this block does:** wrap the path and start two empty result lists (kept in
lockstep — index `i` in each is the same pair).

```python
        if path.suffix == ".csv":
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
```
**▸ What this block does:** open a CSV and read it row by row as dicts (same
`DictReader` pattern as `load_clean_corpus`).

```python
                    noisy = row.get("noisy", row.get("input", "")).strip()
                    clean = row.get("clean", row.get("target", row.get("normalized", ""))).strip()
```
**▸ What this block does:** pull the noisy and clean text, tolerating several
possible column names.
- `row.get("noisy", row.get("input", ""))` — try `"noisy"`, then `"input"`, then
  `""`. So a file with a `noisy` column *or* an `input` column both work.
- `row.get("clean", row.get("target", row.get("normalized", "")))` — try `"clean"`,
  then `"target"`, then `"normalized"`, then `""`.
- `.strip()` — trim whitespace on each.

```python
                    if noisy and clean:
                        noisy_texts.append(noisy)
                        clean_texts.append(clean)
```
**▸ What this block does:** keep the pair only if **both** halves are non-empty (an
empty string is "falsy", so `if noisy and clean` skips half-blank rows), appending
each to its list.

```python
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                noisy = item.get("noisy", item.get("input", "")).strip()
                clean = item.get("clean", item.get("target", item.get("normalized", ""))).strip()
                if noisy and clean:
                    noisy_texts.append(noisy)
                    clean_texts.append(clean)
```
**▸ What this block does:** the JSON branch does the same thing, reading each
object's fields with the same fallback column names.

```python
        logger.info(f"Loaded {len(noisy_texts)} gold standard pairs from {filepath}")
        return noisy_texts, clean_texts
```
**▸ What this block does:** log the count, then return the two parallel lists.
- `logger.info(f"Loaded {len(noisy_texts)} gold standard pairs from {filepath}")` —
  progress log (e.g. `Loaded 15000 gold standard pairs from data/gold.csv`).
- `return noisy_texts, clean_texts` — hand back both lists as a tuple
  `(noisy, clean)`, index-aligned.

### Method `clean_text`

**▸ What this method does (whole function):** light text cleaning for privacy and
tidiness — anonymize @-mentions, replace URLs, and collapse repeated spaces.

```python
    def clean_text(self, text: str) -> str:
        text = _replace_mentions(text)
        text = _replace_urls(text)
        text = " ".join(text.split())
        return text.strip()
```
**▸ line by line:**
- `_replace_mentions(text)` — turn `@username` into `@ANON` (see the helper at the
  bottom of the file).
- `_replace_urls(text)` — turn links into `<URL>`.
- `" ".join(text.split())` — `text.split()` breaks on **any** run of whitespace and
  drops the gaps; joining with single spaces collapses multiple spaces/tabs/newlines
  into one.
- `.strip()` — trim the ends. Returns the cleaned string.

**Worked example**
```
clean_text("@juan   check    https://x.co/ab   grabe")
  → "@ANON check <URL> grabe"
```

### Method `generate_synthetic_pairs`

**▸ What this method does (whole function):** builds the ~1M Stage-1 examples by
running the noise generator over the clean corpus enough times to hit the target
size, computing n\* for each.

```python
    def generate_synthetic_pairs(self, clean_sentences, target_size=1_000_000):
        noisy_texts = []
        clean_texts = []
        noise_levels = []
        passes = max(1, target_size // len(clean_sentences))
        remainder = target_size % len(clean_sentences)
```
**▸ What this block does:** start three empty parallel lists, then figure out how
many full passes over the corpus are needed plus the leftover.
- `noisy_texts = []` / `clean_texts = []` / `noise_levels = []` — the three output
  lists, kept in lockstep (same index = same example).
- `1_000_000` — Python lets you put `_` in numbers for readability (= 1,000,000).
- `target_size // len(clean_sentences)` — `//` is **integer division**; how many
  whole passes reach the target. `max(1, ...)` ensures at least one pass.
- `target_size % len(clean_sentences)` — `%` is **modulo**, the leftover count.

```python
        logger.info(
            f"Generating ~{target_size} synthetic pairs "
            f"({passes} passes + {remainder} extra)"
        )
```
**▸ What this block does:** print a progress message (via the shared logger from
`logging_utils.py`) announcing how much data is about to be generated — e.g.
`Generating ~1000000 synthetic pairs (25 passes + 12345 extra)`. Purely
observability; it doesn't change the data.
- `logger.info(...)` — emit an INFO-level log line.
- `f"...{passes}...{remainder}..."` — an f-string filling in the computed numbers.

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
            noisy = self.noise_gen.apply_noise(clean)
            n_star = compute_noise_level(noisy, clean)
            noisy_texts.append(noisy)
            clean_texts.append(clean)
            noise_levels.append(n_star)

        logger.info(f"Generated {len(noisy_texts)} synthetic pairs")
        return noisy_texts, clean_texts, noise_levels
```
**▸ What this block does:** the whole-passes loop above lands *just under* the target
(because it only does complete passes); this tops up with `remainder` more
randomly-chosen sentences to hit the exact target, corrupting and scoring each the
same way, then returns the three parallel lists.
- `self.rng.sample(list, k)` — pick `k` **distinct** random items from the corpus.
- `min(remainder, len(clean_sentences))` — never ask for more than exist.
- The loop body is identical to the main loop (make noisy → compute n\* → append to
  all three lists).
- `logger.info(f"Generated {len(noisy_texts)} synthetic pairs")` — a progress log
  reporting the final count (e.g. `Generated 1000000 synthetic pairs`).
  `len(noisy_texts)` is how many pairs ended up in the list. Observability only.
- `return noisy_texts, clean_texts, noise_levels` — hand back the three parallel
  lists as a tuple (the caller, usually `preprocessing`/a script, then splits them).

**Worked walkthrough** (tiny numbers)
```
generate_synthetic_pairs(clean_sentences=[A, B, C], target_size=7)

passes    = max(1, 7 // 3) = 2      # two full passes over [A, B, C] = 6 pairs
remainder = 7 % 3           = 1      # 1 more needed

pass 0: corrupt A, B, C  → 3 pairs
pass 1: corrupt A, B, C  → 3 pairs   (different noise; RNG has advanced)
                                     → 6 pairs so far
extra: rng.sample([A,B,C], 1) = [B]  → corrupt B → 1 pair
                                     → 7 pairs total

returns (noisy_texts[7], clean_texts[7], noise_levels[7])
```
(At the real scale the corpus is tens of thousands of sentences and `target_size`
is 1,000,000, so `passes` is a few dozen.)

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
```
**▸ What this block does:** carve the shuffled position list into the train slice and
the val slice.
- `indices[:train_end]` — the first `train_end` shuffled positions → the training
  set's positions.
- `indices[train_end:val_end]` — the next chunk → the validation set's positions.

```python
        splits["train"] = (
            [noisy_texts[i] for i in train_idx],
            [clean_texts[i] for i in train_idx],
            [noise_levels[i] for i in train_idx],
        )
```
**▸ What this block does:** gather the actual data for the training split into a
tuple of three parallel lists.
- `[noisy_texts[i] for i in train_idx]` — pick the noisy texts at those shuffled
  positions. The same `train_idx` is reused for all three lists, so index alignment
  is preserved (position `i` is the *same pair* across noisy/clean/n\*).

```python
        splits["val"] = (
            [noisy_texts[i] for i in val_idx],
            [clean_texts[i] for i in val_idx],
            [noise_levels[i] for i in val_idx],
        )
```
**▸ What this block does:** the identical gather for the validation split, using
`val_idx`.

```python
        if test_ratio > 0:
            test_idx = indices[val_end:]
            splits["test"] = (
                [noisy_texts[i] for i in test_idx],
                [clean_texts[i] for i in test_idx],
                [noise_levels[i] for i in test_idx],
            )
```
**▸ What this block does:** if a test split was requested, everything after `val_end`
becomes the test set (the held-out data used only for final scoring), gathered the
same way.
- `if test_ratio > 0:` — synthetic data uses no test split (it passes `0.0`), so
  this is skipped for Stage-1 data; gold data passes `0.1`, so it runs.
- `indices[val_end:]` — all remaining positions.

```python
        for split_name, (noisy, clean, nl) in splits.items():
            logger.info(f"  {split_name}: {len(noisy)} pairs")

        return splits
```
**▸ What this block does:** log the size of each split, then return the dict.
- `splits.items()` — iterate the dict as `(key, value)` pairs; here the value is
  itself a tuple `(noisy, clean, nl)`, which is **unpacked** inline into three
  variables in the loop header.
- `logger.info(f"  {split_name}: {len(noisy)} pairs")` — one line per split, e.g.
  `train: 12000 pairs`, `val: 1500 pairs`, `test: 1500 pairs`.
- `return splits` — hand back the dict keyed `"train"`, `"val"`, and optionally
  `"test"`, each holding a `(noisy, clean, noise_levels)` tuple — exactly the shape
  `NormalizationDataset` expects.

**Worked walkthrough** (10 examples, 80/10/10)
```
split_data(noisy, clean, n_stars, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
# with 10 examples:

n = 10
indices = [0,1,2,3,4,5,6,7,8,9]
rng.shuffle(indices)  → e.g. [3, 7, 0, 9, 2, 5, 1, 8, 4, 6]   (seeded → same every run)

train_end = int(10 * 0.8) = 8
val_end   = 8 + int(10 * 0.1) = 9

train_idx = indices[:8]   = [3, 7, 0, 9, 2, 5, 1, 8]   → 8 examples
val_idx   = indices[8:9]  = [4]                         → 1 example
test_idx  = indices[9:]   = [6]                         → 1 example

splits["train"] = (noisy at [3,7,0,9,2,5,1,8], clean at same, n* at same)
splits["val"]   = (noisy at [4], ...)
splits["test"]  = (noisy at [6], ...)
```
Because the shuffle is seeded (seed 42), the exact same sentences land in the same
split every run — which is what makes results reproducible.

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
