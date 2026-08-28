# `src/utils/` — the shared toolbox (exhaustive line-by-line)

**Flow position:** used by *every* other folder; the foundation they borrow from.

**Files (dependency order):**
1. `__init__.py` — re-exports
2. `byte_encoding.py` — text ↔ ByT5 byte IDs
3. `logging_utils.py` — progress messages

> **How to read this doc:** each file starts with **Where this fits in TAHIMIK**
> (its real job in the system + what breaks without it). Then, per method, a
> *Variables at a glance* table, an **Example** (concrete input → output), and a
> line-by-line pass where every code block opens with **▸ What this block does**.

---

## `__init__.py`

**Full source:**
```python
from src.utils.byte_encoding import ByteEncoder
from src.utils.logging_utils import setup_logger
```
**▸ What this block does:** lifts two names up to the package level so other code
can import them from `src.utils` directly instead of naming the exact file.

**Where this fits in TAHIMIK:** this is plumbing. When any file writes
`from src.utils import setup_logger`, Python runs this `__init__.py` first, which
makes `ByteEncoder` and `setup_logger` available under the short package name. It
contributes *convenience and a stable import path* — nothing computational.

- **Inputs:** none (runs automatically when `src.utils` is imported)
- **Outputs:** two importable names — `ByteEncoder`, `setup_logger`

---

## `byte_encoding.py` — text ↔ ByT5 byte IDs

**Role:** Convert between human text and the numeric byte-IDs ByT5 expects, and
pad/measure sequences.

### Where this fits in TAHIMIK
The entire project rests on one fact: **ByT5 does not read words — it reads raw
bytes.** Every sentence, clean or noisy, has to become a list of byte-IDs before
any model can touch it. This file is the **plain-Python reference** of exactly how
that conversion works: byte value `b` → ID `b+3`, with `0=pad`, `1=eos`, `2=unk`.

**Honest hot-path note (this is why it can feel "disconnected"):** the *live*
training/inference pipeline does **not** import this file. It tokenizes through
Hugging Face's ByT5 tokenizer inside [`dataset.py`](02-data.md) and `backend/app.py`,
which performs the identical byte scheme with extra machinery. So **nothing in the
running system breaks if this file is deleted.**

**Then why does it matter to you?** Three concrete reasons:
1. **It's the 20-line answer to "how does text become model input?"** — far clearer
   than pointing a panel at the tokenizer's internals. The byte scheme it documents
   is the *same* one the tokenizer uses and the *same* grain `noise_label.py` uses
   to compute n\*.
2. **It's usable standalone** for quick encode/decode in tests or a debugging
   session without loading a 1.2 GB model.
3. **It grounds the whole "byte-level" design decision** you must defend (Filipino
   noise is sub-word, so bytes beat word tokens).

- **Used by:** utility / reference. Not on the training or inference hot path.
- **Connects (conceptually) to:** `dataset.py` (which does this via HF), and
  `noise_label.py` (which also works at the byte grain).

**Terms & abbreviations in this file** *(abbreviations + any technical word you
may not know)*
| Term | Full form / plain meaning |
|---|---|
| UTF-8 | Unicode Transformation Format, 8-bit — standard rule mapping characters → bytes |
| ID | identifier — a number standing for a byte/token |
| EOS | end of sequence — the token (ID 1) marking where text stops |
| pad / padding | filler (ID 0) added so all sequences share a length |
| `str` / `int` / `bytes` | string / integer / byte-string (a sequence of byte values 0–255) |
| decorator | a `@tag` above a function that modifies it (e.g. `@staticmethod`) |
| static method | a method that needs no object — called as `Class.method(...)` |
| constant | a fixed named value that never changes (e.g. `_EOS_ID = 1`) |
| private (`_name`) | a naming convention meaning "internal to this file, don't use outside" |
| slicing | grabbing part of a list with `[start:stop]` (e.g. `[:5]` = first 5) |
| list comprehension | compact "build a list by looping": `[f(x) for x in items]` |
| method | a function that belongs to a class |
| tuple | an ordered, fixed group of values, e.g. `(padded, mask)` |
| concatenate | join two lists/strings end to end with `+` |

### Module-level code

```python
from typing import List
```
**▸ What this block does:** brings in the type-hint helper so the code can label
lists of integers.
- `typing` — a built-in module providing **type hints** (labels describing what
  kind of value is expected). They document intent and let editors catch bugs;
  they do not change how the program runs.
- `List` — the hint for "a list." Used later as `List[int]` = "a list of integers."

```python
_BYT5_OFFSET = 3
_PAD_ID = 0
_EOS_ID = 1
```
**▸ What this block does:** defines ByT5's fixed numbering rules once, so every
method uses the same constants instead of magic numbers.

These three numbers *are* the ByT5 byte contract — the same one the real tokenizer
and the models assume. The leading `_` marks them private to this file.

| Variable | Value | Holds / meaning |
|---|---|---|
| `_BYT5_OFFSET` | `3` | How much every real byte's ID is shifted up, to leave room for the 3 special IDs (0,1,2) |
| `_PAD_ID` | `0` | The ID used for **padding** (filler that makes sequences equal length) |
| `_EOS_ID` | `1` | The ID that marks **E**nd **O**f **S**equence (where real text stops) |

```python
class ByteEncoder:
```
**▸ What this block does:** opens the class that groups all the encode/decode
helpers under one name. All methods are `@staticmethod`, so the class is just a
labeled box — you never create a `ByteEncoder()` object.

---

### Method `encode` — string → list of byte IDs

**▸ What this method does (whole function):** takes a string, converts it to raw
UTF-8 bytes, trims it to fit the length limit, shifts every byte by +3 into ByT5's
ID space, and appends the end-of-sequence marker — returning a ready-to-feed list
of IDs. **This is the "text → numbers" step every model input goes through.**

**Example**
```
encode("Hi")
  "Hi"                → bytes [72, 105]         (H=72, i=105 in UTF-8)
  + offset 3          → [75, 108]
  + EOS (1)           → [75, 108, 1]
returns [75, 108, 1]
```

**Variables at a glance**
| Variable | Type | Holds |
|---|---|---|
| `text` (param) | `str` | The input string to convert |
| `max_length` (param) | `int` (default 1024) | Max number of IDs to output |
| `raw_bytes` | `bytes` | The UTF-8 byte values of `text` |
| `truncated` | `bytes` | `raw_bytes` cut to fit `max_length` |
| `byte_ids` | `List[int]` | The final ID list (offset applied + EOS) |

```python
    @staticmethod
    def encode(text: str, max_length: int = 1024) -> List[int]:
```
**▸ What this block does:** declares the function and its inputs/outputs.
- `@staticmethod` — a **decorator** marking this as a method that needs no `self`
  (no object). You call it as `ByteEncoder.encode("hi")`.
- `text: str` — first parameter, a string.
- `max_length: int = 1024` — an integer that **defaults to 1024** if not supplied.
  1024 is the project's byte cap (≈170 words), set in the configs.
- `-> List[int]` — the **return type hint**: returns a list of integers.

```python
        raw_bytes = text.encode("utf-8")
```
**▸ What this block does:** turns the text into its raw bytes.
- `text.encode("utf-8")` — built-in string method: convert to UTF-8 bytes.
  `"grabe"` → values `[103, 114, 97, 98, 101]`.
- `raw_bytes` — holds those bytes (type `bytes`, behaves like a list of ints 0–255).

```python
        truncated = raw_bytes[: max_length - 1]
```
**▸ What this block does:** caps the length, leaving one slot for the EOS marker.
- `[: max_length - 1]` — **slicing**: keep the first `max_length - 1` bytes.
- The `-1` **reserves a slot** for the EOS token added below. Without it, a
  max-length input would have no room for EOS and the model wouldn't see where the
  sentence ends.

```python
        byte_ids = [b + _BYT5_OFFSET for b in truncated]
```
**▸ What this block does:** shifts every byte into ByT5's ID range (avoiding the
reserved 0/1/2).
- **list comprehension**: "for each `b` in `truncated`, compute `b + 3`."
- `b` — one byte value at a time. `byte_ids` — the shifted list.

```python
        byte_ids.append(_EOS_ID)
```
**▸ What this block does:** marks the end of the sequence with the EOS token (1) so
the model knows the input stopped here.

```python
        return byte_ids
```
**▸ What this block does:** hands the finished ID list back to the caller.

---

### Method `decode` — list of byte IDs → string

**▸ What this method does (whole function):** the reverse of `encode` — turns model
output IDs back into readable text. In the running system the tokenizer does this
(when the backend returns normalized text); here it's the reference version.

**Example**
```
decode([75, 108, 1])
  skip 1 (EOS, ≤2)    → drop it
  [75, 108] − 3       → [72, 105]
  bytes → text        → "Hi"
returns "Hi"
```

**Variables at a glance**
| Variable | Type | Holds |
|---|---|---|
| `byte_ids` (param) | `List[int]` | The IDs to convert back |
| `raw_bytes` | `list[int]` | The recovered original byte values |
| `bid` | `int` | One ID at a time (loop variable) |

```python
    @staticmethod
    def decode(byte_ids: List[int]) -> str:
        raw_bytes = []
        for bid in byte_ids:
            if bid <= 2:
                continue
            raw_bytes.append(bid - _BYT5_OFFSET)
        return bytes(raw_bytes).decode("utf-8", errors="replace")
```
**▸ What this block does:** walk the IDs, drop special tokens, undo the +3 offset,
and rebuild text safely.
- `raw_bytes = []` — collector for recovered byte values.
- `for bid in byte_ids:` — loop over each ID (`bid`).
- `if bid <= 2: continue` — IDs 0/1/2 are pad/eos/unk (not real text); `continue`
  skips them.
- `raw_bytes.append(bid - _BYT5_OFFSET)` — undo the offset to recover the byte.
- `bytes(raw_bytes)` — list of values → a `bytes` object.
- `.decode("utf-8", errors="replace")` — bytes → string; `errors="replace"` inserts
  a `�` for any invalid bytes instead of crashing (a model mid-training can emit
  gibberish, so this must not throw).

---

### Method `pad_sequence` — fix length + build a mask

**▸ What this method does (whole function):** makes a sequence exactly `max_length`
long and produces the matching **attention mask** (1 = real, 0 = padding). This is
the same shape the models need — every batch must be rectangular, and the mask is
what tells the model to ignore the filler.

**Example**
```
pad_sequence([75, 108, 1], max_length=5)
  padded = [75, 108, 1, 0, 0]     (two pad-IDs appended)
  mask   = [ 1,   1, 1, 0, 0]     (last two are padding)
returns ([75,108,1,0,0], [1,1,1,0,0])
```

**Variables at a glance**
| Variable | Type | Holds |
|---|---|---|
| `byte_ids` (param) | `List[int]` | The IDs to pad (already includes EOS) |
| `max_length` (param) | `int` | Target fixed length |
| `seq_len` | `int` | Current length of `byte_ids` |
| `pad_len` | `int` | How many filler slots to add |
| `padded` | `List[int]` | The length-fixed ID list |
| `mask` | `List[int]` | 1 = real position, 0 = padding |

```python
    @staticmethod
    def pad_sequence(byte_ids: List[int], max_length: int) -> tuple[List[int], List[int]]:
        seq_len = len(byte_ids)
        if seq_len >= max_length:
            padded = byte_ids[:max_length]
            mask = [1] * max_length
        else:
            pad_len = max_length - seq_len
            padded = byte_ids + [_PAD_ID] * pad_len
            mask = [1] * seq_len + [0] * pad_len
        return padded, mask
```
**▸ What this block does:** trim-or-pad to a fixed length and emit a parallel mask.
- `seq_len = len(byte_ids)` — current length.
- **too long / exact (`>=`):** slice to `max_length`; `[1] * max_length` (`list *
  int` repeats) makes an all-1s mask — every kept position is real.
- **too short (`else`):** `pad_len` = missing slots; `byte_ids + [_PAD_ID]*pad_len`
  (`list + list` concatenates) appends that many pad-IDs; the mask is 1s for real
  bytes then 0s for padding.
- `return padded, mask` — two values → a **tuple**. The mask half is exactly the
  `attention_mask` you'll see everywhere in the models.

---

### Method `byte_length` — how many bytes a string takes

**▸ What this method does (whole function):** report a string's size in **bytes,
not characters**. TAHIMIK's 1024-byte limit and the n\* calculation are both
byte-based, so this is the "how big is this really?" helper.

**Example**
```
byte_length("grabe 😭")
  "grabe " = 6 bytes,  "😭" = 4 bytes
returns 10
```

```python
    @staticmethod
    def byte_length(text: str) -> int:
        return len(text.encode("utf-8"))
```
**▸ What this block does:** encode to UTF-8 bytes and count them.
- Filipino text with emojis/accents uses multiple bytes per visible character
  (`😭` = 4), which is why byte length ≠ character length and why the project
  measures limits in bytes.

---

## `logging_utils.py` — progress messages

**Role:** Build a configured **logger** — a disciplined `print()` replacement that
adds timestamps, severity levels, and optional saving to a file.

### Where this fits in TAHIMIK
This file is TAHIMIK's **eyes during long-running jobs.** Training a variant is a
multi-hour job (often on Colab); you cannot sit and inspect variables. Instead,
every major component narrates its progress through one of these loggers:

- `preprocessing.py` (`tahimik.data`) logs "Loaded 40,000 clean sentences",
  "Generated 1,000,000 synthetic pairs", split sizes.
- `trainer.py` (`tahimik.trainer`) logs each epoch's train/val loss, the component
  losses (`l_ce`, `l_rate`, `l_ne`), and every "Saved checkpoint (val_loss=…)".
- `evaluation` and `statistical_tests.py` log metric results and each test's
  p-value.

So its contribution to the system is **observability**: the running log stream is
how you know training is healthy (is `l_ne` going down? is a checkpoint being
saved?), and, when something crashes at hour three, *where* it stopped. The
**named** loggers (`tahimik.data`, `tahimik.trainer`, `tahimik.stats`) let you
filter output by component — handy in a debugging session.

**What breaks without it:** no crash — but training and evaluation would run
**silently**. You'd have no visibility into loss curves, no record of checkpoint
saves, and no breadcrumb showing where a failure happened. Not a model bug, but
you'd be flying blind. That's a real cost on a thesis where you must *report* what
happened.

**Example of what it actually prints** (a real line from a training run):
```
[2026-08-28 14:03:01] tahimik.trainer | INFO |   [stage2] Epoch 3/10 (512.4s) — train_loss=0.8130 val_loss=0.7902
[2026-08-28 14:03:01] tahimik.trainer | INFO |     l_ne: train=0.0192 val=0.0210
[2026-08-28 14:03:02] tahimik.trainer | INFO |   Saved checkpoint: checkpoints/tahimik_noise_adaptive/best_stage2.pt (val_loss=0.7902)
```

- **Used by:** nearly every module. Pattern: `logger = setup_logger("tahimik.X")`
  at the top of a file, then `logger.info("message")` where progress is reported.
- **Connects to:** the data pipeline, trainer, and evaluation modules.

**Terms & abbreviations in this file** *(abbreviations + any technical word you
may not know)*
| Term | Full form / plain meaning |
|---|---|
| param / arg | parameter / argument — a function input / the value passed for it |
| logger | an object that prints/saves status messages in a disciplined way |
| logging level | a severity filter: DEBUG < INFO < WARNING < ERROR (least → most serious) |
| handler | a *destination* for messages — the screen or a file |
| formatter | the *layout* of a message line (timestamp, name, level, text) |
| stream | a flowing source/sink of data; `stdout` = the console output stream |
| stdout | standard output — the console/screen stream |
| guard clause | an early `if ...: return` that exits before doing extra work |
| `str \| None` | "a string, or nothing" (the `\|` means "or") |
| default value | a value a parameter takes when the caller doesn't supply one |
| observability | being able to see what a running system is doing |
| UTF-8 | Unicode Transformation Format, 8-bit — text encoding for the log file |

### Module-level imports

```python
import logging
import sys
from pathlib import Path
```
**▸ What this block does:** brings in the logging system, console access, and a
file-path helper.
- `logging` — Python's built-in logging system (loggers, handlers, formatters).
- `sys` — system access; used for `sys.stdout` (the console output stream).
- `from pathlib import Path` — convenient file-path handling (joining folders,
  making directories) across operating systems.

### Function `setup_logger`

**▸ What this function does (whole thing):** returns a ready-to-use logger that
prints nicely formatted, timestamped messages to the screen — and optionally also
writes them to a log file — while guarding against being set up twice.

**Example**
```python
logger = setup_logger("tahimik.trainer")      # console only
logger.info("Epoch 1/10")
# prints: [2026-08-28 14:03:01] tahimik.trainer | INFO | Epoch 1/10

logger = setup_logger("tahimik.trainer", log_file="outputs/run.log")
# same line now ALSO appended to outputs/run.log
```

**Variables at a glance**
| Variable | Type | Holds |
|---|---|---|
| `name` (param) | `str` (default `"tahimik"`) | The logger's label (which module) |
| `log_file` (param) | `str \| None` (default `None`) | Optional file path to also write to |
| `level` (param) | `int` (default `logging.INFO`) | Minimum severity to show |
| `logger` | `logging.Logger` | The logger being built/returned |
| `formatter` | `logging.Formatter` | The line layout |
| `console_handler` | `StreamHandler` | Destination = screen |
| `file_handler` | `FileHandler` | Destination = file (optional) |

```python
def setup_logger(
    name: str = "tahimik",
    log_file: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
```
**▸ What this block does:** declares the function and its three optional inputs.
- `name` — the logger's label; each module passes its own (`"tahimik.trainer"`) so
  log lines say where they came from.
- `log_file: str | None = None` — a path to also save to, or nothing (console only).
- `level: int = logging.INFO` — the verbosity; INFO shows normal progress.
- `-> logging.Logger` — returns a configured logger.

```python
    logger = logging.getLogger(name)
    logger.setLevel(level)
```
**▸ What this block does:** fetch (or create) the named logger and set how verbose
it is.
- `logging.getLogger(name)` — Python keeps one logger per name; this returns the
  existing one or makes it. That's why the same module always gets the same logger.

```python
    if logger.handlers:
        return logger
```
**▸ What this block does:** prevent double setup — if this logger already has
destinations, return it as-is so messages don't print twice.
- Important because modules call `setup_logger` at import time; if two files ask
  for `"tahimik.data"`, the second call must not attach a second console handler
  (which would double every line). A **guard clause**.

```python
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
```
**▸ What this block does:** define the shape of every log line — timestamp, module
name, severity, message — which is what makes the output *filterable and
timestamped* (you can see how long an epoch took, and which component spoke).
- `%(asctime)s` etc. are placeholders filled at runtime. Result:
  `[2026-08-28 14:03:01] tahimik.trainer | INFO | Epoch 1/10`.

```python
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
```
**▸ What this block does:** create the screen destination, apply the format, attach
it — so messages appear in the terminal (or the Colab cell output).
- `StreamHandler(sys.stdout)` — writes to the console. `.addHandler(...)` wires it
  to the logger.

```python
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
```
**▸ What this block does:** if a file path was given, ensure its folder exists and
add a second destination so the *same* messages are also saved to disk — giving you
a permanent record of a run (useful when the Colab session closes).
- `Path(log_file).parent.mkdir(parents=True, exist_ok=True)` — create the folder
  (and any missing parents; don't error if present).
- `FileHandler(log_file, encoding="utf-8")` — writes to the file; UTF-8 keeps
  Filipino characters/emojis correct.

```python
    return logger
```
**▸ What this block does:** return the fully configured logger to the caller.
