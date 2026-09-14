# Session handoff — TAHIMIK documentation & defense prep

**For:** a fresh Claude Code session continuing this work.
**Date of handoff:** 2026-08-28.
**Repo:** `C:\Users\aimee\Downloads\tahimik-tool` (thesis repo; read
`docs/agents/README.md` first).

Read this whole file before acting. It captures who the user is, what was built,
the exact documentation standard to keep consistent, the repo's rules, and what's
likely next.

---

## 1. Who the user is (context that shapes every answer)

- **Undergraduate CS thesis student**, Group 9, Polytechnic University of the
  Philippines. Email `eimzc.mng@gmail.com`; git author `eimimimi`; GitHub `082131`.
- **Has been away from the thesis for months** and forgot much of the detail. Came
  in saying the specs/speckit docs felt "vague and jargony."
- **Relative beginner in ML/deep-learning code.** Explicitly asked for
  beginner-level explanations: every technical term and abbreviation spelled out,
  line-by-line, with worked examples and how each piece connects to the whole system.
- **Preparing for a "tool defense"** in front of **industry-developer panelists**,
  possibly including a **live-debugging** session. Nervous about not knowing the
  system's flow.
- Writes informally (lowercase, some typos); wants substance, not hand-holding
  about tone. Responds well to honesty about weaknesses.

**Tone that worked:** direct, concrete, honest about gaps/risks, beginner-friendly
without being condescending. Validate correct instincts, then add the nuance.

---

## 2. The project in one paragraph

**TAHIMIK** normalizes noisy Filipino/Taglish social-media text (e.g. `slmt` →
`salamat`). Its contribution is **noise-adaptive byte compression**: a small
"noise estimator" predicts how noisy each sentence is, and a "delete gate"
(borrowed from **MrT5**, built on **ByT5**) then compresses clean sentences hard
(for speed) and preserves noisy ones (for accuracy). It's proven as a **controlled
comparison of three models** — ByT5 (no compression, accuracy ceiling), MrT5
(fixed 50% compression), TAHIMIK (adaptive) — under identical config/data/trainer,
so only the compression mechanism varies.

**Critical status fact:** the **implementation and test harness are complete, but
no model is trained** — the gold dataset is still being annotated. So the live tool
(`backend/app.py`) returns **HTTP 503** for `/normalize` until a checkpoint exists.
This is by design, not a bug, and it is the user's biggest **demo risk** for the
defense.

---

## 3. What THIS session produced (all merged to `main`)

Two deliverables, both **documentation only** (no source/behavior changes):

### A. `docs/src-explained/` — beginner line-by-line walkthrough of `src/`
One Markdown file per subfolder, **ordered by data flow**:

| File | Covers |
|---|---|
| `README.md` | Index + concepts primer + **shared glossary** of every abbreviation/term |
| `01-utils.md` | `byte_encoding`, `logging_utils` |
| `02-data.md` | `noise_generator → noise_label → dataset → preprocessing` |
| `03-models.md` | `byt5_baseline → delete_gate → fixed_compression → noise_estimator → noise_adaptive` |
| `04-training.md` | `losses`, `trainer` |
| `05-evaluation.md` | `metrics → efficiency → statistical_tests → annotation` |

### B. `docs/DEFENSE-PREP.md` — tool-defense strategy guide
Two system flows, honest architecture/coupling assessment, design-decision
rationales, live-debugging playbook, how to demo with no checkpoint, anticipated
panel Q&A, and how to own the known `FINDINGS.md` gaps.

### Git state
- Merged via PR **#9** (`docs/src-walkthrough` → `main`), merge commit `289f528`,
  branch deleted. `main` now contains all the docs.
- **Untouched, still uncommitted in the working tree** (intentionally left out):
  `M src/training/losses.py` and `?? .claude/launch.json`. Do **not** commit these
  without asking the user — the `losses.py` change predates this session and its
  intent is unconfirmed.

---

## 4. THE DOCUMENTATION STANDARD (keep this exact format if extending)

The user iterated on the format five times. The final, agreed structure for **each
source file** in a walkthrough doc is:

1. **Role** — one line.
2. **Where this fits in TAHIMIK** — its real system job, who calls it, and
   **what breaks without it** (be honest — e.g. `byte_encoding.py` is off the hot
   path and nothing breaks; say so). This section was the user's key ask: earlier
   drafts explained code but felt "disconnected from the system."
3. **Inputs / Outputs / Used by / Connects to** — bullets.
4. **Terms & abbreviations in this file** — a table covering **abbreviations AND
   any technical word a beginner may not know** (e.g. `EMA`, `broadcasting`,
   `context manager`, not just acronyms). Expand on first inline use too.
5. Per function/method:
   - **▸ What this method does (whole function)** summary.
   - **Variables at a glance** table (name, type/**shape**, holds).
   - **Example** — concrete input → output (the user explicitly wanted these).
   - Line-by-line: **every code block opens with `▸ What this block does`** (a
     group summary), then breaks down every line, variable, method, and operator.

Also: order files **by data flow**, not alphabetically. Flag known `FINDINGS.md`
gaps inline where relevant (e.g. the one-tailed bootstrap in `05`, the
`L_attn_reg` deviation in `04`).

**If asked to extend this to other folders** (`configs`, `scripts`, `backend`,
`tests`, `frontend`, `annotation-platform`) — apply the identical standard.

---

## 5. Repo rules you MUST follow (from `workflow.md` / `contributing.md`)

- **Never commit to `main`.** Branch as `<type>/<short-description>` (e.g.
  `docs/...`, `fix/...`). `main` is **not** branch-protected (free plan), so the
  rule holds by discipline only.
- **Keep the trailer** on assisted commits:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- PR bodies end with the Claude Code trailer.
- **Hyperparameters live in `configs/`**, never inline. **Seed is 42.**
- Features go through the **spec loop** (interrogation → spec → build); model code
  (`src/models/`, `src/training/`) is held to this hardest. Don't invent answers to
  design questions — ask the user.
- `python -m pytest tests/ -v` must pass before merging code.
- No secrets/datasets/checkpoints in git.
- **`specs/FINDINGS.md`** lists every known manuscript-vs-code gap — check it before
  "fixing" anything that looks wrong; it may be intentional/already-tracked.
- Platform: **Windows + PowerShell** primary; a **Bash tool** (Git Bash) is also
  available. Use `gh` for GitHub ops (authed as `082131`).

---

## 6. System facts a new Claude should know (to answer fast)

**Two flows** (keep them separate when explaining):
- **Offline (experiment):** `preprocessing` loads/cleans → `noise_generator` makes
  noise → `noise_label` computes `n*` → `dataset` tensorizes → `trainer` runs
  Stage 1 (synthetic ~1M) then Stage 2 (gold ~15K) → saves `best_stage2.pt` →
  `evaluation` scores accuracy/efficiency/significance.
- **Online (tool):** `frontend` POSTs to `backend/app.py` `/normalize` → lazy-loads
  the chosen variant's checkpoint → `model.generate()` (hard deletion + beam
  search) → returns normalized text. **Returns 503 today (no checkpoint).**

**Key variables:** `n` = *predicted* noise (`noise_scores`); `n*` = *ground-truth*
noise (`noise_level`, edit-distance ratio). `cn` = learned shift strength; `navg` =
running noise average (EMA, momentum 0.99); `k = -30` bounds gate scores;
`d_max = 0.5`; gate sits after encoder **layer 3**.

**Loss:** `L = L_CE + w_rate·L_rate + w_attn_reg·L_attn_reg + L_NE` (terms switch on
per variant). The estimator is trained **only** by L_NE; `n` is `.detach()`ed
elsewhere (gradient isolation — a tested guarantee).

**Architecture notes (honest):** high cohesion, config-driven, "one contract, three
variants." Two coupling seams to name proactively: (1) models reach into Hugging
Face internals (version-fragile), (2) the model↔loss link is an implicit output-dict
contract, not a typed object. `byt5_baseline` returns slightly non-uniform output
keys (`FINDINGS.md` #2).

**Top FINDINGS gaps to own** (all in `specs/FINDINGS.md`): #1–2 bootstrap is
one-tailed & ignores the CI; #4 config runs `byt5-small` (paper says `base`); #5
seeded but not bit-reproducible; #7 `cn` sign unconstrained (could silently invert
adaptivity); #8 `L_attn_reg` deviates from MrT5's paper.

---

## 7. Likely next tasks (what the user may ask for)

- **Extend the walkthrough** to `configs/`, `scripts/`, `backend/`, `tests/`, or the
  two frontends (`frontend/`, `annotation-platform/`) — same standard as §4.
- **Train a tiny demo checkpoint** so the tool responds for the defense: run
  Stage-1-only on a small synthetic set with `byt5-small` to produce a real
  `best_stage1.pt` (the backend accepts it as a fallback). This is the single
  highest-leverage way to defuse the 503 demo risk. Would need `--clean_corpus`
  data and compute (Colab). Confirm scope/data before doing this.
- **Mock-defense Q&A drills** from `docs/DEFENSE-PREP.md` §7.
- **A diagram** of the data→model→train→eval flow (they've been offered one).
- Possibly deal with the stray `src/training/losses.py` edit — **ask first** what it
  is; show the diff before any action.

---

## 8. Pointers

- Study docs: `docs/src-explained/README.md` (start), then `01`–`05`.
- Defense strategy: `docs/DEFENSE-PREP.md`.
- Known gaps: `specs/FINDINGS.md`. Decisions: `specs/DECISIONS.md`.
- Project rules: `docs/agents/workflow.md`, `docs/agents/contributing.md`,
  `docs/WHY.md`,
  `.specify/memory/constitution.md`.
- Code entry points: `scripts/train.py`, `scripts/run_experiment.py`,
  `backend/app.py`.

---

## 9. Working style that fit this user

- Actually **read the code** before explaining — they value accuracy over speed.
- **Beginner-first**: expand every term, give examples, connect to the system.
- **Be honest** about weaknesses, the 503 demo problem, and the FINDINGS gaps —
  they found this more useful than reassurance.
- **One big doc/folder per turn** kept quality high; don't try to do everything at
  once.
- Confirm scope on large/ambiguous asks, but **don't over-ask** — they got mildly
  frustrated with too many clarifying questions; prefer sensible defaults + a note.
